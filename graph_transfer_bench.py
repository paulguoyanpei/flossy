"""CUDA-graph decode + transfer (single process, no verify).

Captures ONE graph that runs the instrumented decode forward AND the cat that
gathers the recorded intermediate tensors into a staging buffer. Because a CUDA
graph reuses the same memory addresses every replay, the recorded intermediates
sit at fixed addresses; the captured cat reads them, so the transfer's GPU-side
gather is *inside* the graph (no per-step CPU launch). Each step then only:
replay() -> async DMA staging->shm -> argmax logits -> next token.

Goal: see whether making decode GPU-bound (graph) shrinks the transfer overhead.

  python graph_transfer_bench.py --model <path> --batch 16 -n 64 [--transfer] [--direct-dma]
"""
from __future__ import annotations

import argparse
import time

import torch

from capture import FlossyContext, instrument
from common import build_prompt_ids, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="essay about computing history")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("-n", "--max-new-tokens", type=int, default=64)
    ap.add_argument("--transfer", action="store_true", help="capture the cat (transfer gather) into the graph")
    ap.add_argument("--no-graph", action="store_true", help="manual decode without graph (coherence check)")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")

    tok, model = load_model(args.model, "cuda:0")
    ids = build_prompt_ids(tok, args.prompt, "cuda:0")
    if args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    B, prompt_len = ids.shape
    N = args.max_new_tokens
    max_len = prompt_len + N
    dev = "cuda:0"

    # instrument; sink just stashes the step's StepBuffer so we can cat it
    captured = {}
    ctx = FlossyContext(role="prover", sink=lambda step, buf: captured.__setitem__("buf", buf))
    instrument(model, ctx)

    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=max_len, device=dev, dtype=torch.float16)

    @torch.inference_mode()
    def forward(cur, pos):
        return model(input_ids=cur, cache_position=pos, past_key_values=cache, use_cache=True).logits

    # ---- prefill (eager) ----
    with torch.inference_mode():
        logits = forward(ids, torch.arange(prompt_len, device=dev))
    next_tok = logits[:, -1:].argmax(-1)  # (B,1)
    gen = [next_tok]

    # static decode buffers
    static_tok = next_tok.clone()
    static_pos = torch.tensor([prompt_len], device=dev)

    def staging_size():
        buf = captured["buf"]
        return sum(buf.fetch(n).numel() for n in buf.order)

    if args.no_graph:
        # ---- manual decode, no graph (coherence check) ----
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(N - 1):
            with torch.inference_mode():
                logits = forward(static_tok, static_pos + i)
            static_tok = logits[:, -1:].argmax(-1)
            gen.append(static_tok)
        torch.cuda.synchronize(); dt = time.time() - t0
        text = tok.decode(torch.cat(gen, dim=1)[0].tolist(), skip_special_tokens=True)
        print(f"[no-graph] {dt/ (N-1)*1000:.2f} ms/step  ({B*(N-1)/dt:.0f} tok/s)")
        print("generation:", repr(text[:200]))
        return

    # ---- capture graph of (forward + optional cat) ----
    # warmup a few decode steps on a side stream for allocator stability
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            with torch.inference_mode():
                logits = forward(static_tok, static_pos)
    torch.cuda.current_stream().wait_stream(s)

    staging = torch.empty(staging_size(), dtype=torch.float16, device=dev) if args.transfer else None
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g):
            static_logits = forward(static_tok, static_pos)
            if args.transfer:
                buf = captured["buf"]
                ordered = [buf.fetch(n).reshape(-1) for n in buf.order]
                torch.cat(ordered, out=staging)

    # pinned shm-like target for the DMA (single-process bench: just a pinned buffer)
    pinned = torch.empty(staging.numel(), dtype=torch.float16, pin_memory=True) if args.transfer else None

    # ---- timed replay loop ----
    # static_tok already holds token 0 (from prefill). Generate tokens 1..N-1.
    static_tok.copy_(gen[0])
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(N - 1):
        static_pos.copy_(torch.tensor([prompt_len + i], device=dev))
        g.replay()  # forward(static_tok, static_pos) [+ cat into staging]
        if args.transfer:
            pinned.copy_(staging, non_blocking=True)  # DMA staging -> CPU
        nxt = static_logits[:, -1:].argmax(-1)
        static_tok.copy_(nxt)  # current token for the next replay
        gen.append(nxt.clone())
    torch.cuda.synchronize(); dt = time.time() - t0

    text = tok.decode(torch.cat(gen, dim=1)[0].tolist(), skip_special_tokens=True)
    mode = "graph+transfer" if args.transfer else "graph"
    print(f"[{mode}] batch={B}  {dt/(N-1)*1000:.2f} ms/step  ({B*(N-1)/dt:.0f} tok/s)")
    print("generation:", repr(text[:200]))


if __name__ == "__main__":
    main()
