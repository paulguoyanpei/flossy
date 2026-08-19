"""Pure-GPU cuda-graph decode floor, sweeping output token count N, with the
graph captured ONCE (excluded from the per-N timing -- the "build once, serve
many" optimization). No transport, no verifier. This is the GPU-inference
baseline to compare the verified end-to-end against.

  python bench_gpu_sweep.py --model <path> --ns 64 128 256 512
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
    ap.add_argument("--ns", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--batch", type=int, default=1, help="batch size (replicates the prompt)")
    args = ap.parse_args()
    dev = "cuda:0"

    tok, model = load_model(args.model, dev)
    ctx = FlossyContext(role="prover", s=16, tol=0.1, sink=lambda step, buf: None)
    instrument(model, ctx)
    ids = build_prompt_ids(tok, args.prompt, dev)
    if args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    B, P = ids.shape
    Nmax = max(args.ns)

    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=P + Nmax,
                        device=dev, dtype=torch.float16)

    @torch.inference_mode()
    def fwd(cur, pos):
        return model(input_ids=cur, cache_position=pos, past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits

    stok = torch.zeros(B, 1, dtype=torch.long, device=dev)
    spos = torch.tensor([P], device=dev)

    # ---- capture ONCE (one-time, excluded from per-N timing) ----
    torch.cuda.synchronize(); t = time.time()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3):
            fwd(stok, spos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        slog = fwd(stok, spos)
    torch.cuda.synchronize()
    cap = time.time() - t
    print(f"graph capture (one-time): {cap:.2f}s\n")
    print(f"{'N':>6}  {'prefill+decode':>14}  {'ms/tok':>7}  {'tok/s':>7}   sample")

    for N in args.ns:
        with torch.inference_mode():
            cache.reset()
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            logits = fwd(ids, torch.arange(P, device=dev))
        nxt = logits[:, -1:].argmax(-1)
        stok.copy_(nxt)
        gen = [nxt]
        for k in range(N - 1):
            spos.fill_(P + k)
            g.replay()
            nxt = slog[:, -1:].argmax(-1)
            stok.copy_(nxt)
            gen.append(nxt.clone())
        torch.cuda.synchronize(); dt = time.time() - t
        text = tok.decode(torch.cat(gen, dim=1)[0].tolist(), skip_special_tokens=True)
        print(f"{N:>6}  {dt:>13.3f}s  {dt / N * 1000:>7.2f}  {N / dt:>7.1f}   {text[:48]!r}")


if __name__ == "__main__":
    main()
