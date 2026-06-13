"""Attribute the CUDA-graph decode overhead by timing graphs that differ ONLY by
the step under test, each with an explicit ``torch.cuda.synchronize`` (no phantom
unsynced numbers). Isolates:

  1. forward only (instrumented model, no cat)         -> the GPU decode floor
  2. forward + in-graph cat -> staging                 -> (2)-(1) = the cat cost
  3. + serial DMA staging -> shm on the replay stream  -> (3)-(2) = DMA on crit path
  4. + overlapped DMA (d2d shadow + side stream)        -> (4)-(2) = DMA not hidden

  python cat_dma_bench.py --model <path> --batch 16 -n 256
"""
from __future__ import annotations

import argparse
import time

import torch

from capture import FlossyContext, instrument
from common import build_prompt_ids, load_model
from transport import SharedRing, plan_grouped


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="essay about computing history")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("-n", "--steps", type=int, default=256)
    ap.add_argument("--n-slots", type=int, default=16)
    ap.add_argument("--shm-name", default="catbench")
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    dev = "cuda:0"

    tok, model = load_model(args.model, dev)
    ids = build_prompt_ids(tok, args.prompt, dev)
    if args.batch > 1:
        ids = ids.repeat(args.batch, 1)
    B, P = ids.shape
    N = args.steps

    captured: dict = {}

    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=P + N,
                        device=dev, dtype=torch.float16)

    def fwd(cur, pos):
        return model(input_ids=cur, cache_position=pos, past_key_values=cache,
                     use_cache=True, logits_to_keep=1).logits

    # prefill -> first generated token
    logits = fwd(ids, torch.arange(P, device=dev))
    cur_tok = logits[:, -1:].argmax(-1)
    static_tok = cur_tok.clone()
    static_pos = torch.zeros(1, dtype=torch.long, device=dev)
    static_pos.fill_(P)

    def capture_fwd() -> torch.cuda.CUDAGraph:
        w = torch.cuda.Stream(); w.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(w):
            for _ in range(2):
                fwd(static_tok, static_pos)
        torch.cuda.current_stream().wait_stream(w)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fwd(static_tok, static_pos)
        torch.cuda.synchronize()
        return g

    # ---- CLEAN (uninstrumented) forward graph, same process ----
    g_clean = capture_fwd()

    # ---- now instrument and rebuild ----
    ctx = FlossyContext(role="prover", sink=lambda st, b: captured.__setitem__("buf", b))
    instrument(model, ctx)

    # warmup decode (populates buf with decode-shaped tensors + allocator stability)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd(static_tok, static_pos)
    torch.cuda.current_stream().wait_stream(s)

    dec_buf = captured["buf"]
    _, ordered, n = plan_grouped(dec_buf, 1, args.n_slots, 1)
    staging = torch.empty(n, dtype=torch.float16, device=dev)
    print(f"batch={B} P={P} steps={N} transfer elems={n} (~{n * 2 / 1e6:.1f} MB/step)\n")

    def capture(do_cat: bool) -> torch.cuda.CUDAGraph:
        w = torch.cuda.Stream(); w.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(w):
            for _ in range(2):
                fwd(static_tok, static_pos)
        torch.cuda.current_stream().wait_stream(w)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fwd(static_tok, static_pos)
            if do_cat:
                gbuf = captured["buf"]
                torch.cat([gbuf.fetch(nm).reshape(-1) for nm in ordered], out=staging)
        torch.cuda.synchronize()
        return g

    def timeit(label: str, g: torch.cuda.CUDAGraph, dma_fn=None) -> float:
        g.replay()  # one untimed warmup replay
        torch.cuda.synchronize()
        t = time.time()
        for k in range(N):
            static_pos.fill_(P + k)
            g.replay()
            if dma_fn is not None:
                dma_fn(k)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"  {label:34s}: {dt / N * 1000:7.3f} ms/step  ({dt:.3f}s)")
        return dt / N * 1000

    g_fwd = capture(do_cat=False)
    ctx.record_scores = False  # drop the explicit QK^T scores (repeat_kv over KV + matmul)
    g_noscore = capture(do_cat=False)
    ctx.record_scores = True
    g_cat = capture(do_cat=True)

    ring = SharedRing.create(args.shm_name, args.n_slots, n, dtype="float16")
    reg = ring.register_cuda()
    side = torch.cuda.Stream()
    shadow = [torch.empty(n, dtype=torch.float16, device=dev) for _ in range(args.n_slots)]

    def dma_serial(k):
        ring.slot(k % args.n_slots)[:n].copy_(staging[:n], non_blocking=True)

    def dma_overlap(k):
        slot = k % args.n_slots
        shadow[slot][:n].copy_(staging[:n])
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ring.slot(slot)[:n].copy_(shadow[slot][:n], non_blocking=True)

    print(f"shm registered for direct DMA: {reg}\n")
    t_clean = timeit("0. forward CLEAN (uninstrumented)", g_clean)
    t_nosc = timeit("1a. instrumented, NO scores", g_noscore)
    t_fwd = timeit("1b. instrumented, +QK^T scores", g_fwd)
    t_cat = timeit("2. forward + cat", g_cat)
    t_ser = timeit("3. + DMA serial (replay stream)", g_cat, dma_serial)
    t_ovl = timeit("4. + DMA overlap (shadow+side)", g_cat, dma_overlap)

    print(f"\n  recording (no scores): {t_nosc - t_clean:+.3f} ms/step  (linears+attn_out refs)")
    print(f"  QK^T scores cost     : {t_fwd - t_nosc:+.3f} ms/step  (repeat_kv over KV + matmul)")
    print(f"  cat cost            : {t_cat - t_fwd:+.3f} ms/step")
    print(f"  DMA on crit path    : {t_ser - t_cat:+.3f} ms/step  (serial)")
    print(f"  DMA not hidden      : {t_ovl - t_cat:+.3f} ms/step  (overlapped)")
    print(f"  overlap saves       : {t_ser - t_ovl:+.3f} ms/step")
    ring.close()


if __name__ == "__main__":
    main()
