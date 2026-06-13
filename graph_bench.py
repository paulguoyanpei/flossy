"""Establish the CUDA-graph floor for pure inference (no instrumentation/transfer).

Compares HF generate with the default dynamic cache vs a static cache +
torch.compile(mode="reduce-overhead") (which uses CUDA graphs to remove the
per-step kernel-launch overhead that makes small-batch decode CPU-bound).

  python graph_bench.py --model <path> --batch 1 16 -n 64
"""
from __future__ import annotations

import argparse
import time

import torch

from common import build_prompt_ids, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="essay about computing history")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("-n", "--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    tok, model = load_model(args.model, "cuda:0")
    base_ids = build_prompt_ids(tok, args.prompt, "cuda:0")

    def bench(ids, n, gen_extra):
        # warmup with the SAME length so the static cache size matches the timed
        # run (otherwise a different cache size triggers a full recompile)
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, **gen_extra)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, **gen_extra)
        torch.cuda.synchronize()
        dt = time.time() - t0
        new = out.shape[1] - ids.shape[1]
        return dt, dt / new * 1000

    for b in args.batch:
        ids = base_ids.repeat(b, 1) if b > 1 else base_ids
        print(f"\n########## BATCH={b} ##########", flush=True)
        dt, ms = bench(ids, args.max_new_tokens, {})
        print(f"  dynamic cache (no graph): {dt:.3f}s  {ms:.2f} ms/step  ({b * args.max_new_tokens / dt:.0f} tok/s)", flush=True)

    # HF-native compiled generation: static cache + compile handled by generate
    print("\n[enabling HF compiled generation (static cache) ...]", flush=True)
    torch.set_float32_matmul_precision("high")
    from transformers import CompileConfig

    model.generation_config.cache_implementation = "static"
    model.generation_config.compile_config = CompileConfig()
    for b in args.batch:
        ids = base_ids.repeat(b, 1) if b > 1 else base_ids
        try:
            dt, ms = bench(ids, args.max_new_tokens, {})
            print(f"BATCH={b} static+compile (CUDA graph): {dt:.3f}s  {ms:.2f} ms/step  "
                  f"({b * args.max_new_tokens / dt:.0f} tok/s)", flush=True)
        except Exception as e:
            import traceback
            print(f"BATCH={b} compile FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
