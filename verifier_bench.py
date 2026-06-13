"""Isolated verifier-compute benchmark: time LeanVerifier.verify_step on
synthetic groups of the REAL shapes, sweeping CPU thread count, at a given batch.
No prover / transport / model forward -- pure verifier cost. (Values are random,
so the verdict is meaningless; only the timing matters.)

  python verifier_bench.py --model <path> --batch 16 -n 64 --threads 4 8 16 32 --scores
"""
from __future__ import annotations

import argparse
import time

import torch

from common import load_model
from lean_verifier import LeanVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("-n", "--steps", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=12)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--layers", type=int, default=0, help="truncate to N layers (0 = all; mimics a shard)")
    ap.add_argument("--s", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--scores", action="store_true", help="include transferred attn_scores (stage-2)")
    args = ap.parse_args()

    _, model = load_model(args.model, "cpu")
    if args.layers > 0:  # mimic a layer-sharded worker
        model.model.layers = model.model.layers[: args.layers]
        model.config.num_hidden_layers = args.layers
    cfg = model.config
    L, B, P, N = cfg.num_hidden_layers, args.batch, args.prompt_len, args.steps
    h, inter = cfg.hidden_size, cfg.intermediate_size
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or h // n_q
    vocab = cfg.vocab_size
    maxkv = P + N

    def make_groups(S):
        g = {
            "q_proj": torch.randn(L, B, S, n_q * hd, dtype=torch.float16),
            "k_proj": torch.randn(L, B, S, n_kv * hd, dtype=torch.float16),
            "v_proj": torch.randn(L, B, S, n_kv * hd, dtype=torch.float16),
            "o_proj": torch.randn(L, B, S, h, dtype=torch.float16),
            "gate_proj": torch.randn(L, B, S, inter, dtype=torch.float16),
            "up_proj": torch.randn(L, B, S, inter, dtype=torch.float16),
            "down_proj": torch.randn(L, B, S, h, dtype=torch.float16),
            "attn_out": torch.randn(L, B, n_q, S, hd, dtype=torch.float16),
            "lm_head": torch.randn(1, B, S, vocab, dtype=torch.float16),
        }
        if args.scores:
            g["attn_scores"] = torch.randn(L, B, n_q, S, maxkv, dtype=torch.float16)
        return g

    pf = make_groups(P)
    dec = make_groups(1)
    print(f"batch={B} layers={L} steps={N} ctx<={maxkv} scores={args.scores}\n")

    for nt in args.threads:
        torch.set_num_threads(nt)
        gen = torch.Generator().manual_seed(0)
        lv = LeanVerifier(model, s=args.s, tol=args.tol, generator=gen)
        lv.set_prompt(torch.zeros(P, dtype=torch.long))
        lv.verify_step(pf)  # prefill (untimed)
        t = time.time()
        for _ in range(N):
            lv.verify_step(dec)
        dt = time.time() - t
        print(f"  threads={nt:3d}: {dt / N * 1000:7.2f} ms/step  ({dt:.2f}s / {N})")


if __name__ == "__main__":
    main()
