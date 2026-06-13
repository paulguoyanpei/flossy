"""Correctness + speed test for the lean verifier (no HF generate).

Captures prover data (GPU), then runs LeanVerifier over the captured log and
checks: (1) it derives the same token sequence as the prover (lockstep via
lm_head argmax), (2) honest data -> 0 failures, (3) a corrupted tensor ->
failure. Also times it against the HF-generate batched verifier.

  python lean_test.py                  # tiny random Qwen3
  python lean_test.py --model <path> -n 32
"""

from __future__ import annotations

import argparse
import time

import torch

from capture import FlossyContext, StepBuffer, instrument
from lean_verifier import LeanVerifier, groups_from_stepbuffer


def build_models(model_id, prover_device):
    if model_id is None:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        cfg = Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                          max_position_embeddings=512, tie_word_embeddings=False)
        torch.manual_seed(0)
        prover = Qwen3ForCausalLM(cfg).eval()
        verifier = Qwen3ForCausalLM(cfg).eval()
        verifier.load_state_dict(prover.state_dict())
        tok = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        prover = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).eval()
        verifier = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    pdt = torch.float16 if prover_device.startswith("cuda") else torch.float32
    return prover.to(prover_device, pdt), verifier.to("cpu", torch.float32), tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=16)
    ap.add_argument("--prover-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=0.1)
    ap.add_argument("--s", type=int, default=10)
    ap.add_argument("--transfer-scores", action="store_true",
                    help="prover ships raw QK^T scores (stage-2); default off = verifier recomputes q.K^T")
    args = ap.parse_args()

    prover, verifier, tok = build_models(args.model, args.prover_device)
    if tok is None:
        ids = torch.tensor([[1, 2, 3, 4, 5]])
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids
    gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)

    # ---- prover capture ----
    log = []

    def sink(step, buf):
        cb = StepBuffer()
        for n in buf.order:
            cb.record(n, buf.fetch(n).detach().to("cpu", torch.float16))
        log.append(cb)

    pctx = FlossyContext(role="prover", s=args.s, tol=args.tol,
                         record_scores=args.transfer_scores, sink=sink)
    instrument(prover, pctx)
    with torch.inference_mode():
        out = prover.generate(ids.to(args.prover_device), **gen)
    prover_tokens = out[0].tolist()
    print(f"prover: {len(log)} steps, {len(prover_tokens)} tokens")

    # ---- lean verify (honest) ----
    torch.set_num_threads(16)
    lv = LeanVerifier(verifier, s=args.s, tol=args.tol)
    lv.set_prompt(ids[0])
    derived = list(ids[0].tolist())
    t0 = time.time()
    for buf in log:
        lv.verify_step(groups_from_stepbuffer(buf))
        derived.append(int(lv.next_tokens[0]))
    dt = time.time() - t0
    derived = derived[: len(prover_tokens)]
    tokens_match = derived == prover_tokens
    print(f"lean:   {lv.n_checks} checks, {lv.n_failures} failures, "
          f"worst {lv.worst_ratio:.3e} @ {lv.worst_name}")
    print(f"        {dt:.3f}s  {len(log)/dt:.1f} steps/s  {dt/len(log)*1000:.1f} ms/step")
    print(f"        tokens match prover: {tokens_match}")

    # ---- attack ----
    bad = log[min(2, len(log) - 1)]
    name = next(n for n in bad.order if "gate_proj" in n)
    t = bad.fetch(name).float()
    bad.tensors[name] = (t + 0.3 * t.std() * torch.randn_like(t)).half()
    lv2 = LeanVerifier(verifier, s=args.s, tol=args.tol)
    lv2.set_prompt(ids[0])
    for buf in log:
        lv2.verify_step(groups_from_stepbuffer(buf))
    print(f"attack ({name}): {lv2.n_failures} failures -> {'REJECT' if lv2.n_failures else 'ACCEPT'}")

    ok = tokens_match and lv.n_failures == 0 and lv2.n_failures > 0
    print("\n" + ("LEAN TEST PASS" if ok else "LEAN TEST FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
