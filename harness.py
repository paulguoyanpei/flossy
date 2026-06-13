"""Single-process correctness harness (no transport).

Runs the prover model (optionally on CUDA/fp16), records every step's tensors
into an in-memory log, then runs a separate fp32 CPU verifier model that
consumes the log and Freivalds-checks every matmul. Validates hook ordering,
the attention interface, lockstep token agreement, and that correct inference
passes all checks.

Usage:
  python harness.py                         # tiny random Qwen3, fast plumbing
  python harness.py --model Qwen/Qwen3-4B-Instruct-2507 --prompt "Hello" -n 16
"""

from __future__ import annotations

import argparse

import torch

from capture import FlossyContext, StepBuffer, instrument


def build_models(model_id: str | None, prover_device: str):
    """Return (prover_model, verifier_model, tokenizer) sharing identical weights."""
    if model_id is None:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=512,
            tie_word_embeddings=False,
        )
        torch.manual_seed(0)
        prover = Qwen3ForCausalLM(config).eval()
        verifier = Qwen3ForCausalLM(config).eval()
        verifier.load_state_dict(prover.state_dict())
        tokenizer = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prover = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).eval()
        verifier = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()

    prover_dtype = torch.float16 if prover_device.startswith("cuda") else torch.float32
    prover = prover.to(device=prover_device, dtype=prover_dtype)
    verifier = verifier.to(device="cpu", dtype=torch.float32)
    return prover, verifier, tokenizer


def make_inputs(tokenizer, prompt: str, device: str):
    if tokenizer is None:
        return torch.tensor([[1, 2, 3, 4, 5]], device=device)
    enc = tokenizer(prompt, return_tensors="pt")
    return enc.input_ids.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id; omit for a tiny random Qwen3")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=16)
    ap.add_argument("--prover-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--s", type=int, default=10)
    args = ap.parse_args()

    prover, verifier, tokenizer = build_models(args.model, args.prover_device)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)

    # ----- PROVER: record every step into an in-memory log -----
    log: list[StepBuffer] = []

    def sink(step: int, buf: StepBuffer) -> None:
        cpu_buf = StepBuffer()
        for name in buf.order:
            cpu_buf.record(name, buf.fetch(name).detach().to("cpu", torch.float32))
        log.append(cpu_buf)

    prover_ctx = FlossyContext(role="prover", s=args.s, tol=args.tol, sink=sink)
    instrument(prover, prover_ctx)

    prover_in = make_inputs(tokenizer, args.prompt, args.prover_device)
    with torch.inference_mode():
        prover_out = prover.generate(prover_in, **gen_kwargs)
    prover_tokens = prover_out[0].tolist()
    print(f"prover: {len(log)} steps recorded, {len(prover_tokens)} total tokens")

    # ----- VERIFIER: consume the log, Freivalds-check everything -----
    def source(step: int) -> StepBuffer:
        if step >= len(log):
            raise IndexError(f"verifier requested step {step} but only {len(log)} recorded")
        return log[step]

    verifier_ctx = FlossyContext(role="verifier", s=args.s, tol=args.tol, source=source)
    instrument(verifier, verifier_ctx)

    verifier_in = make_inputs(tokenizer, args.prompt, "cpu")
    with torch.inference_mode():
        verifier_out = verifier.generate(verifier_in, **gen_kwargs)
    verifier_tokens = verifier_out[0].tolist()

    tokens_match = prover_tokens == verifier_tokens
    print(f"verifier: {verifier_ctx.n_checks} checks, {verifier_ctx.n_failures} failures, "
          f"worst ratio {verifier_ctx.worst_ratio:.2e} (tol {args.tol})")
    print(f"tokens match: {tokens_match}")
    if verifier_ctx.failures:
        for step, name, ratio in verifier_ctx.failures[:20]:
            print(f"  FAIL step={step} {name} ratio={ratio:.3e}")
    if tokenizer is not None:
        print("generation:", repr(tokenizer.decode(prover_tokens[verifier_in.shape[1]:])))

    ok = tokens_match and verifier_ctx.n_failures == 0 and verifier_ctx.n_checks > 0
    print("\n" + ("ALL PASS" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
