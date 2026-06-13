"""Shared CLI / model-loading helpers for the prover and verifier."""

from __future__ import annotations

import argparse

import torch

# HARD NO-FLASH POLICY (process-wide). This verification scheme fundamentally
# cannot use fused flash/mem-efficient attention -- the verifier needs the QK^T
# scores, which those kernels never expose -- so comparing against a flash
# baseline is meaningless. Every entry point (prover, verifier, bench_gpu_sweep,
# cat_dma_bench, verifier_bench) imports this module, so disabling the flash +
# mem-efficient SDPA backends here forces the math kernel everywhere. There is no
# opt-in: flash cannot be selected.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True, help="HF model id or local snapshot dir")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--batch", type=int, default=1, help="batch size (replicates the prompt)")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=32)
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="force at least this many new tokens (set == max for fixed-length benchmarking)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = verifier picks; prover must match")
    ap.add_argument("--port-file", default=None, help="verifier writes / prover reads the chosen port here")
    ap.add_argument("--shm-name", default="flossy_ring")
    ap.add_argument("--slots", type=int, default=32,
                    help="shared-memory ring depth (pipeline depth). Buffers the verifier's "
                         "slot-freeing so it never backpressures the prover; 4 is too tight for "
                         "batched runs (starves the prover ~+1ms/step), 8-16 hides it. Cost is a "
                         "one-time cudaHostRegister of slots*slot_size pinned memory.")
    ap.add_argument("--tol", type=float, default=0.1,
                    help="Freivalds relative tolerance; honest worst is a bounded ~0.05, faults are 0.4-1.0")
    ap.add_argument("--s", type=int, default=10, help="Freivalds projection columns")
    ap.add_argument("--transfer-scores", action="store_true",
                    help="prover ships raw QK^T scores (paper §5.2 stage-2). Helps LONG context "
                         "but costs ~3ms/step at batch16 (explicit repeat_kv over the KV cache + "
                         "QK^T matmul that fused SDPA avoids). Off => verifier recomputes q.K^T.")
    ap.add_argument("--seed", type=int, default=1234)


def load_model(model_id: str, device: str):
    """Load (tokenizer, model). fp16 on CUDA, fp32 on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # keep stdout to just the prover timing + verdict
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tokenizer, model


def build_prompt_ids(tokenizer, prompt: str, device: str) -> torch.Tensor:
    """Apply the chat template if available so prover/verifier agree exactly."""
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def gen_kwargs(max_new_tokens: int, min_new_tokens: int = 0) -> dict:
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, num_beams=1)
    if min_new_tokens > 0:
        kw["min_new_tokens"] = min_new_tokens
    return kw
