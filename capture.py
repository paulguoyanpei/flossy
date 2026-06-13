"""Model-agnostic instrumentation for the FLOSSY prover and verifier.

The same stock Transformers model (Llama, Qwen3, ...) is used on both sides.
We attach behaviour without editing any library file:

* Every target ``nn.Linear`` (q/k/v/o_proj, gate/up/down_proj, lm_head) has its
  ``forward`` replaced.
    - PROVER: run the real matmul, *record* the output.
    - VERIFIER: *skip* the matmul, fetch the prover's output, Freivalds-check it
      against the locally reconstructed input, and return it (passthrough) so
      the residual stream downstream uses the verified GPU values.
* Attention runs through a custom ``AttentionInterface`` implementation.
    - PROVER ("flossy_prover"): compute attention, record raw ``QK^T`` scores
      and the ``softmax @ V`` output.
    - VERIFIER ("flossy_verifier"): Freivalds-check the score and value matmuls
      against the recorded tensors, return the recorded output (passthrough).
* The outer ``*ForCausalLM.forward`` is wrapped to bracket each decode step: the
  prover ships the step's recorded tensors, the verifier receives them before
  the inner modules run.

Greedy decoding + lm_head passthrough keep prover and verifier in lockstep
without sharing tokens, so the per-step tensor sets align by name.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn

from freivalds import DEFAULT_S, DEFAULT_TOL, LinearChecker


# Linear submodule names we verify (suffix match on the dotted module path).
ATTENTION_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
LM_HEAD = ("lm_head",)
TARGET_SUFFIXES = ATTENTION_PROJ + MLP_PROJ + LM_HEAD


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, n_kv, s, d) -> (b, n_kv * n_rep, s, d). Mirrors modeling_llama."""
    batch, num_kv, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv * n_rep, slen, head_dim)


# --------------------------------------------------------------------------- #
# Per-step buffer of named tensors
# --------------------------------------------------------------------------- #
class StepBuffer:
    """Ordered collection of named tensors produced/consumed in one decode step."""

    def __init__(self, tensors: dict[str, torch.Tensor] | None = None) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.order: list[str] = []
        if tensors is not None:
            for name, t in tensors.items():
                self.record(name, t)

    def record(self, name: str, t: torch.Tensor) -> None:
        if name in self.tensors:
            raise KeyError(f"duplicate tensor name in step: {name!r}")
        self.tensors[name] = t
        self.order.append(name)

    def fetch(self, name: str) -> torch.Tensor:
        return self.tensors[name]


# --------------------------------------------------------------------------- #
# Shared context
# --------------------------------------------------------------------------- #
@dataclass
class FlossyContext:
    role: str  # "prover" | "verifier"
    s: int = DEFAULT_S
    tol: float = DEFAULT_TOL
    # prover: called with (step, StepBuffer) to ship the recorded tensors.
    sink: Callable[[int, StepBuffer], None] | None = None
    # verifier: called with (step) -> StepBuffer of received tensors.
    source: Callable[[int], StepBuffer] | None = None
    # verifier: optional callback (step) run after the step's tensors are consumed
    # (e.g. to ack/free the shared-memory slot).
    after_step: Callable[[int], None] | None = None
    checkers: dict[str, LinearChecker] = field(default_factory=dict)
    generator: torch.Generator | None = None
    bverif: object | None = None  # BatchedVerifier (verifier role); checks deferred + batched
    # prover: ship the raw QK^T scores (paper §5.2 stage 2). The eager prover
    # attention computes QK^T anyway (for the output), so recording it is ~free;
    # this only controls whether it's *transferred* (stage 2) or the verifier
    # recomputes q.K^T (stage 1). Transferring adds DMA volume but helps long ctx.
    record_scores: bool = True

    step: int = -1
    current: StepBuffer | None = None
    # verifier stats
    n_checks: int = 0
    n_failures: int = 0
    worst_ratio: float = 0.0
    worst_name: str = ""
    failures: list[tuple[int, str, float]] = field(default_factory=list)

    def begin_step(self) -> None:
        self.step += 1
        if self.role == "prover":
            self.current = StepBuffer()
        else:
            assert self.source is not None
            self.current = self.source(self.step)

    def end_step(self) -> None:
        if self.role == "prover":
            assert self.sink is not None and self.current is not None
            self.sink(self.step, self.current)
        else:
            if self.bverif is not None:
                self.bverif.run()  # batched Freivalds checks for the whole step
            if self.after_step is not None:
                self.after_step(self.step)
        self.current = None

    def record_check(self, name: str, ok: bool, ratio: float) -> None:
        self.n_checks += 1
        if ratio > self.worst_ratio:
            self.worst_ratio = ratio
            self.worst_name = name
        if not ok:
            self.n_failures += 1
            self.failures.append((self.step, name, ratio))


# The active context for the custom attention interfaces (process-global; each
# process has exactly one role).
_ACTIVE_CTX: FlossyContext | None = None


# --------------------------------------------------------------------------- #
# Linear forward replacement
# --------------------------------------------------------------------------- #
def get_target_linears(model: nn.Module) -> dict[str, nn.Linear]:
    """Map flossy name -> linear module for every target projection.

    Name is the dotted path with the redundant prefix trimmed, e.g.
    ``model.layers.5.self_attn.q_proj`` -> ``5.q_proj``; ``lm_head`` -> ``lm_head``.
    """
    out: dict[str, nn.Linear] = {}
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in TARGET_SUFFIXES:
            continue
        layer_idx = getattr(module, "_flossy_layer_idx", None)
        if layer_idx is None:
            # derive layer index from the path: ...layers.<idx>...
            parts = path.split(".")
            layer_idx = next((parts[i + 1] for i, p in enumerate(parts) if p == "layers"), None)
        name = f"{layer_idx}.{leaf}" if layer_idx is not None else leaf
        out[name] = module
    return out


def _make_prover_linear_forward(module: nn.Linear, name: str, ctx: FlossyContext):
    orig_forward = module.forward

    @functools.wraps(orig_forward)
    def forward(x: torch.Tensor) -> torch.Tensor:
        out = orig_forward(x)
        if ctx.current is not None:
            ctx.current.record(name, out)
        return out

    return forward


def _make_verifier_linear_forward(module: nn.Linear, name: str, ctx: FlossyContext):
    def forward(x: torch.Tensor) -> torch.Tensor:
        assert ctx.current is not None, f"no step buffer active for {name}"
        out = ctx.current.fetch(name)  # GPU value (fp16 after transport)
        # defer the Freivalds check; it is run batched across layers in end_step
        ctx.bverif.collect_linear(name, x, out)
        # passthrough so the downstream residual stream uses the verified value
        return out.to(dtype=x.dtype, device=x.device)

    return forward


def install_linears(model: nn.Module, ctx: FlossyContext) -> dict[str, nn.Linear]:
    targets = get_target_linears(model)
    for name, module in targets.items():
        if ctx.role == "prover":
            module.forward = _make_prover_linear_forward(module, name, ctx)
        else:
            if name not in ctx.checkers:
                ctx.checkers[name] = LinearChecker(
                    module.weight,
                    getattr(module, "bias", None),
                    s=ctx.s,
                    tol=ctx.tol,
                    name=name,
                    generator=ctx.generator,
                )
            module.forward = _make_verifier_linear_forward(module, name, ctx)
    if ctx.role == "verifier":
        from batched_check import BatchedVerifier

        ctx.bverif = BatchedVerifier(ctx, targets, ctx.checkers)
    return targets


# --------------------------------------------------------------------------- #
# Attention interfaces
# --------------------------------------------------------------------------- #
def _flossy_prover_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Prover attention: eager (QK^T -> softmax -> .V), computing QK^T ONCE and
    reusing it for both the softmax @ V output and the recorded scores. SDPA never
    returns its internal scores (any backend), so an SDPA path would have to
    recompute QK^T explicitly to ship it; this doesn't -- and since flash/mem-eff
    SDPA is disabled anyway (common.py), there's no fast-path reason to use SDPA.
    Registered with the (graph-safe) sdpa mask; builds its own additive causal /
    static-cache mask from cache_position when that mask is None.
    """
    ctx = _ACTIVE_CTX
    n_rep = module.num_key_value_groups
    key_full = repeat_kv(key, n_rep)
    value_full = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key_full.transpose(-1, -2))  # (b, h, q, kv) raw QK^T
    attn = scores * scaling
    kv_len = key_full.shape[-2]
    q_len = query.shape[-2]
    cache_position = kwargs.get("cache_position")
    if attention_mask is not None:
        m = attention_mask[..., :kv_len]
        attn = attn.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else attn + m
    elif cache_position is not None:
        # causal + static-cache mask, built with pure tensor ops (graph-safe):
        # query at absolute position p attends to kv indices <= p.
        kv_idx = torch.arange(kv_len, device=attn.device)
        future = kv_idx[None, :] > cache_position[:, None]  # (q, kv) True=mask out
        attn = attn.masked_fill(future[None, None], float("-inf"))
    elif q_len > 1:  # prefill fallback: pure causal
        future = torch.ones(q_len, kv_len, dtype=torch.bool, device=attn.device).triu(1)
        attn = attn.masked_fill(future[None, None], float("-inf"))
    probs = torch.softmax(attn.float(), dim=-1).to(value_full.dtype)
    attn_output = torch.matmul(probs, value_full)  # (b, h, q, d)

    if ctx is not None and ctx.current is not None:
        ctx.current.record(f"{module.layer_idx}.attn_out", attn_output)
        if ctx.record_scores:
            raw = scores
            if raw.shape[2] > 1:  # prefill: trim to the causal-valid region (= q)
                raw = raw[..., : raw.shape[2]].contiguous()
            ctx.current.record(f"{module.layer_idx}.attn_scores", raw)

    return attn_output.transpose(1, 2).contiguous(), None


def _flossy_verifier_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    ctx = _ACTIVE_CTX
    assert ctx is not None and ctx.current is not None
    n_rep = module.num_key_value_groups
    key_r = repeat_kv(key, n_rep)
    value_r = repeat_kv(value, n_rep)

    out = ctx.current.fetch(f"{module.layer_idx}.attn_out")  # (b, h, q, d), GPU's softmax @ V

    # Defer the attention check (batched across layers in end_step). The verifier
    # re-derives scores = q @ K^T from its own verified q/K (no raw QK^T from the
    # GPU; KV-cache friendly) and Freivalds-checks the GPU's softmax @ V; a wrong
    # GPU QK^T produces a wrong attn_out and fails this, so QK^T is verified
    # transitively.
    ctx.bverif.collect_attn(
        module.layer_idx, query, key_r, value_r, attention_mask, scaling, out
    )

    attn_output = out.to(dtype=query.dtype, device=query.device)
    return attn_output.transpose(1, 2).contiguous(), None


_REGISTERED = False


def register_attention(model: nn.Module, ctx: FlossyContext) -> None:
    """Register flossy attention interfaces and point the model at the right one."""
    global _ACTIVE_CTX, _REGISTERED
    _ACTIVE_CTX = ctx
    if not _REGISTERED:
        from transformers import AttentionInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

        AttentionInterface.register("flossy_prover", _flossy_prover_attention)
        AttentionInterface.register("flossy_verifier", _flossy_verifier_attention)
        # Prover uses the (graph-safe) sdpa mask -- it builds its own additive mask
        # from cache_position when that's None. Verifier uses the eager additive
        # mask. (flash/mem-efficient SDPA is disabled process-wide, common.py.)
        AttentionMaskInterface.register("flossy_prover", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        AttentionMaskInterface.register("flossy_verifier", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        _REGISTERED = True

    impl = "flossy_prover" if ctx.role == "prover" else "flossy_verifier"
    model.config._attn_implementation = impl
    if getattr(model, "model", None) is not None:
        model.model.config._attn_implementation = impl


# --------------------------------------------------------------------------- #
# Outer forward bracketing (one call == one decode step)
# --------------------------------------------------------------------------- #
def wrap_step_forward(model: nn.Module, ctx: FlossyContext) -> None:
    orig_forward = model.forward

    @functools.wraps(orig_forward)
    def forward(*args, **kwargs):
        ctx.begin_step()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            ctx.end_step()

    model.forward = forward


def instrument(model: nn.Module, ctx: FlossyContext) -> dict[str, nn.Linear]:
    """Install everything for one role. Returns the discovered target linears."""
    register_attention(model, ctx)
    targets = install_linears(model, ctx)
    wrap_step_forward(model, ctx)
    return targets
