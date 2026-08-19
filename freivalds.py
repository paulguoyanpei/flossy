"""Freivalds' algorithm checks for matrix multiplications.

This is the math core of the FLOSSY verifier.  Instead of recomputing an
expensive product ``C = A @ B`` (O(n*m*k)), we draw a random +/-1 projection
``r`` with ``s`` columns and check ``A @ (B @ r) ~= C @ r`` (O(n*k*s)).  If the
GPU produced a wrong ``C`` the check fails with overwhelming probability.

All checks run in fp32 on CPU.  Tolerances are *relative to the L2 norm of the
reference output*, which makes them robust to the fp16 noise coming from the
GPU prover (mirrors ``LinearChecker`` / ``Attention.check`` in
``../flossy/llama3/llama/model.py``).
"""

from __future__ import annotations

import torch


DEFAULT_S = 16  # number of random projection columns ("n_sample" in flossy)
DEFAULT_TOL = 0.05  # max allowed ||A.Br - C.r||_inf / ||C||_2 per row


def make_r(n: int, s: int, *, dtype=torch.float32, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random +/-1 projection matrix of shape ``(n, s)`` on CPU."""
    return torch.randint(0, 2, (n, s), generator=generator, dtype=dtype) * 2 - 1


def _compare(abr: torch.Tensor, cr: torch.Tensor, ref: torch.Tensor, tol: float) -> tuple[bool, float]:
    """Return (passed, worst_ratio).

    ``abr`` and ``cr`` are the two sides of the Freivalds identity (``A @ Br``
    and ``C @ r``); ``ref`` is the transferred output ``C`` whose L2 norm (over
    its last / contracted dimension) sets the per-row scale.
    """
    l2 = ref.float().norm(p=2, dim=-1)
    diff = (abr - cr).abs().amax(dim=-1)
    ratio = diff / (l2 + 1e-9)
    return bool((ratio <= tol).all().item()), float(ratio.max().item())


class LinearChecker:
    """Freivalds checker for a single ``nn.Linear`` (``out = x @ W^T + b``)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        s: int = DEFAULT_S,
        tol: float = DEFAULT_TOL,
        name: str = "",
        generator: torch.Generator | None = None,
    ) -> None:
        out_features, _in_features = weight.shape
        self.name = name
        self.tol = tol
        self.s = s
        w = weight.detach().to("cpu", torch.float32)
        self.r = make_r(out_features, s, generator=generator)  # (out, s)
        # Br = W^T @ r, shape (in, s). Compute as (r^T @ W)^T to avoid
        # materializing a transposed copy of the (potentially huge) weight.
        self.Br = (self.r.t() @ w).t().contiguous()  # (in, s)
        self.br = bias.detach().to("cpu", torch.float32) @ self.r if bias is not None else None  # (s,)

    def check(self, input: torch.Tensor, output: torch.Tensor) -> tuple[bool, float]:
        x = input.reshape(-1, input.shape[-1]).to("cpu", torch.float32)
        y = output.reshape(-1, output.shape[-1]).to("cpu", torch.float32)
        abr = x @ self.Br
        if self.br is not None:
            abr = abr + self.br
        cr = y @ self.r
        return _compare(abr, cr, y, self.tol)


def attention_score_check(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    *,
    s: int = DEFAULT_S,
    tol: float = DEFAULT_TOL,
    generator: torch.Generator | None = None,
) -> tuple[bool, float]:
    """Verify ``scores == query @ key^T`` (raw, pre-scaling).

    Shapes: ``query`` ``(b, h, q, d)``, ``key`` ``(b, h, kv, d)`` (already
    repeated to the query head count), ``scores`` ``(b, h, q, kv)``.
    """
    query = query.to("cpu", torch.float32)
    key = key.to("cpu", torch.float32)
    scores = scores.to("cpu", torch.float32)
    kv_len = key.shape[-2]
    r1 = make_r(kv_len, s, generator=generator)  # (kv, s)
    kr = key.transpose(-1, -2) @ r1  # (b, h, d, s)
    abr = query @ kr  # (b, h, q, s)
    cr = scores @ r1  # (b, h, q, s)
    return _compare(abr, cr, scores, tol)


def attention_value_check(
    probs: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    s: int = DEFAULT_S,
    tol: float = DEFAULT_TOL,
    generator: torch.Generator | None = None,
) -> tuple[bool, float]:
    """Verify ``out == probs @ value`` where ``probs`` is the softmax matrix.

    Shapes: ``probs`` ``(b, h, q, kv)`` (recomputed on CPU), ``value``
    ``(b, h, kv, d)`` (repeated), ``out`` ``(b, h, q, d)``.
    """
    probs = probs.to("cpu", torch.float32)
    value = value.to("cpu", torch.float32)
    out = out.to("cpu", torch.float32)
    head_dim = value.shape[-1]
    r2 = make_r(head_dim, s, generator=generator)  # (d, s)
    vr = value @ r2  # (b, h, kv, s)
    abr = probs @ vr  # (b, h, q, s)
    cr = out @ r2  # (b, h, q, s)
    return _compare(abr, cr, out, tol)


def _selftest() -> int:
    torch.manual_seed(0)
    failures = 0

    def report(label: str, ok: bool, ratio: float, expect_ok: bool) -> None:
        nonlocal failures
        good = ok == expect_ok
        failures += 0 if good else 1
        print(f"[{'PASS' if good else 'FAIL'}] {label}: ok={ok} ratio={ratio:.2e} (expected ok={expect_ok})")

    # --- Linear (fp16 prover, fp32 checker) ---
    n, in_f, out_f = 32, 4096, 4096
    W = torch.randn(out_f, in_f, dtype=torch.float16)
    b = torch.randn(out_f, dtype=torch.float16)
    x = torch.randn(n, in_f, dtype=torch.float16)
    y = (x.float() @ W.float().t() + b.float()).half()  # fp16 "GPU" output
    chk = LinearChecker(W, b, name="lin")
    report("linear correct", *chk.check(x, y), expect_ok=True)
    # A faulty matmul yields errors that scale with the output magnitude; a
    # single tiny element flip (below tol * ||row||_2) is correctly *not*
    # flagged, so corrupt a whole row to an unrelated value of similar norm.
    y_bad = y.clone()
    y_bad[3] = torch.randn_like(y_bad[3]) * y_bad[3].float().std()
    report("linear corrupted", *chk.check(x, y_bad), expect_ok=False)

    # --- Attention score & value ---
    bsz, h, q, kv, d = 2, 8, 5, 17, 64
    query = torch.randn(bsz, h, q, d, dtype=torch.float16)
    key = torch.randn(bsz, h, kv, d, dtype=torch.float16)
    value = torch.randn(bsz, h, kv, d, dtype=torch.float16)
    scores = (query.float() @ key.float().transpose(-1, -2)).half()
    report("attn score correct", *attention_score_check(query, key, scores), expect_ok=True)
    scores_bad = scores.clone()
    scores_bad[0, 0, 0, 0] += 50.0
    report("attn score corrupted", *attention_score_check(query, key, scores_bad), expect_ok=False)

    probs = torch.softmax(scores.float() / (d**0.5), dim=-1)
    out = (probs @ value.float()).half()
    report("attn value correct", *attention_value_check(probs, value, out), expect_ok=True)
    out_bad = out.clone()
    out_bad[0, 0, 0, 0] += 50.0
    report("attn value corrupted", *attention_value_check(probs, value, out_bad), expect_ok=False)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(1 if _selftest() else 0)
    print("usage: python freivalds.py --selftest")
