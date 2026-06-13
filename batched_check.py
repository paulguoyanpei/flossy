"""Batched Freivalds verification across transformer layers.

The per-layer checks are tiny matmuls, so running ~250 of them per step one at a
time is dominated by Python/framework overhead. Every layer has identical
shapes, so we instead *defer* the checks during the residual-stream
reconstruction and run them **stacked across all layers** as a few batched
``bmm``s -- collapsing ~250 Python calls into ~8.

Setup stacks each projection's precomputed ``Br`` / ``r`` (from the per-layer
``LinearChecker``s) into one tensor of shape ``(L, in, s)`` / ``(L, out, s)``.
Per step we collect each layer's reconstructed input and the GPU output (in
layer order, which is the module execution order), then:

    ABr = bmm(X, Br) (+ bias·r)      Cr = bmm(Y, r)
    fail if  max_s|ABr - Cr| > tol · ||Y||_2   (per row)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freivalds import make_r


def _layer_of(name: str) -> tuple[int, str]:
    parts = name.split(".")
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    return -1, name  # e.g. "lm_head" (its own singleton group)


class BatchedVerifier:
    def __init__(self, ctx, targets: dict, checkers: dict) -> None:
        self.ctx = ctx
        self.s = ctx.s
        self.tol = ctx.tol
        self.generator = ctx.generator

        # group projections by type, ordered by layer index
        by_type: dict[str, list[tuple[int, str]]] = {}
        for name in targets:
            layer, proj = _layer_of(name)
            by_type.setdefault(proj, []).append((layer, name))

        self.groups: dict[str, dict] = {}
        for proj, items in by_type.items():
            items.sort()
            names = [n for _, n in items]
            chs = [checkers[n] for n in names]
            Br = torch.stack([c.Br for c in chs])  # (L, in, s)
            r = torch.stack([c.r for c in chs])  # (L, out, s)
            has_bias = any(c.br is not None for c in chs)
            br = (
                torch.stack([c.br if c.br is not None else torch.zeros(self.s) for c in chs]).unsqueeze(1)
                if has_bias
                else None
            )  # (L, 1, s)
            self.groups[proj] = {"names": names, "Br": Br, "r": r, "br": br}
        self.reset()

    def reset(self) -> None:
        self._lin: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {p: [] for p in self.groups}
        self._attn: list[tuple] = []

    # ---- collection (called from the verifier forward hooks) ----
    def collect_linear(self, name: str, x: torch.Tensor, out: torch.Tensor) -> None:
        _, proj = _layer_of(name)
        self._lin[proj].append((x, out))

    def collect_attn(self, layer: int, q, k_r, v_r, mask, scaling, out) -> None:
        self._attn.append((layer, q, k_r, v_r, mask, scaling, out))

    # ---- batched checks (called once per step in end_step) ----
    def run(self) -> None:
        for proj, g in self.groups.items():
            items = self._lin[proj]
            if not items:
                continue
            X = torch.stack([x.reshape(-1, x.shape[-1]).to(torch.float32) for x, _ in items])  # (L,n,in)
            Y = torch.stack([o.reshape(-1, o.shape[-1]).to(torch.float32) for _, o in items])  # (L,n,out)
            abr = torch.bmm(X, g["Br"])  # (L,n,s)
            if g["br"] is not None:
                abr = abr + g["br"]
            cr = torch.bmm(Y, g["r"])  # (L,n,s)
            self._record_batch(g["names"], abr, cr, Y)

        if self._attn:
            self._run_attn()
        self.reset()

    def _record_batch(self, names: list[str], abr: torch.Tensor, cr: torch.Tensor, ref: torch.Tensor) -> None:
        # abr, cr: (L, n, s); ref: (L, n, out). Ratio per layer = worst over rows.
        l2 = ref.norm(p=2, dim=-1)  # (L, n)
        diff = (abr - cr).abs().amax(dim=-1)  # (L, n)
        ratio = (diff / (l2 + 1e-9)).amax(dim=-1)  # (L,)
        self._emit(names, ratio)

    def _run_attn(self) -> None:
        self._attn.sort(key=lambda a: a[0])
        names = [f"{a[0]}.attn_out" for a in self._attn]
        Q = torch.stack([a[1].to(torch.float32) for a in self._attn])  # (L,b,h,q,d)
        K = torch.stack([a[2].to(torch.float32) for a in self._attn])  # (L,b,h,kv,d)
        V = torch.stack([a[3].to(torch.float32) for a in self._attn])  # (L,b,h,kv,d)
        O = torch.stack([a[6].to(torch.float32) for a in self._attn])  # (L,b,h,q,d)
        mask, scaling = self._attn[0][4], self._attn[0][5]

        scores = torch.matmul(Q, K.transpose(-1, -2)) * scaling  # (L,b,h,q,kv)
        if mask is not None:
            scores = scores + mask[..., : K.shape[-2]].to(torch.float32).unsqueeze(0)
        probs = F.softmax(scores, dim=-1)

        rv = make_r(V.shape[-1], self.s, generator=self.generator)  # (d, s)
        abr = torch.matmul(probs, torch.matmul(V, rv))  # (L,b,h,q,s)
        cr = torch.matmul(O, rv)  # (L,b,h,q,s)
        l2 = O.norm(p=2, dim=-1)  # (L,b,h,q)
        diff = (abr - cr).abs().amax(dim=-1)  # (L,b,h,q)
        ratio = (diff / (l2 + 1e-9)).flatten(1).amax(dim=-1)  # (L,)
        self._emit(names, ratio)

    def _emit(self, names: list[str], ratio: torch.Tensor) -> None:
        ctx = self.ctx
        ratios = ratio.tolist()
        ctx.n_checks += len(ratios)
        worst_i = int(ratio.argmax())
        if ratios[worst_i] > ctx.worst_ratio:
            ctx.worst_ratio = ratios[worst_i]
            ctx.worst_name = names[worst_i]
        if ratio.max().item() > ctx.tol:  # only walk per-layer on a failure
            for name, rt in zip(names, ratios):
                if rt > ctx.tol:
                    ctx.n_failures += 1
                    ctx.failures.append((ctx.step, name, rt))
