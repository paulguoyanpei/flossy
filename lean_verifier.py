"""Lean verifier: verify a step without running the HF ``generate`` forward.

The residual stream never needs the sequential model forward: every block's
contribution is a transferred GPU output, so

    h_i = embed(tokens) + sum_{j<i} (o_proj_out_j + down_proj_out_j)

is a prefix sum over the (transferred) per-layer block outputs. Given ``h_i``
for every layer, all per-layer check inputs are reconstructable with only the
cheap nonlinear ops (RMSNorm, RoPE, softmax), and every layer is independent —
so the whole step is a handful of batched ops over the layer dimension, with no
HF generate loop / causal-mask vmap / per-module dispatch.

This removes the framework overhead. Assumes the Llama/Qwen3 decoder structure
(pre-norm, gated MLP, optional q/k norm); reuses the loaded model's embed/rope
modules and norm/projection weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from batched_check import BatchedVerifier
from freivalds import make_r


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def groups_from_stepbuffer(buf) -> dict:
    """Build the grouped-by-projection dict (proj -> (count, *shape)) from a
    per-name StepBuffer, by stacking over layers. Used for in-memory tests; the
    two-process path gets the same dict zero-copy via ``transport.read_grouped``.
    """
    tmp = {}
    for name in buf.order:
        if "." in name:
            li, proj = name.split(".", 1)
            tmp.setdefault(proj, []).append((int(li), buf.fetch(name)))
        else:
            tmp.setdefault(name, []).append((-1, buf.fetch(name)))
    return {proj: torch.stack([t for _, t in sorted(items)]) for proj, items in tmp.items()}


def _batched_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim. x: (L, ..., d); weight: (L, d) -> broadcast."""
    var = x.pow(2).mean(-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    wshape = [weight.shape[0]] + [1] * (x.ndim - 2) + [weight.shape[-1]]
    return xn * weight.view(*wshape)


class LeanVerifier:
    def __init__(self, model, s: int, tol: float, generator=None, layers: range | None = None,
                 checks_lmhead: bool | None = None):
        cfg = model.config
        self.cfg = cfg
        self.s = s
        self.tol = tol
        self.generator = generator
        self.eps = cfg.rms_norm_eps
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        self.n_rep = self.n_heads // self.n_kv
        self.scaling = self.head_dim**-0.5

        base = model.model
        self.embed = base.embed_tokens
        self.rotary = base.rotary_emb
        self.final_norm = base.norm
        decoder = base.layers
        self.n_layers_full = len(decoder)
        # contiguous layer slice this verifier owns (all layers in single-process);
        # a multi-process worker owns [lo, hi) and reconstructs the boundary prefix.
        self.layer_ids = list(layers) if layers is not None else list(range(len(decoder)))
        self.lo, self.hi = self.layer_ids[0], self.layer_ids[-1] + 1
        self.full = self.layer_ids == list(range(len(decoder)))
        # only the worker owning the final slice checks lm_head; all workers still
        # derive the next token from the transferred lm_head (lockstep embedding).
        self.checks_lmhead = self.full if checks_lmhead is None else checks_lmhead

        sa0 = decoder[self.layer_ids[0]].self_attn
        self.has_qk_norm = hasattr(sa0, "q_norm")

        def stack_w(getter):
            return torch.stack([getter(decoder[i]).detach().float() for i in self.layer_ids])

        self.W_input_ln = stack_w(lambda l: l.input_layernorm.weight)
        self.W_post_ln = stack_w(lambda l: l.post_attention_layernorm.weight)
        if self.has_qk_norm:
            self.W_q_norm = stack_w(lambda l: l.self_attn.q_norm.weight)
            self.W_k_norm = stack_w(lambda l: l.self_attn.k_norm.weight)

        # reuse BatchedVerifier only for its grouped Br/r + stats emission
        from capture import FlossyContext

        self.ctx = FlossyContext(role="verifier", s=s, tol=tol, generator=generator)
        targets = {}
        checkers = {}
        from freivalds import LinearChecker

        for i in self.layer_ids:
            sa, mlp = decoder[i].self_attn, decoder[i].mlp
            for proj, mod in [("q_proj", sa.q_proj), ("k_proj", sa.k_proj), ("v_proj", sa.v_proj),
                              ("o_proj", sa.o_proj), ("gate_proj", mlp.gate_proj),
                              ("up_proj", mlp.up_proj), ("down_proj", mlp.down_proj)]:
                name = f"{i}.{proj}"
                targets[name] = mod
                checkers[name] = LinearChecker(mod.weight, getattr(mod, "bias", None),
                                               s=s, tol=tol, name=name, generator=generator)
        if self.checks_lmhead:
            targets["lm_head"] = model.lm_head
            checkers["lm_head"] = LinearChecker(model.lm_head.weight, None, s=s, tol=tol,
                                                name="lm_head", generator=generator)
        self.bv = BatchedVerifier(self.ctx, targets, checkers)

        self.cache_k = None  # (L, n_kv, kv_len, head_dim)
        # FLOSSY cache-reuse (paper Sec 5.2): the value-check projection r is fixed
        # for the whole request (secret -> still sound, paper App. D) so V.r can be
        # cached incrementally -- each step only the new rows v_t.r are computed and
        # appended, instead of recomputing the full V.r (the dominant ~L*head_dim*s
        # term). cache_Vr is kept at the n_kv (pre-GQA) level.
        self.rv = make_r(self.head_dim, self.s, generator=self.generator)  # (head_dim, s), fixed
        self.cache_Vr = None  # (L, n_kv, kv_len, s)
        # Stage 2 (paper 5.2, opt-in): when the prover ships raw QK^T scores, the
        # verifier checks them via a cached K^T b instead of recomputing q.K^T.
        # cache_Kb accumulates sum_j k_j^T b_j; b_stacked keeps the per-position
        # binary challenges to form <s_t, b>.
        self.cache_Kb = None  # (L, n_kv, head_dim, s)
        self.b_stacked = None  # (kv_len, s), per-position +/-1
        self.pos = 0
        self.next_tokens: torch.Tensor | None = None

    # convenience pass-throughs to the shared stats
    @property
    def n_checks(self):
        return self.ctx.n_checks

    @property
    def n_failures(self):
        return self.ctx.n_failures

    @property
    def worst_ratio(self):
        return self.ctx.worst_ratio

    @property
    def worst_name(self):
        return self.ctx.worst_name

    @property
    def failures(self):
        return self.ctx.failures

    @property
    def step(self):
        return self.ctx.step

    def set_prompt(self, input_ids: torch.Tensor) -> None:
        self.next_tokens = input_ids.reshape(-1)

    @torch.inference_mode()
    def verify_step(self, groups: dict) -> None:
        """Verify one step. ``groups[proj]`` is a (count, *per_layer_shape) tensor
        (contiguous view into the shared-memory slot), already stacked over layers.
        """
        self.ctx.step += 1
        lo, hi = self.lo, self.hi
        L = hi - lo
        # linear groups are (Lf, B, S, out): B = batch, S = new positions; Lf is the
        # FULL set of transferred layers (every worker reads the whole slot). The
        # prompt is shared across the batch (greedy, replicated), so h0 is
        # reconstructed once and broadcast over B; every per-element activation is
        # still checked (B folds into the Freivalds rows / the attention batch dim).
        B, S = groups["q_proj"].shape[1], groups["q_proj"].shape[2]
        tokens = self.next_tokens
        position_ids = torch.arange(self.pos, self.pos + S).unsqueeze(0)

        # --- Residual stream for this worker's slice [lo, hi): h_i = embed +
        # boundary prefix below the slice + a local cumsum within it (not a full
        # redundant prefix-sum; single-process => lo=0 so prefix=0). ---
        o_full = groups["o_proj"].float()       # (Lf, B, S, hidden)
        down_full = groups["down_proj"].float()
        if self.pos == 0:                       # prefill: shared prompt tokens (S,)
            h0 = self.embed(tokens).float().unsqueeze(0)   # (1, S, hidden) -> bcast over B
        else:                                   # decode: per-batch next tokens (B,)
            h0 = self.embed(tokens).float().unsqueeze(1)   # (B, 1, hidden)
        prefix = (o_full[:lo].sum(0) + down_full[:lo].sum(0)) if lo else 0.0  # (B,S,hidden) or 0
        o_out = o_full[lo:hi]                    # (L, B, S, hidden)
        down_out = down_full[lo:hi]
        deltas = o_out + down_out
        local_cs = torch.cumsum(deltas, dim=0)  # (L, B, S, hidden)
        h_in = torch.empty_like(deltas)
        h_in[0] = h0 + prefix
        if L > 1:
            h_in[1:] = h0 + prefix + local_cs[:-1]

        # --- run batched checks over this worker's layers (B*S folds into rows) ---
        def gm(proj):  # owned slice: (L, B, S, dim), fp32
            return groups[proj][lo:hi].float()

        x_norm1 = _batched_rmsnorm(h_in, self.W_input_ln, self.eps)  # (L,B,S,hidden)
        self._check("q_proj", x_norm1, gm("q_proj"))
        self._check("k_proj", x_norm1, gm("k_proj"))
        self._check("v_proj", x_norm1, gm("v_proj"))

        x_norm2 = _batched_rmsnorm(h_in + o_out, self.W_post_ln, self.eps)
        gate_out, up_out = gm("gate_proj"), gm("up_proj")
        self._check("gate_proj", x_norm2, gate_out)
        self._check("up_proj", x_norm2, up_out)
        self._check("down_proj", F.silu(gate_out) * up_out, down_out)

        attn_out = gm("attn_out").view(L, B, self.n_heads, S, self.head_dim)  # (L,B,h,S,hd)
        o_in = attn_out.transpose(2, 3).reshape(L, B, S, -1)  # (L,B,S,h*hd)
        self._check("o_proj", o_in, o_out)

        # --- attention check (scores from q/K, or transferred + cached K^T b) ---
        self._attention_check(gm, position_ids, B, S, attn_out, "attn_scores" in groups)

        # --- lm_head: only the owning worker checks it; ALL derive the next token ---
        lm_out = groups["lm_head"].float()      # (1, B, S_kept, vocab)
        if self.checks_lmhead:
            h_final = h0 + prefix + local_cs[-1]            # residual after the slice
            if hi < self.n_layers_full:                     # add the suffix above hi
                h_final = h_final + o_full[hi:].sum(0) + down_full[hi:].sum(0)
            S_kept = lm_out.shape[2]
            hf = self.final_norm(h_final[:, -S_kept:].to(self.final_norm.weight.dtype)).float()
            self._check("lm_head", hf.unsqueeze(0), lm_out)
        self.next_tokens = lm_out[0, :, -1].argmax(-1)   # (B,)

        self.pos += S
        self.bv.reset()

    def _check(self, proj, X, Y):
        g = self.bv.groups[proj]
        # X: (Lg, S, in), Y: (Lg, S, out) -> flatten S into rows
        Xr = X.reshape(X.shape[0], -1, X.shape[-1])
        Yr = Y.reshape(Y.shape[0], -1, Y.shape[-1])
        abr = torch.bmm(Xr, g["Br"])
        if g["br"] is not None:
            abr = abr + g["br"]
        cr = torch.bmm(Yr, g["r"])
        self.bv._record_batch(g["names"], abr, cr, Yr)

    def _attention_check(self, g, position_ids, B, S, attn_out, has_scores):
        # All attention tensors carry an explicit batch dim B (attention does not
        # mix batch elements), so caches are (L, B, ...). Per-layer ratios reduce
        # over B as well, so a fault in any batch element fails its layer's check.
        L = len(self.layer_ids)
        cos, sin = self.rotary(attn_out, position_ids)  # (1, S, head_dim)
        cos = cos.float().view(1, 1, 1, S, self.head_dim)  # bcast (L,B,h,S,hd)
        sin = sin.float().view(1, 1, 1, S, self.head_dim)

        q = g("q_proj").view(L, B, S, self.n_heads, self.head_dim)
        k = g("k_proj").view(L, B, S, self.n_kv, self.head_dim)
        v = g("v_proj").view(L, B, S, self.n_kv, self.head_dim)
        if self.has_qk_norm:
            q = _batched_rmsnorm(q, self.W_q_norm, self.eps)
            k = _batched_rmsnorm(k, self.W_k_norm, self.eps)
        q = q.transpose(2, 3)  # (L, B, n_heads, S, hd)
        k = k.transpose(2, 3)  # (L, B, n_kv, S, hd)
        v = v.transpose(2, 3)  # (L, B, n_kv, S, hd)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # --- cache reuse: append the new V.r rows only (paper 5.2) ---
        new_Vr = torch.matmul(v, self.rv)  # (L, B, n_kv, S, s)
        if self.cache_Vr is None:
            self.cache_Vr = new_Vr
        else:
            self.cache_Vr = torch.cat([self.cache_Vr, new_Vr], dim=3)
        kv = self.cache_Vr.shape[3]

        if has_scores:
            # Stage 2: GPU shipped raw QK^T -> check it via cached K^T b, then
            # softmax the (verified) transferred scores. Avoids the O(L*head_dim)
            # q.K^T recompute (residual is O(L*s) like the paper).
            b_new = make_r(S, self.s, generator=self.generator)  # (S, s), +/-1 per new pos
            new_Kb = torch.einsum("lbnsd,se->lbnde", k, b_new)  # (L, B, n_kv, hd, s)
            if self.cache_Kb is None:
                self.cache_Kb = new_Kb
                self.b_stacked = b_new
            else:
                self.cache_Kb = self.cache_Kb + new_Kb
                self.b_stacked = torch.cat([self.b_stacked, b_new], dim=0)
            s_raw = g("attn_scores")[..., :kv]  # (L, B, n_heads, S, kv), trim static-cache pad
            # QK check: | <s_raw, b> - q (K^T b) | < tol * ||s_raw||_2
            sb = torch.einsum("lbhsk,ke->lbhse", s_raw, self.b_stacked)  # (L,B,h,S,s)
            Kb = self.cache_Kb.repeat_interleave(self.n_rep, dim=2)  # (L,B,h,hd,s)
            qKb = torch.einsum("lbhsd,lbhde->lbhse", q, Kb)  # (L,B,h,S,s)
            qk_l2 = s_raw.norm(p=2, dim=-1)  # (L,B,h,S)
            qk_diff = (sb - qKb).abs().amax(dim=-1)  # (L,B,h,S)
            qk_ratio = (qk_diff / (qk_l2 + 1e-9)).flatten(1).amax(dim=-1)  # (L,)
            self.bv._emit([f"{i}.attn_scores" for i in self.layer_ids], qk_ratio)
            scores = s_raw * self.scaling
        else:
            # Stage 1: recompute scores from q/K (no transfer).
            if self.cache_k is None:
                self.cache_k = k
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=3)
            K = self.cache_k.repeat_interleave(self.n_rep, dim=2)  # (L, B, n_heads, kv, hd)
            scores = torch.matmul(q, K.transpose(-1, -2)) * self.scaling  # (L,B,h,S,kv)
        if S > 1:
            qp = torch.arange(self.pos, self.pos + S).unsqueeze(-1)
            kp = torch.arange(kv).unsqueeze(0)
            scores = scores + torch.where(kp <= qp, 0.0, float("-inf"))
        probs = F.softmax(scores, dim=-1)

        # value check via the cached V.r (no full V.r recompute): abr = P @ (V.r)
        Vr = self.cache_Vr.repeat_interleave(self.n_rep, dim=2)  # (L, B, n_heads, kv, s)
        abr = torch.matmul(probs, Vr)  # (L,B,h,S,s)
        cr = torch.matmul(attn_out, self.rv)  # (L,B,h,S,s)
        names = [f"{i}.attn_out" for i in self.layer_ids]
        l2 = attn_out.norm(p=2, dim=-1)  # (L,B,h,S)
        diff = (abr - cr).abs().amax(dim=-1)  # (L,B,h,S)
        ratio = (diff / (l2 + 1e-9)).flatten(1).amax(dim=-1)  # (L,)
        self.bv._emit(names, ratio)
