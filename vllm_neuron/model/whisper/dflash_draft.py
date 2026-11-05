# SPDX-License-Identifier: Apache-2.0
"""DFlash block-parallel draft for a Whisper (encoder-decoder) target.

The draft is a small Qwen3-style stack (DFlash release format) with two contexts:

* the **decoder context**: the target decoder hidden states at ``target_layer_ids``
  (the output of those layers, before the final layer norm), fused by ``fc`` and
  ``hidden_norm`` and projected into each draft layer's K/V with RoPE at the token
  position (standard DFlash);
* the **audio context** (``dflash_config.audio_context = "own_fc"``): the target encoder
  output ``e`` ``[1500, d]``, ``audio_stack`` consecutive frames concatenated,
  ``A = audio_norm(audio_fc(e.reshape(1500 / stack, stack * d)))``. Each draft layer
  projects ``A`` with its own ``k_proj`` / ``k_norm`` / ``v_proj`` and **no RoPE**.
  These K/V depend only on the encoder output, so they are computed one time per
  request, after the encoder, like Whisper's cross-attention K/V.

Every block query sees all audio positions, the decoder context before the block
(positions ``< seq_len``) and the whole block (non-causal).

Storage (model-owned, one region per request slot, the target's cross-KV slots):

* ``ctx_k_{i}`` / ``ctx_v_{i}``: ``[slots * max_ctx, kv_heads_local * head_dim]``; the
  context K/V of position ``p`` of slot ``s`` is row ``s * max_ctx + p``. A verify step
  writes the K/V of all its target rows. The rows of rejected drafts are hidden by the
  ``< seq_len`` mask and written over by the next step.
* ``audio_k_{i}`` / ``audio_v_{i}``: ``[slots, audio_positions, kv_heads_local * head_dim]``.

The block's own K/V are not stored. Attention is plain torch (16 queries per request
over ``audio + max_ctx + 16`` keys, fp32 softmax). The token embedding and LM head are
the target's (``bind_target``): the tied ``decoder.embed_tokens`` and the vocab-parallel
``lm_head`` with its pad-column mask.

Tensor parallelism: q heads, kv heads and the MLP intermediate dim are split over the
TP ranks (row-parallel ``o_proj`` / ``down_proj`` with an all-reduce). ``fc``,
``audio_fc`` and the norms are replicated.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.model.qwen3.dflash_model import (
    _correct_target_positions_for_async,
    _current_seq_lens,
)

AUDIO_FRAMES = 1500


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm as in the training reference: normalize in fp32, cast, then scale."""
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(dt)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    """Neox (rotate-half) RoPE tables ``[T, head_dim]`` for integer ``positions`` ``[T]``."""
    inv = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    f = positions.float().unsqueeze(-1) * inv
    emb = torch.cat([f, f], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` ``[T, heads, d]``, ``cos`` / ``sin`` ``[T, d]``."""
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def load_draft_config(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)


class _Layer(nn.Module):
    def __init__(self, H, nh_l, nkv_l, hd, I_l, dtype):
        super().__init__()

        def p(*shape):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        self.input_layernorm = p(H)
        self.post_attention_layernorm = p(H)
        self.q_norm = p(hd)
        self.k_norm = p(hd)
        # [in, out] weights of this rank.
        self.wq = p(H, nh_l * hd)
        self.wk = p(H, nkv_l * hd)
        self.wv = p(H, nkv_l * hd)
        self.wo = p(nh_l * hd, H)
        self.wg = p(H, I_l)
        self.wu = p(H, I_l)
        self.wd = p(I_l, H)


class WhisperDFlashDraft(nn.Module):
    """The compiled DFlash proposal graph for a Whisper target (see the module docstring)."""

    # The proposer reads these.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(
        self,
        cfg: dict,
        num_slots: int,
        max_ctx: int = 448,
        world: int = 1,
        rank: int = 0,
        tp_group=None,
        dtype: torch.dtype = torch.bfloat16,
        on_device_sampling: bool = True,
    ):
        super().__init__()
        dc = cfg.get("dflash_config", {}) or {}
        self.H = cfg["hidden_size"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim", self.H // self.nh)
        self.n_layers = cfg["num_hidden_layers"]
        self.inter = cfg["intermediate_size"]
        self.eps = float(cfg["rms_norm_eps"])
        theta = cfg.get("rope_theta")
        if theta is None:
            theta = (cfg.get("rope_parameters") or {}).get("rope_theta")
        if theta is None:
            raise ValueError("The DFlash draft config has no rope_theta.")
        self.theta = float(theta)
        self.block_size = int(cfg.get("block_size", dc.get("block_size")))
        self.mask_token_id = int(dc.get("mask_token_id", cfg.get("mask_token_id")))
        self.target_layer_ids = list(dc.get("target_layer_ids", cfg.get("target_layer_ids")))
        self.target_hidden = int(cfg.get("target_hidden_size", self.H))
        self.vocab_size = int(cfg["vocab_size"])
        self.audio_context = dc.get("audio_context", "none")
        if self.audio_context not in ("none", "own_fc"):
            raise NotImplementedError(
                f"Whisper DFlash audio_context={self.audio_context!r} is not supported "
                "(supported: own_fc, none)."
            )
        self.audio_stack = int(dc.get("audio_stack", 4))
        self.n_audio = AUDIO_FRAMES // self.audio_stack if self.audio_context != "none" else 0
        self.num_speculative_tokens = self.block_size - 1

        if self.nh % world or self.nkv % world or self.inter % world:
            raise ValueError(
                f"Whisper DFlash draft heads {self.nh}/{self.nkv} and ffn {self.inter} "
                f"must divide by TP {world}."
            )
        self.world, self.rank, self.tp_group = world, rank, tp_group
        self.nh_l, self.nkv_l, self.inter_l = self.nh // world, self.nkv // world, self.inter // world
        self.rep = self.nh_l // self.nkv_l
        self.num_slots = num_slots
        self.max_ctx = max_ctx

        def p(*shape):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        # The fc weight carries the dtype (see dtype_): a torch.dtype attribute is
        # saved as _Missing in a precompile package.
        nf = len(self.target_layer_ids)
        self.fc = p(nf * self.target_hidden, self.H)
        self.hidden_norm = p(self.H)
        self.norm = p(self.H)
        if self.audio_context == "own_fc":
            self.audio_fc = p(self.audio_stack * self.target_hidden, self.H)
            self.audio_norm = p(self.H)
        self.layers = nn.ModuleList(
            [_Layer(self.H, self.nh_l, self.nkv_l, self.hd, self.inter_l, dtype) for _ in range(self.n_layers)]
        )
        kvw = self.nkv_l * self.hd
        for i in range(self.n_layers):
            for kind in ("k", "v"):
                self.register_buffer(
                    f"ctx_{kind}_{i}", torch.zeros(num_slots * max_ctx, kvw, dtype=dtype), persistent=False
                )
                if self.n_audio:
                    self.register_buffer(
                        f"audio_{kind}_{i}", torch.zeros(num_slots, self.n_audio, kvw, dtype=dtype),
                        persistent=False,
                    )
        self.sampler = None
        if on_device_sampling:
            from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
            from vllm_neuron.nn.sampler import Sampler

            self.sampler = Sampler(OnDeviceSamplingConfig(all_greedy=True), process_group=tp_group)
        self.embed_weight = None
        self.lm_weight = None
        self.lm_mode = "fp32"
        self.lm_in_out = False
        self.per_rank = self.vocab_size
        self.vocab_pad = 0
        # WHISPER_DFLASH_LM: "bf16" (default) runs the draft LM head as a bf16 GEMM on a bf16
        # weight (no fp32 upcast of the vocab x d weight); "target" uses the target's LM-head
        # precision. Draft logits only pick the proposals; the verifier decides the output.
        self.draft_lm = os.environ.get("WHISPER_DFLASH_LM", "bf16")
        if self.draft_lm not in ("bf16", "target"):
            raise ValueError(f"WHISPER_DFLASH_LM={self.draft_lm!r}: use bf16 or target.")
        # With one request slot (+ scratch), read slot 0 directly as the target does: a
        # gather by slot is an indirect DMA.
        self.single_slot = num_slots <= 2

    @property
    def dtype_(self) -> torch.dtype:
        return self.fc.dtype

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def load_weights(self, path: str) -> None:
        """Load the release safetensors and keep this rank's shard of each tensor."""
        from safetensors import safe_open

        r = self.rank
        hd = self.hd
        qs = slice(r * self.nh_l * hd, (r + 1) * self.nh_l * hd)
        ks = slice(r * self.nkv_l * hd, (r + 1) * self.nkv_l * hd)
        fs = slice(r * self.inter_l, (r + 1) * self.inter_l)
        seen = set()
        with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
            keys = set(f.keys())

            def get(k):
                seen.add(k)
                return f.get_tensor(k).to(self.dtype_)

            self.fc.copy_(get("fc.weight").t())
            self.hidden_norm.copy_(get("hidden_norm.weight"))
            self.norm.copy_(get("norm.weight"))
            if self.audio_context == "own_fc":
                self.audio_fc.copy_(get("audio_fc.weight").t())
                self.audio_norm.copy_(get("audio_norm.weight"))
            for i, L in enumerate(self.layers):
                pre = f"layers.{i}."
                L.input_layernorm.copy_(get(pre + "input_layernorm.weight"))
                L.post_attention_layernorm.copy_(get(pre + "post_attention_layernorm.weight"))
                L.q_norm.copy_(get(pre + "self_attn.q_norm.weight"))
                L.k_norm.copy_(get(pre + "self_attn.k_norm.weight"))
                L.wq.copy_(get(pre + "self_attn.q_proj.weight")[qs].t())
                L.wk.copy_(get(pre + "self_attn.k_proj.weight")[ks].t())
                L.wv.copy_(get(pre + "self_attn.v_proj.weight")[ks].t())
                L.wo.copy_(get(pre + "self_attn.o_proj.weight")[:, qs].t())
                L.wg.copy_(get(pre + "mlp.gate_proj.weight")[fs].t())
                L.wu.copy_(get(pre + "mlp.up_proj.weight")[fs].t())
                L.wd.copy_(get(pre + "mlp.down_proj.weight")[:, fs].t())
        extra = keys - seen
        if extra:
            raise RuntimeError(f"Whisper DFlash draft: unexpected checkpoint tensors {sorted(extra)}")

    def bind_target(self, target: nn.Module) -> None:
        """Share the target's tied token embedding and vocab-parallel LM head (no copy).

        The LM head keeps the target's storage (fp32 or bf16, ``[in, out]`` or ``[out, in]``)
        and its pad columns (vocab padded to a multiple of TP), so the draft logits are the
        target's logits of the draft hidden states, with the pad columns at -inf.
        """
        target = getattr(target, "_orig_mod", target)
        self.embed_weight = target.decoder.embed_tokens.weight
        self.lm_weight = target.lm_head.weight
        self.lm_mode = target.lm_mode
        self.lm_in_out = getattr(target.lm_head, "weight_layout", "out_in") == "in_out"
        self.per_rank = target.per_rank
        self.vocab_pad = target.vocab_pad
        if target.config.vocab_size != self.vocab_size:
            raise ValueError(
                f"Draft vocab {self.vocab_size} != target vocab {target.config.vocab_size}"
            )

    # The runner asks the drafter for paged-KV layers. The draft keeps its own K/V.
    def get_kv_spec(self):
        from vllm_neuron.model.kv_cache import KVSpec

        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches) -> None:
        return None

    # ------------------------------------------------------------------ pieces
    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world == 1:
            return x
        from torch.distributed._functional_collectives import all_reduce

        return all_reduce(x, reduceOp="sum", group=self.tp_group)

    def _k(self, L: _Layer, x: torch.Tensor) -> torch.Tensor:
        """``x`` ``[T, H]`` -> normed keys ``[T, nkv_l, hd]`` (no RoPE)."""
        k = torch.matmul(x, L.wk).view(x.shape[0], self.nkv_l, self.hd)
        return rms_norm(k, L.k_norm, self.eps)

    def audio_context_states(self, enc: torch.Tensor) -> torch.Tensor:
        """Encoder output ``[n, 1500, H_t]`` -> fused audio context ``[n, positions, H]``."""
        n = enc.shape[0]
        x = enc.to(self.dtype_).reshape(n, self.n_audio, self.audio_stack * self.target_hidden)
        return rms_norm(torch.matmul(x, self.audio_fc), self.audio_norm, self.eps)

    @torch.no_grad()
    def write_audio_kv(self, enc: torch.Tensor, slots: torch.Tensor) -> None:
        """Write the audio K/V of each clip ``enc[j]`` into slot ``slots[j]`` of every layer."""
        if not self.n_audio:
            return
        n = enc.shape[0]
        A = self.audio_context_states(enc).reshape(n * self.n_audio, self.H)
        slots = slots.to(torch.int64)
        for i, L in enumerate(self.layers):
            k = self._k(L, A).reshape(n, self.n_audio, self.nkv_l * self.hd)
            v = torch.matmul(A, L.wv).reshape(n, self.n_audio, self.nkv_l * self.hd)
            getattr(self, f"audio_k_{i}").index_put_((slots,), k)
            getattr(self, f"audio_v_{i}").index_put_((slots,), v)

    @torch.no_grad()
    def write_context_kv(self, feats: torch.Tensor, positions: torch.Tensor, rows: torch.Tensor) -> None:
        """Project target features ``[T, n_feat * H_t]`` at ``positions`` into buffer ``rows``."""
        ctx = rms_norm(torch.matmul(feats.to(self.dtype_), self.fc), self.hidden_norm, self.eps)
        cos, sin = rope_cos_sin(positions, self.hd, self.theta, ctx.dtype)
        T = ctx.shape[0]
        for i, L in enumerate(self.layers):
            k = apply_rope(self._k(L, ctx), cos, sin).reshape(T, -1)
            v = torch.matmul(ctx, L.wv)
            getattr(self, f"ctx_k_{i}").index_put_((rows,), k)
            getattr(self, f"ctx_v_{i}").index_put_((rows,), v)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.to(torch.int64), self.embed_weight)

    def block_hidden(
        self, ids: torch.Tensor, q_pos: torch.Tensor, slots: torch.Tensor, seq_lens: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the draft block. ``ids`` / ``q_pos`` ``[B * S]``, ``slots`` / ``seq_lens`` ``[B]``.

        Returns the final hidden states ``[B * S, H]`` (after ``norm``).
        """
        B = slots.shape[0]
        T = q_pos.shape[0]
        S = T // B
        hd, nkv, rep, C = self.hd, self.nkv_l, self.rep, self.max_ctx
        if h is None:
            h = self.embed(ids)
        h = h.to(self.dtype_)
        cos, sin = rope_cos_sin(q_pos, hd, self.theta, h.dtype)
        slots = slots.to(torch.int64)
        j = torch.arange(C, device=h.device, dtype=torch.int32)
        ctx_ok = j.view(1, C) < seq_lens.to(torch.int32).view(B, 1)
        neg = torch.full((), float("-inf"), dtype=torch.float32, device=h.device)
        zero = torch.zeros((), dtype=torch.float32, device=h.device)
        bias = torch.cat(
            [
                torch.zeros(B, self.n_audio, dtype=torch.float32, device=h.device),
                torch.where(ctx_ok, zero, neg),
                torch.zeros(B, S, dtype=torch.float32, device=h.device),
            ],
            dim=1,
        ).view(B, 1, 1, -1)
        scale = hd ** -0.5
        for i, L in enumerate(self.layers):
            hn = rms_norm(h, L.input_layernorm, self.eps)
            q = torch.matmul(hn, L.wq).view(T, self.nh_l, hd)
            q = apply_rope(rms_norm(q, L.q_norm, self.eps), cos, sin)
            kb = apply_rope(self._k(L, hn), cos, sin)
            vb = torch.matmul(hn, L.wv).view(T, nkv, hd)
            ks = []
            vs = []
            for kind, out in (("k", ks), ("v", vs)):
                a = getattr(self, f"audio_{kind}_{i}") if self.n_audio else None
                c = getattr(self, f"ctx_{kind}_{i}").view(self.num_slots, C, nkv, hd)
                if self.single_slot:
                    if a is not None:
                        out.append(a[:1].view(1, self.n_audio, nkv, hd).expand(B, -1, -1, -1))
                    out.append(c[:1].expand(B, -1, -1, -1))
                else:
                    if a is not None:
                        out.append(a.index_select(0, slots).view(B, self.n_audio, nkv, hd))
                    out.append(c.index_select(0, slots))
            ks.append(kb.view(B, S, nkv, hd))
            vs.append(vb.view(B, S, nkv, hd))
            K = torch.cat(ks, dim=1).permute(0, 2, 1, 3).float()  # [B, nkv, Nk, hd]
            V = torch.cat(vs, dim=1).permute(0, 2, 1, 3).float()
            Q = q.view(B, S, nkv, rep, hd).permute(0, 2, 3, 1, 4).reshape(B, nkv, rep * S, hd).float()
            sc = torch.matmul(Q, K.transpose(-1, -2)) * scale + bias
            pr = torch.softmax(sc, dim=-1)
            o = torch.matmul(pr, V).to(h.dtype)  # [B, nkv, rep * S, hd]
            o = o.view(B, nkv, rep, S, hd).permute(0, 3, 1, 2, 4).reshape(T, self.nh_l * hd)
            h = h + self._all_reduce(torch.matmul(o, L.wo))
            m = rms_norm(h, L.post_attention_layernorm, self.eps)
            m = torch.matmul(F.silu(torch.matmul(m, L.wg)) * torch.matmul(m, L.wu), L.wd)
            h = h + self._all_reduce(m)
        return rms_norm(h, self.norm, self.eps)

    def logits_local(self, hidden: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        """fp32 logits of this rank's vocab slice, pad columns at -inf (as the target)."""
        w = self.lm_weight
        if self.draft_lm == "bf16" and w.dtype == torch.bfloat16:
            x = hidden.to(torch.bfloat16)
        elif self.lm_mode == "fp32":
            x, w = hidden.to(torch.float32), w
        elif self.lm_mode == "bf16":
            x = hidden.to(torch.bfloat16)
        else:  # bf16w: bf16 storage, fp32 GEMM
            x, w = hidden.to(torch.float32), w.to(torch.float32)
        local = torch.matmul(x, w) if self.lm_in_out else F.linear(x, w)
        local = local.to(torch.float32)
        if self.vocab_pad:
            off = rank.to(device=local.device, dtype=torch.int32).reshape(()) * self.per_rank
            idx = torch.arange(self.per_rank, device=local.device, dtype=torch.int32) + off
            neg = torch.full((), float("-inf"), dtype=local.dtype, device=local.device)
            local = torch.where(idx < self.vocab_size, local, neg)
        return local

    def _sample(self, hidden: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
        local = self.logits_local(hidden, rank)
        if self.sampler is None:
            return torch.argmax(local, dim=-1).to(torch.int32)
        return self.sampler(local, None, logit_mask=None, tp_rank=rank).to(torch.int32)

    # ------------------------------------------------------------------ graph
    @torch.no_grad()
    def forward(
        self,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
        request_slots: torch.Tensor,
        rank: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor,
        prev_num_draft_tokens: torch.Tensor,
        req_indices_per_token: torch.Tensor,
        cur_num_draft_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One proposal: write the context K/V of the target rows, run the block, sample K drafts.

        Returns ``(bonus + drafts [B, 1 + K], drafts [B, K])``.
        """
        B = last_token_indices.shape[0]
        dev = target_hidden_states.device
        S = self.block_size
        V = self.vocab_size
        C = self.max_ctx
        raw = raw_sampled_token_ids.to(torch.int32)
        valid_count = ((raw != -1) & (raw < V)).sum(dim=1)
        # Async scheduling: the positions assume full acceptance of the previous step.
        positions = _correct_target_positions_for_async(
            target_positions.to(torch.int64),
            prev_sampled_token_ids,
            prev_num_draft_tokens,
            req_indices_per_token,
            V,
        )
        # A padded row can get a negative position (see the target correction).
        positions = positions.clamp(min=0, max=C - 1)
        bonus_pos = positions[last_token_indices]
        seq_lens = _current_seq_lens(
            bonus_pos, raw, prev_sampled_token_ids, prev_num_draft_tokens, cur_num_draft_tokens,
            V, self.num_speculative_tokens, valid_per_req=valid_count,
        ).clamp(max=C)

        slots = request_slots[:B].to(torch.int64)
        if self.single_slot:
            # One request: it is in slot 0 (the target reads slot 0 the same way).
            slots = torch.zeros_like(slots)
        # Context rows: the request of each target row; rows after the last request are padding.
        T = target_hidden_states.shape[0]
        rows_i = torch.arange(T, device=dev)
        req = (rows_i[:, None] > last_token_indices[None, :]).sum(dim=1)
        is_pad = req >= B
        req = req.clamp(max=B - 1)
        row_slot = torch.where(
            is_pad, torch.full_like(req, self.num_slots - 1), slots.index_select(0, req)
        )
        self.write_context_kv(target_hidden_states, positions, row_slot * C + positions)

        # Block [bonus, mask, ..., mask] at positions seq_len .. seq_len + S - 1.
        last_valid = (valid_count - 1).clamp(min=0).to(torch.int64)
        bonus = raw.gather(1, last_valid.view(B, 1)).view(B)
        bonus = torch.where(valid_count > 0, bonus, torch.zeros_like(bonus))
        ids = torch.cat(
            [bonus.view(B, 1), torch.full((B, S - 1), self.mask_token_id, dtype=torch.int32, device=dev)],
            dim=1,
        ).reshape(-1)
        q_pos = (seq_lens.view(B, 1) + torch.arange(S, device=dev).view(1, S)).reshape(-1)
        hidden = self.block_hidden(ids, q_pos, slots, seq_lens)
        keep = (torch.arange(B, device=dev).view(B, 1) * S + torch.arange(1, S, device=dev).view(1, S - 1)).reshape(-1)
        drafts = self._sample(hidden.index_select(0, keep), rank).view(B, S - 1)
        return torch.cat([bonus.view(B, 1).to(torch.int32), drafts], dim=1), drafts


class DraftAudioKV(nn.Module):
    """Compiled graph: the draft audio K/V of each clip, from the target encoder output."""

    def __init__(self, owner: WhisperDFlashDraft):
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def forward(self, encoder_hidden: torch.Tensor, write_block_ids: torch.Tensor) -> torch.Tensor:
        self._owner.write_audio_kv(encoder_hidden, write_block_ids)
        return encoder_hidden[:1, :1, :1]
