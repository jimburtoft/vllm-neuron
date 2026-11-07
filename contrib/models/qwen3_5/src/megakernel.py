# SPDX-License-Identifier: Apache-2.0
"""Opt-in decode megakernel for Qwen3.5-35B-A3B (``QWEN3_5_DECODE_MEGAKERNEL=1``).

On a decode step with one token per request, the 40 decoder layers, the final norm and (with on-device sampling)
the vocab-parallel lm_head run as one NKI kernel per TP rank (``kernels/megakernel``) instead of ~40 x 10 graph
ops and 80 all-reduces. Prefill, chunked prefill, and every configuration the kernel does not cover keep the
model's own path; with the flag unset nothing here runs.

What the kernel reads and writes, all in place and zero-copy:

* the attention layers' paged KV caches ``[blocks, 1, 256, 256]`` (new K/V row written at ``slot_mapping``);
* the DeltaNet layers' state pages, in the layout ``Qwen3_5GatedDeltaNet.bind_state`` binds (two float32 halves,
  conv window then recurrent state), so prefill and the fallback path read what the kernel wrote and vice versa;
* the model's weights. A few are copied once at load into the kernel's layout (``prepare``): the per-core DeltaNet
  in_proj / conv weights, the fused attention qkv, fp32 router and norm weights, the fused shared-expert gate/up,
  and the lm_head tile layout; the routed experts, out projections and caches are read as they are.

Coverage (checked at trace time by ``DecodeMegakernel.applies``; anything else falls back): the 35B-A3B shapes,
TP=4 without expert parallelism, LNC=2, KV block size 256 (``--block-size 256``), batch buckets up to 128 with
state-slot rows (below; up to 8 without), one token per request (no speculative decoding).

State-slot rows: with the dedicated state pool (no prefix caching, no speculative decoding) and a largest decode
bucket >= ``--max-num-seqs``, the runner asks the model to run decode with "kernel row k = state slot k"
(``use_state_slot_rows``). It then hands out the lowest free slot and pads the decode batch to cover the largest
slot in it, and the kernel reads and writes the state of pages 1..B with static contiguous DMAs instead of one
runtime-offset DMA per request. Here the batch rows are permuted into slot order (small per-row tensors only; no
state moves) and the outputs gathered back; kernel rows without a request keep their state bit-exactly.
"""

from __future__ import annotations

import torch

# 35B-A3B shapes the kernel is written for.
_SHAPES = dict(
    hidden_size=2048,
    num_hidden_layers=40,
    head_dim=256,
    num_attention_heads=16,
    num_key_value_heads=2,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)
_PATTERN = ["linear_attention"] * 3 + ["full_attention"]
TP = 4
LNC = 2
KV_BLOCK = 256
MAX_BATCH = 128       # with state-slot rows
PAGE_MAX_BATCH = 8    # per-request state pages (no state-slot rows)
_GDN_KEYS = ("Win", "Wout", "cw", "alog", "dtb", "nw", "sa", "sb")


def config_supported(config, world_size: int) -> str | None:
    """None if the megakernel covers this checkpoint and TP degree, else the reason it does not."""
    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    if not config.is_moe:
        return "dense checkpoint"
    for name, want in _SHAPES.items():
        if getattr(config, name, None) != want:
            return f"{name}={getattr(config, name, None)} (needs {want})"
    if list(config.layer_types) != _PATTERN * 10:
        return "layer pattern"
    moe = config.moe
    if (moe.num_experts, moe.num_experts_per_tok, moe.moe_intermediate_size,
            moe.shared_expert_intermediate_size) != (256, 8, 512, 512):
        return "MoE shapes"
    if world_size != TP:
        return f"tp_size={world_size} (needs {TP})"
    if get_logical_nc_config() != LNC:
        return f"logical NeuronCore config {get_logical_nc_config()} (needs {LNC})"
    if config.vocab_size % (TP * LNC):
        return "vocab size"
    return None


class DecodeMegakernel:
    """Holds references to the model's modules (not an ``nn.Module``: nothing is registered twice).

    ``prepare`` adds the kernel-layout weight copies as attributes (``mk_*``) of the modules that own the
    originals; ``__call__`` runs one decode step.
    """

    def __init__(self, text_model, tp_group, use_lm_head: bool):
        self.text_model = text_model
        self.tp_group = tp_group
        self.use_lm_head = use_lm_head
        self.lm_head = None
        self.slot_rows = False   # set by the runner (``Qwen3_5ForCausalLM.use_state_slot_rows``)
        self.state_pool_ok = True   # set at ``bind_state_cache``

    def check_state_pool(self, state_a) -> None:
        """The kernel updates the state in place. A float32 view of vLLM's shared int8 KV pool
        (``VLLM_NEURON_SHARED_STATE_POOL=1`` or prefix caching) is a view with another dtype, whose in-place
        update the compiler does not support; only the dedicated state pool is covered."""
        base = state_a._base
        self.state_pool_ok = base is None or base.dtype == state_a.dtype

    # ── load time ────────────────────────────────────────────────────────

    def prepare(self, lm_head) -> None:
        """Build the kernel-layout copies (once, after the weights are loaded)."""
        from .kernels.megakernel.mk_lmhead import lm_layout

        self.lm_head = lm_head

        def cpu(t):
            return t.detach().to("cpu")

        def put(module, name, tensor, like):
            # A plain attribute, not a registered buffer: the copies are not part of the state dict.
            setattr(module, name, tensor.contiguous().to(like.device))

        with torch.no_grad():
            for layer in self.text_model.layers:
                put(layer, "mk_ln1", 1.0 + cpu(layer.input_layernorm.weight).float(), layer.input_layernorm.weight)
                put(layer, "mk_ln2", 1.0 + cpu(layer.post_attention_layernorm.weight).float(),
                    layer.post_attention_layernorm.weight)
                m = layer.mlp
                put(m, "mk_wr", cpu(m.router_weight).float().t(), m.router_weight)
                put(m, "mk_wsgu", torch.cat([cpu(m.shared_gate_proj_weight), cpu(m.shared_up_proj_weight)], 1),
                    m.shared_gate_proj_weight)
                put(m, "mk_wsg", cpu(m.shared_expert_gate_weight).t(), m.shared_expert_gate_weight)
                if layer.is_linear_attention:
                    self._prepare_gdn(layer.linear_attn, cpu, put)
                else:
                    a = layer.self_attn
                    put(a, "mk_wqkv", torch.cat([cpu(a.q_proj_weight), cpu(a.k_proj_weight), cpu(a.v_proj_weight)], 1),
                        a.q_proj_weight)
                    put(a, "mk_qnw", 1.0 + cpu(a.q_norm.weight).float(), a.q_norm.weight)
                    put(a, "mk_knw", 1.0 + cpu(a.k_norm.weight).float(), a.k_norm.weight)
            norm = self.text_model.norm
            put(norm, "mk_lnf", 1.0 + cpu(norm.weight).float(), norm.weight)
            if self.use_lm_head:
                put(lm_head, "mk_lmw", lm_layout(cpu(lm_head.weight), LNC), lm_head.weight)

    @staticmethod
    def _prepare_gdn(g, cpu, put) -> None:
        """Per-core in_proj columns [q_c | k_c | v_c | z_c | b_c | a_c] and conv weights [c, 128, K, tiles] fp32."""
        kd, vd, hv = g.key_dim, g.value_dim, g.num_v_heads
        kc, vc, hc = kd // LNC, vd // LNC, hv // LNC
        qkv, z, ba = cpu(g.in_proj_qkv_weight), cpu(g.in_proj_z_weight), cpu(g.in_proj_ba_weight)
        conv = cpu(g.conv1d_weight).float()   # [conv_dim, K], local [q | k | v] channels
        win, cw = [], []
        for c in range(LNC):
            cols = [(c * kc, kc), (kd + c * kc, kc), (2 * kd + c * vc, vc)]
            parts = [qkv[:, o:o + n] for o, n in cols]
            parts += [z[:, c * vc:(c + 1) * vc], ba[:, c * hc:(c + 1) * hc], ba[:, hv + c * hc:hv + (c + 1) * hc]]
            win.append(torch.cat(parts, 1))
            w = torch.cat([conv[o:o + n] for o, n in cols], 0)          # [128 * ct, K]
            ct = w.shape[0] // 128
            cw.append(w.reshape(ct, 128, g.conv_kernel_size).permute(1, 2, 0))
        put(g, "mk_win", torch.stack(win, 0).reshape(-1), g.in_proj_qkv_weight)
        put(g, "mk_cw", torch.stack(cw, 0).reshape(-1), g.dt_bias)
        put(g, "mk_nw", cpu(g.norm_weight).float(), g.norm_weight)

    # ── trace time ───────────────────────────────────────────────────────

    def applies(self, num_tokens: int, attn_metadata: dict) -> bool:
        """Whether this decode graph is one the kernel covers (decided while tracing: static shapes only)."""
        layers = self.text_model.layers
        attn = next(L.self_attn for L in layers if not L.is_linear_attention)
        gdn = next(L.linear_attn for L in layers if L.is_linear_attention)
        num_reqs = attn_metadata[attn.layer_name]["block_table_tensor"].shape[0]
        if num_tokens != num_reqs or num_reqs > (MAX_BATCH if self._slot_rows_apply(num_reqs) else PAGE_MAX_BATCH):
            return False
        if attn.k_cache is None or attn.k_cache.shape[1:] != (1, KV_BLOCK, 256):
            return False
        sa = gdn.state_a
        if sa is None or sa.dim() != 3 or sa.shape[2] != 128:
            return False
        if not self.state_pool_ok:
            return False
        conv_rows, rec_rows = gdn.conv_numel // 128, gdn.rec_numel // 128
        if not (conv_rows <= sa.shape[1] and conv_rows + rec_rows <= 2 * sa.shape[1]):
            return False
        return all(getattr(L.mlp, "ep_degree", 1) == 1 for L in layers)

    def __call__(self, hidden_states, positions, position_embeddings, attn_metadata):
        """One decode step. Returns (normed hidden [B, H], rank logits [B, V/TP] or None)."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .kernels.megakernel.entry import decode_step, decode_step_lm

        layers = self.text_model.layers
        B = hidden_states.shape[0]
        dev = hidden_states.device
        dt = hidden_states.dtype
        attn0 = next(L.self_attn for L in layers if not L.is_linear_attention)
        md = attn_metadata[attn0.layer_name]
        num_blocks = attn0.k_cache.shape[0]

        bt = md["block_table_tensor"].to(torch.int32)
        ctx = bt.shape[1] * KV_BLOCK
        sm = md["slot_mapping"].reshape(B).to(torch.int32)
        live_kv = (sm > 0) & (sm < num_blocks * KV_BLOCK)
        # Dead (padded) rows write their (finite) K/V into the null block 0, which they also read, masked, as in
        # ``Qwen3_5Attention._write_kv``.
        slot = torch.where(live_kv, sm, torch.zeros_like(sm))
        pos = positions.reshape(B).to(torch.int32).clamp(min=0, max=ctx - 1)

        cos, sin = position_embeddings
        gdn0 = next(L.linear_attn for L in layers if L.is_linear_attention)
        gat = None
        if self._slot_rows_apply(B):
            # Kernel row k = state slot k = state page 1 + k (``use_state_slot_rows``; the runner pads the batch
            # to cover the largest slot). Permute the per-row inputs into slot order; act[k] = 1 if slot k decodes.
            idx, live = gdn0.state_indices(attn_metadata[gdn0.layer_name], B)
            ks = idx.to(torch.long) - 1
            ar = torch.arange(B, device=dev, dtype=torch.long)
            perm = ((ks.reshape(1, B) == ar.reshape(B, 1)) & live.reshape(1, B)).to(torch.float32)   # [k, b]
            act = perm.sum(dim=1)
            inv = (perm * ar.reshape(1, B).to(torch.float32)).sum(dim=1).to(torch.long)   # batch row of k
            gat = torch.where(live & (ks < B), ks, torch.zeros_like(ks))                   # kernel row of b
            hidden_states = torch.index_select(hidden_states, 0, inv)
            bt = torch.index_select(bt, 0, inv)
            slot = torch.where(act > 0.5, torch.index_select(slot, 0, inv), torch.zeros_like(slot))
            pos = torch.index_select(pos, 0, inv)
            cos = torch.index_select(cos, 0, inv)
            sin = torch.index_select(sin, 0, inv)
            gmap = [0] * sum(1 for L in layers if L.is_linear_attention)
            pages = act.contiguous()
        else:
            # DeltaNet state pages per KV-cache group (each group has its own block table), as in
            # ``Qwen3_5GatedDeltaNet.state_indices``: dead rows read the null page 0 and write an out-of-range
            # page, which the kernel's DMA drops, so the null page stays zero.
            groups: dict[int, int] = {}
            gmap, page_rows = [], []
            for L in layers:
                if not L.is_linear_attention:
                    continue
                g = L.linear_attn
                gmd = attn_metadata[g.layer_name]
                key = id(gmd)
                if key not in groups:
                    groups[key] = len(groups)
                    idx, live = g.state_indices(gmd, B)
                    idx = idx.to(torch.int32)
                    widx = torch.where(live, idx, torch.full_like(idx, g.state_a.shape[0]))
                    page_rows.append(torch.stack([idx, widx], 0))
                gmap.append(groups[key])
            pages = torch.stack(page_rows, 0).contiguous()   # [groups, 2 (read, write), B]

        cosb = torch.cat([cos, cos], dim=-1).float().contiguous()
        sinb = torch.cat([sin, sin], dim=-1).float().contiguous()
        hv_c = gdn0.num_v_heads // LNC
        JM = min(hv_c * B, 128)   # block mask rows (the recurrence runs in groups of <= 128 state columns)
        ident = torch.eye(128, dtype=dt, device=dev)
        blk = torch.eye(JM, dtype=dt, device=dev).repeat_interleave(128, dim=1).contiguous()
        bmask = torch.eye(B, dtype=torch.float32, device=dev).repeat_interleave(8, dim=1).contiguous()

        kw = {}
        ig = ia = 0
        for li, L in enumerate(layers):
            m = L.mlp
            kw.update({f"ln1_{li}": L.mk_ln1, f"ln2_{li}": L.mk_ln2, f"Wr_{li}": m.mk_wr, f"Wsgu_{li}": m.mk_wsgu,
                       f"Wsd_{li}": m.shared_down_proj_weight, f"Wsg_{li}": m.mk_wsg,
                       f"gu_{li}": m.experts_gate_up_weight, f"dn_{li}": m.experts_down_weight})
            if L.is_linear_attention:
                g = L.linear_attn
                vals = (g.mk_win, g.out_proj_weight, g.mk_cw, g.A_log, g.dt_bias, g.mk_nw, g.state_a, g.state_b)
                kw.update({f"{k}_{ig}": v for k, v in zip(_GDN_KEYS, vals)})
                ig += 1
            else:
                a = L.self_attn
                kw.update({f"Wqkv_{ia}": a.mk_wqkv, f"Wo_{ia}": a.o_proj_weight, f"qnw_{ia}": a.mk_qnw,
                           f"knw_{ia}": a.mk_knw, f"kc_{ia}": a.k_cache, f"vc_{ia}": a.v_cache})
                ia += 1
        common = dict(x_rows=hidden_states.contiguous(), cosb=cosb, sinb=sinb, bt=bt.contiguous(),
                      slot=slot.contiguous(), pos=pos.contiguous(), pages=pages, ident_in=ident, mk_in=blk,
                      bmask_in=bmask, gmap=tuple(gmap), ranks=tuple(self.tp_group.ranks), **kw)
        if self.use_lm_head:
            lmw = self.lm_head.mk_lmw
            outs = wrap_nki(decode_step_lm)[LNC](lnf=self.text_model.norm.mk_lnf, lmw=lmw,
                                                 vocab=self.lm_head.weight.shape[0], **common)
            if gat is not None:
                return torch.index_select(outs[0], 0, gat), torch.index_select(outs[1], 0, gat)
            return outs[0], outs[1]
        outs = wrap_nki(decode_step)[LNC](**common)
        h = outs[0] if gat is None else torch.index_select(outs[0], 0, gat)
        return self.text_model.norm(h), None

    def _slot_rows_apply(self, B: int) -> bool:
        """Slot-row mode for this graph: enabled by the runner, and pages 1..B exist in every state pool."""
        if not self.slot_rows:
            return False
        return all(L.linear_attn.state_a.shape[0] > B for L in self.text_model.layers if L.is_linear_attention)
