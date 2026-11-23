# SPDX-License-Identifier: Apache-2.0
"""Isaac-0.5 robot-action path: VLM prefill -> post-norm final hidden states, and the DiT flow expert.

Runs the vllm-neuron Isaac text decoder (``Qwen3_5MoeForCausalLM`` layers + final norm, TP over the vLLM TP group)
outside the vLLM scheduler: one observation per call, no paged KV cache kept (GQA K/V go to a scratch cache), GDN
state in the scratch request slot. The host builds the inputs exactly as the HF reference does
(``Qwen35VLAModel.forward``): ``inputs_embeds`` (text + vision tower + proprio vector MLP, NTP-truncated) and the
3-D mRoPE ``position_ids`` (non-spatial tokens collapsed to (t, t, t)).

Used by ``pipeline_isaac_action.IsaacActionPipeline`` (vLLM-Omni diffusion stage, public 0.29 plugin).
"""

from __future__ import annotations

import torch


def vlm_prefill_hidden(model, inputs_embeds: torch.Tensor, rot_pos: torch.Tensor, md: dict, slots: torch.Tensor,
                       n_real: int | None = None):
    """model: Qwen3_5MoeForCausalLM (text decoder, weights loaded, KV bound). inputs_embeds [T, H] bf16 (T padded to a
    prefill bucket, padding rows zero), rot_pos [3, T] int (mRoPE; padding repeats the last real position),
    md: attention metadata of one prefill request. Returns post-final-norm hidden states [T, H] (all positions)."""
    T = inputs_embeds.shape[0]
    tp = model.world_size
    # 1-D positions for KV slots, causal masks and GDN padding detection are the TOKEN INDEX (as in vLLM); the mRoPE
    # axes (image tokens share t; positions jump after images) only drive cos/sin. Padding rows repeat the last real
    # index (n_real - 1), which is how the decoder detects them.
    ti = torch.arange(T, device=inputs_embeds.device, dtype=torch.int32)
    positions = torch.minimum(ti, (n_real - 1) * torch.ones_like(ti)) if n_real is not None else ti
    attn_metadata = {layer.self_attn.layer_name: md for layer in model.layers if layer.is_full}
    sp = tp > 1 and model.prefill_sp
    if sp:
        assert T % tp == 0, (T, tp)
        r = model.tp_group.rank_in_group
        h = inputs_embeds.reshape(tp, T // tp, -1)[r]  # SP-local rows (the decoder all-gathers inside the mixers)
    else:
        h = inputs_embeds
    rot = getattr(model, "_action_mrotary", None)
    if rot is None:  # text-only model view: build the interleaved-mRoPE rotary once (same config, bf16 inv_freq)
        from .model_vl import MRotary

        rot = MRotary(model.config, [11, 11, 10])
        model._action_mrotary = rot
    cos, sin = rot(rot_pos)
    for layer in model.layers:
        h = layer(h, positions, cos, sin, attn_metadata, False, slots, model.gdn_mask_pack, False, sp or tp == 1)
    h = model.norm(h)
    if sp:
        h = model.tp_group.all_gather(h, dim=0)
    return h


def vlm_prefill_hidden_packed(model, inputs_embeds: torch.Tensor, rot_pos: torch.Tensor, md: dict, slots: torch.Tensor,
                              seg: int, n_real: torch.Tensor):
    """B observations packed in one prefill (T = B * seg tokens, observation b at [b * seg, b * seg + L_b)).
    Uses the decoder's packed-prefill path: block-diagonal causal attention over live keys (no KV cache),
    GDN conv / recurrent state reset at every observation start (seg is a multiple of 128), MoE on real tokens only.
    n_real [B] int32 (device): real tokens per observation. slots [B]: GDN state rows (scratch rows, nothing kept).
    Returns post-final-norm hidden states [T, H]."""
    T = inputs_embeds.shape[0]
    B = T // seg
    tp = model.world_size
    dev = inputs_embeds.device
    ti = torch.arange(T, device=dev, dtype=torch.int32)
    loc = ti - (ti // seg) * seg  # position inside the observation
    nr = n_real.to(torch.int32).repeat_interleave(seg)
    positions = torch.minimum(loc, nr - 1)  # padding repeats the observation's last real position
    real = loc < nr
    nc = T // 128
    start = real & (loc == 0)
    keep = 1.0 - start.reshape(nc, 128)[:, 0].float()
    pk = {
        "real": real,
        "lo": (ti - positions).float(),
        "e": (torch.arange(B, device=dev, dtype=torch.long) * seg + n_real.long() - 1),
        "chunk_keep": keep.view(nc, 1, 1).expand(nc, 128, 1).contiguous(),
        "skip_state_write": True,  # nothing is kept between plans (no decode follows)
    }
    attn_metadata = {layer.self_attn.layer_name: md for layer in model.layers if layer.is_full}
    sp = tp > 1 and model.prefill_sp
    if sp:
        assert T % tp == 0, (T, tp)
        r = model.tp_group.rank_in_group
        h = inputs_embeds.reshape(tp, T // tp, -1)[r]
    else:
        h = inputs_embeds
    rot = getattr(model, "_action_mrotary", None)
    if rot is None:
        from .model_vl import MRotary

        rot = MRotary(model.config, [11, 11, 10])
        model._action_mrotary = rot
    cos, sin = rot(rot_pos)
    for layer in model.layers:
        h = layer(h, positions, cos, sin, attn_metadata, False, slots, model.gdn_mask_pack, False, sp or tp == 1, pk=pk)
    h = model.norm(h)
    if sp:
        h = model.tp_group.all_gather(h, dim=0)
    return h
