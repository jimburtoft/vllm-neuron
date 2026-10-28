# SPDX-License-Identifier: Apache-2.0
"""Functional entry point of the Whisper self-attention decode kernel."""

import torch

from vllm_neuron._kernels.whisper.cross_attention_decode import xkernel_dmask
from vllm_neuron._kernels.whisper.self_attention_decode import (
    self_attn_vb,
    whisper_self_attn_decode,
    whisper_self_attn_decode_torch,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def self_attn_mask_bias(positions: torch.Tensor, heads: int, ctx: int) -> torch.Tensor:
    """Return the ``[B * heads, ctx]`` fp32 additive mask of the kernel (0 valid, -1e30 masked).

    Build it one time per step.
    """
    B = positions.shape[0]
    col = torch.arange(ctx, device=positions.device)
    valid = col.view(1, ctx) <= positions.to(torch.int64).view(B, 1)
    zero = torch.zeros((), dtype=torch.float32, device=positions.device)
    neg = torch.full((), -1e30, dtype=torch.float32, device=positions.device)
    bias = torch.where(valid, zero, neg)
    return bias.view(B, 1, ctx).expand(B, heads, ctx).reshape(B * heads, ctx)


def whisper_self_attention_decode(
    q: torch.Tensor,
    k_slots: torch.Tensor,
    v_slots: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    ctx: int,
    qmask: torch.Tensor | None,
    maskb: torch.Tensor | None,
    group: int,
    s_rows: int = 1,
) -> torch.Tensor:
    """Return the self-attention context of B decode rows, ``[B, H, 64]``.

    Each row attends over the positions ``0 .. positions[row]`` of its own slot (up to ``ctx``).
    The NKI kernel runs on a Neuron device; the torch reference runs on CPU.

    Args:
        q: ``[B, H, 64]``, scaled by ``head_dim ** -0.5``. H must be even.
        k_slots: ``[S, CS, H * 64]`` slot self-K, after the write of this step.
        v_slots: ``[S, CS, H * 64]`` slot self-V.
        slots: ``[B]`` int, the slot of each row.
        positions: ``[B]`` int, the position of each row.
        ctx: the decode-context bucket (up to 512 and the slot length).
        qmask: the ``xkernel_qmask(H, group)`` tensor on the device (kernel only).
        maskb: the ``self_attn_mask_bias(positions, H, ctx)`` tensor (kernel only).
        group: rows per kernel group (``xkernel_group``).
        s_rows: query rows per slot (a verify step: K + 1). ``q`` and ``positions`` then have
            ``B * s_rows`` rows in token order and ``slots`` has ``B``. ``group * s_rows * H``
            must be at most 128, and ``qmask`` must be built for ``group * s_rows`` rows.

    Example:
        >>> # ctx = whisper_self_attention_decode(q, slot_k, slot_v, slots, pos, 128, qm, mb, 4)
    """
    BT, H, D = q.shape
    B = BT // s_rows
    S, CS = k_slots.shape[0], k_slots.shape[1]
    if not can_run_kernel(q):
        tok_slots = slots.view(-1).repeat_interleave(s_rows) if s_rows > 1 else slots
        return whisper_self_attn_decode_torch(
            q, k_slots.view(S, CS, H, D), v_slots.view(S, CS, H, D), tok_slots, positions, ctx
        )
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    vb = self_attn_vb(H)
    q_t = q.reshape(BT * H // 2, 2 * D).t().contiguous()
    dmask = xkernel_dmask(H, group * s_rows, vb, device=q.device)
    out = wrap_nki(whisper_self_attn_decode)[get_logical_nc_config()](
        q_t=q_t,
        k_cache=k_slots.view(S, CS * H * D),
        v_cache=v_slots.view(S, CS * H * D),
        slots=slots.to(torch.int32).view(1, B),
        qmask=qmask,
        dmask=dmask,
        maskb=maskb,
        G=group,
        VB=vb,
        S=s_rows,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.view(BT, H, D)


__all__ = ["whisper_self_attention_decode", "self_attn_mask_bias"]
