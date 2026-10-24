# SPDX-License-Identifier: Apache-2.0
"""Functional entry point of the Whisper cross-attention decode kernel."""

import torch

from vllm_neuron._kernels.whisper.cross_attention_decode import (
    whisper_xattn_decode,
    whisper_xattn_decode_torch,
    xkernel_dmask,
    xkernel_group,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel


def whisper_cross_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    qmask: torch.Tensor | None,
    group: int,
    k_pairs: int = 5,
    v_heads: int = 10,
    pv_block: int = 5,
    dmask: torch.Tensor | None = None,
    hw_dge: int = 0,
) -> torch.Tensor:
    """Return the cross-attention context of B decode rows, ``[B, H, 64]``.

    Each row attends over the 1500 encoder positions of its own slot. The NKI kernel
    runs on a Neuron device; the torch reference runs on CPU.

    Args:
        q: ``[B, H, 64]``, scaled by ``head_dim ** -0.5``.
        k_cache: ``[S, H, 64, 1536]`` cross K^T of all slots.
        v_cache: ``[S, H, 1536, 64]`` cross V of all slots.
        slots: ``[B]`` int32, the slot of each row.
        qmask: the ``xkernel_qmask(H, group)`` tensor on the device (kernel only).
        group: rows per kernel group (``xkernel_group``).
        k_pairs: head pairs per K DMA (divides H / 2).
        v_heads: heads per V DMA (divides H).
        pv_block: units per blocked-PV matmul (divides ``v_heads``); 0 selects the
            one-column PV.
        dmask: the ``xkernel_dmask(H, group, pv_block)`` tensor on the device.
        hw_dge: 1 loads K and V with hardware DGE.

    Example:
        >>> # ctx = whisper_cross_attention_decode(q, k, v, slots, qmask, 4)
    """
    if not can_run_kernel(q):
        return whisper_xattn_decode_torch(q, k_cache, v_cache, slots)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    B, H, D = q.shape
    S = k_cache.shape[0]
    units = group * H
    # Column c of q_t holds the units (2c, 2c + 1): the even unit on partitions 0-63.
    q_t = q.reshape(B * H // 2, 2 * D).t().contiguous()
    pv = 1 if pv_block > 0 else 0
    vb = pv_block if pv_block > 0 else 5
    if dmask is None:
        dmask = xkernel_dmask(H, group, vb, device=q.device)
    out = wrap_nki(whisper_xattn_decode)[1](
        q_t=q_t,
        k_cache=k_cache.reshape(S, -1),
        v_cache=v_cache.reshape(S, -1),
        slots=slots.to(torch.int32).view(1, B),
        qmask=qmask,
        dmask=dmask,
        G=group,
        KP=k_pairs,
        VH=v_heads,
        PV=pv,
        VB=vb,
        HW=hw_dge,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    if pv == 1:
        return out.view(B, H, D)
    # out [B/G, 4, U/4 * 64]: unit u = 4 f + qd is at [g, qd, f].
    out = out.view(B // group, 4, units // 4, D).transpose(1, 2)
    return out.reshape(B, H, D)


__all__ = ["whisper_cross_attention_decode", "xkernel_group"]
