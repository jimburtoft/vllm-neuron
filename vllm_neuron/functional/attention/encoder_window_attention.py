# SPDX-License-Identifier: Apache-2.0
"""Block-local bidirectional attention for block-packed encoders (128-token blocks, head_dim 64).

``encoder_window_attention`` computes, for every block b and head h,

    out[b, h] = (softmax(scale * q_bh @ k_bh^T + mask_b) @ v_bh)^T          # [64, 128], BNdS

straight from the fused QKV projection output, with the block-local segment mask given as two
rank-16 factor tensors (``build_segment_factors``). It is a drop-in replacement for

    permute q/k/v to [nb * NH, 128, 64] -> NF.flash_attention(..., causal_mask=False, tp_q=True,
    tp_k=True, tp_out=True, bound_min, bound_max) -> view [nb, NH, 64, 128]

when every row's ``[bound_min, bound_max)`` range is the contiguous span of the segment the row
belongs to, and padding rows have an empty range. ``plan_audio_blocks`` produces exactly that layout.
Padding rows get a finite (uniform) softmax instead of zeros; no real token reads them.

On Neuron the NKI kernel in ``encoder_window_attention_kernel.py`` runs; elsewhere a torch
implementation of the same math runs.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.hardware_config import get_logical_nc_config

logger = logging.getLogger(__name__)

BLOCK = 128
HEAD_DIM = 64
SEG_ROWS = 16
MAX_SEGMENTS = SEG_ROWS - 1
MASK_BIG = 4096.0

_wrapped_kernel = None
try:
    import nki

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .encoder_window_attention_kernel import encoder_window_attention_kernel

    _wrapped_kernel = wrap_nki(nki.jit()(encoder_window_attention_kernel))
except Exception as e:  # noqa: BLE001
    logger.debug("encoder_window_attention kernel unavailable, torch fallback only: %s", e)


def build_segment_factors(bound_min: Tensor, bound_max: Tensor, dtype=torch.bfloat16) -> tuple[Tensor, Tensor]:
    """Turn per-row ``[bound_min, bound_max)`` bounds into rank-16 mask factors.

    Args:
        bound_min, bound_max: ``[nb, 128, 1]`` int32, block-local, as from
            ``AudioBlockAssignment.block_bounds``.

    Returns:
        ``(seg_q, seg_k)``, each ``[nb, 16, 128]`` in ``dtype``, with
        ``seg_q[b].T @ seg_k[b] == 0`` where query and key share a segment and ``-4096`` elsewhere
        (every entry is a single product, so the zeros are exact).

    A segment is identified by its start ``s`` (a valid row with ``bound_min == s == its own index``);
    a valid row's segment id is the number of segment starts below its ``bound_min``. Up to
    ``MAX_SEGMENTS`` (15) segments per block; ``plan_audio_blocks`` makes at most
    ``slots_per_block`` (9 at block 128).
    """
    nb, S, _ = bound_min.shape
    lo = bound_min[..., 0].to(torch.int32)
    hi = bound_max[..., 0].to(torch.int32)
    valid = hi > lo                                                   # [nb, S]
    pos = torch.arange(S, device=lo.device, dtype=torch.int32)
    is_start = valid & (lo == pos.view(1, S))                         # [nb, S]
    # seg_id[b, i] = #starts strictly below lo[b, i]
    below = (pos.view(1, 1, S) < lo.view(nb, S, 1)) & is_start.view(nb, 1, S)
    seg_id = below.to(torch.int32).sum(-1)                           # [nb, S]
    r = torch.arange(SEG_ROWS, device=lo.device, dtype=torch.int32).view(1, SEG_ROWS, 1)
    sid = seg_id.view(nb, 1, S)
    v = valid.view(nb, 1, S)
    # key factor: one-hot of its segment (row 15 = padding key)
    key_in = torch.where(v, sid == r, r == MAX_SEGMENTS)
    seg_k = key_in.to(dtype)
    # query factor: -4096 except at its own segment row (padding query: -4096 everywhere)
    q_own = v & (sid == r)
    seg_q = torch.where(q_own, torch.zeros((), dtype=dtype, device=lo.device),
                        torch.full((), -MASK_BIG, dtype=dtype, device=lo.device)).to(dtype)
    return seg_q, seg_k


def _torch_impl(qkv: Tensor, seg_q: Tensor, seg_k: Tensor, scale: float, num_heads: int) -> Tensor:
    nb, S, F = qkv.shape
    hd = num_heads * HEAD_DIM
    q = qkv[..., :hd].reshape(nb, S, num_heads, HEAD_DIM).permute(0, 2, 1, 3).float()
    k = qkv[..., hd:2 * hd].reshape(nb, S, num_heads, HEAD_DIM).permute(0, 2, 1, 3).float()
    v = qkv[..., 2 * hd:].reshape(nb, S, num_heads, HEAD_DIM).permute(0, 2, 1, 3).float()
    bias = torch.einsum("brq,brk->bqk", seg_q.float(), seg_k.float()).unsqueeze(1)
    s = (q @ k.transpose(-1, -2) + bias) * scale
    p = torch.softmax(s, dim=-1).to(qkv.dtype).float()
    o = p @ v                                                        # [nb, NH, S, D]
    return o.transpose(-1, -2).to(qkv.dtype).contiguous()             # [nb, NH, D, S]


def can_use_encoder_window_attention_kernel(qkv: Tensor, num_heads: int) -> bool:
    if _wrapped_kernel is None or not can_run_kernel(qkv):
        return False
    nb, S, F = qkv.shape
    if S != BLOCK or F != 3 * num_heads * HEAD_DIM or num_heads % 2 != 0:
        return False
    G = min(8, num_heads)
    if num_heads % G != 0:
        return False
    lnc = get_logical_nc_config()
    if lnc > 1 and nb % lnc != 0 and (num_heads // G) % lnc != 0:
        return False
    return True


def encoder_window_attention(
    qkv: Tensor, seg_q: Tensor, seg_k: Tensor, scale: float, num_heads: int
) -> Tensor:
    """See module docstring. Returns ``[nb, num_heads, 64, 128]`` (BNdS)."""
    if can_use_encoder_window_attention_kernel(qkv, num_heads):
        return _wrapped_kernel[get_logical_nc_config()](
            qkv=qkv, seg_q=seg_q, seg_k=seg_k, scale=float(scale), num_heads=int(num_heads)
        )
    return _torch_impl(qkv, seg_q, seg_k, scale, num_heads)
