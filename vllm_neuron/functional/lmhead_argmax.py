# SPDX-License-Identifier: Apache-2.0
"""Fused lm_head GEMM + greedy argmax: ``argmax(x @ W, -1)`` without materializing the logits.

``lmhead_argmax(x, weight)`` returns the int32 token ids ``torch.argmax(bf16(x @ W), -1)`` (first index of the
maximum, like torch) for an ``in_out`` lm_head weight ``W [H, V]`` (``ColumnParallelLinear(weight_layout="in_out")``,
tensor-parallel size 1). On Neuron the NKI kernel in ``lmhead_argmax_kernel.py`` streams W once and never writes the
``[B, V]`` logits to HBM; elsewhere (CPU mode, unsupported shapes) a torch implementation of the same math runs.

The caller is responsible for routing only greedy sampling without logit masks or logits gathering here; see
``qwen3_asr/model_bf16.py``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from vllm_neuron import envs
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

_P = 128

_wrapped_kernel = None
try:
    import nki

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .lmhead_argmax_kernel import lmhead_argmax_kernel

    _wrapped_kernel = wrap_nki(nki.jit()(lmhead_argmax_kernel))
except Exception as e:  # noqa: BLE001
    logger.debug("lmhead_argmax kernel unavailable, torch fallback only: %s", e)


def _grid() -> int:
    return 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()


def can_use_lmhead_argmax_kernel(x: Tensor, weight: Tensor) -> bool:
    """True when the NKI kernel supports ``x [B, H]`` x ``weight [H, V]`` on this device and LNC."""
    if _wrapped_kernel is None or not can_run_kernel(x):
        return False
    if x.dim() != 2 or weight.dim() != 2:
        return False
    B, H = x.shape
    if weight.shape[0] != H or H % _P != 0 or B < 1 or B > _P:
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    V = weight.shape[1]
    g = _grid()
    return V % g == 0 and V // g >= 8


def _torch_impl(x: Tensor, weight: Tensor) -> Tensor:
    logits = torch.matmul(x, weight)
    return torch.argmax(logits, dim=-1).to(torch.int32)


def combine_programs(val: Tensor, idx: Tensor, vocab_per_program: int) -> Tensor:
    """Merge per-program ``(val, idx)`` ``[n, B, 1]`` into global ids ``[B]`` int32 (lowest program wins ties).

    Example:
        >>> v = torch.tensor([[[1.0], [3.0]], [[2.0], [3.0]]]); i = torch.tensor([[[5], [7]], [[1], [0]]], dtype=torch.int32)
        >>> combine_programs(v, i, 10).tolist()
        [11, 7]
    """
    n = val.shape[0]
    if n == 1:
        return idx[0, :, 0].to(torch.int32)
    v = val[..., 0]                                                         # [n, B]
    off = torch.arange(n, device=idx.device, dtype=torch.int32).view(n, 1) * vocab_per_program
    gi = idx[..., 0].to(torch.int32) + off                                  # [n, B]
    best = torch.argmax(v, dim=0, keepdim=True)                             # first program with the max
    return torch.gather(gi, 0, best)[0]


def lmhead_argmax(x: Tensor, weight: Tensor) -> Tensor:
    """Greedy token ids ``[B]`` int32 for hidden states ``x [B, H]`` and lm_head weight ``[H, V]`` (in_out).

    Example:
        >>> x = torch.randn(2, 128, dtype=torch.bfloat16); w = torch.randn(128, 300, dtype=torch.bfloat16)
        >>> bool((lmhead_argmax(x, w) == torch.argmax(x @ w, -1)).all())
        True
    """
    if can_use_lmhead_argmax_kernel(x, weight):
        g = _grid()
        val, idx = _wrapped_kernel[g](x=x, w=weight)
        return combine_programs(val, idx, weight.shape[1] // g)
    return _torch_impl(x, weight)
