# SPDX-License-Identifier: Apache-2.0
"""Fused 3-layer audio conv frontend (conv1 -> GELU -> conv2 -> GELU -> conv3 -> GELU).

``audio_conv_frontend(mel, w1, b1, w2, b2, w3, b3)`` computes, per base chunk,

    gelu(conv3(gelu(conv2(gelu(conv1(mel))))))     # every conv: 3x3, stride 2, padding 1, + bias

for ``mel [n, H, W]`` (H mel bins, W frames per chunk) and returns ``[n, C, H/8, ceil(W/8)]`` (NCHW).
GELU is the exact (erf) form in fp32, rounded to the activation dtype.

Weight layouts (packed once at load time, see ``conv3x3_packed_weight_loader``):
  w1 [9, C]       tap-major, tap t = 3 * kr + kc
  w2, w3 [C, 9, C] as [ci, tap, co]

On Neuron the NKI kernel in ``audio_conv_frontend_kernel.py`` runs (intermediates stay in SBUF);
elsewhere a torch implementation of the same math runs.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

_wrapped_kernel = None
try:
    import nki

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .audio_conv_frontend_kernel import qwen3_asr_conv_frontend_kernel

    _wrapped_kernel = wrap_nki(nki.jit()(qwen3_asr_conv_frontend_kernel))
except Exception as e:  # noqa: BLE001
    logger.debug("audio_conv_frontend kernel unavailable, torch fallback only: %s", e)


def pack_conv3x3_weight(w: Tensor) -> Tensor:
    """``[co, ci, 3, 3]`` -> ``[ci, 9, co]``."""
    co, ci, kh, kw = w.shape
    assert kh == 3 and kw == 3, "3x3 conv weight expected"
    return w.permute(1, 2, 3, 0).reshape(ci, 9, co).contiguous()


def unpack_conv3x3_weight(w: Tensor) -> Tensor:
    """``[ci, 9, co]`` -> ``[co, ci, 3, 3]``."""
    ci, _, co = w.shape
    return w.reshape(ci, 3, 3, co).permute(3, 0, 1, 2).contiguous()


def _gelu(x: Tensor, dtype) -> Tensor:
    xf = x.float()
    return (0.5 * xf * (1.0 + torch.erf(xf * 0.7071067811865476))).to(dtype)


def _conv1_addmm(mel: Tensor, w1: Tensor, b1: Tensor) -> Tensor:
    """conv1 as 3x3 patches @ w1 + b1 (the stock encoder's addmm form, same rounding)."""
    n, H, W = mel.shape
    oh, ow = (H + 1) // 2, (W + 1) // 2
    p = F.pad(mel, (1, 1, 1, 1))
    patches = torch.stack([p[:, kr:kr + 2 * oh:2, kc:kc + 2 * ow:2] for kr in range(3) for kc in range(3)], dim=-1)
    y = torch.addmm(b1, patches.reshape(-1, 9), w1).view(n, oh, ow, w1.shape[1])
    return y.permute(0, 3, 1, 2)


def _torch_impl(mel, w1, b1, w2, b2, w3, b3) -> Tensor:
    dt = mel.dtype
    x = _gelu(_conv1_addmm(mel, w1, b1), dt)
    x = _gelu(F.conv2d(x, unpack_conv3x3_weight(w2), b2, stride=2, padding=1), dt)
    x = _gelu(F.conv2d(x, unpack_conv3x3_weight(w3), b3, stride=2, padding=1), dt)
    return x


def can_use_audio_conv_frontend_kernel(mel: Tensor, w1: Tensor) -> bool:
    if _wrapped_kernel is None or not can_run_kernel(mel):
        return False
    if mel.dtype != torch.bfloat16 or w1.dtype != torch.bfloat16:
        return False
    n, H, W = mel.shape
    C = w1.shape[1]
    if H % 8 != 0 or W % 2 != 0 or C > 512 or W > 1022:
        return False
    if (W + 1) // 2 > 512 or ((W + 1) // 2 + 1) // 2 > 32 or H // 4 > 32:
        return False
    lnc = get_logical_nc_config()
    if lnc > 1 and n % lnc != 0:
        return False
    return True


def audio_conv_frontend(mel: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, w3: Tensor,
                        b3: Tensor) -> Tensor:
    """See module docstring. ``mel [n, H, W]`` -> ``[n, C, H/8, ceil(W/8)]``."""
    if can_use_audio_conv_frontend_kernel(mel, w1):
        xpad = F.pad(mel, (1, 1, 1, 1))
        return _wrapped_kernel[get_logical_nc_config()](
            xpad=xpad, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3
        )
    return _torch_impl(mel, w1, b1, w2, b2, w3, b3)
