# SPDX-License-Identifier: Apache-2.0
"""FP8 KV cache for MuseGlimmer (Task 028).

Enabled by the vLLM cache dtype (``--kv-cache-dtype fp8``): the runner then
allocates float8_e4m3fn K/V caches and ``bind_kv_cache`` calls :func:`setup`.

Scales (one scalar per layer, as the nkilib decode megakernel requires):

- **K: analytic, no calibration.** K is stored after the weightless per-head
  QK RMSNorm and RoPE. RMSNorm gives ``||k||^2 = d * m / (m + eps) < d`` and
  RoPE rotates pairs, so ``|k_i| < sqrt(d)`` (11.31 at d=128). The dequant
  scale ``K_MARGIN * sqrt(d) / FP8_MAX`` (K_MARGIN 1.125 covers bf16 rounding)
  can never clip. It is a Python float folded into
  ``softmax_scale`` (``scaling * k_dq``), so it is a compile-time constant that
  does not depend on calibration data.
- **V: calibrated.** Per-layer ``v_absmax`` from a JSON file
  (``MUSE_GLIMMER_KV_SCALES``, written by ``harness/t028_calib_kv.py``) times
  ``MUSE_GLIMMER_KV_V_MARGIN`` (default 2.0). The V dequant multiplies the
  attention output before the output gate. The gate sits between attention and
  o_proj, so the V scale cannot fold into ``W_out`` as in llama3.
  The V scales are ``[1, 1]`` fp32 **buffers** (``kv_v_qscale`` / ``kv_v_dq``)
  filled at weight load (:func:`load_scales`), so they are graph inputs: the
  NEFFs do not depend on the calibration, and a new calibration needs no
  recompile. The K scale is a constant (``kv_k_qscale`` buffer for the kernel,
  a Python float in ``softmax_scale``).

FP8 e4m3 is floating point: the scale only places the exponent range, so a
margin costs precision only for values that fall into the subnormal range
(e4m3 spans 2^-6 .. 240, a ratio of ~15000). Headroom matters on trn2: the
torch dtype is e4m3fn (max 448) and the compiler reinterprets it as trn2 e4m3
(max 240, exponent 1111 = inf/NaN). Our own writes clamp at 240, but a clamp
inside a kernel that resolves the bound from the e4m3fn dtype would let
240..448 through, which trn2 reads as inf/NaN. A 2x V margin keeps calibrated
values at <= 120, and K cannot exceed 240 by construction.

Conventions (same as llama3 / the kernel):

- ``*_qscale`` = quantize multiplier (``x_fp8 = clamp(x * qscale, +-FP8_MAX)``),
  passed to the kernel as ``k_scale`` / ``v_scale`` ``[1, 1]`` fp32.
- ``*_dq`` = dequant multiplier = 1 / qscale.
- Writes: values already in fp8 (decode: the megakernel returns quantized
  active K/V) are stored as is; bf16 values (prefill) are quantized here.
- Prefill attention (``flash_attention``) reads the bf16 K/V just computed, not
  the cache, so it needs no scales. Segmented prefill would read the cache, but
  the nkilib segmented kernel cannot read a non-packed fp8 cache and the runner
  disables it; FP8 KV therefore requires single-shot prefill (checked in
  :func:`setup`). The segmented call in the model still folds the scales, for
  a future packed layout / the torch fallback.
"""

from __future__ import annotations

import json
import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

SCALES_ENV = "MUSE_GLIMMER_KV_SCALES"
V_MARGIN_ENV = "MUSE_GLIMMER_KV_V_MARGIN"
FUSED_MASK_ENV = "MUSE_GLIMMER_KV_FP8_FUSED_MASK"
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_max() -> float:
    from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

    return float(FP8_CLAMP_MAX)


def is_fp8(t: torch.Tensor | None) -> bool:
    return t is not None and t.dtype in FP8_DTYPES


def fused_mask_allowed() -> bool:
    return os.environ.get(FUSED_MASK_ENV, "0").strip().lower() in ("1", "true", "yes")


K_MARGIN = 1.125


def k_dequant_scale(head_dim: int) -> float:
    """Analytic K dequant scale: |k| < sqrt(head_dim) after weightless RMSNorm + RoPE."""
    return K_MARGIN * math.sqrt(head_dim) / fp8_max()


def load_v_absmax(num_layers: int) -> list[float] | None:
    path = os.environ.get(SCALES_ENV, "").strip()
    if not path:
        return None
    with open(path) as f:
        js = json.load(f)
    layers = js["layers"]
    if len(layers) != num_layers:
        raise ValueError(f"{SCALES_ENV}={path}: {len(layers)} layers, model has {num_layers}")
    return [float(layers[i]["v_absmax"]) for i in range(num_layers)]


def quantize(x: torch.Tensor, qscale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    m = fp8_max()
    return (x.to(torch.float32) * qscale).clamp(-m, m).to(dtype)


def register_buffers(attn) -> None:
    """``[1, 1]`` fp32 scale buffers (meta at construction; real values from :func:`load_scales`)."""
    for n in ("kv_k_qscale", "kv_v_qscale", "kv_v_dq"):
        attn.register_buffer(n, torch.ones(1, 1, dtype=torch.float32), persistent=False)


def load_scales(layers) -> None:
    """Called from ``load_weights`` (CPU tensors, before ``.to(device)``). Harmless for a bf16 cache."""
    m = fp8_max()
    vmax = load_v_absmax(len(layers))
    margin = float(os.environ.get(V_MARGIN_ENV, "2.0"))
    for i, l in enumerate(layers):
        a = l.self_attn
        k_dq = k_dequant_scale(a.head_dim)
        v_dq = 1.0 if vmax is None else max(vmax[i] * margin, 1e-6) / m
        a.k_dq = k_dq
        a.v_dq = v_dq
        a.kv_k_qscale = torch.full((1, 1), 1.0 / k_dq, dtype=torch.float32)
        a.kv_v_qscale = torch.full((1, 1), 1.0 / v_dq, dtype=torch.float32)
        a.kv_v_dq = torch.full((1, 1), v_dq, dtype=torch.float32)
    layers[0].self_attn._kv_scales_src = (os.environ.get(SCALES_ENV, "") or None, margin)


def to_cache(attn, x: torch.Tensor, which: str, cache: torch.Tensor) -> torch.Tensor:
    """Values for ``cache``: already-fp8 values pass through; others are quantized if the cache is fp8."""
    if not is_fp8(cache):
        return x.to(cache.dtype)
    if is_fp8(x):
        # Decode: the megakernel already quantized the active K/V. Re-clamp to the trn2
        # range in case its clamp used the e4m3fn bound (448); exact for in-range values.
        m = fp8_max()
        return x.to(torch.float32).clamp(-m, m).to(cache.dtype)
    return quantize(x, attn.kv_k_qscale if which == "k" else attn.kv_v_qscale, cache.dtype)


def _check_single_shot_prefill() -> None:
    """FP8 KV needs single-shot prefill (max_num_batched_tokens >= max_model_len).

    The runner disables segmented prefill for a non-packed FP8 cache. A chunked prompt
    would then run its later chunks through flash_attention over the chunk alone,
    ignoring the cached prefix (verified on CPU: wrong tokens from step 0). Fail loudly.
    """
    try:
        from vllm.config import get_current_vllm_config

        vc = get_current_vllm_config()
        mnbt = vc.scheduler_config.max_num_batched_tokens
        mml = vc.model_config.max_model_len
    except Exception:  # pragma: no cover - config not available (unit use)
        return
    if mnbt is not None and mml is not None and mnbt < mml:
        raise ValueError(
            f"MuseGlimmer FP8 KV cache needs single-shot prefill: max_num_batched_tokens={mnbt} < "
            f"max_model_len={mml}. Chunked prefill would need segmented attention over the FP8 "
            "cache, which the runner disables for a non-packed FP8 cache. Set "
            "max_num_batched_tokens >= max_model_len or use --kv-cache-dtype auto.")


def setup(layers) -> bool:
    """Called from ``bind_kv_cache`` (outside the trace). Sets per-layer FP8 KV state.

    Returns True when the caches are FP8.
    """
    fp8 = [is_fp8(l.self_attn.k_cache) for l in layers]
    if not any(fp8):
        for l in layers:
            l.self_attn.kv_fp8 = False
        return False
    if not all(fp8) or not all(is_fp8(l.self_attn.v_cache) for l in layers):
        raise ValueError("MuseGlimmer FP8 KV: every layer's K and V cache must be FP8")
    _check_single_shot_prefill()
    src, margin = getattr(layers[0].self_attn, "_kv_scales_src", (None, None))
    if src is None:
        logger.warning(
            "MuseGlimmer FP8 KV: %s was not set at weight load; V scale is 1.0 (values below "
            "~2e-3 lose precision). Calibrate with harness/t028_calib_kv.py.", SCALES_ENV)
    for l in layers:
        l.self_attn.kv_fp8 = True
    logger.info(
        "MuseGlimmer FP8 KV: %d layers, k_dq=%.6g (analytic), v_dq range [%.4g, %.4g] "
        "(margin %s, scales=%s), fused_mask=%s", len(layers), layers[0].self_attn.k_dq,
        min(l.self_attn.v_dq for l in layers), max(l.self_attn.v_dq for l in layers),
        margin, src, fused_mask_allowed())
    return True
