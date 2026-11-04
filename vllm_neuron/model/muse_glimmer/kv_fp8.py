# SPDX-License-Identifier: Apache-2.0
"""FP8 KV cache for MuseGlimmer.

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
  (``MUSE_GLIMMER_KV_SCALES``, computed offline by a calibration pass over natural prompts) times
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
from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor

logger = logging.getLogger(__name__)

SCALES_ENV = "MUSE_GLIMMER_KV_SCALES"
V_MARGIN_ENV = "MUSE_GLIMMER_KV_V_MARGIN"
FUSED_MASK_ENV = "MUSE_GLIMMER_KV_FP8_FUSED_MASK"
PACKED_ENV = "MUSE_GLIMMER_KV_PACKED"
BLOCK_WRITE_ENV = "MUSE_GLIMMER_PREFILL_KV_BLOCK_WRITE"
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def packed_mode() -> str:
    """``MUSE_GLIMMER_KV_PACKED``: ``off`` (default), ``global`` or ``all`` (``1``).

    The packed layout stores a layer's K blocks as ``[block // 2, d, 2]``: two
    consecutive slots share one bf16-wide element, so the decode kernel loads K with
    a DMA transpose instead of an fp8 -> bf16 cast plus a PE transpose. V is never
    packed and the byte footprint is unchanged. Every layer owns its own blocks of a
    shared buffer, so the layout is chosen per layer: ``global`` packs the
    full-attention layers only (the sliding-window kernel is slower packed at small
    batch on trn2), ``all`` packs every layer.
    """
    v = os.environ.get(PACKED_ENV, "0").strip().lower()
    v = {"": "off", "0": "off", "false": "off", "1": "all", "true": "all", "yes": "all"}.get(v, v)
    if v not in ("off", "global", "all"):
        raise ValueError(f"{PACKED_ENV}={v!r} not in (0, global, all)")
    return v


INKERNEL_ENV = "MUSE_GLIMMER_KV_INKERNEL"
INKERNEL_MAX_BS_ENV = "MUSE_GLIMMER_KV_INKERNEL_MAX_BS"


def inkernel_max_bs() -> int:
    """``MUSE_GLIMMER_KV_INKERNEL_MAX_BS`` (default 8): largest decode batch bucket that writes K/V in the kernel.

    The in-kernel write takes the separate K/V scatter off the serial path (-1.7 ms per step at
    batch 1 on trn2), but at batch 64-128 with long context it lengthens the attention kernel,
    which is already the bottleneck, by more than the deferred scatter costs (about +1 ms).
    Larger buckets keep the deferred write. ``0`` = no limit.
    """
    v = os.environ.get(INKERNEL_MAX_BS_ENV, "").strip()
    return int(v) if v else 8


def inkernel_mode() -> str:
    """``MUSE_GLIMMER_KV_INKERNEL``: ``off`` (``0``), ``all`` (``1``, default) or ``packed``.

    Decode writes the new K/V inside the attention megakernel: the kernel runs with
    ``update_cache=True`` and the step's slot mapping, so the layer's K/V write is an
    aliased in-place update of its cache (the kernel's parity-split pair write for a
    packed K cache) instead of the deferred ``index_put_`` scatter after the layer loop.
    ``packed``: only the layers with a packed K cache; the others keep the deferred write.
    """
    v = os.environ.get(INKERNEL_ENV, "1").strip().lower()
    v = {"": "off", "0": "off", "false": "off", "1": "all", "true": "all", "yes": "all"}.get(v, v)
    if v not in ("off", "all", "packed"):
        raise ValueError(f"{INKERNEL_ENV}={v!r} not in (0, all, packed)")
    return v


def packed_requested() -> bool:
    """True when ``MUSE_GLIMMER_KV_PACKED`` asks the runner for the packed K allocation."""
    return packed_mode() != "off"


def block_write_mode() -> str:
    """``MUSE_GLIMMER_PREFILL_KV_BLOCK_WRITE``: ``packed`` (default), ``all`` or ``off``.

    Full-prefill KV write in whole cache blocks (one scatter row per block and head)
    for layers whose prefill slots are linear (no SWA ring). ``packed``: only for a
    packed FP8 K cache, where the per-token write must read the stored pairs.
    """
    v = os.environ.get(BLOCK_WRITE_ENV, "packed").strip().lower() or "packed"
    v = {"1": "all", "true": "all", "0": "off", "false": "off"}.get(v, v)
    if v not in ("packed", "all", "off"):
        raise ValueError(f"{BLOCK_WRITE_ENV}={v!r} not in (packed, all, off)")
    return v


def packed_view(k4: torch.Tensor) -> torch.Tensor:
    """Packed K layout ``[blocks, nkh, block // 2, d, 2]`` of the bound 4-D K cache (a view, same bytes)."""
    nb, nkh, bs, d = k4.shape
    return k4.view(nb, nkh, bs // 2, d, 2)


def write_packed_k(k4: torch.Tensor, k_rows: torch.Tensor, b: torch.Tensor, h: torch.Tensor,
                   p: torch.Tensor, unique_pairs: bool = False, ring_modulus: int | None = None) -> None:
    """Write K rows into the packed cache (in place).

    ``k4`` is the packed K cache bound as its 4-D ``[blocks, nkh, block, d]`` view (see
    :func:`bind_packed_views`): packed element o of a block holds slots 2o / 2o + 1 as
    ``[d, 2]`` and occupies 4-D rows 2o and 2o + 1. Writing through that view uses the
    same (block, head, row) scatter as the unpacked cache, so the compiled graph updates
    the shared KV buffer in place (other views of it made the graph copy the buffer).

    ``k_rows`` [N, d] (cache dtype) go to (block ``b``, head ``h``, slot ``p``): element
    ``p // 2``, lane ``p % 2``. A single-lane store does not lower on Neuron, so a row
    writes its whole pair: the other lane comes from the row of this write that targets
    the partner slot, or else from the cache. Lane parity comes from the slot, so any
    slot order works. Partner rows are searched at fixed offsets: ``t +- 1``
    (consecutive slots of one sequence) and, with ``ring_modulus=R``, ``t +- (1 - R)``
    (the SWA ring maps position q to slot q % R with R odd, so the two slots of a pair
    can be the first and the last kept position of a prompt). ``unique_pairs=True``
    (decode, one token per request) skips the search. Rows in the null block (padding)
    never pair.
    """
    o = p // 2
    lane = p % 2
    d = k_rows.shape[-1]
    bb, hh, rr = torch.cat([b, b]), torch.cat([h, h]), torch.cat([2 * o, 2 * o + 1])
    n = k_rows.shape[0]
    g = k4[(bb, hh, rr)].to(torch.bfloat16)  # [2N, d]; fp8 -> bf16 -> fp8 is exact
    old = torch.cat([g[:n], g[n:]], dim=-1).view(n, d, 2)
    kb = k_rows.to(torch.bfloat16)
    is0 = (lane == 0).unsqueeze(-1)
    other0, other1 = old[..., 0], old[..., 1]
    if not unique_pairs:
        t = torch.arange(n, device=kb.device)
        offs = [1] if ring_modulus is None else [1, 1 - int(ring_modulus)]
        part = torch.zeros_like(kb)
        found = torch.zeros(n, dtype=torch.bool, device=kb.device)
        for off in offs:
            # lane-0 rows look forward by +off, lane-1 rows backward by -off
            c = torch.where(lane == 0, t + off, t - off)
            inr = (c >= 0) & (c < n)
            c = c.clamp(0, n - 1)
            m = inr & (b[c] == b) & (h[c] == h) & (o[c] == o) & (lane[c] != lane) & (b != 0) & ~found
            part = torch.where(m.unsqueeze(-1), kb[c], part)
            found = found | m
        f = found.unsqueeze(-1)
        other0 = torch.where(f, part, other0)
        other1 = torch.where(f, part, other1)
    lane0 = torch.where(is0, kb, other0)
    lane1 = torch.where(is0, other1, kb)
    val = torch.stack([lane0, lane1], dim=-1).reshape(n, 2 * d).to(k4.dtype)  # [N, d, 2] -> rows 2o | 2o + 1
    k4.index_put_((bb, hh, rr), torch.cat([val[:, :d], val[:, d:]]))


def bind_packed_views(layers) -> None:
    """Rebind each layer's packed 5-D K cache as a 4-D ``[blocks, nkh, block, d]`` view.

    The view object is shared by every layer on the same buffer (as the runner does for its
    slices): distinct per-layer views of one buffer would be overlapping graph inputs.
    Called from ``bind_kv_cache`` (outside the trace) after :func:`setup`.
    """
    memo: dict = {}
    for l in layers:
        a = l.self_attn
        k5, v = a.k_cache, a.v_cache
        if k5 is None or v is None or k5.dim() != v.dim() + 1:
            continue
        key = (id(k5), tuple(k5.shape))
        if key not in memo:
            nb, nkh = k5.shape[0], k5.shape[1]
            memo[key] = k5.view(nb, nkh, v.shape[2], v.shape[3])
        a.k_cache = memo[key]
        a.k_cache_packed = k5  # the runner's packed tensor itself (in-kernel decode write target)


def write_kv_blocks(k_cache: torch.Tensor, v_cache: torch.Tensor, k_q: torch.Tensor, v_q: torch.Tensor,
                    slot_mapping: torch.Tensor, block_size: int, packed: bool) -> None:
    """Full-prefill KV write in whole blocks (one ``index_put_`` row per block and head).

    ``k_q`` / ``v_q``: [nkh * T, d] head-major, already in the cache dtype; T is a whole
    number of blocks and the step is one single-shot prefill (tokens start at position
    0, so token chunk c fills one block, the block of its first slot). Padding chunks go
    to the null block. The pad tail of the last real block holds pad values at slots past
    the prompt; the decode mask stops at the sequence length, so they are not read before
    decode writes them. A packed K block gets both lanes of every pair from the new
    tokens, so the write does not read the cache. ``k_cache`` is the 4-D K view (packed
    or not), so one (block, head) row is a contiguous [block, d] slab in both layouts.
    """
    T = slot_mapping.shape[0]
    d = k_q.shape[-1]
    nkh = k_q.shape[0] // T
    nblk = T // block_size
    safe = torch.where(slot_mapping < 0, torch.zeros_like(slot_mapping), slot_mapping)
    if not torch.compiler.is_dynamo_compiling() and not isinstance(slot_mapping, _FakeTensor):
        blocks = (safe // block_size).view(nblk, block_size)
        pos = (safe % block_size).view(nblk, block_size)
        off = torch.arange(block_size, device=pos.device).view(1, -1)
        ok = (blocks == 0) | ((blocks == blocks[:, :1]) & (pos == off))
        assert bool(ok.all()), "MuseGlimmer block KV write: prefill slots are not block-aligned"
    blk = (safe.view(nblk, block_size)[:, 0] // block_size)
    index = (blk.repeat(nkh), torch.arange(nkh, dtype=blk.dtype, device=blk.device).repeat_interleave(nblk))
    v_cache.index_put_(index, v_q.reshape(nkh * nblk, block_size, d))
    if packed:
        # packed bytes of a block, written through the 4-D K view as [block, d] rows
        k_rows = k_q.reshape(nkh * nblk, block_size // 2, 2, d).to(torch.bfloat16)
        k_rows = k_rows.transpose(-1, -2).contiguous().to(k_cache.dtype).view(nkh * nblk, block_size, d)
    else:
        k_rows = k_q.reshape(nkh * nblk, block_size, d)
    k_cache.index_put_(index, k_rows)




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


CHUNKED_ENV = "MUSE_GLIMMER_KV_FP8_CHUNKED"


def fp8_chunked_enabled() -> bool:
    """``MUSE_GLIMMER_KV_FP8_CHUNKED`` (default 1): chunked prefill over the unpacked FP8 cache.

    Later prefill chunks gather the prior K/V from the FP8 cache, dequantize them and run flash
    attention with them as the prefix (``model_bf16._fp8_chunked_prefill``), so the FP8 cache
    no longer needs ``max_num_batched_tokens >= max_model_len``. ``0`` keeps the old rule.
    """
    return os.environ.get(CHUNKED_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


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
    if mnbt is not None and mml is not None and mnbt < mml and fp8_chunked_enabled():
        # Chunked prefill reads the prior chunks through the segmented path, which the runner
        # turns on only when mnbt is a supported segment size (its default). Without it later
        # chunks would attend to themselves alone (wrong tokens, no error).
        from vllm_neuron.utils.bucket_utils import SUPPORTED_KV_SEGMENT_SIZES

        if mnbt not in SUPPORTED_KV_SEGMENT_SIZES:
            raise ValueError(
                f"MuseGlimmer FP8 KV chunked prefill needs max_num_batched_tokens in "
                f"{sorted(SUPPORTED_KV_SEGMENT_SIZES)} (the segment size); got {mnbt} < "
                f"max_model_len={mml}.")
    if mnbt is not None and mml is not None and mnbt < mml and not fp8_chunked_enabled():
        raise ValueError(
            f"MuseGlimmer FP8 KV cache needs single-shot prefill: max_num_batched_tokens={mnbt} < "
            f"max_model_len={mml}. Chunked prefill would need segmented attention over the FP8 "
            "cache, which the runner disables for a non-packed FP8 cache. Set "
            "max_num_batched_tokens >= max_model_len or use --kv-cache-dtype auto.")
    # Speculative decoding: the scheduler caps a prefill step at mnbt - draft slots (DFlash:
    # 15) and rounds it down to the KV block size (scheduler._align_token_budget_to_kv_blocks;
    # DFlash unifies the groups to block 64). At mnbt == mml a prompt longer than that budget
    # is chunked, and the FP8 cache has no segmented prefill: wrong tokens from step 0
    # (CPU check). The runner clamps its own mnbt to mml, so mnbt = mml + 64 keeps the
    # prefill buckets and makes the scheduler budget mml.
    spec = getattr(vc, "speculative_config", None)
    draft = int(getattr(spec, "max_num_new_slots_for_drafting", 0) or 0) if spec is not None else 0
    if draft and not fp8_chunked_enabled() and mnbt is not None and mml is not None and (mnbt - draft) // 64 * 64 < mml:
        raise ValueError(
            f"MuseGlimmer FP8 KV cache with speculative decoding needs max_num_batched_tokens >= "
            f"max_model_len + 64 (draft slots {draft}, rounded to the KV block), e.g. "
            f"{mml + 64}; got max_num_batched_tokens={mnbt}, max_model_len={mml}. Otherwise "
            "prompts near max_model_len are chunked, and the FP8 cache has no segmented prefill.")


def setup(layers) -> bool:
    """Called from ``bind_kv_cache`` (outside the trace). Sets per-layer FP8 KV state.

    Returns True when the caches are FP8.
    """
    fp8 = [is_fp8(l.self_attn.k_cache) for l in layers]
    if not any(fp8):
        if packed_requested():
            logger.warning("MuseGlimmer: %s=1 has no effect with a bf16 KV cache", PACKED_ENV)
        for l in layers:
            l.self_attn.kv_fp8 = False
            l.self_attn.kv_packed = False
            # Default-on applies to the FP8 cache (measured); a bf16 cache writes in the kernel only on request.
            l.self_attn.kv_inkernel = os.environ.get(INKERNEL_ENV) is not None and inkernel_mode() == "all"
        return False
    if not all(fp8) or not all(is_fp8(l.self_attn.v_cache) for l in layers):
        raise ValueError("MuseGlimmer FP8 KV: every layer's K and V cache must be FP8")
    _check_single_shot_prefill()
    src, margin = getattr(layers[0].self_attn, "_kv_scales_src", (None, None))
    if src is None:
        logger.warning(
            "MuseGlimmer FP8 KV: %s was not set at weight load; V scale is 1.0 (values below "
            "~2e-3 lose precision). Calibrate the per-layer V absmax offline.", SCALES_ENV)
    packed = [l.self_attn.k_cache.dim() == l.self_attn.v_cache.dim() + 1 for l in layers]
    if any(packed) and not all(packed):
        raise ValueError("MuseGlimmer FP8 KV: the packed K layout must cover every layer")
    if packed_requested() and not packed[0]:
        raise ValueError(
            f"{PACKED_ENV}=1 but the runner allocated an unpacked K cache (the packed layout needs "
            "--kv-cache-dtype fp8 (e4m3) and the V1 model runner)")
    if packed[0] and layers[0].self_attn.k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("MuseGlimmer packed FP8 KV needs float8_e4m3fn")
    mode = packed_mode()
    ik = inkernel_mode()
    for l in layers:
        l.self_attn.kv_fp8 = True
        l.self_attn.kv_packed = packed[0] and (mode == "all" or l.self_attn.sliding_window is None)
        l.self_attn.kv_inkernel = ik == "all" or (ik == "packed" and l.self_attn.kv_packed)
    logger.info(
        "MuseGlimmer FP8 KV: %d layers, k_dq=%.6g (analytic), v_dq range [%.4g, %.4g] "
        "(margin %s, scales=%s), fused_mask=%s, packed K=%s (%d layers), prefill block write=%s, "
        "in-kernel decode write=%s", len(layers),
        layers[0].self_attn.k_dq, min(l.self_attn.v_dq for l in layers), max(l.self_attn.v_dq for l in layers),
        margin, src, fused_mask_allowed(), packed_mode() if packed[0] else "off",
        sum(l.self_attn.kv_packed for l in layers), block_write_mode(),
        f"{ik} ({sum(l.self_attn.kv_inkernel for l in layers)} layers)")
    return True
