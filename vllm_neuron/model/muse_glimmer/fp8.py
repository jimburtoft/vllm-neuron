# SPDX-License-Identifier: Apache-2.0
"""Load-time FP8 self-quantization for MuseGlimmer.

``MUSE_GLIMMER_FP8`` selects which linears are stored in FP8 (comma list):

    ""      -> BF16 everywhere (default)
    "mlp"   -> gate/up/down of every decoder MLP in FP8
    "attn"  -> qkv, output-gate and o_proj of every decoder attention in FP8

Scheme: nkilib ``QuantizationType.ROW`` on trn2.

- Weights: float8 e4m3 with one fp32 dequant scale per **output channel**,
  ``scale = absmax / 240`` (trn2's nkilib kernels use the legacy
  ``nl.float8_e4m3``, max 240; a value <= 240 has the same bit pattern in
  ``torch.float8_e4m3fn``, so clamping to 240 and storing as e4m3fn is exact).
  Scales are pre-broadcast to ``[128, N]`` as the kernel requires.
- Activations: no calibration.
  * decode (TKG, T <= 96): the kernel multiplies bf16 activations by the fp8
    weights and applies the per-channel scale on PSUM eviction (weight-only).
  * prefill / large-T (CTE): activations are quantized per row (per token)
    at run time, by ``NF.rmsnorm_quant(ROW, NO_NORM)`` for the MLP input and
    in-kernel for the intermediate.

The intermediate size per rank is zero-padded to a multiple of
``FP8_I_ALIGN`` (4992 -> 5120 at TP=4): the nkilib CTE ROW kernel returns
wrong results when the per-core intermediate (I/2 under I-sharding) is not a
multiple of 128 (NKI simulator: cos 0.986 at I=4992 and 2496, 0.9996 at
4864/5120). Zero gate/up columns and zero down rows contribute exactly 0.

The checkpoint stays BF16. Each weight and its scale have separate loaders
that run the same deterministic quantization, so the loader order across
threads does not matter.
"""

from __future__ import annotations

import os

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

FP8_ENV = "MUSE_GLIMMER_FP8"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 240.0  # trn2 nl.float8_e4m3
PMAX = 128
FP8_I_ALIGN = 256
_MIN_SCALE = 1e-12
_VALID = {"mlp", "attn"}


def fp8_modules() -> frozenset[str]:
    raw = os.environ.get(FP8_ENV, "").strip().lower()
    mods = frozenset(m.strip() for m in raw.split(",") if m.strip())
    bad = mods - _VALID
    if bad:
        raise ValueError(f"{FP8_ENV}={raw!r}: unknown entries {sorted(bad)}; valid: {sorted(_VALID)}")
    return mods


def round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def quantize_per_out_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``w`` [K, N] (contraction K, output N) -> (fp8 [K, N], fp32 scale [1, N])."""
    wf = w.to(torch.float32)
    scale = (wf.abs().amax(dim=0, keepdim=True) / FP8_MAX).clamp(min=_MIN_SCALE)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


def _shard_kn(slice_obj, rank: int, num_shards: int, shard_size: int, shard_out: bool,
              pad_to: int) -> torch.Tensor:
    """Checkpoint ``[N_full, K_full]`` (HF Linear layout) -> rank shard as ``[K, N]``, zero-padded.

    ``shard_out=True``: shard the output dim N (column parallel, gate/up); pad N.
    ``shard_out=False``: shard the contraction dim K (row parallel, down); pad K.
    """
    r = rank % num_shards
    s, e = r * shard_size, (r + 1) * shard_size
    t = slice_obj[s:e, :] if shard_out else slice_obj[:, s:e]
    t = t.T  # [K, N]
    K, N = t.shape
    if shard_out and N < pad_to:
        t = torch.cat([t, t.new_zeros(K, pad_to - N)], dim=1)
    elif not shard_out and K < pad_to:
        t = torch.cat([t, t.new_zeros(pad_to - K, N)], dim=0)
    return t.contiguous()


def fp8_weight_loader(*, num_shards: int, shard_size: int, shard_out: bool, pad_to: int):
    def transform(slices, rank):
        q, _ = quantize_per_out_channel(_shard_kn(slices[0], rank, num_shards, shard_size, shard_out, pad_to))
        return q

    return SafetensorsWeightLoader(transform=transform)


def fp8_scale_loader(*, num_shards: int, shard_size: int, shard_out: bool, pad_to: int):
    def transform(slices, rank):
        _, s = quantize_per_out_channel(_shard_kn(slices[0], rank, num_shards, shard_size, shard_out, pad_to))
        return s.expand(PMAX, s.shape[1]).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def wrap_fp8(loader: SafetensorsWeightLoader, want: str) -> SafetensorsWeightLoader:
    """Quantize the ``[K, N]`` output of an existing (sharding / fused-QKV) BF16 loader.

    ``want="q"`` -> fp8 weight; ``want="s"`` -> ``[128, N]`` fp32 per-output-channel scale.
    """
    base = loader.transform or (lambda slices, rank: slices[0][:])

    def transform(slices, rank):
        q, sc = quantize_per_out_channel(base(slices, rank))
        return q if want == "q" else sc.expand(PMAX, sc.shape[1]).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def dequant(w: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """CPU / torch-fallback helper: fp8 ``[K, N]`` x ``[128, N]`` scale -> ``dtype``."""
    return (w.to(torch.float32) * scale[0:1].to(torch.float32)).to(dtype)
