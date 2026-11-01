# SPDX-License-Identifier: Apache-2.0
"""MuseGlimmer decode fusions.

``MUSE_GLIMMER_DECODE_FUSE`` = comma list of the knobs below. Unset = ``gate`` (the default,
used for decode batch buckets >= ``MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS``, default 16); ``off`` = the old path.

- ``norm``  (018a): fold ``input_layernorm`` into the decode megakernel (``rmsnorm_X_enabled``). The centered
  weight is passed as ``gamma = 1 + w`` in **fp32** (``nkilib`` loads gamma in its own dtype and multiplies in
  fp32, so ``1 + w`` is not rounded to bf16; simulator: bit-identical to the unfused fp32 torch norm at B=1).
- ``ffn``   (018b): fold ``post_attention_layernorm``'s residual add and ``pre_feedforward_layernorm`` into
  ``NF.mlp`` (``fused_add_tensor`` + ``RMS_NORM`` with fp32 ``gamma = 1 + w``, ``store_fused_add_result`` for
  the new residual). FP8 ROW MLP decode (TKG) only.
- ``gate``  (022): sigmoid output gate and o_proj inside the megakernel
  (``nki_attn_gate_tkg``, ``gate_decode.gated_attention_decode``). BF16 or FP8 ROW (``MUSE_GLIMMER_FP8=..,attn``)
  qkv / gate / o_proj weights.
- ``early_gate`` (022 cheap version): compute ``sigmoid(x @ Wgate)`` before the megakernel call, so the gate
  weight DMA is issued ahead of attention (the compiler may overlap it with the attention KV traffic).

Decode only (``max_query_len <= decode_token_threshold``) and B*S <= 128 (megakernel single tile);
everything else takes the shipping path.
"""
from __future__ import annotations

import os

import torch

DECODE_FUSE_ENV = "MUSE_GLIMMER_DECODE_FUSE"
_VALID = {"norm", "ffn", "gate", "early_gate"}


DEFAULT_FLAGS = "gate"  # in-kernel gate on by default (batch buckets >= gate_min_bs())


def decode_fuse_flags() -> frozenset:
    v = os.environ.get(DECODE_FUSE_ENV, DEFAULT_FLAGS).strip().lower()
    if v in ("", "0", "off", "none"):
        return frozenset()
    flags = frozenset(x.strip() for x in v.replace("+", ",").split(",") if x.strip())
    bad = flags - _VALID
    if bad:
        raise ValueError(f"{DECODE_FUSE_ENV}: unknown {sorted(bad)} (valid: {sorted(_VALID)})")
    if "gate" in flags and "early_gate" in flags:
        raise ValueError(f"{DECODE_FUSE_ENV}: 'gate' and 'early_gate' are exclusive")
    return flags


GATE_MIN_BS_ENV = "MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS"


GATE_MIN_BS_FP8_ENV = "MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS_FP8"


def gate_min_bs(fp8_attn: bool = False) -> int:
    """Smallest decode batch bucket that takes the in-kernel gate (+3 % at B=1, -5..-15 % at 16..64 on trn2).

    ``fp8_attn`` (``MUSE_GLIMMER_FP8`` includes ``attn``): separate threshold
    ``MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS_FP8`` (default 1). With FP8 ROW attention weights the unfused path runs
    the gate and o_proj as separate qkv / o_proj kernels, so the in-kernel gate is used at every bucket.
    """
    if fp8_attn:
        return int(os.environ.get(GATE_MIN_BS_FP8_ENV, "1"))
    return int(os.environ.get(GATE_MIN_BS_ENV, "16"))


def centered_gamma(norm) -> torch.Tensor:
    """fp32 ``[1, H]`` gamma = 1 + w for a MuseGlimmerCenteredRMSNorm (traced: one tiny elementwise op)."""
    return (1.0 + norm.weight.to(torch.float32)).reshape(1, -1)
