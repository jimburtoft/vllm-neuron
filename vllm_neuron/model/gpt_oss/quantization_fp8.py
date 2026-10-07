# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization for the GPT-OSS MoE expert weights on Trn2.

GPT-OSS ships MXFP4. Trn2 cannot do MX *compute* -- ``nki.isa.nc_matmul_mx`` is a
NeuronCore-v4 instruction and nkilib's MX MoE kernels require ``nc_version >= gen4`` --
so the Trn2 path dequantizes MXFP4 to BF16 on the host
(``weight_loaders_bf16._dequantize_mxfp4_to_bf16``) and the device sees full BF16.

Trn2 *can* do plain (non-MX) FP8 matmul: ``float8_e4m3`` is in its valid ``nc_matmul``
stationary-dtype list, and the non-MX MoE kernels accept FP8 ROW weights because their
``>= gen4`` assert sits inside an ``is_mx_kernel`` branch keyed on the weight dtype.
Requantizing the dequantized experts to FP8 halves the expert-weight bytes, which is
what decode is bound by.

**The requantization is bit-exact.** MXFP4 dequantizes to ``FP4_LUT[i] * 2**(e - 127)``,
and the 16-entry LUT is ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` and negatives: every entry needs
at most 2 mantissa bits, and FP8 E4M3 has 3. A power-of-two scale shifts only the
exponent, so it cannot perturb the mantissa, and E4M3 is a strict superset of MXFP4's
value set under such a scale. Measured over all 48 expert tensors of
``openai/gpt-oss-20b``: 48/48 round-trip with ``torch.equal`` true. The scale exponents
observed span 2**-15 to 2**4, well inside E4M3's range.

A naive ``amax / fp8_max`` scale would lose about 1e-2 relative L2 for nothing, so the
scale is always rounded *up* to a power of two. The cost is at most one bit of headroom.

**Dtype: the 240-vs-448 trap.** Trn2 interprets FP8 E4M3 with max 240 (nkilib's
``DtypeMode.NON_OCP``), while the OCP interpretation used by Trn3 has max 448. The torch
dtype whose native max is 240 is ``float8_e4m3fnuz``, but the NKI bridge
(``libtorch_neuronx_lite.nki.nki_dtype``) does not accept it and raises
"Unsupported torch dtype for NKI". The bridge accepts ``float8_e4m3`` and reinterprets it
as NKI ``float8_e4m3`` (max 240) when the platform target is Trn2. So the storage dtype
is ``torch.float8_e4m3fn`` while the numeric range quantized into is 240: the carrier and
the range have to be set independently, which is why they are separate constants below.

Do not apply a 448/240 downscale here. ``llama3/quantization.py`` does that to rescale a
ModelOpt checkpoint calibrated for the 448 range; these weights are quantized from BF16
in this module, so there is nothing to rescale and the factor would silently shrink every
weight.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Storage dtype the NKI bridge accepts. On Trn2 it is reinterpreted as NKI
# ``float8_e4m3``, whose max is 240 -- see the module docstring.
FP8_TRN2_DTYPE = torch.float8_e4m3fn

# Numeric range NKI uses for that dtype on Trn2. Quantizing to 448 would overflow it.
FP8_TRN2_MAX = 240.0

# Suffixes of the expert weights this module quantizes.
_EXPERT_WEIGHT_SUFFIXES = (
    "mlp.experts.gate_up_proj_weight",
    "mlp.experts.down_proj_weight",
)


def pow2_scale(amax: torch.Tensor, fp8_max: float = FP8_TRN2_MAX) -> torch.Tensor:
    """Return the smallest power-of-two scale ``s`` with ``amax / s <= fp8_max``.

    Rounding up to a power of two is what makes an MXFP4-sourced weight round-trip
    bit-exact: it shifts only the exponent. See the module docstring.

    The result is also clamped to the smallest *normal* power of two. Only an all-zero
    (padded) channel can reach that clamp, and without it such a channel yields a
    subnormal scale, which is a hazard: accelerator datapaths commonly flush subnormals
    to zero, BF16 staging underflows them outright, and a reciprocal of a flushed zero
    poisons a tile with Inf. Clamping to 1.0 instead would be wrong -- it silently zeroes
    genuinely small channels.

    Args:
        amax: Absolute maximum per channel, any float dtype.
        fp8_max: Largest representable magnitude of the target FP8 range.

    Returns:
        The scale, float32, same shape as ``amax``. Both MoE kernels consume float32
        scales.
    """
    amax = amax.to(torch.float32)
    raw = amax.clamp(min=torch.finfo(torch.float32).tiny) / fp8_max
    scale = torch.exp2(torch.ceil(torch.log2(raw)))
    min_normal_pow2 = torch.exp2(
        torch.ceil(torch.log2(torch.tensor(torch.finfo(torch.float32).tiny)))
    )
    return torch.maximum(scale, min_normal_pow2.to(scale.dtype))


def quantize_expert_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one expert weight to FP8 with a per-output-channel power-of-two scale.

    The contracted dimension is 1 for both projections, matching the layouts the kernels
    consume::

        gate_up : [E, H, 2I]  -- contract H -> scale [E, 1, 2I]
        down    : [E, I, H]   -- contract I -> scale [E, 1, H]

    The scale must be per *output* channel, because the kernels apply the dequantization
    elementwise on the matmul output; a per-input-channel scale cannot be undone there.

    Args:
        weight: Expert weight ``[E, contracted, out_channels]``, any float dtype.

    Returns:
        ``(quantized, scale)``. ``quantized`` is ``FP8_TRN2_DTYPE`` and ``scale`` is
        float32 of shape ``[E, 1, out_channels]``. Dequantization is exactly
        ``quantized.float() * scale``.

    Raises:
        ValueError: If the scale is not per-output-channel for this weight, or if the
            round-trip is not bit-exact.
    """
    weight_f32 = weight.to(torch.float32)
    amax = weight_f32.abs().amax(dim=1, keepdim=True)
    scale = pow2_scale(amax)
    quantized = (
        (weight_f32 / scale).clamp(-FP8_TRN2_MAX, FP8_TRN2_MAX).to(FP8_TRN2_DTYPE)
    )

    expected = (weight.shape[0], 1, weight.shape[-1])
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale {tuple(scale.shape)} is not per-output-channel for a weight of "
            f"shape {tuple(weight.shape)} (expected {expected})"
        )

    dequantized = (quantized.to(torch.float32) * scale).to(weight.dtype)
    if not torch.equal(dequantized, weight):
        n_differ = int((dequantized != weight).sum())
        raise ValueError(
            f"FP8 requantization is not bit-exact ({n_differ} elements differ). An "
            f"MXFP4-sourced weight must round-trip exactly under a power-of-two scale; "
            f"check that the scale is a power of two and the clamp is "
            f"{FP8_TRN2_MAX:.0f}."
        )
    return quantized, scale


def quantize_expert_state_dict(
    state_dict: dict[str, torch.Tensor],
    keep_bf16_for_prefill: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Replace the MoE expert weights in a host state dict with FP8 ones.

    The extra parameters are returned separately rather than placed in the state dict,
    because ``load_sharded_pipelined`` requires a checkpoint key for every parameter it
    walks and the checkpoint has neither FP8 scales nor a second weight copy. The caller
    registers them on the modules after ``load_state_dict``.

    Quantizing here, on the loaded tensor, is deliberate: the weight loaders emit a
    different geometry than the parameter declaration (H-sharded and padded, against
    I-sharded), and ``load_state_dict(assign=True)`` replaces the parameter object
    without checking the shape. A scale derived from the declaration would therefore not
    correspond to the weight the kernel sees, which produces fluent but wrong output
    rather than an error. Deriving the weight and its scale from the same tensor removes
    that class of issue.

    Args:
        state_dict: Rank-sharded checkpoint state dict. Not modified.
        keep_bf16_for_prefill: Also return the original BF16 tensor under a
            ``..._weight_bf16`` key, so prefill can keep using it. FP8 prefill is slower
            than BF16 (the per-channel FP8 path in ``bwmm_shard_on_I`` upcasts every
            weight tile back to BF16 before the matmul), while decode is bound by weight
            DMA and gains from FP8. Keeping both costs one extra BF16 copy of the expert
            weights in HBM, which reduces KV-cache capacity.

    Returns:
        ``(new_state_dict, extra_params)``. ``extra_params`` maps each parameter path to
        its tensor: a ``..._scale`` for every quantized weight, plus a
        ``..._weight_bf16`` for each when ``keep_bf16_for_prefill`` is set.
    """
    out = dict(state_dict)
    extra_params: dict[str, torch.Tensor] = {}
    n_quantized = 0

    for key, weight in state_dict.items():
        if not key.endswith(_EXPERT_WEIGHT_SUFFIXES) or weight.dtype == FP8_TRN2_DTYPE:
            continue
        # load_sharded_pipelined may already have moved these to the Neuron device,
        # where a bf16 -> fp32 cast raises "Expected self.dtype() == dst.dtype()". Pull
        # the tensor back with a same-dtype copy first, then cast on the host.
        if weight.device.type != "cpu":
            weight = weight.to("cpu")
        try:
            quantized, scale = quantize_expert_weight(weight)
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from e

        out[key] = quantized
        extra_params[key.replace("_weight", "_scale")] = scale
        if keep_bf16_for_prefill:
            extra_params[key.replace("_weight", "_weight_bf16")] = weight
        n_quantized += 1

    logger.info(
        "Quantized %d GPT-OSS expert tensors to %s (clamp %.0f); BF16 prefill copy: %s",
        n_quantized,
        FP8_TRN2_DTYPE,
        FP8_TRN2_MAX,
        "kept" if keep_bf16_for_prefill else "dropped",
    )
    return out, extra_params
