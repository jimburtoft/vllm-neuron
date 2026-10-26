"""FP8 ROW quantization for GPT-OSS MoE expert weights on TRN2.

Companion to `weight_loaders_bf16.py`.

Background
----------
GPT-OSS checkpoints are MXFP4. TRN2 has no MX matmul (`nki.isa.nc_matmul_mx` is a
NeuronCore-v4 instruction, and the nkilib MoE kernels require gen4 for MX weights), so
the BF16 path dequantizes MXFP4 -> BF16 on the host and the device holds full BF16
expert weights. TRN2 does support plain (non-MX) FP8 matmul: `float8_e4m3` is a valid
`nc_matmul` dtype and the nkilib MoE kernels' gen4 check applies only to MX weights.
Storing the experts as FP8 halves expert-weight memory and DMA traffic.

Power-of-two scales are exact
-----------------------------
MXFP4 dequantizes to `FP4_LUT[i] * 2^(scale-127)` with the LUT
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}` (and negatives). Every entry needs at most 2 mantissa
bits; FP8 E4M3 has 3. A power-of-two rescale only shifts the exponent, so every
MXFP4-derived value is representable in FP8 E4M3 and the round trip is bit-exact
(scale exponents for gpt-oss-20b span 2^-15..2^4, well inside E4M3's range). A plain
`amax / fp8_max` scale is not a power of two and loses precision for no benefit, so
the scale is always rounded up to a power of two (at most 1 bit of headroom).

Dtype: 240 vs 448
-----------------
TRN2 uses NKI `float8_e4m3` (max 240). The OCP `float8_e4m3fn` interpretation (max
448) is rejected by the trn2 compiler and accepted on trn3. Weights are quantized
directly from BF16 into the 240 range; there is no 448-calibrated checkpoint to
downscale, so no 240/448 factor is applied. See `FP8_TRN2_DTYPE` for the torch carrier
dtype.

Scale layouts
-------------
The decode and prefill kernels take different scale shapes:

    decode  (moe_tkg via moe_block_tkg), QuantizationType.ROW:
        gate_up : [E, 2, I]
        down    : [E, H]
    prefill (moe_cte):
        gate_up : [E, 1, 2*I]
        down    : [E, 1, H]

Scales are stored in the prefill layout and reshaped for decode at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# torch <-> NKI FP8 dtype mapping.
#
# The torch storage dtype handed to the kernel is `torch.float8_e4m3fn`, while the numeric
# range on TRN2 is 240. The NKI dtype bridge (`libtorch_neuronx_lite/nki/nki_dtype.py`)
# accepts only `float8_e4m3fn` and `float8_e5m2` from torch and maps `float8_e4m3fn` to
# NKI `float8_e4m3` (max 240) when the platform target is trn2. `torch.float8_e4m3fnuz`
# (the torch dtype with max 240) is not accepted by the bridge
# ("Unsupported torch dtype for NKI: torch.float8_e4m3fnuz").
#
# So weights are stored as `float8_e4m3fn` but quantized to the 240 range; clamping to 448
# would overflow the trn2 interpretation. Dtype and max are therefore separate constants.
FP8_TRN2_DTYPE = torch.float8_e4m3fn   # storage/carrier dtype the NKI bridge accepts
FP8_TRN2_MAX = 240.0                   # numeric range NKI uses for it on trn2

# The torch dtype whose native max is 240. Not accepted by the NKI bridge; kept for
# documentation and so tests can assert it is not used.
FP8_E4M3FNUZ_DTYPE = torch.float8_e4m3fnuz
FP8_E4M3FNUZ_MAX = 240.0

# OCP interpretation (max 448), used for the same carrier dtype on TRN3. Not used here.
FP8_OCP_DTYPE = torch.float8_e4m3fn
FP8_OCP_MAX = 448.0


def pow2_scale(amax: torch.Tensor, fp8_max: float = FP8_TRN2_MAX) -> torch.Tensor:
    """Smallest power-of-two scale `s` with `amax / s <= fp8_max`.

    Rounding up to a power of two makes an MXFP4-sourced weight round-trip
    bit-exactly: it shifts only the exponent. See the module docstring.

    Returns float32 (scales are consumed as f32 by both kernels).
    """
    # Two constraints:
    #   (a) the scale is the smallest pow2 with amax/s <= fp8_max (exactness);
    #   (b) the scale is a normal f32.
    # All-zero (padded) channels would otherwise get a subnormal scale (~2^-133): datapaths
    # may flush denormals to zero, a bf16 intermediate underflows them, and the reciprocal of a
    # flushed zero produces Inf. Clamping the scale to the smallest normal power of two is a
    # no-op for every real channel and only affects the all-zero ones. (Clamping to 1.0
    # instead would zero out genuinely small channels.)
    amax = amax.to(torch.float32)
    raw = amax.clamp(min=torch.finfo(torch.float32).tiny) / fp8_max
    scale = torch.exp2(torch.ceil(torch.log2(raw)))
    min_normal_pow2 = torch.exp2(
        torch.ceil(torch.log2(torch.tensor(torch.finfo(torch.float32).tiny)))
    )
    return torch.maximum(scale, min_normal_pow2.to(scale.dtype))


def quantize_row_fp8(
    w: torch.Tensor,
    *,
    channel_dim: int = -1,
    dtype: torch.dtype = FP8_TRN2_DTYPE,
    fp8_max: float = FP8_TRN2_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (ROW) FP8 quantization with a power-of-two scale.

    Args:
        w: weight tensor, any float dtype (typically the BF16 output of the MXFP4 dequant).
        channel_dim: dim reduced to form each scale, i.e. one scale per *other* index.

    Returns:
        (q, scale) where `q` is `dtype` and `scale` is float32 with `channel_dim` kept as 1.
        Dequantization is exactly `q.float() * scale`.
    """
    wf = w.to(torch.float32)
    amax = wf.abs().amax(dim=channel_dim, keepdim=True)
    scale = pow2_scale(amax, fp8_max)
    q = (wf / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return q, scale


def quantize_static_fp8(
    w: torch.Tensor,
    *,
    dtype: torch.dtype = FP8_TRN2_DTYPE,
    fp8_max: float = FP8_TRN2_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor (STATIC) FP8 quantization with a power-of-two scale.

    Also bit-exact for MXFP4-sourced weights. Kept for completeness; ROW is preferred
    because it needs no activation calibration and has documented scale shapes.
    """
    wf = w.to(torch.float32)
    scale = pow2_scale(wf.abs().amax(), fp8_max)
    q = (wf / scale).clamp(-fp8_max, fp8_max).to(dtype)
    return q, scale.reshape(1)


def verify_exact(w_ref: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> bool:
    """True iff dequantizing `q` reproduces `w_ref` exactly.

    Holds for MXFP4-sourced weights; False indicates a non-pow2 scale, the wrong FP8
    dtype, or a source that is not MXFP4-derived.
    """
    deq = (q.to(torch.float32) * scale).to(w_ref.dtype)
    return torch.equal(deq, w_ref)


def quantize_expert_state_dict(state_dict: dict, model=None, keep_bf16: bool = False,
                               gate_up_major: bool = False):
    """Quantize MoE expert weights to FP8 ROW inside a host-side state dict.

    Returns `(new_state_dict, scales)` where `new_state_dict` has the expert weights
    replaced by FP8 tensors, and `scales` maps a parameter path (e.g.
    `model.layers.0.mlp.experts.gate_up_proj_scale`) to its tensor. With `keep_bf16`,
    `scales` also holds the original BF16 weights under `*_weight_bf16` for prefill. These
    are returned separately because they must be registered on the modules after
    load_state_dict: `load_sharded_pipelined(self, ...)` requires a checkpoint key for every
    parameter, and the checkpoint has none for them.

    The scale is derived from the tensor the kernel will actually use. The weight loaders
    emit a geometry that differs from the parameter declaration, and
    `load_state_dict(..., assign=True)` replaces the parameter without a shape check, so a
    scale derived from the declaration would not match the loaded weight. The state dict
    is where the loaded tensor is available before device placement makes a bf16->fp8 cast
    fail ("Expected self.dtype() == dst.dtype()").

    Contracted dim is 1 for both projections:
        gate_up : [E, H, 2I] -- contract H -> scale [E, 1, 2I]
        down    : [E, I, H]  -- contract I -> scale [E, 1, H]
    Per-output-channel is required because the kernels apply the dequant elementwise on
    the matmul output (`mlp_tkg_gate_up_projection.py`); a per-input-channel scale cannot
    be undone there.
    """
    out = dict(state_dict)
    scales: dict = {}
    n = 0
    for key in list(state_dict.keys()):
        if not key.endswith(("mlp.experts.gate_up_proj_weight",
                             "mlp.experts.down_proj_weight")):
            continue
        w = state_dict[key]
        if w.dtype == FP8_TRN2_DTYPE:
            continue
        # `load_sharded_pipelined(..., device)` may have placed these on the Neuron device,
        # where a bf16->fp32 cast raises "Expected self.dtype() == dst.dtype()". Copy to
        # host in the same dtype first, then cast.
        w = w.to("cpu") if w.device.type != "cpu" else w
        wf = w.to(torch.float32)
        amax = wf.abs().amax(dim=1, keepdim=True)
        scale = pow2_scale(amax, FP8_TRN2_MAX)
        q = (wf / scale).clamp(-FP8_TRN2_MAX, FP8_TRN2_MAX).to(FP8_TRN2_DTYPE)

        # Guard: the scale must be per-output-channel and the round trip exact.
        if (scale.shape[1] != 1 or scale.shape[0] != w.shape[0]
                or scale.shape[-1] != w.shape[-1]):
            raise ValueError(
                f"{key}: scale {tuple(scale.shape)} is not per-output-channel for a weight "
                f"of shape {tuple(w.shape)} (expected [{w.shape[0]}, 1, {w.shape[-1]}])")
        deq = (q.to(torch.float32) * scale).to(w.dtype)
        if not torch.equal(deq, w):
            raise ValueError(
                f"{key}: FP8 requantization not bit-exact "
                f"({int((deq != w).sum())} elements differ). MXFP4-sourced weights must "
                f"round-trip exactly under a power-of-two scale.")

        if gate_up_major and key.endswith("mlp.experts.gate_up_proj_weight"):
            # Decode-only FP8 copy in gate/up-major layout [E, 2, H, I], so each moe_tkg
            # weight-tile DMA reads contiguous rows instead of short rows interleaved at a
            # 2*I stride. Requires an nkilib moe_tkg with SUPPORTS_GATE_UP_MAJOR, which
            # detects the layout from shape[1] == 2. Prefill uses the BF16 copy, or (without
            # the copy) this same tensor through a bwmm_shard_on_I that also reads the layout
            # (the caller only asks for it then). Stored in
            # the declared [E, H, 2I] shape; the model reshapes it to [E, 2, H, I] at the
            # kernel call.
            E_, H_, I2_ = q.shape
            q = q.reshape(E_, H_, 2, I2_ // 2).permute(0, 2, 1, 3).contiguous().reshape(E_, H_, I2_)
        out[key] = q
        scales[key.replace("_weight", "_scale")] = scale
        if keep_bf16:
            # Hybrid mode: keep the BF16 tensor for prefill (moe_cte). Prefill is
            # compute-bound and bwmm_shard_on_I upcasts each FP8 tile to bf16, which adds
            # TTFT; decode is DMA-bound and uses FP8.
            scales[key.replace("_weight", "_weight_bf16")] = w
        n += 1
    import logging
    logging.getLogger(__name__).info(
        "[gpt_oss fp8] quantized %d expert tensors to %s (clamp %.0f)",
        n, FP8_TRN2_DTYPE, FP8_TRN2_MAX)
    return out, scales


def quantize_experts_in_place(experts) -> None:
    """Quantize a GptOssExperts block's expert weights to FP8 ROW, in place, post-load.

    Weight and scale are both derived from the runtime tensor, so the scale always
    matches the geometry the kernel sees. `load_weights` uses `quantize_expert_state_dict`;
    this helper is the module-level equivalent.

    Contracted dim is 1 for both projections, matching the layouts the kernels consume:
        gate_up : [E, H, 2I] -- contract H -> scale [E, 1, 2I]
        down    : [E, I, H]  -- contract I -> scale [E, 1, H]
    A per-output-channel scale is required because the kernels apply the dequant
    elementwise on the matmul output.
    """
    for name, scale_attr in (
        ("gate_up_proj_weight", "gate_up_proj_scale"),
        ("down_proj_weight", "down_proj_scale"),
    ):
        w = getattr(experts, name)
        if w.dtype == FP8_TRN2_DTYPE:      # already quantized
            continue
        # Quantize on CPU: a bf16 -> fp8 cast on the Neuron device raises
        # "Expected self.dtype() == dst.dtype()".
        wf = w.detach().to("cpu", torch.float32)
        amax = wf.abs().amax(dim=1, keepdim=True)
        scale = pow2_scale(amax, FP8_TRN2_MAX)
        q = (wf / scale).clamp(-FP8_TRN2_MAX, FP8_TRN2_MAX).to(FP8_TRN2_DTYPE)

        # Guard: the scale must be per-output-channel and the round trip exact.
        if scale.shape[1] != 1 or scale.shape[0] != w.shape[0] or scale.shape[-1] != w.shape[-1]:
            raise ValueError(
                f"{name}: scale shape {tuple(scale.shape)} is not per-output-channel for a "
                f"weight of shape {tuple(w.shape)} (expected [{w.shape[0]}, 1, {w.shape[-1]}])"
            )
        w_cpu = w.detach().to("cpu")
        deq = (q.to(torch.float32) * scale).to(w_cpu.dtype)
        if not torch.equal(deq, w_cpu):
            n_bad = int((deq != w_cpu).sum())
            raise ValueError(
                f"{name}: FP8 requantization was not bit-exact ({n_bad} elements differ). "
                f"MXFP4-sourced weights must round-trip exactly under a power-of-two scale; "
                f"check the scale is pow2 and the dtype is float8_e4m3fn clamped to 240."
            )

        # Replace the BF16 weight with the FP8 one and create/overwrite the scale. The scale
        # parameter is created here, not in __init__, so its shape always matches the
        # runtime weight.
        setattr(experts, name, torch.nn.Parameter(q, requires_grad=False))
        setattr(experts, scale_attr, torch.nn.Parameter(scale, requires_grad=False))
