# Copied from dstair/qwen3.5-moe-fp8-trainium2 moe_w8.py (Apache-2.0, see NOTICE.dstair); import path adapted.
"""CPU-side FP8 weight prep for the fused MoE decode kernels.

Converts the official Qwen3.5 MoE FP8 checkpoint (OCP E4M3FN bytes + one BF16
`weight_scale_inv` per 128x128 block) into the operands the kernels take: trn2
legacy-E4M3 bytes plus coalesced scale tables. Two conversions:

* ``block_pow2_coalesced`` (default): `requantize_official_fp8_pow2`. Lossless
  apart from the smallest-normal / subnormal edge. It keeps Qwen's 128x128
  scales and halves only the blocks that use E4M3FN-only codes (>240), doubling
  their scale to compensate. Pairs with
  `nki_moe_fused_w8_fp8_block_coalesced`.
* ``block_ob_coalesced`` (opt-in): `requantize_official_fp8_output_block`. One
  scale per 128 output rows, which lets the `_ob` kernel accumulate the whole
  contraction in PSUM at B=128. Costs ~4% rel-L2 per MoE layer against the
  official weights.
"""

from dataclasses import dataclass
import math
import os

import torch


BLOCK_SIZE = 128
LEGACY_E4M3_MAX = 240.0
FP8_IMPLS = ("block_pow2_coalesced", "block_ob_coalesced")


def _byte_values(raw):
    if raw.element_size() != 1:
        raise TypeError(f"FP8 storage must contain one-byte elements, got {raw.dtype}")
    return raw.contiguous().view(torch.uint8)


def decode_e4m3fn(raw, *, reject_nan=True):
    """Decode raw safetensors F8_E4M3 (OCP E4M3FN) bytes to float32.

    This is implemented from the bit layout and does not depend on a PyTorch
    float8 dtype. E4M3FN uses exponent 15 for finite values through 448; only
    exponent=15,mantissa=7 is NaN.
    """
    bits = _byte_values(raw)
    sign = (bits >> 7).to(torch.bool)
    exponent = ((bits >> 3) & 0x0F).to(torch.int32)
    mantissa = (bits & 0x07).to(torch.int32)
    is_nan = (exponent == 0x0F) & (mantissa == 0x07)
    if reject_nan and bool(is_nan.any()):
        count = int(is_nan.sum())
        raise ValueError(f"F8_E4M3 tensor contains {count} NaN encoding(s)")

    exp_f = exponent.to(torch.float32)
    mant_f = mantissa.to(torch.float32)
    subnormal = mant_f * (2.0 ** -9)
    normal = (1.0 + mant_f / 8.0) * torch.pow(2.0, exp_f - 7.0)
    value = torch.where(exponent == 0, subnormal, normal)
    value = torch.where(sign, -value, value)
    if not reject_nan:
        value = torch.where(is_nan, torch.full_like(value, float("nan")), value)
    return value


def _legacy_positive_values(device):
    codes = torch.arange(0x78, dtype=torch.uint8, device=device)
    exponent = ((codes >> 3) & 0x0F).to(torch.int32)
    mantissa = (codes & 0x07).to(torch.float32)
    subnormal = mantissa * (2.0 ** -9)
    normal = (1.0 + mantissa / 8.0) * torch.pow(
        2.0, exponent.to(torch.float32) - 7.0
    )
    return codes, torch.where(exponent == 0, subnormal, normal)


def validate_legacy_e4m3_bytes(raw):
    """Reject bytes that legacy Trn2 E4M3 interprets as Inf/NaN."""
    bits = _byte_values(raw)
    unsafe = ((bits >> 3) & 0x0F) == 0x0F
    if bool(unsafe.any()):
        count = int(unsafe.sum())
        raise ValueError(f"legacy E4M3 tensor contains {count} unsafe encoding(s)")


def encode_legacy_e4m3(values):
    """Round float values to finite Trn2 legacy E4M3 and return int8 bytes."""
    if not torch.is_floating_point(values):
        values = values.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot encode non-finite values as legacy E4M3")

    magnitude = values.abs().float().clamp_max(LEGACY_E4M3_MAX)
    codes, positive = _legacy_positive_values(magnitude.device)
    midpoints = (positive[:-1] + positive[1:]) * 0.5
    selected = torch.bucketize(magnitude.reshape(-1), midpoints, right=False)

    # bucketize chooses the lower code at a midpoint. E4M3 uses round-to-nearest
    # ties-to-even, so move to the upper code when the lower significand is odd.
    if selected.numel():
        lower = selected.clamp_max(midpoints.numel() - 1)
        tied = (
            (selected < positive.numel() - 1)
            & (magnitude.reshape(-1) == midpoints.index_select(0, lower))
        )
        lower_is_odd = (selected & 1) != 0
        selected = selected + (tied & lower_is_odd).to(selected.dtype)

    encoded = codes.index_select(0, selected).reshape(values.shape)
    sign = torch.signbit(values) & (magnitude != 0)
    encoded = encoded | (sign.to(torch.uint8) << 7)
    validate_legacy_e4m3_bytes(encoded)
    return encoded.contiguous().view(torch.int8)


def decode_legacy_e4m3(raw):
    """Decode finite legacy E4M3 bytes, rejecting its reserved exponent."""
    validate_legacy_e4m3_bytes(raw)
    bits = _byte_values(raw)
    sign = (bits >> 7).to(torch.bool)
    exponent = ((bits >> 3) & 0x0F).to(torch.int32)
    mantissa = (bits & 0x07).to(torch.float32)
    subnormal = mantissa * (2.0 ** -9)
    normal = (1.0 + mantissa / 8.0) * torch.pow(
        2.0, exponent.to(torch.float32) - 7.0
    )
    value = torch.where(exponent == 0, subnormal, normal)
    return torch.where(sign, -value, value)


def _scale_grid_shape(shape, block_size):
    rows, cols = shape[-2:]
    return (
        math.ceil(rows / block_size),
        math.ceil(cols / block_size),
    )


def expand_block_scales(scales, weight_shape, block_size=BLOCK_SIZE):
    """Expand [...,row_blocks,col_blocks] scales to a weight tensor shape."""
    expected = (*weight_shape[:-2], *_scale_grid_shape(weight_shape, block_size))
    if tuple(scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scales.shape)} does not match expected {expected}"
        )
    expanded = scales.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )
    return expanded[..., : weight_shape[-2], : weight_shape[-1]]


def dequantize_official_fp8(raw, scales, block_size=BLOCK_SIZE):
    """Reconstruct an official block-scaled E4M3FN tensor as float32."""
    if scales.dtype != torch.bfloat16:
        raise TypeError(f"official weight_scale_inv must be BF16, got {scales.dtype}")
    decoded = decode_e4m3fn(raw)
    return decoded * expand_block_scales(
        scales.float(), tuple(raw.shape), block_size
    )


@dataclass
class QuantizationStats:
    dot: float = 0.0
    source_sq: float = 0.0
    target_sq: float = 0.0
    error_sq: float = 0.0
    count: int = 0
    exact_count: int = 0
    block_count: int = 0
    shifted_block_count: int = 0
    clipped_count: int = 0

    def update(self, source, target):
        source = source.double()
        target = target.double()
        self.dot += float((source * target).sum())
        self.source_sq += float(source.square().sum())
        self.target_sq += float(target.square().sum())
        self.error_sq += float((source - target).square().sum())
        self.count += source.numel()
        self.exact_count += int((source == target).sum())

    def merge(self, other):
        self.dot += other.dot
        self.source_sq += other.source_sq
        self.target_sq += other.target_sq
        self.error_sq += other.error_sq
        self.count += other.count
        self.exact_count += other.exact_count
        self.block_count += other.block_count
        self.shifted_block_count += other.shifted_block_count
        self.clipped_count += other.clipped_count
        return self

    @property
    def cosine(self):
        denom = math.sqrt(self.source_sq * self.target_sq)
        return self.dot / denom if denom else 1.0

    @property
    def normalized_rmse(self):
        return math.sqrt(self.error_sq / max(self.source_sq, 1e-30))

    @property
    def exact_fraction(self):
        return self.exact_count / self.count if self.count else 1.0

    @property
    def shifted_block_fraction(self):
        if not self.block_count:
            return 0.0
        return self.shifted_block_count / self.block_count


def requantize_official_fp8_pow2(
    raw,
    source_scales,
    block_size=BLOCK_SIZE,
):
    """Convert E4M3FN blocks to native E4M3 with an exact scale exponent shift.

    A source block containing any E4M3FN-only finite code is divided by two
    while its external BF16 scale is doubled. This maps values 256..448 to
    native E4M3 values 128..224. All values except the smallest normal and
    subnormal codes remain exactly representable. Blocks without extended
    codes retain their bytes and scale bit-for-bit.
    """
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )

    result = torch.empty_like(raw, dtype=torch.int8)
    result_scales = torch.empty_like(source_scales)
    stats = QuantizationStats()
    leading_shape = raw.shape[:-2]
    leading_count = math.prod(leading_shape) if leading_shape else 1
    raw_flat = raw.reshape(leading_count, *raw.shape[-2:])
    source_scale_flat = source_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    result_flat = result.reshape(leading_count, *raw.shape[-2:])
    result_scale_flat = result_scales.reshape(
        leading_count, *source_scales.shape[-2:]
    )
    rows, cols = raw.shape[-2:]

    for leading in range(leading_count):
        for row_block, row_start in enumerate(range(0, rows, block_size)):
            row_end = min(row_start + block_size, rows)
            for col_block, col_start in enumerate(range(0, cols, block_size)):
                col_end = min(col_start + block_size, cols)
                source_raw = raw_flat[
                    leading, row_start:row_end, col_start:col_end
                ]
                source_normalized = decode_e4m3fn(source_raw)
                source_scale = source_scale_flat[
                    leading, row_block, col_block
                ]
                source_scale_f32 = source_scale.float()
                if not bool(torch.isfinite(source_scale_f32)):
                    raise ValueError(
                        "official weight_scale_inv contains a non-finite value"
                    )

                magnitude = _byte_values(source_raw) & 0x7F
                shifted = bool((magnitude >= 0x78).any())
                stats.block_count += 1
                if shifted:
                    stats.shifted_block_count += 1
                    shifted_values = source_normalized * 0.5
                    stats.clipped_count += int(
                        (shifted_values.abs() > LEGACY_E4M3_MAX).sum()
                    )
                    quantized = encode_legacy_e4m3(shifted_values)
                    stored_scale = (source_scale_f32 * 2.0).to(
                        torch.bfloat16
                    )
                    if (
                        not bool(torch.isfinite(stored_scale.float()))
                        or bool(
                            stored_scale.float() * 0.5
                            != source_scale_f32
                        )
                    ):
                        raise ValueError(
                            "doubling weight_scale_inv is not exactly "
                            "representable as finite BF16"
                        )
                else:
                    quantized = _byte_values(source_raw).contiguous().view(
                        torch.int8
                    )
                    stored_scale = source_scale

                validate_legacy_e4m3_bytes(quantized)
                source = source_normalized * source_scale_f32
                reconstructed = (
                    decode_legacy_e4m3(quantized) * stored_scale.float()
                )
                result_flat[
                    leading, row_start:row_end, col_start:col_end
                ] = quantized
                result_scale_flat[
                    leading, row_block, col_block
                ] = stored_scale
                stats.update(source, reconstructed)

    if stats.clipped_count:
        raise RuntimeError(
            f"power-of-two E4M3 conversion clipped {stats.clipped_count} values"
        )
    return result, result_scales, stats


def requantize_official_fp8_output_block(
    raw,
    source_scales,
    block_size=BLOCK_SIZE,
):
    """Requantize to legacy E4M3 with ONE scale per 128-output-row block.

    Unlike ``requantize_official_fp8_pow2`` (a scale per 128x128 block), the
    scale here depends only on the OUTPUT-row block and is shared across every
    input-column block. Reduction B relies on exactly this input-independence:
    because ``sum_i(partial_i * s[o]) == s[o] * sum_i partial_i``, the coalesced
    kernel's per-block scale-adds become a single post-accumulation scale, so
    the native-FP8 contraction can accumulate in PSUM. The trade is FP8
    accuracy — a whole 128-output-channel x full-input group shares one absmax
    scale instead of one per 128x128 tile.

    Returns ``(int8_bytes, bf16_scales, stats)`` with the SAME scale-grid shape
    as the source (values repeated across the input-block axis) so the existing
    ``pack_coalesced_block_scales`` / kernel scale indexing works unchanged.
    """
    if source_scales.dtype != torch.bfloat16:
        raise TypeError(
            f"official weight_scale_inv must be BF16, got {source_scales.dtype}"
        )
    expected = (*raw.shape[:-2], *_scale_grid_shape(raw.shape, block_size))
    if tuple(source_scales.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(source_scales.shape)} does not match expected {expected}"
        )

    source = dequantize_official_fp8(raw, source_scales, block_size)
    result = torch.empty_like(raw, dtype=torch.int8)
    result_scales = torch.empty_like(source_scales)
    stats = QuantizationStats()

    leading_shape = raw.shape[:-2]
    leading_count = math.prod(leading_shape) if leading_shape else 1
    rows, cols = raw.shape[-2:]
    row_blocks, col_blocks = _scale_grid_shape(raw.shape, block_size)
    source_flat = source.reshape(leading_count, rows, cols)
    result_flat = result.reshape(leading_count, rows, cols)
    scale_flat = result_scales.reshape(leading_count, row_blocks, col_blocks)

    for leading in range(leading_count):
        for row_block, row_start in enumerate(range(0, rows, block_size)):
            row_end = min(row_start + block_size, rows)
            block = source_flat[leading, row_start:row_end, :]
            scale = block.abs().amax().clamp_min(1e-12) / LEGACY_E4M3_MAX
            stored_scale = scale.to(torch.bfloat16)
            stored_f32 = stored_scale.float()
            quantized = encode_legacy_e4m3(block / stored_f32)
            reconstructed = decode_legacy_e4m3(quantized) * stored_f32
            result_flat[leading, row_start:row_end, :] = quantized
            scale_flat[leading, row_block, :] = stored_scale
            stats.block_count += col_blocks
            stats.update(block, reconstructed)

    return result, result_scales, stats


def assert_output_block_scale_grid(gate_up_scales, down_scales):
    """Fail loudly if these grids are NOT input-block-independent.

    ``nki_moe_fused_w8_fp8_block_coalesced_ob`` (Reduction B1) accumulates the whole
    contraction natively in PSUM and post-scales each output block ONCE, reading the
    scale at ``gate_scale_index = output_block`` with the input tile hard-wired to 0
    (see kernels/moe_fused_w8_35b.py's ``coarse_scale`` branch). That rewrite is only valid
    because

        sum_i (partial_i * s[o]) == s[o] * sum_i partial_i

    which requires ``s`` to be independent of the input block ``i``.

    The hazard this guards is silent, which is why it is checked rather than
    documented: ``requantize_official_fp8_output_block`` deliberately returns the SAME
    SHAPE as the per-128x128 grid (values repeated along the input-block axis) so the
    packing and kernel indexing work unchanged. So handing the B1 kernel a
    ``block_pow2_coalesced`` weight set compiles, runs, produces finite output of
    ordinary magnitude -- and is wrong, because 15 of every 16 input blocks get
    scaled by the wrong factor.

    Checked as a PROPERTY of the tensor rather than as provenance (a marker, a dtype,
    an env var). The property is the actual precondition: it cannot be forged, cannot
    be lost through save/load or ``.clone()``, and admits any producer that satisfies
    it. Both grids carry the contraction axis LAST -- gate/up ``[E,2,I/128,H/128]``
    contract over H, down ``[E,H/128,I/128]`` contracts over I -- so the requirement
    is constancy along ``dim=-1``.

    Cheap: one comparison over a grid of a few thousand elements, once at weight load.
    """
    for name, grid, contraction in (
        ("gate/up", gate_up_scales, "H (hidden)"),
        ("down", down_scales, "I (intermediate)"),
    ):
        if grid.shape[-1] == 1:
            continue                      # single input block: trivially constant
        first = grid[..., :1]
        if not torch.equal(grid, first.expand_as(grid)):
            varying = (grid != first).any(dim=-1).sum().item()
            raise ValueError(
                f"block_ob_coalesced requires input-block-independent scales, but the "
                f"{name} scale grid {tuple(grid.shape)} VARIES along its last axis "
                f"(the {contraction} contraction axis): {varying} output blocks differ "
                f"across input blocks. This is what a per-128x128 grid "
                f"(requantize_official_fp8_pow2 / block_pow2_coalesced) looks like, and "
                f"the B1 kernel would silently mis-scale every input block but the "
                f"first -- finite, plausible, wrong. Use "
                f"requantize_official_fp8_output_block for this impl, or use "
                f"fp8_impl='block_pow2_coalesced'."
            )


def pack_coalesced_block_scales(gate_up_scales, down_scales):
    """Pack block scales in projection, contraction, output-block order.

    Input grids retain the official projection orientation:
    gate/up `[E,2,I/128,H/128]` and down `[E,H/128,I/128]`.
    Each scalar is repeated over the NKI partition dimension so one expert's
    complete tables can be loaded and converted to FP32 once.
    """
    if gate_up_scales.ndim != 4 or gate_up_scales.shape[1] != 2:
        raise ValueError(
            "gate/up block scales must have shape [E,2,I/128,H/128]"
        )
    if down_scales.ndim != 3:
        raise ValueError(
            "down block scales must have shape [E,H/128,I/128]"
        )
    experts, _, intermediate_blocks, hidden_blocks = gate_up_scales.shape
    if tuple(down_scales.shape) != (
        experts,
        hidden_blocks,
        intermediate_blocks,
    ):
        raise ValueError("gate/up and down block-scale grids are inconsistent")
    if gate_up_scales.dtype != torch.bfloat16:
        raise TypeError("gate/up block scales must be BF16")
    if down_scales.dtype != torch.bfloat16:
        raise TypeError("down block scales must be BF16")

    gate_up_flat = (
        gate_up_scales.permute(0, 1, 3, 2)
        .reshape(experts, -1)
        .contiguous()
    )
    down_flat = (
        down_scales.transpose(1, 2)
        .reshape(experts, -1)
        .contiguous()
    )
    return (
        gate_up_flat.unsqueeze(1)
        .expand(experts, BLOCK_SIZE, gate_up_flat.shape[1])
        .contiguous(),
        down_flat.unsqueeze(1)
        .expand(experts, BLOCK_SIZE, down_flat.shape[1])
        .contiguous(),
    )


class OfficialFP8ExpertReader:
    """Lazy reader for only the routed experts in an official FP8 checkpoint."""

    def __init__(self, checkpoint):
        from .st_reader import build_weight_map

        self.checkpoint = checkpoint
        self.weight_map = build_weight_map(checkpoint)
        # The checkpoint also carries `mtp.layers.0.*` (the multi-token-prediction
        # head), so match the decoder stack explicitly rather than by index order.
        marker = "layers.0.input_layernorm.weight"
        samples = [
            key
            for key in self.weight_map
            if key.endswith(marker) and not key.startswith("mtp.")
            and ".mtp." not in key
        ]
        if len(samples) != 1:
            raise KeyError(
                f"expected exactly one decoder-stack {marker!r} key in the "
                f"official FP8 checkpoint, found {samples}"
            )
        self.prefix = samples[0][: -len(marker)]
        self._handles = {}

    def _get(self, key, expected_dtype):
        from .st_reader import SafeReader

        if key not in self.weight_map:
            raise KeyError(f"official FP8 checkpoint is missing {key!r}")
        filename = self.weight_map[key]
        if filename not in self._handles:
            self._handles[filename] = SafeReader(
                os.path.join(self.checkpoint, filename)
            )
        reader = self._handles[filename]
        actual_dtype = reader.get_dtype(key)
        if actual_dtype != expected_dtype:
            raise TypeError(
                f"{key} must have safetensors dtype {expected_dtype}, "
                f"found {actual_dtype}"
            )
        return reader.get_tensor(key)

    def load_layer(
        self,
        layer,
        expert_start,
        expert_end,
        fp8_impl="block_pow2_coalesced",
    ):
        """Return one layer's kernel operands for experts [expert_start, expert_end).

        ``fp8_impl`` picks the conversion (see the module docstring):
        ``block_pow2_coalesced`` (default, lossless) for
        `nki_moe_fused_w8_fp8_block_coalesced`, or ``block_ob_coalesced`` for
        `nki_moe_fused_w8_fp8_block_coalesced_ob`.

        Returns ``(operands, stats)``: a dict with `w8_gate_up` [E,H,2,I] and
        `w8_down` [E,I,H] (legacy-E4M3 bytes as int8) plus the coalesced
        `w8_gate_up_scale` / `w8_down_scale` [E,128,N] bf16 tables, and the
        aggregate `QuantizationStats` against the official dequantized weights.
        """
        if fp8_impl not in FP8_IMPLS:
            raise ValueError(f"fp8_impl must be one of {FP8_IMPLS}, got {fp8_impl!r}")
        requantize = (
            requantize_official_fp8_pow2
            if fp8_impl == "block_pow2_coalesced"
            else requantize_official_fp8_output_block
        )
        gate_up_weights = []
        gate_up_scales = []
        down_weights = []
        down_scales = []
        aggregate = QuantizationStats()
        layer_prefix = f"{self.prefix}layers.{layer}.mlp.experts."

        for expert in range(expert_start, expert_end):
            expert_prefix = f"{layer_prefix}{expert}."
            projections = []
            projection_scales = []
            for projection in ("gate_proj", "up_proj"):
                weight_key = expert_prefix + projection + ".weight"
                raw = self._get(weight_key, "F8_E4M3")
                source_scales = self._get(weight_key + "_scale_inv", "BF16")
                converted, scales, stats = requantize(raw, source_scales)
                self._check_quality(weight_key, fp8_impl, stats)
                aggregate.merge(stats)
                projections.append(converted.transpose(0, 1).contiguous())
                projection_scales.append(scales)
            gate_up_weights.append(torch.stack(projections, dim=1))
            gate_up_scales.append(torch.stack(projection_scales, dim=0))

            weight_key = expert_prefix + "down_proj.weight"
            raw = self._get(weight_key, "F8_E4M3")
            source_scales = self._get(weight_key + "_scale_inv", "BF16")
            converted, scales, stats = requantize(raw, source_scales)
            self._check_quality(weight_key, fp8_impl, stats)
            aggregate.merge(stats)
            down_weights.append(converted.transpose(0, 1).contiguous())
            down_scales.append(scales)

        gate_up_scale_grid = torch.stack(gate_up_scales, dim=0)
        down_scale_grid = torch.stack(down_scales, dim=0)
        if fp8_impl == "block_ob_coalesced":
            assert_output_block_scale_grid(gate_up_scale_grid, down_scale_grid)
        gate_up_scale_table, down_scale_table = pack_coalesced_block_scales(
            gate_up_scale_grid, down_scale_grid
        )
        converted = {
            "w8_gate_up": torch.stack(gate_up_weights, dim=0),
            "w8_gate_up_scale": gate_up_scale_table,
            "w8_down": torch.stack(down_weights, dim=0),
            "w8_down_scale": down_scale_table,
        }
        return converted, aggregate

    @staticmethod
    def _check_quality(key, fp8_impl, stats):
        # pow2 is lossless apart from the subnormal edge; the output-block
        # requant shares one absmax scale across a whole 128-row group.
        min_cos, max_nrmse = (
            (0.99999, 0.005) if fp8_impl == "block_pow2_coalesced" else (0.9995, 0.035)
        )
        if stats.cosine < min_cos or stats.normalized_rmse > max_nrmse:
            raise ValueError(
                f"{key} failed the {fp8_impl} quality gate: "
                f"cosine={stats.cosine:.7f}, "
                f"nrmse={stats.normalized_rmse:.5%}"
            )

    def close(self):
        for reader in self._handles.values():
            reader.close()
        self._handles.clear()


def build_local_affinities(top_ids, top_weights, expert_start, local_experts):
    """Sum global top-k routes into a dense local affinity matrix.

    `scatter_add_` deliberately preserves duplicate-route semantics and makes
    the result independent of top-k slot ordering.
    """
    local = top_ids.to(torch.long) - expert_start
    on_rank = (local >= 0) & (local < local_experts)
    safe = local.clamp(0, local_experts - 1)
    affinities = torch.zeros(
        top_ids.shape[0],
        local_experts,
        dtype=top_weights.dtype,
        device=top_weights.device,
    )
    affinities.scatter_add_(1, safe, top_weights * on_rank.to(top_weights.dtype))
    return affinities


def fused_moe_reference(
    hidden, gate_up, down, gate_up_scales, down_scales, affinities
):
    """CPU float32 reference for the fused kernels on the SAME kernel operands.

    Dequantizes the legacy-E4M3 bytes with the packed scale tables (either
    fp8_impl), then computes `sum_e affinity[:, e] * down_e(silu(gate_e(x)) * up_e(x))`.
    """
    experts, hidden_size, _, intermediate = gate_up.shape
    hb, ib = hidden_size // BLOCK_SIZE, intermediate // BLOCK_SIZE
    # gate/up table packs [E, 2, I/128, H/128] as projection, contraction (H), output (I).
    gu_s = gate_up_scales[:, 0, :].float().reshape(experts, 2, hb, ib)
    # down table packs [E, H/128, I/128] as contraction (I), output (H).
    dn_s = down_scales[:, 0, :].float().reshape(experts, ib, hb)

    x = hidden.float().cpu()
    out = torch.zeros(x.shape[0], hidden_size, dtype=torch.float32)
    gate_up_f = decode_legacy_e4m3(gate_up.cpu())   # [E, H, 2, I]
    down_f = decode_legacy_e4m3(down.cpu())         # [E, I, H]
    aff = affinities.float().cpu()

    def expand(s):
        return s.repeat_interleave(BLOCK_SIZE, 0).repeat_interleave(BLOCK_SIZE, 1)

    for e in range(experts):
        if not bool((aff[:, e] != 0).any()):
            continue
        gate = x @ (gate_up_f[e, :, 0, :] * expand(gu_s[e, 0]))    # [B, I]
        up = x @ (gate_up_f[e, :, 1, :] * expand(gu_s[e, 1]))
        act = torch.nn.functional.silu(gate) * up
        out += aff[:, e : e + 1] * (act @ (down_f[e] * expand(dn_s[e])))  # [B, H]
    return out

