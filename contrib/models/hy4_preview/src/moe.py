# SPDX-License-Identifier: Apache-2.0
"""Hy4's routed + shared MoE, with fp8 experts, in torch.

256 routed experts (top-8) of ``moe_intermediate_size`` 2048 plus one shared
expert of the same width. The routed experts are 744B of the model's 770B
parameters, so they are the one thing that has to stay 8-bit to fit a
trn2.48xlarge: ``convert_mxfp8_to_neuron_fp8.py`` stores them as Trn2-range
fp8 with one fp32 scale per output row.

**Layout: pure tensor parallelism over the experts' intermediate dimension.**
Every rank holds all 256 experts and ``I / world_size`` (32 at world 64) of each
one's columns, and the combine is the same all-reduce the attention output
already pays. That is chosen for bring-up, not speed:

* it needs exactly one prefill and one decode graph -- expert parallelism makes
  ``ep_rank`` a trace constant and multiplies the graph count by ``ep_degree``;
* it does not depend on Trn2 accepting fp8 weights in ``moe_cte`` /
  ``moe_tkg`` (``NF.moe_tkg`` rejects ``float8`` outright today), so nothing
  here is a kernel guess.

The cost is that every rank evaluates all 256 experts on every token and masks
the 248 unselected ones with a zero affinity -- exact, but 32x the routed
FLOPs. At decode that is free (the step reads every local expert's weights
either way); at a 2048-token prefill it is ~0.6 TFLOP per rank per layer. The
kernel path (``moe_cte`` shard_on_block FP8 for prefill, ``moe_tkg`` for decode)
needs ``moe_intermediate_size / tp_degree`` to be a multiple of 128, i.e.
``ep_degree >= 4`` at world 64, and is the next step.

Routing is DeepSeek-V3's, *not* what the fused ``moe_block_tkg`` router does:

    s = sigmoid(x @ Wr.T)                      # fp32, over all 256
    pick top-8 of (s + e_score_correction_bias)  # bias chooses, never weighs
    w = s[picked] / sum(s[picked]) * routed_scaling_factor

The kernel's ``router_bias`` is added to the logits and so changes the weights
as well as the choice; that is why routing runs here and not in a kernel.

The routed SwiGLU is clamped (``gate <= 10``, ``-10 <= up <= 10``); the shared
expert and the dense layer-0 MLP are not.

**Trn3: native MXFP8 experts** (``neuron_quantization.format == "mxfp8_experts"``,
written by ``convert_mxfp8_to_neuron_fp8.py --keep-mx-experts``). The published
OCP e4m3fn + e8m0 expert tensors are loaded without requantisation, tiled at load
time into nkilib ``moe_tkg``'s MX layouts (``mx_experts.py``), and run through
``NF.moe_tkg``: selective over the top-k at decode, all-expert dynamic at prefill.
Same TP-over-I layout as above. Trn3 only (the MX kernels need gen4); the two
fp8-row paths above are what Trn2 runs.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from . import mx_experts as mx
from .config import HYV4Config

FP8 = torch.float8_e4m3fn

# ``config.neuron_quantization["format"]`` values (written by the converter).
FORMAT_FP8_ROW = "fp8_row_experts"
FORMAT_MXFP8 = "mxfp8_experts"
# Testing only: skip the Trn3 platform check for the MXFP8 expert format.
MX_FORCE_ENV = "HY4_MX_EXPERTS_FORCE"


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """fp8 -> uint8 view, so CPU permutes and copies never touch fp8 kernels."""
    return t.view(torch.uint8) if t.dtype == FP8 else t


def _expert_gate_up_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``[E, 2I, H]`` fp8 -> this rank's ``[E, H, 2, I/tp]`` fp8.

    Rows ``[0, I)`` of each expert are gate and ``[I, 2I)`` are up (contiguous
    halves, ``chunk(2)`` in ``HYV4Experts._apply_gate``). Stored expert-major so
    that one expert is one contiguous ``H * 2 * I_r`` read: decode gathers only
    the selected experts (see ``HYV4MoE._forward_selected``), and prefill runs
    one batched matmul over all of them.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        num_experts, two_i, hidden = source.get_shape()
        if two_i != 2 * intermediate:
            raise ValueError(
                f"gate_up_proj has {two_i} rows, expected {2 * intermediate}"
            )
        start = rank * shard
        gate = _as_bytes(source[:, start : start + shard, :])
        up = _as_bytes(
            source[:, intermediate + start : intermediate + start + shard, :]
        )
        out = torch.stack((gate, up), dim=1)  # [E, 2, I_r, H]
        return out.permute(0, 3, 1, 2).contiguous().view(FP8)

    return SafetensorsWeightLoader(transform=transform)


def _expert_gate_up_scale_loader(
    intermediate: int, shard: int
) -> SafetensorsWeightLoader:
    """``[E, 2I]`` row scales -> ``[1, E * 2 * I/tp]``, in the weight's column order."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        start = rank * shard
        gate = source[:, start : start + shard]
        up = source[:, intermediate + start : intermediate + start + shard]
        return torch.stack((gate, up), dim=1).reshape(1, -1).float()

    return SafetensorsWeightLoader(transform=transform)


def _expert_down_loader(shard: int) -> SafetensorsWeightLoader:
    """``down_proj_t [E, I, H]`` fp8 -> this rank's ``[E, I/tp, H]`` fp8.

    The converter stores the down weight transposed precisely so that this is a
    row slice: sharding HF's ``[E, H, I]`` on its last dim is either ``E * H``
    32-byte reads or every rank reading the whole 3 GiB tensor per layer.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        start = rank * shard
        return source[:, start : start + shard, :]

    return SafetensorsWeightLoader(transform=transform)


# ── Trn3: OCP MXFP8 experts, loaded as published ──────────────────────────────
#
# Each transform takes the checkpoint tensor exactly as ModelOpt wrote it (the
# converter's ``--keep-mx-experts`` copies it through), slices this rank's
# ``I_r`` intermediate columns -- the gate and up halves separately, as in
# ``_expert_gate_up_loader`` -- and tiles the slice into the ``moe_tkg`` MX
# layout (see ``mx_experts.py``). No value is changed: the fp8 bytes and e8m0
# exponents are only permuted (and packed four fp8 to a uint32 word).


def _mx_gate_up_rows(source, intermediate: int, shard: int, rank: int, what: str):
    """``[E, 2I, X]`` -> this rank's ``[E, 2 * I_r, X]``: gate rows, then up rows."""
    num_experts, two_i = source.get_shape()[:2]
    if two_i != 2 * intermediate:
        raise ValueError(f"{what} has {two_i} rows, expected {2 * intermediate}")
    start = rank * shard
    gate = _as_bytes(source[:, start : start + shard, :])
    up = _as_bytes(source[:, intermediate + start : intermediate + start + shard, :])
    return torch.cat((gate, up), dim=1)


def mx_gate_up_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``gate_up_proj [E, 2I, H]`` e4m3fn -> uint32 ``[E, 128, 2, H/512, I_r]``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        rows = _mx_gate_up_rows(source, intermediate, shard, rank, "gate_up_proj")
        return mx.tile_gate_up_weight(rows)

    return SafetensorsWeightLoader(transform=transform)


def mx_gate_up_scale_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``gate_up_proj_scale [E, 2I, H/32]`` e8m0 -> uint8 ``[E, 16, 2, H/512, I_r]``."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        rows = _mx_gate_up_rows(source, intermediate, shard, rank, "gate_up_proj_scale")
        return mx.tile_gate_up_scale(rows)

    return SafetensorsWeightLoader(transform=transform)


def mx_down_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``down_proj [E, H, I]`` e4m3fn -> uint32 ``[E, I_p, ceil(I_r/512), H]``.

    The shard is the last dim of the published layout, so each rank reads
    ``E * H`` runs of ``I_r`` bytes; the converter does not transpose it here
    (``--keep-mx-experts`` copies the tensors unchanged).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        if source.get_shape()[-1] != intermediate:
            raise ValueError(
                f"down_proj has {source.get_shape()[-1]} columns, expected {intermediate}"
            )
        start = rank * shard
        return mx.tile_down_weight(_as_bytes(source[:, :, start : start + shard]))

    return SafetensorsWeightLoader(transform=transform)


def mx_down_scale_loader(intermediate: int, shard: int) -> SafetensorsWeightLoader:
    """``down_proj_scale [E, H, I/32]`` e8m0 -> uint8 ``[E, I_p/8, ceil(I_r/512), H]``."""
    if shard % mx.MX_BLOCK:
        raise ValueError(f"per-rank I={shard} does not split the 32-wide MX blocks")

    def transform(slices: list, rank: int) -> torch.Tensor:
        (source,) = slices
        if source.get_shape()[-1] * mx.MX_BLOCK != intermediate:
            raise ValueError(
                f"down_proj_scale has {source.get_shape()[-1]} columns, expected "
                f"{intermediate // mx.MX_BLOCK}"
            )
        lo = rank * shard // mx.MX_BLOCK
        return mx.tile_down_scale(source[:, :, lo : lo + shard // mx.MX_BLOCK])

    return SafetensorsWeightLoader(transform=transform)


def require_mx_platform() -> None:
    """Raise unless this host targets Trn3 (the MX ``moe_tkg`` kernels are gen4-only).

    Evaluated once, at model construction, never inside a traced forward
    (``get_platform_target()`` builds a torchbind object; see
    ``vllm_neuron/model/gpt_oss/factory.py:100``). ``HY4_MX_EXPERTS_FORCE=1``
    skips the check -- for CPU testing only; on a Trn2 device the kernel would
    still reject the MX weights at trace time.
    """
    if os.environ.get(MX_FORCE_ENV) == "1":
        return
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        target = str(get_platform_target())
    except Exception as exc:  # no runtime / no device: report it as such
        raise RuntimeError(
            f"neuron_quantization.format={FORMAT_MXFP8!r} needs a Trn3 platform "
            f"target, and the platform could not be detected ({exc!r}). "
            f"Set {MX_FORCE_ENV}=1 only for CPU tests."
        ) from exc
    if not target.startswith("trn3"):
        raise RuntimeError(
            f"neuron_quantization.format={FORMAT_MXFP8!r} (native MXFP8 experts, "
            f"nkilib MX moe_tkg) runs on Trn3 only, but the platform target is "
            f"{target!r}. On Trn2 convert the checkpoint without --keep-mx-experts "
            f"(format {FORMAT_FP8_ROW!r})."
        )


def _moe_tkg_api():
    """Imported on first use so the Trn2 paths (and their CPU tests) need no nkilib."""
    import vllm_neuron.functional as NF
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    return NF.moe_tkg, ActFnType, ExpertAffinityScaleMode


class HYV4MLP(nn.Module):
    """Plain SwiGLU, intermediate dim TP-sharded. Dense layer 0 and the shared expert.

    Returns this rank's partial sum; the caller reduces.
    """

    def __init__(self, hidden: int, intermediate: int, world_size: int, dtype):
        super().__init__()
        if intermediate % world_size:
            raise ValueError(
                f"intermediate_size={intermediate} not divisible by {world_size}"
            )
        inter = intermediate // world_size
        self.gate_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        self.up_proj_weight = nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        for param in (self.gate_proj_weight, self.up_proj_weight):
            set_weight_loader(
                param,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=inter,
                    num_shards=world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.down_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=inter,
                num_shards=world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.down_proj_weight.dtype)
        return (
            F.silu(x @ self.gate_proj_weight) * (x @ self.up_proj_weight)
        ) @ self.down_proj_weight


def route_topk(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (``[T, K]`` expert ids, ``[T, K]`` fp32 weights). DeepSeek-V3 routing."""
    logits = hidden_states.float() @ router_weight.float().t()
    scores = torch.sigmoid(logits)
    _, idx = torch.topk(scores + correction_bias.float(), top_k, dim=-1)
    # Each slot's score by a one-hot over experts, not ``torch.gather(scores, 1,
    # idx)``: with the selective decode MoE in the graph, the gather of the
    # sigmoid scores made neuronx-cc 2.28 fail with NCC_IESL002 ("Expected to
    # find a definition before read for shared tensor logistic.N"; trn2.48xlarge,
    # 2-layer sweep: every formulation with the gather failed, every one with
    # the one-hot compiled). Same values: each one-hot row selects one score.
    experts = torch.arange(scores.shape[-1], device=idx.device, dtype=idx.dtype)
    one_hot = (idx.unsqueeze(-1) == experts).to(scores.dtype)  # [T, K, E]
    weights = (one_hot * scores.unsqueeze(1)).sum(dim=-1)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, weights * routed_scaling_factor


def route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> torch.Tensor:
    """Dense ``[T, E]`` affinities, zero outside each token's top-k.

    Returned dense rather than as (indices, weights) because the prefill expert
    compute is dense over experts; built from a one-hot instead of a scatter,
    which keeps the trace free of ``index_put``.
    """
    idx, weights = route_topk(
        hidden_states, router_weight, correction_bias, top_k,
        routed_scaling_factor, norm_topk_prob,
    )
    # A comparison, not ``F.one_hot``: the capture backend hands topk's indices
    # back as int32, which ``one_hot`` rejects.
    experts = torch.arange(router_weight.shape[0], device=idx.device, dtype=idx.dtype)
    one_hot = (idx.unsqueeze(-1) == experts).to(weights.dtype)  # [T, K, E]
    return (one_hot * weights.unsqueeze(-1)).sum(dim=1)


# Up to this many tokens the routed experts are evaluated per (token, selected
# expert) -- gathering only those experts' weights -- instead of densely over
# all of them. Measured (4 layers, 4K context, HY4_DEC_CP=1): at 1 row the
# selective path is ~2.8x faster than dense, at 4 rows dense is ~8% faster, so
# the gathered path is used for 1-2 row graphs only. ``=0`` restores dense everywhere.
SELECTIVE_MOE_MAX_TOKENS = int(os.environ.get("HY4_SELECTIVE_MOE_MAX_TOKENS", "2"))

# MXFP8 experts (Trn3): up to this many tokens ``moe_tkg`` runs selective (only
# the ``T * top_k`` routed experts' weights are read; the kernel takes T <= 128);
# above it, all-expert with dynamic control flow, which skips experts no token
# in a block was routed to. The all-expert MX kernel needs T % 32 == 0 for bf16
# input (nkilib ``all_expert_mx_utils.py:453``), so prefill is padded with zero-
# affinity rows to a multiple of 32.
MX_SELECTIVE_MAX_TOKENS = int(os.environ.get("HY4_MX_SELECTIVE_MAX_TOKENS", "16"))
MX_ALL_EXPERT_T_MULTIPLE = 32


def _mx_valid_block_size(tokens: int, block: int) -> bool:
    """nkilib ``all_expert_mx_utils._is_valid_block_size``, restated (SDK 2.33)."""
    if block <= 0 or tokens % block or block > tokens // 2:
        return False
    if block <= 32:
        return block % 4 == 0
    if block <= 128:
        return block % 32 == 0
    return block % 128 == 0


def mx_block_size(tokens: int) -> int:
    """Largest valid all-expert-dynamic block size <= 128 for ``tokens`` rows."""
    for block in (128, 64, 32, 16, 8, 4):
        if _mx_valid_block_size(tokens, block):
            return block
    raise ValueError(f"no valid moe_tkg all-expert block size for T={tokens}")


def _quant_format(config) -> str:
    return (getattr(config, "neuron_quantization", None) or {}).get(
        "format", FORMAT_FP8_ROW
    )


class HYV4MoE(nn.Module):
    """Router + 256 fp8 experts (dense over experts, masked) + shared expert.

    Takes the *normed* hidden states and returns this rank's partial sum.
    With ``neuron_quantization.format == "mxfp8_experts"`` (Trn3) the routed
    experts are the published MXFP8 tensors run by nkilib ``moe_tkg`` instead.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.world_size = get_tp_group().world_size
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.swiglu_limit = config.swiglu_limit
        intermediate = config.moe_intermediate_size
        if intermediate % self.world_size:
            raise ValueError(
                f"moe_intermediate_size={intermediate} not divisible by "
                f"tp={self.world_size}"
            )
        self.inter_per_rank = intermediate // self.world_size
        h, e, i_r = self.hidden_size, self.num_experts, self.inter_per_rank

        fmt = _quant_format(config)
        if fmt not in (FORMAT_FP8_ROW, FORMAT_MXFP8):
            raise ValueError(f"unknown neuron_quantization.format {fmt!r}")
        self.mx_experts = fmt == FORMAT_MXFP8
        if self.mx_experts:
            self._init_mx_experts(intermediate)
        else:
            self._init_fp8_row_experts(intermediate)

        # Replicated, as stored: ``[E, H]`` bf16 and ``[E]`` fp32.
        self.router_weight = nn.Parameter(torch.empty(e, h, dtype=self.dtype))
        self.e_score_correction_bias = nn.Parameter(torch.empty(e, dtype=torch.float32))

        self.shared_experts = (
            HYV4MLP(
                h, intermediate * config.n_shared_experts, self.world_size, self.dtype
            )
            if config.n_shared_experts
            else None
        )

    def _init_fp8_row_experts(self, intermediate: int) -> None:
        """Trn2: per-output-row fp8 (``fp8_row_experts``)."""
        h, e, i_r = self.hidden_size, self.num_experts, self.inter_per_rank
        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(e, h, 2, i_r, dtype=FP8), requires_grad=False
        )
        self.experts_gate_up_scale = nn.Parameter(
            torch.empty(1, e * 2 * i_r, dtype=torch.float32), requires_grad=False
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(e, i_r, h, dtype=FP8), requires_grad=False
        )
        self.experts_down_scale = nn.Parameter(
            torch.empty(e, 1, h, dtype=torch.float32), requires_grad=False
        )
        set_weight_loader(
            self.experts_gate_up_weight, _expert_gate_up_loader(intermediate, i_r)
        )
        set_weight_loader(
            self.experts_gate_up_scale, _expert_gate_up_scale_loader(intermediate, i_r)
        )
        set_weight_loader(self.experts_down_weight, _expert_down_loader(i_r))
        # Per output row, so the scale of an input-dim shard is the whole scale.
        set_weight_loader(
            self.experts_down_scale,
            SafetensorsWeightLoader(
                transform=lambda s, rank: s[0][:].unsqueeze(1).float()
            ),
        )

    def _init_mx_experts(self, intermediate: int) -> None:
        """Trn3: OCP MXFP8 in nkilib ``moe_tkg`` MX layouts (``mxfp8_experts``).

        uint32 words of four e4m3fn bytes; ``moe_tkg_wrapper`` views them as
        ``float8_e4m3fn_x4`` (``vllm_neuron/functional/moe/moe_tkg_wrapper.py:53-56``).
        """
        h, e, i_r = self.hidden_size, self.num_experts, self.inter_per_rank
        mx.check_dims(h, i_r)
        shapes = mx.expected_shapes(e, h, i_r)
        self.experts_gate_up_weight = nn.Parameter(
            torch.empty(shapes["expert_gate_up_weights"], dtype=torch.uint32),
            requires_grad=False,
        )
        self.experts_gate_up_scale = nn.Parameter(
            torch.empty(shapes["expert_gate_up_weights_scale"], dtype=torch.uint8),
            requires_grad=False,
        )
        self.experts_down_weight = nn.Parameter(
            torch.empty(shapes["expert_down_weights"], dtype=torch.uint32),
            requires_grad=False,
        )
        self.experts_down_scale = nn.Parameter(
            torch.empty(shapes["expert_down_weights_scale"], dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_loader(
            self.experts_gate_up_weight, mx_gate_up_loader(intermediate, i_r)
        )
        set_weight_loader(
            self.experts_gate_up_scale, mx_gate_up_scale_loader(intermediate, i_r)
        )
        set_weight_loader(self.experts_down_weight, mx_down_loader(intermediate, i_r))
        set_weight_loader(
            self.experts_down_scale, mx_down_scale_loader(intermediate, i_r)
        )

    def forward(self, normed: torch.Tensor) -> torch.Tensor:
        normed = normed.to(self.dtype)
        if self.mx_experts:
            out = self._forward_mx(normed)
        elif normed.shape[0] <= SELECTIVE_MOE_MAX_TOKENS:
            out = self._forward_selected(normed)
        else:
            out = self._forward_dense(normed)
        if self.shared_experts is not None:
            out = out + self.shared_experts(normed)
        return out

    def _forward_mx(self, normed: torch.Tensor) -> torch.Tensor:
        """Routed experts through nkilib MX ``moe_tkg`` (Trn3).

        Routing stays in torch (DeepSeek-V3 bias-chooses-only, see the module
        docstring); the kernel gets dense ``[T, E]`` fp32 affinities (zero off
        each token's top-k, ``routed_scaling_factor`` applied) and the ``[T, K]``
        int32 ids, and applies the affinity after the down projection
        (POST_SCALE). The activations are MX-quantised inside the kernel, in
        the shuffled H order that makes each MX block 32 consecutive natural H
        (``mx_experts.shuffle_hidden``); the output comes back shuffled.

        OCP vs Trn2 fp8: ``moe_tkg``'s ``dtype_mode`` (default NON_OCP) selects
        the fp8 tile dtype for ROW/STATIC quantisation only (nkilib
        ``moe_tkg.py:178-183``, used at ``mlp_tkg_gate_up_projection.py:351`` and
        ``mlp_tkg_down_projection.py:273``). The MX path never reads it: the
        weights arrive as ``float8_e4m3fn_x4`` (``moe_tkg_wrapper.py:53-56``) and
        the activation/intermediate tiles are allocated as ``nl.float8_e4m3fn``
        (``selective_expert_mx_impl.py:200,260``,
        ``all_expert_mx_impl.py:531,1917``), so the OCP checkpoint is decoded
        as OCP. ``NF.moe_tkg`` does not expose ``dtype_mode`` either.
        """
        moe_tkg, ActFnType, ExpertAffinityScaleMode = _moe_tkg_api()
        tokens = normed.shape[0]
        idx, weights = route_topk(
            normed,
            self.router_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )  # [T, K], [T, K] fp32
        experts = torch.arange(self.num_experts, device=idx.device, dtype=idx.dtype)
        one_hot = (idx.unsqueeze(-1) == experts).to(weights.dtype)  # [T, K, E]
        affinities = (one_hot * weights.unsqueeze(-1)).sum(dim=1)  # [T, E] fp32
        expert_index = idx.to(torch.int32)
        hidden = mx.shuffle_hidden(normed)

        limit = self.swiglu_limit if self.swiglu_limit > 0 else None
        common = dict(
            expert_gate_up_weights=self.experts_gate_up_weight,
            expert_down_weights=self.experts_down_weight,
            expert_gate_up_weights_scale=self.experts_gate_up_scale,
            expert_down_weights_scale=self.experts_down_scale,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            gate_clamp_upper_limit=limit,
            gate_clamp_lower_limit=None,
            up_clamp_upper_limit=limit,
            up_clamp_lower_limit=-limit if limit is not None else None,
        )
        if tokens <= MX_SELECTIVE_MAX_TOKENS:
            out = moe_tkg(
                hidden_input=hidden,
                expert_affinities=affinities,
                expert_index=expert_index,
                is_all_expert=False,
                **common,
            )
        else:
            padded = -(-tokens // MX_ALL_EXPERT_T_MULTIPLE) * MX_ALL_EXPERT_T_MULTIPLE
            if padded != tokens:
                # Pad rows get zero affinity, so no expert is computed for them
                # (the dynamic algorithm skips them) and their output is dropped.
                pad = padded - tokens
                hidden = F.pad(hidden, (0, 0, 0, pad))
                affinities = F.pad(affinities, (0, 0, 0, pad))
                expert_index = F.pad(expert_index, (0, 0, 0, pad))
            # Every rank holds all experts (TP over I), so the local expert
            # range starts at 0: ``rank_id`` is the EP rank, as in gpt-oss
            # ``_ep_rank_id`` with ep_degree == 1 (model_mxfp4.py:1401-1404).
            rank_id = torch.zeros((1, 1), dtype=torch.int32, device=normed.device)
            out = moe_tkg(
                hidden_input=hidden,
                expert_affinities=affinities,
                expert_index=expert_index,
                is_all_expert=True,
                rank_id=rank_id,
                is_all_expert_dynamic=True,
                block_size=mx_block_size(padded),
                **common,
            )[:tokens]
        return mx.unshuffle_hidden(out).to(self.dtype)

    def _forward_selected(self, normed: torch.Tensor) -> torch.Tensor:
        """Routed experts per (token, selected expert): reads ``T * top_k`` experts.

        The dense path reads (and converts) all 256 local experts on every step;
        at decode only ``T * 8`` of them carry a non-zero affinity. Here the
        selected experts' fp8 weights and row scales are gathered (one contiguous
        block per expert, see ``_expert_gate_up_loader``), converted to bf16, and
        multiplied per pair. Both row scales are per output channel of their own
        expert, so they are applied to the matmul outputs in fp32 -- the same
        product as dequantising the weight first, without materialising it.
        """
        tokens = normed.shape[0]
        e, h, i_r, k = self.num_experts, self.hidden_size, self.inter_per_rank, self.top_k
        idx, weights = route_topk(
            normed,
            self.router_weight,
            self.e_score_correction_bias,
            k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )  # [T, K], [T, K] fp32
        flat = idx.reshape(-1).to(torch.long)  # [T*K]
        pairs = flat.shape[0]

        w_gu = torch.index_select(self.experts_gate_up_weight, 0, flat)  # [P, H, 2, I_r]
        s_gu = torch.index_select(
            self.experts_gate_up_scale.view(e, 2 * i_r), 0, flat
        )  # [P, 2*I_r]
        x = normed.repeat_interleave(k, dim=0)  # [P, H]
        gate_up = torch.einsum(
            "ph,phc->pc", x, w_gu.to(self.dtype).view(pairs, h, 2 * i_r)
        )
        gate_up = (gate_up.reshape(pairs, 2 * i_r).float() * s_gu).view(pairs, 2, i_r)
        act = self._act(gate_up[:, 0, :], gate_up[:, 1, :])
        act = act * weights.reshape(pairs, 1)  # [P, I_r] fp32

        w_d = torch.index_select(self.experts_down_weight, 0, flat)  # [P, I_r, H]
        s_d = torch.index_select(self.experts_down_scale, 0, flat)  # [P, 1, H]
        down = torch.einsum("pi,pih->ph", act.to(self.dtype), w_d.to(self.dtype))
        down = down.float() * s_d.view(pairs, h)
        return down.view(tokens, k, h).sum(dim=1).to(self.dtype)

    def _act(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Clamped SwiGLU on fp32 ``gate``/``up`` for the selective path."""
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # SiLU spelt out: the dense path's ``F.silu`` hit NCC_IESL002 in the
        # gathered graph (as did a ``bmm`` formulation of the two matmuls).
        return gate * torch.sigmoid(gate) * up

    def _forward_dense(self, normed: torch.Tensor) -> torch.Tensor:
        tokens = normed.shape[0]
        e, i_r = self.num_experts, self.inter_per_rank

        affinities = route(
            normed,
            self.router_weight,
            self.e_score_correction_bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )  # [T, E] fp32

        # gate/up: the per-row scale is per *output* column here, so it applies
        # after the matmul and the weight only needs a dtype convert.
        w_gu = self.experts_gate_up_weight.to(self.dtype).view(e, self.hidden_size, 2 * i_r)
        gate_up = torch.einsum("th,ehc->tec", normed, w_gu)
        gate_up = (gate_up.float().reshape(tokens, -1) * self.experts_gate_up_scale).view(
            tokens, e, 2, i_r
        )
        gate, up = gate_up[:, :, 0, :], gate_up[:, :, 1, :]
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        act = F.silu(gate) * up * affinities.unsqueeze(-1)  # [T, E, I_r]

        # down: the scale is per output channel *per expert*, so it cannot be
        # pulled out of the sum over experts -- dequantise the weight instead.
        # The product is formed in fp32: a bf16 scale alone would cost 0.4% per row.
        down = (self.experts_down_weight.float() * self.experts_down_scale).to(
            self.dtype
        )
        return act.to(self.dtype).reshape(tokens, e * i_r) @ down.view(
            e * i_r, self.hidden_size
        )
