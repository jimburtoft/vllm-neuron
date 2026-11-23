# SPDX-License-Identifier: Apache-2.0
"""
Mellum BF16 Implementation
==========================

Mellum (``MellumForCausalLM``, e.g. ``JetBrains/Mellum2.1-12B-A2.5B-Thinking``) for the Neuron
backend. Supported parallelism: TP, SP (prefill), optional EP for the experts, DP.

Architecture = Qwen3-MoE attention + MoE with a hybrid sliding-window schedule:

  * GQA 32 Q / 4 KV heads, head_dim 128, per-head QK RMSNorm BEFORE RoPE, no bias.
  * 3 sliding-window (1024) layers then 1 full-attention layer, repeated.
  * RoPE per layer type: sliding = default theta 5e5; full = YaRN (factor 16, original 8192,
    attention_factor 1.2773 multiplied into cos/sin, as in transformers).
  * Every layer is a sparse MoE: 64 experts, top-8, softmax in fp32, renormalized top-k,
    SiLU GLU, no shared expert.
  * Pre-norm residual blocks (input_layernorm -> attn, post_attention_layernorm -> MoE).
    RMSNorm is ``w * norm(x)`` (no Gemma-style ``1 + w``).

Reused from ``model/gemma3`` (same branch): the attention module (QK-norm + per-layer sliding
window in prefill and in the fused decode kernel), the two-mask decode precompute with the SWA
``swa_kv_pos_offset`` rebase, and the deferred / coalesced KV write-back. The SWA scheduler
features (``VLLM_NEURON_SWA_*``) are model independent and apply here too.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code.
  # <-- MODEL-SPECIFIC: ...    Mellum-specific.

MODEL ENV GATES (read at trace time; Dynamo folds them):
  MELLUM_MOE_DECODE   "auto" (default): fp32 router, then the routed-expert gather when
                      T * top_k <= local experts, else every local expert on every token ("dense").
                      "gather" / "dense" force one; "kernel": fused moe_block_tkg (bf16 router;
                      needs I/rank % 128 == 0).
  MELLUM_MOE_PREFILL  "kernel" (default): blockwise moe_cte kernel (MELLUM_MOE_CTE_IO picks its
                      hidden-state dtype). "torch": dense torch MoE.
  MELLUM_DECODE_ATTN  "stock" (default): fused attention_block_tkg decode kernel.
                      "torch": the plugin's torch reference of the same kernel.
"""

from __future__ import annotations

import logging
import math
import os

import nki.language as nl
import torch
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.model.gemma3.model import Gemma3Attention, Gemma3ForCausalLM
from vllm_neuron.model.gemma3.model import (
    _coalesce_kv_writes,
    _precompute_decode_attn_masks,
    _resolve_schedule_probe_layers,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_grouped_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import MellumConfig

logger = logging.getLogger(__name__)

# moe_block_tkg (decode) limits that NF.moe_block_tkg does not check itself.
_DECODE_KERNEL_MAX_TOKENS = 128
_DECODE_KERNEL_I_ALIGNMENT = 128
# Routed-expert gather: at most this many expert slices per index_select (larger gathers
# stall neuronx-cc's ModuleForkPass), and above this many slices per step the dense path
# reads no more than the gather.
_GATHER_CHUNK = 64
# The prefill expert kernel reads H in PSUM tiles of 512.
_H_ALIGN = 512


# =============================================================================
# Section 1: RMSNorm
# <-- MODEL-SPECIFIC: transformers' MellumRMSNorm normalizes in fp32, casts to the input dtype,
#     THEN multiplies by the weight: ``w * norm(x).to(dtype)``.
# =============================================================================


class MellumRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


# =============================================================================
# Section 2: RoPE per layer type (default on sliding layers, YaRN on full layers)
# Computed lazily in forward (the model is built on the meta device).
# =============================================================================


def _yarn_inv_freq(config: MellumConfig, device) -> torch.Tensor:
    """transformers ``_compute_yarn_parameters`` (inverse frequencies only)."""
    dim = config.head_dim
    base = config.full_rope_theta
    factor = config.full_rope_factor
    orig = config.full_rope_original_max_position

    def corr_dim(num_rot):
        return (dim * math.log(orig / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(config.full_rope_beta_fast), corr_dim(config.full_rope_beta_slow)
    if config.full_rope_truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low), 0, 1
    )
    extra_factor = 1 - ramp
    return inv_inter * (1 - extra_factor) + inv_extra * extra_factor


class MellumRotaryEmbedding(nn.Module):
    """Returns ``{"sliding": (cos, sin), "global": (cos, sin)}`` (keys used by the Gemma3 backbone code)."""

    def __init__(self, config: MellumConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.full_scale = float(config.full_rope_attention_factor) if config.full_rope_type == "yarn" else 1.0

    def _inv_freq(self, kind: str, device) -> torch.Tensor:
        c = self.config
        if kind == "global" and c.full_rope_type == "yarn":
            return _yarn_inv_freq(c, device)
        base = c.full_rope_theta if kind == "global" else c.sliding_rope_theta
        return 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim))

    def _cos_sin(self, kind, position_ids, device, dtype, scale):
        inv = self._inv_freq(kind, device)
        freqs = (inv[:, None].float() @ position_ids[None, :].float()).transpose(0, 1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        if scale != 1.0:
            cos, sin = cos * scale, sin * scale
        return cos.to(dtype), sin.to(dtype)

    def forward(self, position_ids, device, dtype):
        return {
            "sliding": self._cos_sin("sliding", position_ids, device, dtype, 1.0),
            "global": self._cos_sin("global", position_ids, device, dtype, self.full_scale),
        }


# =============================================================================
# Section 3: Attention (Gemma3 module with Mellum norms; head_dim 128 -> stock fused decode)
# =============================================================================


class MellumAttention(Gemma3Attention):
    """Gemma3's TP / SWA / QK-norm attention, configured by MellumConfig.

    MellumConfig.query_pre_attn_scalar == head_dim, so the softmax scale is head_dim ** -0.5.
    head_dim 128 lets the fused decode kernel fold the output projection.
    """

    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = MellumRMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = MellumRMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

    def forward_decode(self, hidden_states, positions, position_embeddings, attn_mask, attn_metadata, kv_writeback=None):
        if os.environ.get("MELLUM_DECODE_ATTN", "stock") == "torch":
            return self._forward_decode_torch_baseline(
                hidden_states, positions, position_embeddings, attn_mask, attn_metadata, kv_writeback=kv_writeback
            )
        return self._forward_decode_stock(
            hidden_states, positions, position_embeddings, attn_mask, attn_metadata, kv_writeback=kv_writeback
        )


# =============================================================================
# Section 4: Sparse MoE (64 experts, top-8, softmax + renorm, SiLU GLU, no shared expert)
# =============================================================================


def _per_expert_gate_up_loader(
    intermediate: int, hidden: int, i_tp: int, num_shards: int, hidden_padded: int, i_padded: int
):
    """``[gate_0..gate_{E-1}, up_0..up_{E-1}]`` each ``[I, H]`` -> ``[E, Hp, 2 * Ip]`` (gate slice first; H and I_tp zero-padded)."""

    def transform(slices, rank):
        n = len(slices) // 2
        start = (rank % num_shards) * i_tp
        out = []
        for e in range(n):
            g, u = slices[e], slices[n + e]
            for s in (g, u):
                if tuple(s.get_shape()) != (intermediate, hidden):
                    raise ValueError(f"expert gate/up shape {list(s.get_shape())} != [{intermediate}, {hidden}]")
            gs, us = g[start : start + i_tp, :], u[start : start + i_tp, :]
            if i_padded > i_tp:
                gs = torch.nn.functional.pad(gs, (0, 0, 0, i_padded - i_tp))
                us = torch.nn.functional.pad(us, (0, 0, 0, i_padded - i_tp))
            w = torch.cat([gs, us], dim=0).t()
            if hidden_padded > hidden:
                w = torch.nn.functional.pad(w, (0, 0, 0, hidden_padded - hidden))
            out.append(w)
        return torch.stack(out, 0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _per_expert_down_loader(
    intermediate: int, hidden: int, i_tp: int, num_shards: int, hidden_padded: int, i_padded: int
):
    """``[down_0..down_{E-1}]`` each ``[H, I]`` -> ``[E, Ip, Hp]`` (H and I_tp zero-padded)."""

    def transform(slices, rank):
        start = (rank % num_shards) * i_tp
        out = []
        for s in slices:
            if tuple(s.get_shape()) != (hidden, intermediate):
                raise ValueError(f"expert down shape {list(s.get_shape())} != [{hidden}, {intermediate}]")
            w = s[:, start : start + i_tp].t()
            if hidden_padded > hidden or i_padded > i_tp:
                w = torch.nn.functional.pad(w, (0, hidden_padded - hidden, 0, i_padded - i_tp))
            out.append(w)
        return torch.stack(out, 0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


class MellumSparseMoeBlock(nn.Module):
    """Router + routed experts. Owns post_attention_layernorm (the decode kernel fuses it).

    >>> PARALLELISM <<<
    - No EP: every rank holds all experts; TP shards the expert intermediate size.
    - EP (``--enable-expert-parallel``): EP rank k holds experts [k*E_L, (k+1)*E_L); the EP-TP
      sub-group shards the intermediate size.
    - Prefill is sequence parallel: all-gather tokens -> experts -> reduce-scatter.
    - Decode: every rank has all tokens -> partial outputs -> all-reduce.
    """

    def __init__(self, config: MellumConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        from vllm.config import get_current_vllm_config

        pc = get_current_vllm_config().parallel_config
        ep_enabled = pc.enable_expert_parallel
        if ep_enabled and pc.data_parallel_size > 1:
            raise NotImplementedError("Mellum: expert parallelism across data-parallel replicas is not implemented")
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
            get_neuron_ep_tp_group,
        )

        self.ep_degree = get_neuron_ep_degree() if ep_enabled else 1
        if self.ep_degree > 1:
            self.ep_rank = get_neuron_ep_rank()
            self.ep_tp_group = get_neuron_ep_tp_group()
        else:
            self.ep_rank = 0
            self.ep_tp_group = self.tp_group
        self.tp_degree = self.ep_tp_group.world_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.rms_norm_eps = config.rms_norm_eps
        self.dtype = config.torch_dtype
        if self.num_experts % self.ep_degree:
            raise ValueError(f"num_experts {self.num_experts} not divisible by EP degree {self.ep_degree}")
        if self.intermediate_size % self.tp_degree:
            raise ValueError(f"moe_intermediate_size {self.intermediate_size} not divisible by MoE TP {self.tp_degree}")
        if not self.norm_topk_prob:
            raise NotImplementedError("Mellum on Neuron implements norm_topk_prob=True routing only")
        self.num_local_experts = self.num_experts // self.ep_degree
        self.i_tp = self.intermediate_size // self.tp_degree
        # MELLUM_MOE_PREFILL_BLOCK: token block size of the blockwise prefill MoE kernels (multiple of 256).
        self.block_size = int(os.environ.get("MELLUM_MOE_PREFILL_BLOCK", "256"))
        # The prefill expert kernel (bwmm) tiles H by 512 (H=2304 is not a multiple). Like gpt_oss, keep the
        # experts zero-padded to Hp inside the MoE block only; the residual stream stays at H.
        # MELLUM_MOE_PAD_H=0 keeps H unpadded (then prefill must use MELLUM_MOE_PREFILL=torch).
        pad = os.environ.get("MELLUM_MOE_PAD_H", "1") == "1"
        self.hp = ((self.hidden_size + _H_ALIGN - 1) // _H_ALIGN) * _H_ALIGN if pad else self.hidden_size
        # MELLUM_MOE_PAD_I=1: zero-pad the expert width per rank to a multiple of 128 (exact: silu(0) * 0 = 0).
        # Off by default: it adds decode bytes and did not fix the TP-sharded prefill kernel issue below.
        self.i_real = self.i_tp
        if os.environ.get("MELLUM_MOE_PAD_I", "0") == "1" and self.i_tp % _DECODE_KERNEL_I_ALIGNMENT:
            self.i_tp = ((self.i_tp + _DECODE_KERNEL_I_ALIGNMENT - 1) // _DECODE_KERNEL_I_ALIGNMENT) * _DECODE_KERNEL_I_ALIGNMENT

        self.post_attention_layernorm = MellumRMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)
        # MELLUM_MOE_FP8=1: FP8 ROW copy of the experts for decode at large batch (see _decode_fp8).
        self.fp8_decode = os.environ.get("MELLUM_MOE_FP8", "0") == "1"
        # MELLUM_MOE_FP8_ROUTED=1 (no EP only): FP8 128x128-block-scale copy of the experts for the routed-expert
        # decode kernel (swiglu_moe_fp8.py) at small decode batches (T <= MELLUM_MOE_FP8_ROUTED_MAX_T, default 32).
        self.swm_enabled = os.environ.get("MELLUM_MOE_FP8_ROUTED", "0") == "1"
        # Decision is on the padded decode batch (num_seqs bucket 1/2/4/../32/64). Measured (no EP, 128/1k): bucket 32
        # kernel 1,095 tok/s vs dense 665; bucket 64 kernel 1,196 vs dense 2,670 -> default 32.
        self.swm_max_t = int(os.environ.get("MELLUM_MOE_FP8_ROUTED_MAX_T", "32"))
        self.swm_ep_kl = int(os.environ.get("MELLUM_MOE_EP_KL", "6"))
        # MELLUM_MOE_EM=1: expert-major NKI decode kernel (moe_decode_kernel.py): each local expert loaded once per step,
        # all decode rows (<= 128) applied; FP8 weights when MELLUM_MOE_EM_FP8=1 (default), else bf16.
        self.em_enabled = os.environ.get("MELLUM_MOE_EM", "0") == "1"
        self.em_fp8 = os.environ.get("MELLUM_MOE_EM_FP8", "1") == "1"
        # T > 128: the kernel is called once per 128-row chunk (weights re-streamed per chunk; the kernel is
        # tensor-engine bound with DMA ~9 % busy, so the re-read is cheap). MELLUM_MOE_EM_MAX_T caps its use.
        # Measured (trn2, KV cap 0.42, mns 512): through the 256 bucket the chunked kernel wins (c256 +14 %); at the
        # 512 bucket the 4 kernel copies' scratch plus KV 0.42 ran out of HBM (nrt_tensor_allocate status=4) -> 256.
        self.em_max_t = int(os.environ.get("MELLUM_MOE_EM_MAX_T", "256"))
        # MELLUM_MOE_FP8_ONLY=1 (with MELLUM_MOE_EM=1, EP only): keep a single FP8 copy of the experts (padded Hp layout
        # with per-column gate|up and per-H-column down scales). Prefill uses it through nkilib bwmm_shard_on_I(_hybrid)
        # FP8 (stock 2.33: fails at I/core 448; branch jimburtoft/nki-library@mellum-trn2-2.33: direct FP8 matmuls);
        # decode uses the expert-major kernel on the same
        # buffers. The bf16 experts are freed (-1 copy, ~3 GB/core).
        self.fp8_only = self.em_enabled and self.em_fp8 and os.environ.get("MELLUM_MOE_FP8_ONLY", "0") == "1"
        # >>> PARALLELISM: router replicated on every rank <<<
        self.router_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype))
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.hp, 2 * self.i_tp, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.i_tp, self.hp, dtype=self.dtype)
        )

        gu = _per_expert_gate_up_loader(
            self.intermediate_size, self.hidden_size, self.i_real, self.tp_degree, self.hp, self.i_tp
        )
        dn = _per_expert_down_loader(
            self.intermediate_size, self.hidden_size, self.i_real, self.tp_degree, self.hp, self.i_tp
        )
        if self.ep_degree > 1:
            local = list(range(self.ep_rank * self.num_local_experts, (self.ep_rank + 1) * self.num_local_experts))
            gu = expert_parallel_grouped_loader(local, gu, self.num_experts)
            dn = expert_parallel_grouped_loader(local, dn, self.num_experts)
        set_weight_loader(self.gate_up_proj_weight, gu)
        set_weight_loader(self.down_proj_weight, dn)

    # ── helpers ──────────────────────────────────────────────────────────

    def _ep_rank_tensor(self):
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

        return get_neuron_ep_group().rank_in_group_tensor

    def _decode_weights(self, path: str):
        """Expert weights for the torch decode paths, without the zero H padding when that is faster.

        MELLUM_MOE_DECODE_HSLICE: "auto" (default) slices H for the dense path only; "1" / "0" force it.
        Measured on trn2 (EP4, TP4): dense at batch 64 2,728 tok/s sliced vs 1,980 padded; the gather at
        batch 1 36 tok/s sliced (index_select over a non-contiguous view) vs 103 padded.
        """
        hs = os.environ.get("MELLUM_MOE_DECODE_HSLICE", "auto")
        use = (path == "dense") if hs == "auto" else hs == "1"
        if self.hp > self.hidden_size and use:
            return self.gate_up_proj_weight[:, : self.hidden_size, :], self.down_proj_weight[..., : self.hidden_size], False
        return self.gate_up_proj_weight, self.down_proj_weight, True

    def _pad_h(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (0, self.hp - self.hidden_size)) if self.hp > self.hidden_size else x

    def _unpad_h(self, y: torch.Tensor) -> torch.Tensor:
        return y[..., : self.hidden_size] if self.hp > self.hidden_size else y

    def _route(self, x: torch.Tensor):
        """fp32 logits -> softmax -> top-k -> renorm. Returns (weights fp32 [T,k], ids [T,k])."""
        probs = torch.softmax(x.float() @ self.router_weight.float().T, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topv = topv / topv.sum(-1, keepdim=True)
        return topv, topi

    def _decode_mode(self, T: int) -> str:
        # Default "gather": fp32 router + routed experts only. The fused kernel ("kernel") computes the router
        # logits in bf16 (an fp32 router input fails its dtype check) and failed the fp32 first-divergence
        # gate on trn2 (gap 0.40 / 1.05 vs <= 0.23 for gather at the same EP4 layout); it also needs
        # I/rank % 128 == 0 (EP4 on one device).
        # "auto": the gather reads T * top_k expert slices, the dense path reads each local expert once.
        mode = os.environ.get("MELLUM_MOE_DECODE", "auto")
        if mode == "auto":
            if self.swm_enabled and T <= self.swm_max_t:
                return "routed_fp8"
            if self.fp8_only:
                return "expert_major"  # the bf16 experts are gone: every decode batch uses the FP8 kernel
            if self.em_enabled and T <= self.em_max_t and T * self.top_k > self.num_local_experts:
                return "expert_major"
            if self.fp8_decode and T * self.top_k > self.num_local_experts:
                return "fp8"
            return "gather" if T * self.top_k <= self.num_local_experts else "dense"
        return mode

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_decode: bool,
        positions: torch.Tensor | None = None,
        segmented: bool = False,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        if is_decode:
            mode = self._decode_mode(hidden_states.shape[0])
            if mode == "kernel":
                out = self._unpad_h(self._decode_kernel(hidden_states))
            elif mode == "fp8":
                out = self._decode_fp8(hidden_states)
            elif mode == "routed_fp8":
                out = self._decode_routed_fp8(hidden_states)
            elif mode == "expert_major":
                out = self._decode_expert_major(hidden_states)
            elif mode == "dense":
                out = self._decode_dense(hidden_states)
            else:
                out = self._decode_gather(hidden_states)
            if self.world_size > 1:
                out = self.tp_group.all_reduce(out)
            return out
        return self._prefill(hidden_states, positions, segmented)

    def _decode_kernel(self, hidden_states):
        """Fused RMSNorm + router + all local experts (nkilib moe_block_tkg)."""
        if can_run_kernel(hidden_states) and self.i_tp % _DECODE_KERNEL_I_ALIGNMENT:
            raise ValueError(
                f"moe_block_tkg needs the per-rank expert intermediate size ({self.i_tp}) to be a multiple of "
                f"{_DECODE_KERNEL_I_ALIGNMENT}. Enable expert parallelism or use MELLUM_MOE_DECODE=gather."
            )
        rank_id = (
            self._ep_rank_tensor().reshape(1, 1).to(torch.int32)
            if self.ep_degree > 1
            else torch.zeros((1, 1), dtype=torch.int32, device=hidden_states.device)
        )
        return NF.moe_block_tkg(
            inp=self._pad_h(hidden_states).unsqueeze(0),
            gamma=self._pad_h(self.post_attention_layernorm.weight).unsqueeze(0).to(torch.float32),
            router_weights=self._pad_h(self.router_weight).T,
            expert_gate_up_weights=self.gate_up_proj_weight.reshape(self.num_local_experts, self.hp, 2, self.i_tp),
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.top_k,
            eps=self.rms_norm_eps,
            # <-- MODEL-SPECIFIC: give every routing argument (the wrapper defaults are SIGMOID / no renorm).
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,  # softmax before top-k (not the RMSNorm)
            norm_topk_prob=self.norm_topk_prob,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
            hidden_actual=self.hidden_size,  # RMSNorm over the real H, not the padded Hp
            is_all_expert=True,
            skip_router_logits=True,
        )

    def _decode_fp8(self, hidden_states):
        """fp32 torch router + nkilib moe_tkg (all local experts) on FP8 ROW expert weights.

        MELLUM_MOE_FP8=1 keeps an FP8 copy of the experts for decode (per-output-channel power-of-two scales,
        TRN2 float8_e4m3 range 240); prefill keeps the bf16 experts. Routing stays fp32 (the fused
        moe_block_tkg router runs in bf16 and failed the accuracy gate).
        Measured on trn2 (TP4 EP4, 2048 decode bucket, 128/1k): slower than the bf16 dense einsum at every batch
        that uses it (c64 3,270 -> 1,956 tok/s, c256 3,902 -> 2,078). Off by default; kept for further kernel work.
        """
        x = self.post_attention_layernorm(hidden_states)
        topv, topi = self._route(x)
        T = x.shape[0]
        aff = torch.zeros(T, self.num_experts, dtype=torch.float32, device=x.device).scatter(1, topi, topv)
        rank_id = (
            self._ep_rank_tensor().reshape(1, 1).to(torch.int32)
            if self.ep_degree > 1
            else torch.zeros((1, 1), dtype=torch.int32, device=x.device)
        )
        out = NF.moe_tkg(
            hidden_input=self._pad_h(x),
            expert_gate_up_weights=self.gu_fp8.reshape(self.num_local_experts, self.hp, 2, self.i_tp),
            expert_down_weights=self.dn_fp8,
            expert_affinities=aff.to(self.dtype),
            expert_index=topi.to(torch.int32),
            is_all_expert=True,
            rank_id=rank_id,
            expert_gate_up_weights_scale=self.gu_fp8_scale,
            expert_down_weights_scale=self.dn_fp8_scale,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=nl.bfloat16,
        )
        return self._unpad_h(out)

    def quantize_fp8_decode_copy(self, device=None):
        """Build the FP8 decode copy of the experts (called after the bf16 weights are loaded).

        Quantized on the host (a bf16 -> fp8 cast on the Neuron device fails), then moved to the device.
        """
        dev = device if device is not None else self.gate_up_proj_weight.device

        def q(w, red_dim):
            wf = w.detach().cpu().to(torch.float32)
            amax = wf.abs().amax(dim=red_dim, keepdim=True).clamp(min=1e-12)
            scale = torch.exp2(torch.ceil(torch.log2(amax / 240.0)))
            qw = (wf / scale).clamp(-240.0, 240.0).to(torch.float8_e4m3fn)
            return qw.to(dev), scale.squeeze(red_dim).to(dev)
        # gate_up [E, Hp, 2*I] -> per output channel (reduce over H): scale [E, 2*I] -> [E, 2, I]
        gu_q, gu_s = q(self.gate_up_proj_weight, 1)
        # down [E, I, Hp] -> per output channel (reduce over I): scale [E, Hp]
        dn_q, dn_s = q(self.down_proj_weight, 1)
        self.gu_fp8 = nn.Parameter(gu_q, requires_grad=False)
        self.dn_fp8 = nn.Parameter(dn_q, requires_grad=False)
        self.gu_fp8_scale = nn.Parameter(gu_s.reshape(self.num_local_experts, 2, self.i_tp).contiguous(), requires_grad=False)
        self.dn_fp8_scale = nn.Parameter(dn_s.contiguous(), requires_grad=False)

    def _decode_routed_fp8(self, hidden_states):
        """fp32 torch router + the fused FP8 routed-expert SwiGLU decode kernel (swiglu_moe_fp8.py).

        Streams only the T * top_k routed (expert, 64-row block) units, FP8 with 128x128 block scales, at
        520-630 GB/s on trn2; it has no expert dedupe, so it is used only for small decode batches. No EP: every
        rank holds all 64 experts with I sharded 4 ways (224, zero-padded to 256 in the FP8 copy: exact).
        """
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from . import swiglu_moe_fp8 as swm

        x = self.post_attention_layernorm(hidden_states)
        topv, topi = self._route(x)
        T = x.shape[0]
        if self.ep_degree > 1:
            # Measured on trn2 (EP4, 2048 bucket, 128/1k): +36 % at c1 (TPOT 9.6 -> 7.0 ms), +9 % at c4, but slower
            # than bf16 dense at c8 / c16 (-31 % / -58 %), so with EP keep MELLUM_MOE_FP8_ROUTED_MAX_T <= 4.
            # EP: keep each token's routes that land on this rank, compacted to a fixed width KL (MELLUM_MOE_EP_KL,
            # default 6). Local-route counts per token are ~Binomial(8, 1/4): P(> 6) ~ 4e-4 per token-rank. Routes
            # beyond KL are dropped (weight lost); slots without a local route point at local expert 0 with weight 0
            # (their units are still streamed).
            KL = self.swm_ep_kl
            lo = self._ep_rank_tensor().to(topi.dtype) * self.num_local_experts
            local = topi - lo
            mine = (local >= 0) & (local < self.num_local_experts)
            # stable compaction: order key = (not mine, position) -> mine first
            pos = torch.arange(self.top_k, device=topi.device, dtype=topi.dtype).unsqueeze(0).expand_as(topi)
            key = (~mine).to(topi.dtype) * self.top_k + pos
            order = torch.argsort(key, dim=-1)[:, :KL]
            m_sel = torch.gather(mine, 1, order)
            topi = torch.where(m_sel, torch.gather(local, 1, order), torch.zeros_like(order))
            topv = torch.where(m_sel, torch.gather(topv, 1, order), torch.zeros_like(topv[:, :KL]))
        K_eff = topi.shape[1]
        lnc = get_logical_nc_config()
        k = wrap_nki(swm.swiglu_moe_fp8)
        k = k[lnc] if lnc > 1 else k
        out = k(
            x.to(torch.bfloat16).contiguous(),
            topi.to(torch.int32).contiguous(),
            topv.to(torch.bfloat16).contiguous(),
            self.swm_blob,
            NB=self.swm_nb,
            group=swm.auto_group(T, K_eff, self.swm_nb),
            ring=swm.RING,
            act=0,
            limit=0.0,
            split=1,
        )
        out = out if isinstance(out, torch.Tensor) else out[0]
        return out.to(self.dtype)

    def _decode_expert_major(self, hidden_states):
        """fp32 torch router -> [T, E_local] affinities -> expert-major NKI kernel (moe_decode_kernel.py)."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .moe_decode_kernel import mellum_moe_decode

        x = self.post_attention_layernorm(hidden_states)
        topv, topi = self._route(x)
        T = x.shape[0]
        aff = torch.zeros(T, self.num_experts, dtype=torch.float32, device=x.device).scatter(1, topi, topv)
        if self.ep_degree > 1:
            lo = self._ep_rank_tensor().to(torch.long) * self.num_local_experts
            cols = torch.arange(self.num_local_experts, device=x.device, dtype=torch.long) + lo
            aff = aff.index_select(1, cols)
        lnc = get_logical_nc_config()
        k = wrap_nki(mellum_moe_decode)
        k = k[lnc] if lnc > 1 else k
        # MELLUM_MOE_EM_DR: FP8 double_row (1: gate/up, 2: + down per slot, 3: + down two-pass). dr 2 / 3 measured slower than dr 1. Standalone trn2: dr 1 -14 % at R 16
        # (250.7 -> 216.3 us), -3 % at R 64, -5 % at R 128; dr 2 no faster than dr 1. Per-token e4m3 activation
        # quantization: cos 0.99929 / 0.99899 vs 0.99999 for dr 0. Default 0 (accuracy gate not run).
        kw = dict(nbuf=int(os.environ.get("MELLUM_MOE_EM_NBUF", "2")), dedupe=1, dr=int(os.environ.get("MELLUM_MOE_EM_DR", "0")),
                  pipe=int(os.environ.get("MELLUM_MOE_EM_PIPE", "0")))
        if self.em_fp8:
            kw.update(gu_scale=self.em_gu_scale, d_scale=self.em_d_scale)
        xb = x.to(torch.bfloat16).contiguous()
        if T > 128 and os.environ.get("MELLUM_MOE_EM_GATHER", "1") == "1":
            return self._decode_expert_major_gathered(xb, aff, k_lnc=lnc).to(self.dtype)
        outs = []
        for c0 in range(0, T, 128):
            c1 = min(T, c0 + 128)
            o = k(xb[c0:c1].contiguous(), aff[c0:c1].contiguous(), self.em_gu, self.em_dn, **kw)
            outs.append(o if isinstance(o, torch.Tensor) else o[0])
        out = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
        return out.to(self.dtype)

    def _decode_expert_major_gathered(self, xb, aff, k_lnc):
        """T > 128: the routed (row, local expert) pairs, sorted by expert, are cut into G groups of Cg slots; each
        group runs one kernel pass against its expert's weights (indirect DMA by group expert id). No capacity limit:
        an expert with more than Cg rows uses several groups (its weights are re-read once per extra group).

        Sizing: P = T * min(top_k, E_local) pair slots (upper bound on routed local pairs); groups are aligned to
        expert boundaries, so G = ceil(P / Cg) + E_local (+1) groups always suffice. Unused groups have weight 0.
        Above 256 rows (MELLUM_MOE_EM_XS / _YBF = auto): x stays resident in SBUF and each group selects its rows with a
        one-hot matmul (no per-group HBM gather + transpose), and the scatter-add accumulator is bf16 (half the
        read-modify-write bytes; a row gets <= top_k partials). c512 +3 %, GSM8K unchanged.
        Groups are built in torch (argsort / gather, no scatter). Default (MELLUM_MOE_EM_DYN=1, MELLUM_MOE_EM_XIN=1):
        the kernel runs S0 groups unrolled plus a runtime loop up to the used group count, gathers each group's rows
        itself and scatter-adds the results into [T, H] by DMA. A torch index_select gather and a one-hot combine
        cost more than the expert work at T 256-512. Real routing on GSM8K uses ~15-19 groups of 128 per rank.
        """
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .moe_decode_kernel import mellum_moe_decode_gathered, mellum_moe_decode_gathered_dyn

        T, H = xb.shape
        E = self.num_local_experts
        # Group size: 64 rows up to T 256, 128 above (model-path timing with real routing: T256 Cg64 933 vs Cg128
        # 1,090 us per layer; T512 Cg128 1,431 vs Cg64 1,560 us).
        _cg = os.environ.get("MELLUM_MOE_EM_GROUP", "auto")
        Cg = (64 if T <= 256 else 128) if _cg == "auto" else int(_cg)
        dyn = os.environ.get("MELLUM_MOE_EM_DYN", "1") == "1"
        S0 = int(os.environ.get("MELLUM_MOE_EM_S0", "16"))
        K = min(self.top_k, E)
        G = -(-T * K // Cg) + E + (1 if dyn else 0)
        routed = aff > 0  # [T, E]
        cnt = routed.sum(0).to(torch.int32)  # [E] rows per expert
        gpe = (cnt + Cg - 1) // Cg  # groups per expert
        gstart = torch.cumsum(gpe, 0) - gpe  # first group of each expert
        # pair order: by expert, then row (expert-major flatten of the routed mask)
        rm = routed.t().contiguous()  # [E, T]
        key = (~rm).to(torch.int32) * T + torch.arange(T, device=xb.device, dtype=torch.int32).unsqueeze(0)  # [E, T]
        rows_by_e = torch.argsort(key, dim=1)  # [E, T]: expert e's routed rows first
        # slot s of group g of expert e: row rows_by_e[e, (g - gstart[e]) * Cg + s] if that index < cnt[e]
        g_ids = torch.arange(G, device=xb.device, dtype=torch.int32)
        g_exp = (torch.searchsorted(torch.cumsum(gpe, 0), g_ids, right=True)).clamp(max=E - 1)  # [G] expert of group
        local_g = g_ids - gstart[g_exp]  # [G] group index within its expert
        sidx = local_g.unsqueeze(1) * Cg + torch.arange(Cg, device=xb.device, dtype=torch.int32).unsqueeze(0)  # [G, Cg]
        in_e = (sidx < cnt[g_exp].unsqueeze(1)) & (g_ids < gpe.sum()).unsqueeze(1)
        sidx_c = sidx.clamp(max=T - 1).to(torch.int64)
        slot_row = torch.gather(rows_by_e[g_exp.to(torch.int64)], 1, sidx_c)  # [G, Cg]
        xin = dyn and os.environ.get("MELLUM_MOE_EM_XIN", "1") == "1"
        xg = None if xin else xb.index_select(0, slot_row.reshape(-1)).reshape(G, Cg, H)
        w = aff[slot_row, g_exp.to(torch.int64).unsqueeze(1).expand(G, Cg)]  # [G, Cg]
        wslot = torch.where(in_e, w, torch.zeros_like(w))
        if self.em_fp8:
            wslot = wslot * self.em_d_scale[g_exp.to(torch.int64)]  # [G, 1]
        eidx = g_exp.to(torch.int32).reshape(1, G).contiguous()
        if xin:
            # in-kernel row gather + DMA scatter-add combine (torch index_select / one-hot combine cost more than
            # the expert work); only the used groups run
            from .moe_decode_kernel import mellum_moe_decode_gathered_dyn_x

            used = gpe.sum().to(torch.int32).clamp(min=S0)
            nrun = (S0 + (used - S0 + 1) // 2 * 2).reshape(1, 1).to(torch.int32)
            srow = torch.where(in_e, slot_row, torch.full_like(slot_row, T)).to(torch.int32).contiguous()
            k = wrap_nki(mellum_moe_decode_gathered_dyn_x)
            k = k[k_lnc] if k_lnc > 1 else k
            kw = dict(S0=S0, xt=int(os.environ.get("MELLUM_MOE_EM_XT", "0")), oh=int(os.environ.get("MELLUM_MOE_EM_OH", "0")), wq=int(os.environ.get("MELLUM_MOE_EM_WQ", "0")), xs=int(os.environ.get("MELLUM_MOE_EM_XS", "auto").replace("auto", "1" if T > 256 else "0")), abl=int(os.environ.get("MELLUM_MOE_EM_ABL", "0")), sc=int(os.environ.get("MELLUM_MOE_EM_SC", "0")), ybf=int(os.environ.get("MELLUM_MOE_EM_YBF", "auto").replace("auto", "1" if T > 256 else "0")))
            if self.em_fp8:
                kw.update(gu_scale=self.em_gu_scale)
            o = k(xb, wslot.float().contiguous(), self.em_gu, self.em_dn, eidx, nrun, slot_row.to(torch.int32).contiguous(), srow, **kw)
            o = o if isinstance(o, torch.Tensor) else o[0]
            if kw["sc"]:
                # per-slot outputs [NC, G*Cg, H]; row t's slots: inverse table built without scatter. Slot s of row t
                # (k-th routed local expert of t, in expert order) is at flat position pos[t, k]; pad -> a zero row.
                o = o.reshape(-1, G * Cg, H).sum(0)  # [G*Cg, H] (skipped groups uninitialized -> never indexed)
                o = torch.cat([o, torch.zeros(1, H, dtype=o.dtype, device=o.device)], 0)
                # rank of t among expert e's routed rows = cumsum of routed[:, e] up to t (exclusive)
                rk = torch.cumsum(routed.to(torch.int32), 0) - routed.to(torch.int32)  # [T, E]
                flat = (gstart.unsqueeze(0) + rk // Cg) * Cg + rk % Cg  # [T, E] slot position if routed
                flat = torch.where(routed, flat, torch.full_like(flat, G * Cg))
                # keep the K smallest (all routed ones; pad entries sort last)
                kk = min(self.top_k, E)
                inv = torch.topk(flat, kk, dim=1, largest=False).values  # [T, K]
                return o.index_select(0, inv.reshape(-1).to(torch.int64)).reshape(T, kk, H).float().sum(1).to(torch.bfloat16)
            return o.reshape(-1, T, H).sum(0).to(torch.bfloat16)
        if dyn:
            # run only the used groups: S0 unrolled + a runtime loop (step 2) up to the used count
            used = gpe.sum().to(torch.int32).clamp(min=S0)
            nrun = (S0 + (used - S0 + 1) // 2 * 2).reshape(1, 1).to(torch.int32)
            k = wrap_nki(mellum_moe_decode_gathered_dyn)
            k = k[k_lnc] if k_lnc > 1 else k
            kw = dict(S0=S0)
            if self.em_fp8:
                kw.update(gu_scale=self.em_gu_scale)
            o = k(xg.transpose(1, 2).contiguous(), wslot.float().contiguous(), self.em_gu, self.em_dn, eidx, nrun, **kw)
        else:
            k = wrap_nki(mellum_moe_decode_gathered)
            k = k[k_lnc] if k_lnc > 1 else k
            kw = dict(nbuf=int(os.environ.get("MELLUM_MOE_EM_NBUF", "2")), eidx=eidx)
            if self.em_fp8:
                kw.update(gu_scale=self.em_gu_scale)
            o = k(xg.transpose(1, 2).contiguous(), wslot.float().contiguous(), self.em_gu, self.em_dn, **kw)
        o = o if isinstance(o, torch.Tensor) else o[0]
        o = o.sum(0).reshape(G * Cg, H)
        # groups the dynamic kernel skipped are never written (uninitialized HBM): zero them before the combine
        o = torch.where(in_e.reshape(-1, 1), o, torch.zeros_like(o))
        sel = ((slot_row.reshape(1, -1) == torch.arange(T, device=xb.device).unsqueeze(1)) & in_e.reshape(1, -1)).to(torch.bfloat16)
        return sel @ o.to(torch.bfloat16)

    def build_expert_major_weights(self, device=None):
        """Real-H (2304) expert weights for the expert-major kernel: gate|up [E, H, 2I], down [E, I, H]; FP8 (amax/240
        per gate/up output column, per-expert scalar for down) or bf16. Host-side, once at load."""
        dev = device if device is not None else self.gate_up_proj_weight.device
        H, I = self.hidden_size, self.i_real
        if self.fp8_only:
            # single FP8 copy in the prefill kernel's padded layout: gate|up [E, Hp, 2I] (gate cols 0..I-1, up I..2I-1)
            # scale [E, 1, 2I] per output column; down [E, I, Hp], scale [E, 1, Hp] per output column. The decode
            # kernel reads the same buffers (first H rows/cols) with gate|up scale [E, 2I] and a per-expert down
            # scalar -> keep the down scale per expert (constant over H) so both kernels agree.
            gu = self.gate_up_proj_weight.detach().cpu().float()  # [E, Hp, 2 I_tp], gate | up halves
            dn = self.down_proj_weight.detach().cpu().float()  # [E, I_tp, Hp]
            E = gu.shape[0]
            # Pad I so each of the 2 physical cores gets a multiple of 128 (bwmm_shard_on_I's per-tile FP8 scale DMA
            # inside the hybrid dynamic loop fails to compile otherwise: NCC_IGCA108). Exact: zero gate/up columns
            # and zero down rows contribute 0 (scales for the pad set to 1).
            ip = ((self.i_tp + 255) // 256) * 256 if os.environ.get("MELLUM_MOE_FP8_PAD_I", "1") == "1" else self.i_tp
            if ip > self.i_tp:
                pd = ip - self.i_tp
                g_, u_ = gu[:, :, : self.i_tp], gu[:, :, self.i_tp :]
                z = torch.zeros(E, gu.shape[1], pd)
                gu = torch.cat([g_, z, u_, z], dim=2)
                dn = torch.cat([dn, torch.zeros(E, pd, dn.shape[2])], dim=1)
            self.em_ip = ip
            gs = gu.abs().amax(1) / 240.0  # [E, 2I]
            gs = torch.where(gs > 0, gs, torch.ones_like(gs))
            ds = (dn.abs().reshape(E, -1).amax(1, keepdim=True) / 240.0).clamp(min=1e-12)  # [E, 1]
            gq = (gu / gs[:, None, :]).clamp(-240, 240).to(torch.float8_e4m3fn)
            dq = (dn / ds[:, :, None]).clamp(-240, 240).to(torch.float8_e4m3fn)
            self.em_gu = nn.Parameter(gq.contiguous().to(dev), requires_grad=False)  # [E, Hp, 2I]
            self.em_dn = nn.Parameter(dq.contiguous().to(dev), requires_grad=False)  # [E, I, Hp]
            self.em_gu_scale = nn.Parameter(gs.contiguous().to(dev), requires_grad=False)  # [E, 2I]
            self.em_d_scale = nn.Parameter(ds.contiguous().to(dev), requires_grad=False)  # [E, 1]
            self.pf_gu_scale = nn.Parameter(gs.reshape(E, 1, -1).contiguous().to(dev), requires_grad=False)
            self.pf_d_scale = nn.Parameter(ds.expand(E, self.hp).reshape(E, 1, -1).contiguous().to(dev), requires_grad=False)
            # drop the bf16 experts
            self.gate_up_proj_weight = nn.Parameter(torch.empty(0, dtype=self.dtype, device=dev), requires_grad=False)
            self.down_proj_weight = nn.Parameter(torch.empty(0, dtype=self.dtype, device=dev), requires_grad=False)
            return
        gu = self.gate_up_proj_weight.detach().cpu()[:, :H, :]
        gu = torch.cat([gu[:, :, :I], gu[:, :, self.i_tp:self.i_tp + I]], dim=2).float()  # [E, H, 2I]
        dn = self.down_proj_weight.detach().cpu()[:, :I, :H].float()  # [E, I, H]
        if self.em_fp8:
            gs = (gu.abs().amax(1) / 240.0).clamp(min=1e-12)  # [E, 2I]
            ds = (dn.abs().reshape(dn.shape[0], -1).amax(1, keepdim=True) / 240.0).clamp(min=1e-12)  # [E, 1]
            gq = (gu / gs[:, None, :]).clamp(-240, 240).to(torch.float8_e4m3fn)
            dq = (dn / ds[:, :, None]).clamp(-240, 240).to(torch.float8_e4m3fn)
            self.em_gu = nn.Parameter(gq.contiguous().to(dev), requires_grad=False)
            self.em_dn = nn.Parameter(dq.contiguous().to(dev), requires_grad=False)
            self.em_gu_scale = nn.Parameter(gs.contiguous().to(dev), requires_grad=False)
            self.em_d_scale = nn.Parameter(ds.contiguous().to(dev), requires_grad=False)
        else:
            self.em_gu = nn.Parameter(gu.to(torch.bfloat16).contiguous().to(dev), requires_grad=False)
            self.em_dn = nn.Parameter(dn.to(torch.bfloat16).contiguous().to(dev), requires_grad=False)

    def build_routed_fp8_blob(self, device=None):
        """Quantize this rank's experts to FP8 e4m3 (128x128 block scales amax/240, TRN2 e4m3 max 240) and pack them
        into the kernel's unit blob. Host-side, once at load."""
        from . import swiglu_moe_fp8 as swm

        dev = device if device is not None else self.gate_up_proj_weight.device
        H, I = self.hidden_size, self.i_real
        Ip = ((I + 127) // 128) * 128
        gu = self.gate_up_proj_weight.detach().cpu().float()[:, :H, :]
        dn = self.down_proj_weight.detach().cpu().float()[:, :I, :H]
        wg = gu[:, :, :I].transpose(1, 2)
        wu = gu[:, :, self.i_tp : self.i_tp + I].transpose(1, 2)
        wd = dn.transpose(1, 2)
        if Ip > I:
            wg = torch.nn.functional.pad(wg, (0, 0, 0, Ip - I))
            wu = torch.nn.functional.pad(wu, (0, 0, 0, Ip - I))
            wd = torch.nn.functional.pad(wd, (0, Ip - I))

        def qblock(w):  # [E, R, C] fp32 -> fp8 codes [E, R, C], scales [E, R/128, C/128] (dequant = q * s)
            E, R, C = w.shape
            b = w.view(E, R // 128, 128, C // 128, 128)
            amax = b.abs().amax(dim=(2, 4)).clamp(min=1e-12)
            s = amax / 240.0  # full e4m3 range per block (a power-of-two scale only helps MX-sourced weights)
            q = (b / s[:, :, None, :, None]).clamp(-240.0, 240.0).view(E, R, C).to(torch.float8_e4m3fn)
            return q, s

        qg, sg = qblock(wg.contiguous())
        qu, su = qblock(wu.contiguous())
        qd, sd = qblock(wd.contiguous())
        blob = swm.pack(qg, qu, qd, sg, su, sd)
        self.swm_nb = Ip // swm.IB
        self.swm_blob = nn.Parameter(blob.to(dev), requires_grad=False)

    def _local_route(self, topv, topi):
        """With EP: keep only this rank's experts (others get weight 0, slot 0)."""
        if self.ep_degree == 1:
            return topv, topi
        lo = self._ep_rank_tensor().to(topi.dtype) * self.num_local_experts
        local = topi - lo
        mine = (local >= 0) & (local < self.num_local_experts)
        return torch.where(mine, topv, torch.zeros_like(topv)), torch.where(mine, local, torch.zeros_like(local))

    def _decode_gather(self, hidden_states):
        """Read only the T * top_k routed expert slices (index_select, chunked)."""
        x = self.post_attention_layernorm(hidden_states)
        T = x.shape[0]
        topv, topi = self._local_route(*self._route(x))
        w_gu, w_dn, padded = self._decode_weights("gather")
        if padded:
            x = self._pad_h(x)
        hx = x.shape[-1]
        idx = topi.reshape(-1)
        n = T * self.top_k
        xr = x.unsqueeze(1).expand(T, self.top_k, hx).reshape(n, 1, hx)
        ys = []
        for c0 in range(0, n, _GATHER_CHUNK):
            c1 = min(c0 + _GATHER_CHUNK, n)
            ic = idx[c0:c1]
            gu = torch.bmm(xr[c0:c1], w_gu.index_select(0, ic))
            h = torch.nn.functional.silu(gu[..., : self.i_tp]) * gu[..., self.i_tp :]
            ys.append(torch.bmm(h, w_dn.index_select(0, ic)))
        y = (torch.cat(ys, 0) if len(ys) > 1 else ys[0]).reshape(T, self.top_k, hx)
        out = (y.float() * topv.unsqueeze(-1)).sum(1).to(self.dtype)
        return self._unpad_h(out) if padded else out

    def _dense(self, x, topv, topi):
        """Every local expert on every token, weighted by the (local) affinity."""
        T = x.shape[0]
        aff = torch.zeros(T, self.num_experts, dtype=torch.float32, device=x.device).scatter(1, topi, topv)
        if self.ep_degree > 1:
            lo = self._ep_rank_tensor().to(torch.long) * self.num_local_experts
            cols = torch.arange(self.num_local_experts, device=x.device, dtype=torch.long) + lo
            aff = aff.index_select(1, cols)
        w_gu, w_dn, padded = self._decode_weights("dense")
        gu = torch.einsum("th,ehi->eti", self._pad_h(x) if padded else x, w_gu)
        h = torch.nn.functional.silu(gu[..., : self.i_tp]) * gu[..., self.i_tp :]
        y = torch.einsum("eti,eih->eth", h, w_dn)
        out = (y.float() * aff.t().unsqueeze(-1)).sum(0).to(self.dtype)
        return self._unpad_h(out) if padded else out

    def _decode_dense(self, hidden_states):
        x = self.post_attention_layernorm(hidden_states)
        topv, topi = self._route(x)
        return self._dense(x, topv, topi)

    def _prefill(self, hidden_states, positions=None, segmented=False):
        """SP shard -> norm + router -> all-gather -> experts -> reduce-scatter."""
        x = self.post_attention_layernorm(hidden_states)
        # MELLUM_MOE_PREFILL: "kernel" (default, nkilib blockwise moe_cte) or "torch" (dense torch MoE).
        mode = os.environ.get("MELLUM_MOE_PREFILL", "kernel")
        if mode == "diff":
            # Diagnostic: the kernel on layers MELLUM_MOE_DIFF_LAYER="lo:hi" only, the torch MoE elsewhere.
            lo, hi = (int(v) for v in os.environ.get("MELLUM_MOE_DIFF_LAYER", "-1:-1").split(":"))
            mode = "kernel" if lo <= self._layer_idx <= hi else "torch"
        if mode == "torch":
            topv, topi = self._route(x)
            if self.world_size > 1:
                x = self.tp_group.all_gather(x, dim=0)
                topv = self.tp_group.all_gather(topv, dim=0)
                topi = self.tp_group.all_gather(topi, dim=0)
            out = self._dense(x, topv, topi)
            if self.world_size > 1:
                out = self.tp_group.reduce_scatter(out, dim=0)
            return out

        if os.environ.get("MELLUM_MOE_ROUTER", "kernel") == "torch":
            topv, topi = self._route(x)
            aff = torch.zeros(x.shape[0], self.num_experts, dtype=torch.float32, device=x.device).scatter(1, topi, topv)
        else:
          aff = NF.router(
            hidden_states=x,
            router_weights=self.router_weight.T,
            top_k=self.top_k,
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
          )
        if self.world_size > 1:
            aff = self.tp_group.all_gather(aff, dim=0)
            x = self.tp_group.all_gather(x, dim=0)
        # Padding rows take no expert blocks (gpt_oss pattern). The runner pads a prefill step by repeating
        # the last real position, and with packed prefill (several prompts per step) positions restart at 0
        # for each prompt. So a token is padding when it repeats the previous token's position, unless that
        # position is 0 (a prompt boundary after a one-token prompt). A one-token final prompt leaves its
        # padding marked real, which only costs compute.
        padding_mask = None
        if positions is not None and os.environ.get("MELLUM_MOE_PAD_MASK", "1") == "1":
            padding_mask = _real_token_mask(positions.reshape(-1))
        if self.ep_degree > 1:
            ep_rank = self._ep_rank_tensor().to(torch.int32)
            local_ids = torch.arange(self.num_local_experts, device=x.device, dtype=torch.int32) + (
                ep_rank * self.num_local_experts
            )
            aff = NF.get_local_expert_affinities(aff, local_ids)
        mask_pad = os.environ.get("MELLUM_MOE_MASK_PAD_ROUTE", "auto")
        if mask_pad == "auto":
            mask_pad = "1" if os.environ.get("MELLUM_MOE_PREFILL_IMPL", "auto") in ("auto", "hybrid") and self.tp_degree == 1 else "0"
        if padding_mask is not None and mask_pad == "1":
            # Padding tokens get no routes (build_blockwise_mapping routes them before applying padding_mask).
            aff = aff * padding_mask.unsqueeze(1).to(aff.dtype)
        aff_masked, tok_pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            # MELLUM_MOE_MAP_SHARD=1: split the mapping across the MoE TP ranks. Default 0: every rank maps
            # all local experts (identical result, measured on trn2).
            tp_degree=self.tp_degree if os.environ.get("MELLUM_MOE_MAP_SHARD", "0") == "1" else 1,
            padding_mask=padding_mask,
        )
        # MELLUM_MOE_CTE_IO: hidden-state dtype in and out of the blockwise kernel. The kernel accumulates a token's
        # top-k expert contributions by read-modify-write on its output buffer, so with bf16 I/O the sum is rounded
        # to bf16 up to top_k (8) times. With the experts' I sharded across ranks (TP4 without EP) that moved a real
        # near-tie on trn2 (fp32 gap 1.74 under segmented prefill); fp32 I/O removed it. "auto" (default): fp32 when
        # the experts are I-sharded, bf16 with whole experts per rank (EP4, validated in bf16). "fp32" / "bf16" force.
        io = os.environ.get("MELLUM_MOE_CTE_IO", "auto")
        # FP8-only: fp32 I/O also with EP (bf16 I/O flipped a long-prompt near-tie, fp32 gate gap 1.74, 2 of 3 compiles;
        # fp32 I/O: PASS, gap 0.12; -1 % throughput, +5-12 % TTFT).
        cte_io_fp32 = io == "fp32" or (io == "auto" and (self.tp_degree > 1 or self.fp8_only))
        # MELLUM_MOE_PREFILL_IMPL: "auto" (default): "hybrid" with whole experts per rank (EP), "block" when the
        # experts are I-sharded (TP without EP). "block": bwmm shard_on_block; it computes every allocated block
        # (worst case: all top-k experts of a token on one rank), so at EP4 about 70 % of its blocks are empty, and
        # it spills at I = 896. "hybrid": bwmm shard_on_I with a static loop of MELLUM_MOE_PREFILL_NSB blocks, then
        # a dynamic loop over the non-empty blocks. Measured on trn2 (TP4, one request, TTFT): EP4 4k 534 -> 206 ms,
        # 1k 182 -> 80 ms; EP1 is slower with hybrid (4k 223 -> 384 ms), so EP1 keeps shard_on_block.
        impl_name = os.environ.get("MELLUM_MOE_PREFILL_IMPL", "auto")
        if impl_name == "auto":
            impl_name = "hybrid" if self.tp_degree == 1 else "block"
        if self.fp8_only and impl_name == "block":
            # bwmm_shard_on_block ignores FP8 dequant scales; FP8 prefill must use shard_on_I(_hybrid)
            impl_name = "hybrid" if self.tp_degree == 1 else "shard_on_i"
        impl = {
            "block": MoECTEImplementation.shard_on_block,
            "shard_on_i": MoECTEImplementation.shard_on_i,
            "hybrid": MoECTEImplementation.shard_on_i_hybrid,
        }[impl_name]
        extra = {}
        if impl_name == "hybrid":
            nsb = int(os.environ.get("MELLUM_MOE_PREFILL_NSB", "-1"))
            if nsb < 0:
                # expected non-empty blocks per rank, rounded down to even (the 2 cores alternate blocks)
                T = x.shape[0]  # already all-gathered (full step)
                exp_blocks = -(-T * self.top_k // (self.ep_degree * self.block_size))
                nsb = max(2, 2 * (exp_blocks // 2))
            extra["num_static_block"] = nsb
        out = NF.moe_cte(
            implementation=impl,
            **extra,
            conditions=conditions,
            hidden_states=self._pad_h(x).float() if cte_io_fp32 else self._pad_h(x),
            expert_affinities_masked=aff_masked,
            gate_up_proj_weight=(self.em_gu if self.fp8_only else self.gate_up_proj_weight).reshape(
                self.num_local_experts, self.hp, 2, self.em_ip if self.fp8_only else self.i_tp),
            down_proj_weight=self.em_dn if self.fp8_only else self.down_proj_weight,
            **({"gate_up_proj_scale": self.pf_gu_scale, "down_proj_scale": self.pf_d_scale} if self.fp8_only else {}),
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=tok_pos_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            # compute_dtype=fp32 gave up to 30 % per-token error at I=896 on trn2 (nkilib 2.33); keep bf16.
            compute_dtype=nl.bfloat16,
        )
        out = self._unpad_h(out)
        if cte_io_fp32:
            out = out.to(self.dtype)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out


def _real_token_mask(pos: torch.Tensor) -> torch.Tensor:
    """[T] bool: False for padding tokens (repeated position after the last real token of a prefill step)."""
    rep = torch.zeros_like(pos, dtype=torch.bool)
    rep[1:] = (pos[1:] == pos[:-1]) & (pos[1:] != 0)
    return ~rep


# =============================================================================
# Section 5: Decoder layer (pre-norm; the MoE block applies post_attention_layernorm)
# =============================================================================


class MellumDecoderLayer(nn.Module):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.self_attn = MellumAttention(config, layer_idx=layer_idx)
        self.mlp = MellumSparseMoeBlock(config)
        self.mlp._layer_idx = layer_idx
        self.layer_idx = layer_idx

    def forward(self, hidden_states, positions, position_embeddings, attn_mask=None, attn_metadata=None, kv_writeback=None):
        m = attn_metadata[f"layers.{self.layer_idx}.self_attn"]
        is_decode = m["max_query_len"] <= m["decode_token_threshold"]
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            positions=positions,
            position_embeddings=position_embeddings,
            attn_mask=attn_mask,
            attn_metadata=attn_metadata,
            kv_writeback=kv_writeback,
        )
        hidden_states = residual + hidden_states
        segmented = bool(m.get("kv_segment_size"))
        return hidden_states + self.mlp(hidden_states, is_decode=is_decode, positions=positions, segmented=segmented)


# =============================================================================
# Section 6: Backbone (Gemma3 backbone flow without the embedding scale)
# =============================================================================


class MellumModel(nn.Module):
    def __init__(self, config: MellumConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.kv_ptr_group_of = None
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList([MellumDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = MellumRotaryEmbedding(config)
        self.sliding_probe_layer, self.global_probe_layer = _resolve_schedule_probe_layers(config)
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(self, input_ids, positions, attn_metadata=None, rank=None, inputs_embeds=None, is_token_ids=None):
        m0 = attn_metadata["layers.0.self_attn"]
        is_prefill = m0["max_query_len"] > m0["decode_token_threshold"]

        hidden_states = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        if is_prefill and self.world_size > 1 and inputs_embeds is not None and is_token_ids is not None:
            n = hidden_states.shape[0]
            inputs_embeds = inputs_embeds[self.rank * n : (self.rank + 1) * n]
            is_token_ids = is_token_ids[self.rank * n : (self.rank + 1) * n]
        hidden_states = NF.merge_prompt_embeds(hidden_states, inputs_embeds, is_token_ids)

        rope = self.rotary_emb(positions, device=hidden_states.device, dtype=hidden_states.dtype)

        sliding_mask = global_mask = None
        if not is_prefill:
            sliding_mask, global_mask = _precompute_decode_attn_masks(
                config=self.config,
                attn_metadata=attn_metadata,
                positions=positions,
                num_q_heads_per_rank=self.layers[0].self_attn.num_attention_heads_per_rank,
                sliding_probe_layer=self.sliding_probe_layer,
                global_probe_layer=self.global_probe_layer,
            )

        # Deferred + coalesced KV write-back (see gemma3 model "X5"). Prefill defers only on the
        # single-shot path (enforced per layer in forward_prefill).
        if is_prefill:
            coalesce = os.environ.get("GEMMA3_PREFILL_KV_COALESCE", "1") == "1"
        else:
            coalesce = os.environ.get("GEMMA3_KV_COALESCE", "1") == "1"
        kv_writeback = [] if coalesce else None
        if not is_prefill:
            attn_metadata["_g3_kv_chain"] = {}

        for layer in self.layers:
            sliding = layer.self_attn.is_sliding
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=rope["sliding" if sliding else "global"],
                attn_mask=sliding_mask if sliding else global_mask,
                attn_metadata=attn_metadata,
                kv_writeback=kv_writeback,
            )

        if not is_prefill:
            attn_metadata.pop("_g3_kv_chain", None)
        if kv_writeback:
            _coalesce_kv_writes(kv_writeback, self.kv_ptr_group_of)

        hidden_states = self.norm(hidden_states)
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states, []


# =============================================================================
# Section 7: LM head + weight loading. forward / get_kv_spec / bind_kv_cache are the Gemma3
# implementations (model independent apart from the backbone class).
# =============================================================================


def _check_packed_prefill() -> None:
    """Packed prefill (VLLM_NEURON_MAX_PREFILLS_PER_BATCH > 1) needs per-request attention bounds and whole prompts.

    Turns on the per-request attention bounds of the shared Gemma3 prefill attention, and rejects the
    configurations where a prompt can be split across steps or share cached blocks (the bounds assume every
    packed prompt starts at position 0 of the step).
    """
    n = int(os.environ.get("VLLM_NEURON_MAX_PREFILLS_PER_BATCH", "1") or 1)
    if n <= 1:
        return
    from vllm.config import get_current_vllm_config

    vc = get_current_vllm_config()
    sc, mc = vc.scheduler_config, vc.model_config
    problems = []
    if sc.max_num_batched_tokens < mc.max_model_len:
        problems.append(
            f"max_num_batched_tokens ({sc.max_num_batched_tokens}) < max_model_len ({mc.max_model_len}) "
            "allows segmented prefill"
        )
    if getattr(vc.cache_config, "enable_prefix_caching", False):
        problems.append("prefix caching is enabled")
    if vc.speculative_config is not None:
        problems.append("speculative decoding is configured")
    if getattr(vc.parallel_config, "data_parallel_size", 1) > 1:
        problems.append("data parallelism is enabled")
    if problems:
        raise ValueError(
            "Mellum packed prefill (VLLM_NEURON_MAX_PREFILLS_PER_BATCH="
            f"{n}) needs whole-prompt prefill: " + "; ".join(problems)
        )
    os.environ["GEMMA3_PACKED_PREFILL"] = "1"
    logger.info("Mellum packed prefill: up to %d prompts per prefill step", n)


class MellumForCausalLM(Gemma3ForCausalLM):
    def __init__(self, config: MellumConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model = MellumModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or (
            nc is not None and nc.debug_logits_dir is not None
        )
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
        _check_packed_prefill()
        logger.info("Mellum config: %s", config.summary())

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(MellumConfig.from_configs(hf_config, neuron_config))

    def _build_weight_mappings(self, prefix: str = "model.") -> dict:
        m: dict = {
            "model.embed_tokens.weight": f"{prefix}embed_tokens.weight",
            "model.norm.weight": f"{prefix}norm.weight",
            "lm_head.weight": f"{prefix}embed_tokens.weight" if self.config.tie_word_embeddings else "lm_head.weight",
        }
        E = self.config.num_experts
        for i in range(len(self.model.layers)):
            hf, our = f"{prefix}layers.{i}", f"model.layers.{i}"
            m[f"{our}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ]
            m[f"{our}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"
            m[f"{our}.self_attn.q_norm.weight"] = f"{hf}.self_attn.q_norm.weight"
            m[f"{our}.self_attn.k_norm.weight"] = f"{hf}.self_attn.k_norm.weight"
            m[f"{our}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            m[f"{our}.mlp.post_attention_layernorm.weight"] = f"{hf}.post_attention_layernorm.weight"
            m[f"{our}.mlp.router_weight"] = f"{hf}.mlp.gate.weight"
            m[f"{our}.mlp.gate_up_proj_weight"] = [f"{hf}.mlp.experts.{e}.gate_proj.weight" for e in range(E)] + [
                f"{hf}.mlp.experts.{e}.up_proj.weight" for e in range(E)
            ]
            m[f"{our}.mlp.down_proj_weight"] = [f"{hf}.mlp.experts.{e}.down_proj.weight" for e in range(E)]
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        result = checkpoint.load_sharded_pipelined(
            self.rank, self.world_size, self, self._build_weight_mappings(), device, strict=False
        )
        missing = list(result.missing_keys or [])
        n = len(self.model.layers)

        def _ignored(k: str) -> bool:
            # MTP head (not used), and layers beyond num_hidden_layers (reduced-layer bring-up).
            if k.startswith("mtp."):
                return True
            parts = k.split(".")
            return len(parts) > 2 and parts[1] == "layers" and parts[2].isdigit() and int(parts[2]) >= n

        unexpected = [k for k in (result.unexpected_keys or []) if not _ignored(k)]
        if missing or unexpected:
            raise RuntimeError(
                f"Mellum weights do not match the checkpoint: missing params {missing[:10]} ({len(missing)}), "
                f"unused checkpoint keys {unexpected[:10]} ({len(unexpected)})"
            )
        sd = result.state_dict
        for k, t in sd.items():
            if t.dtype != self.config.torch_dtype:
                sd[k] = t.to(self.config.torch_dtype)
        self.load_state_dict(sd, strict=False, assign=True)
        for layer in self.model.layers:
            if getattr(layer.mlp, "fp8_decode", False):
                layer.mlp.quantize_fp8_decode_copy(device)
            if getattr(layer.mlp, "em_enabled", False):
                layer.mlp.build_expert_major_weights(device)
            if getattr(layer.mlp, "swm_enabled", False):
                layer.mlp.build_routed_fp8_blob(device)
