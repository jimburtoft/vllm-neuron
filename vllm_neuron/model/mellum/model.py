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
  MELLUM_MOE_PREFILL  "auto" (default): blockwise moe_cte kernel when I/rank % 128 == 0 (EP4 on one
                      device), else the dense torch MoE. "kernel" / "torch" force a path.
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


def _per_expert_gate_up_loader(intermediate: int, hidden: int, i_tp: int, num_shards: int, hidden_padded: int):
    """``[gate_0..gate_{E-1}, up_0..up_{E-1}]`` each ``[I, H]`` -> ``[E, Hp, 2 * I_tp]`` (gate slice first, H zero-padded)."""

    def transform(slices, rank):
        n = len(slices) // 2
        start = (rank % num_shards) * i_tp
        out = []
        for e in range(n):
            g, u = slices[e], slices[n + e]
            for s in (g, u):
                if tuple(s.get_shape()) != (intermediate, hidden):
                    raise ValueError(f"expert gate/up shape {list(s.get_shape())} != [{intermediate}, {hidden}]")
            w = torch.cat([g[start : start + i_tp, :], u[start : start + i_tp, :]], dim=0).t()
            if hidden_padded > hidden:
                w = torch.nn.functional.pad(w, (0, 0, 0, hidden_padded - hidden))
            out.append(w)
        return torch.stack(out, 0).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _per_expert_down_loader(intermediate: int, hidden: int, i_tp: int, num_shards: int, hidden_padded: int):
    """``[down_0..down_{E-1}]`` each ``[H, I]`` -> ``[E, I_tp, Hp]`` (H zero-padded)."""

    def transform(slices, rank):
        start = (rank % num_shards) * i_tp
        out = []
        for s in slices:
            if tuple(s.get_shape()) != (hidden, intermediate):
                raise ValueError(f"expert down shape {list(s.get_shape())} != [{hidden}, {intermediate}]")
            w = s[:, start : start + i_tp].t()
            if hidden_padded > hidden:
                w = torch.nn.functional.pad(w, (0, hidden_padded - hidden))
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
        self.block_size = 256
        # The prefill expert kernel (bwmm) tiles H by 512 (H=2304 is not a multiple). Like gpt_oss, keep the
        # experts zero-padded to Hp inside the MoE block only; the residual stream stays at H.
        # MELLUM_MOE_PAD_H=0 keeps H unpadded (then prefill must use MELLUM_MOE_PREFILL=torch).
        pad = os.environ.get("MELLUM_MOE_PAD_H", "1") == "1"
        self.hp = ((self.hidden_size + _H_ALIGN - 1) // _H_ALIGN) * _H_ALIGN if pad else self.hidden_size

        self.post_attention_layernorm = MellumRMSNorm(self.hidden_size, self.rms_norm_eps, self.dtype)
        # >>> PARALLELISM: router replicated on every rank <<<
        self.router_weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, dtype=self.dtype))
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.hp, 2 * self.i_tp, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.num_local_experts, self.i_tp, self.hp, dtype=self.dtype)
        )

        gu = _per_expert_gate_up_loader(self.intermediate_size, self.hidden_size, self.i_tp, self.tp_degree, self.hp)
        dn = _per_expert_down_loader(self.intermediate_size, self.hidden_size, self.i_tp, self.tp_degree, self.hp)
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
            return "gather" if T * self.top_k <= self.num_local_experts else "dense"
        return mode

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, hidden_states: torch.Tensor, is_decode: bool, positions: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        if is_decode:
            mode = self._decode_mode(hidden_states.shape[0])
            if mode == "kernel":
                out = self._unpad_h(self._decode_kernel(hidden_states))
            elif mode == "dense":
                out = self._decode_dense(hidden_states)
            else:
                out = self._decode_gather(hidden_states)
            if self.world_size > 1:
                out = self.tp_group.all_reduce(out)
            return out
        return self._prefill(hidden_states, positions)

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
        x = self._pad_h(x)
        idx = topi.reshape(-1)
        n = T * self.top_k
        xr = x.unsqueeze(1).expand(T, self.top_k, self.hp).reshape(n, 1, self.hp)
        ys = []
        for c0 in range(0, n, _GATHER_CHUNK):
            c1 = min(c0 + _GATHER_CHUNK, n)
            ic = idx[c0:c1]
            gu = torch.bmm(xr[c0:c1], self.gate_up_proj_weight.index_select(0, ic))
            h = torch.nn.functional.silu(gu[..., : self.i_tp]) * gu[..., self.i_tp :]
            ys.append(torch.bmm(h, self.down_proj_weight.index_select(0, ic)))
        y = (torch.cat(ys, 0) if len(ys) > 1 else ys[0]).reshape(T, self.top_k, self.hp)
        return self._unpad_h((y.float() * topv.unsqueeze(-1)).sum(1).to(self.dtype))

    def _dense(self, x, topv, topi):
        """Every local expert on every token, weighted by the (local) affinity."""
        T = x.shape[0]
        aff = torch.zeros(T, self.num_experts, dtype=torch.float32, device=x.device).scatter(1, topi, topv)
        if self.ep_degree > 1:
            lo = self._ep_rank_tensor().to(torch.long) * self.num_local_experts
            cols = torch.arange(self.num_local_experts, device=x.device, dtype=torch.long) + lo
            aff = aff.index_select(1, cols)
        gu = torch.einsum("th,ehi->eti", self._pad_h(x), self.gate_up_proj_weight)
        h = torch.nn.functional.silu(gu[..., : self.i_tp]) * gu[..., self.i_tp :]
        y = torch.einsum("eti,eih->eth", h, self.down_proj_weight)
        return self._unpad_h((y.float() * aff.t().unsqueeze(-1)).sum(0).to(self.dtype))

    def _decode_dense(self, hidden_states):
        x = self.post_attention_layernorm(hidden_states)
        topv, topi = self._route(x)
        return self._dense(x, topv, topi)

    def _prefill(self, hidden_states, positions=None):
        """SP shard -> norm + router -> all-gather -> experts -> reduce-scatter."""
        x = self.post_attention_layernorm(hidden_states)
        # The blockwise kernel gave wrong prefill output on trn2 with I/rank = 224 (TP4 without EP) under
        # segmented prefill, while the torch MoE and the kernel at I/rank = 896 (EP4) passed. Default to the
        # torch MoE when I/rank is not a multiple of 128.
        mode = os.environ.get("MELLUM_MOE_PREFILL", "auto")
        if mode == "auto":
            mode = "kernel" if self.i_tp % _DECODE_KERNEL_I_ALIGNMENT == 0 else "torch"
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
        # Padding rows (after the last real token) take no expert blocks (gpt_oss pattern). Positions are
        # increasing over the real tokens of a single-request prefill chunk and zero-padded after them.
        padding_mask = None
        if positions is not None and os.environ.get("MELLUM_MOE_PAD_MASK", "1") == "1":
            pos = positions.reshape(-1)
            last_real = torch.argmax(pos)
            padding_mask = torch.arange(pos.shape[0], device=pos.device) <= last_real
        if self.ep_degree > 1:
            ep_rank = self._ep_rank_tensor().to(torch.int32)
            local_ids = torch.arange(self.num_local_experts, device=x.device, dtype=torch.int32) + (
                ep_rank * self.num_local_experts
            )
            aff = NF.get_local_expert_affinities(aff, local_ids)
        aff_masked, tok_pos_to_id, block_to_expert, conditions = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )
        out = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=self._pad_h(x),
            expert_affinities_masked=aff_masked,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(self.num_local_experts, self.hp, 2, self.i_tp),
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=tok_pos_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        )
        out = self._unpad_h(out)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out


# =============================================================================
# Section 5: Decoder layer (pre-norm; the MoE block applies post_attention_layernorm)
# =============================================================================


class MellumDecoderLayer(nn.Module):
    def __init__(self, config: MellumConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = MellumRMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.self_attn = MellumAttention(config, layer_idx=layer_idx)
        self.mlp = MellumSparseMoeBlock(config)
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
        return hidden_states + self.mlp(hidden_states, is_decode=is_decode, positions=positions)


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
