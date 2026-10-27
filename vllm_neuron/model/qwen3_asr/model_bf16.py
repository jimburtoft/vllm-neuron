# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR composed model (BF16).

Top-level ``Qwen3ASRForConditionalGeneration``: audio encoder + Qwen3 dense text
decoder + (audio-placeholder) fusion.

ASR's decoder is standard dense Qwen3 (GQA 16Q/8KV, head_dim=128, QK-norm, SiLU,
standard 1D RoPE).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ... touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Qwen3-ASR-specific. Change when porting.
"""

from __future__ import annotations

import logging
import os
import time as _time

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, EncoderNeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.parallel.neuron_parallel_state import shard_select
from vllm_neuron.functional.attention.attention_decode import (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER._swizzle_packed_k,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER._unswizzle_packed_k,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.packed_fp8_viable_for_bucket,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.dtype_utils import (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.FP8_CLAMP_MAX,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.validate_fp8_segmented_supported,
)
from vllm_neuron.utils.weight_loader import (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.fused_qkv_weight_loader,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.set_weight_loader,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.sharding_weight_loader,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.sharding_weight_loader_with_padding,
)

from ..kv_cache import build_per_head_slot_mapping, paged_slot_indices
from ..qwen3_vl.utils.decode_kv import build_per_head_block_table
from .audio_encoder_bf16 import Qwen3ASREncoder
from .config import Qwen3ASRConfig, Qwen3ASRTextConfig
from .utils.audio_preprocessing import plan_audio_blocks
from .utils.merge_audio_embeds import merge_audio_embeddings
from vllm_neuron.functional.lmhead_argmax import lmhead_argmax

logger = logging.getLogger(__name__)

# HF checkpoint prefixes. The thinker text decoder lives under `thinker.model.*`
# and its head under `thinker.lm_head.*`
HF_TEXT_PREFIX = "thinker.model"
HF_LM_HEAD_KEY = "thinker.lm_head.weight"


_FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# lm_head + greedy sampling backend. "stock" (default): lm_head matmul -> [B, V] logits -> Sampler
# (argmax). "fused": one NKI kernel streams the lm_head weight once and returns the argmax token ids
# without writing the logits. Fused is used only where it is exactly the stock math: TP1, on-device
# all-greedy sampling, no logits gather (logprobs / debug dump), no logit_mask; otherwise stock runs.
_LMHEAD_BACKENDS = ("stock", "fused")


def get_lmhead_backend() -> str:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.v = os.environ.get("VLLM_NEURON_QWEN3_ASR_LMHEAD", "stock").strip().lower()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.if v not in _LMHEAD_BACKENDS:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        f"VLLM_NEURON_QWEN3_ASR_LMHEAD={v!r}; use one of {_LMHEAD_BACKENDS}."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.return v


def _write_packed_k_pairs(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.k_cache: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.k_new: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.block_indices: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.position_indices: torch.Tensor,
) -> None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Write new K rows into the packed FP8 K cache, one token pair per row.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.The packed K cache is ``[num_blocks, kv_heads, block_size // 2, d, 2]``.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Lane ``p % 2`` of packed row ``p // 2`` holds the token at in-block
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.position ``p``. A per-token scatter into one lane does not lower on Neuron,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.so each new token writes its whole ``[d, 2]`` pair instead:

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.* its own lane holds the new token;
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.* the other lane holds the partner token (position ``p ^ 1``) when that
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.  token is written in the same call, else the stored value.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Both tokens of a pair then write identical bytes, so the duplicate indices
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.in the ``index_put_`` are harmless. vLLM gives the tokens of one sequence
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.consecutive rows and consecutive slots, so the partner of a row, if
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.present, is the row just before or after it. This needs no pair-aligned
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.row count, unlike ``scatter_packed_k``, and touches only the new tokens'
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.pairs, unlike the whole-cache un-swizzle/re-swizzle round trip.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Args:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_cache: packed K cache, fp8 ``[nb, nkh, block_size // 2, d, 2]``.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_new: new K ``[nkh, T, d]`` in the compute dtype, already rounded
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        to the cache dtype (quantized, then cast back).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_indices: ``[T]`` block of each token (null block for padding).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_indices: ``[T]`` in-block position of each token.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.nkh, T, d = k_new.shape
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.slots = block_indices * (2 * k_cache.shape[2]) + position_indices
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.lane = position_indices % 2  # [T]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Partner of row t: row t+1 when t is lane 0 and slot+1 follows, row t-1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# when t is lane 1 and slot-1 precedes. The shifted copies repeat the edge
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# row, which never matches slot +/- 1. Pad rows all map to the same null
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# slot, so they never pair either.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.nxt = torch.cat([slots[1:], slots[-1:]])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.prv = torch.cat([slots[:1], slots[:-1]])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.has_next = (lane == 0) & (nxt == slots + 1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.has_prev = (lane == 1) & (prv == slots - 1)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.k_next = torch.cat([k_new[:, 1:], k_new[:, -1:]], dim=1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.k_prev = torch.cat([k_new[:, :1], k_new[:, :-1]], dim=1)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.head_idx = torch.arange(nkh, dtype=torch.long, device=slots.device)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.head_idx = head_idx.repeat_interleave(T)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.blk = block_indices.repeat(nkh)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.outer = (position_indices // 2).repeat(nkh)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.index = (blk, head_idx, outer)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Only the gather and the scatter touch fp8 (the scatter_packed_k pattern).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# The merge runs in the compute dtype; fp8 -> bf16 -> fp8 is exact.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.stored = k_cache[index].to(k_new.dtype).view(nkh, T, d, 2)  # [nkh, T, d, 2]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.lane_b = lane.view(1, T, 1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.stored_other = torch.where(lane_b == 0, stored[..., 1], stored[..., 0])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.other = torch.where(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    has_next.view(1, T, 1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_next,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    torch.where(has_prev.view(1, T, 1), k_prev, stored_other),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.lane0 = torch.where(lane_b == 0, k_new, other)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.lane1 = torch.where(lane_b == 0, other, k_new)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.pairs = torch.stack([lane0, lane1], dim=-1).reshape(nkh * T, d, 2)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.k_cache.index_put_(index, pairs.to(k_cache.dtype).contiguous())


# =============================================================================
# Text decoder
# =============================================================================


class Qwen3ASRTextRMSNorm(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""RMS Normalization — standard, no padding."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.variance_epsilon = eps

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    input_dtype = hidden_states.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = hidden_states.to(torch.float32)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    variance = hidden_states.pow(2).mean(-1, keepdim=True)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return (self.weight * hidden_states).to(input_dtype)


class Qwen3ASRTextRotaryEmbedding(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Standard RoPE for the Qwen3-ASR decoder (no Llama3-style scaling).

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.<-- MODEL-SPECIFIC: rope_theta=1_000_000, no rope_scaling.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRTextConfig):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.rope_theta = config.rope_theta
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.head_dim = config.head_dim

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return 1.0 / (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.rope_theta
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ** (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            / self.head_dim
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> tuple[torch.Tensor, torch.Tensor]:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    inv_freq = self._compute_inv_freq(device)  # [Hd/2]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions_expanded = position_ids[None, :].float()  # [1, T]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class Qwen3ASRTextAttention(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Multi-head attention with TP head sharding and per-head QK-norm.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.>>> PARALLELISM: TP + SP <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Q/K/V heads sharded across TP ranks
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- KV heads replicated when fewer than TP size (GQA)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Prefill: all-gather from SP → QKV (with fused input norm) → QK-norm →
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.  RoPE → attn → O proj → reduce-scatter to SP
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Decode: fused megakernel (with fused input norm) → all-reduce (no SP)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.<-- MODEL-SPECIFIC:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- GQA: 16 Q heads / 8 KV heads (Qwen3-ASR-1.7B)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Per-head QK-norm (head_dim RMSNorm) applied BEFORE RoPE
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- No attention bias, no sinks, no sliding window
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Standard RoPE (rotate_half)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRTextConfig, layer_idx: int):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.layer_idx = layer_idx
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.head_dim = config.head_dim
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.dtype = config.torch_dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.hidden_size = config.hidden_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.num_attention_heads = config.num_attention_heads
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.num_key_value_heads = config.num_key_value_heads
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.scaling = config.head_dim**-0.5

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: TP group setup <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.tp_group = get_tp_group()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.world_size = self.tp_group.world_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: Head sharding <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.world_size >= self.num_key_value_heads:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_key_value_heads_per_rank = 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_kv_replicas = self.world_size // self.num_key_value_heads
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_key_value_heads_per_rank = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.num_key_value_heads // self.world_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_kv_replicas = 1

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.num_key_value_groups = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: QKV weight shapes <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    q_size = self.num_attention_heads_per_rank * self.head_dim
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    kv_size = self.num_key_value_heads_per_rank * self.head_dim
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    qkv_size = q_size + 2 * kv_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    o_proj_in_features = self.num_attention_heads * self.head_dim // self.world_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.qkv_proj_weight = nn.Parameter(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.o_proj_weight = nn.Parameter(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.q_size = q_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.kv_size = kv_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.qkv_split_indices = [q_size, q_size + kv_size]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Pre-attention input RMSNorm
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.input_layernorm = Qwen3ASRTextRMSNorm(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.hidden_size, config.rms_norm_eps, self.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: Per-head QK-norm (on head_dim, not hidden_size)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.q_norm = Qwen3ASRTextRMSNorm(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.head_dim, config.rms_norm_eps, self.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.k_norm = Qwen3ASRTextRMSNorm(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.head_dim, config.rms_norm_eps, self.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # KV caches bound externally
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.k_cache = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.v_cache = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 3D views of the caches, for the nkh == 1 decode kernel. bind_kv_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # makes them, because the decode graph must not make the view itself.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.k_cache_3d = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.v_cache_3d = None

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Packed FP8 K cache layout. bind_kv_cache sets it from the cache rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # (packed K is one dim larger than V). Off for a bf16 cache.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.fp8_packed = False

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # FP8 KV quantization scales (quantize as tensor * scale). The
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # checkpoint has none, so they stay 1.0: bf16 -> fp8 directly, clamped
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # to +/-FP8_CLAMP_MAX. The tensors are created in bind_kv_cache, on the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # cache device, only for an FP8 cache. The floats fold the dequant into
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the softmax scale and W_out at trace time.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.register_buffer("k_scale", None, persistent=False)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.register_buffer("v_scale", None, persistent=False)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.k_scale_float = 1.0
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.v_scale_float = 1.0

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self._setup_weight_loaders()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.@property
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def kv_is_fp8(self) -> bool:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return self.k_cache is not None and self.k_cache.dtype in _FP8_KV_DTYPES

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _quantize_kv(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """bf16 -> fp8 cache dtype: (x * scale), clamped to the fp8 range."""
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return (x * scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _packed_viable(self, B: int, S_decode: int, S_ctx: int) -> bool:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Whether the packed FP8 decode kernel can run at this bucket.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    The kernel folds the KV heads into the batch, so check the resize at
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    both the folded and the unfolded batch: the mask helper sees the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    unfolded one. Both must keep block_len >= 2.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    nkh = self.num_key_value_heads_per_rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_len = self.v_cache.shape[-2] if self.v_cache is not None else 0
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return packed_fp8_viable_for_bucket(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        block_len=block_len,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        bs=B,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        q_head=self.num_attention_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        s_active=S_decode,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        s_prior=S_ctx,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ) and packed_fp8_viable_for_bucket(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        block_len=block_len,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        bs=B * nkh,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        q_head=self.num_key_value_groups,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        s_active=S_decode,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        s_prior=S_ctx,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _setup_weight_loaders(self):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """>>> PARALLELISM: TP sharding of QKV and O-proj weights. <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    <-- MODEL-SPECIFIC: Separate Q, K, V (no bias). HF stores transposed.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.qkv_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        fused_qkv_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            q_size=self.q_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            kv_size=self.kv_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            shard_dim=1,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            is_storage_transposed=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            num_kv_replicas=self.num_kv_replicas,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.o_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sharding_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            shard_dim=0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            shard_size=self.num_attention_heads * self.head_dim // self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            is_storage_transposed=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ── Forward dispatch ──────────────────────────────────────────────────

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.LongTensor | None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_embeddings: tuple[torch.Tensor, torch.Tensor],
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """>>> PARALLELISM: Dispatch to prefill or decode. <<<

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    This module owns ``input_layernorm`` and folds it into the QKV kernel on
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    each path, so the caller must not normalize ``hidden_states`` first.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    layer_name = f"layers.{self.layer_idx}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    max_query_len = attn_metadata[layer_name]["max_query_len"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if max_query_len <= decode_token_threshold:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return self.forward_decode(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states, positions, position_embeddings, attn_metadata
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # >>> PARALLELISM: All-gather from SP <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # The gather precedes the fused norm, so each rank normalizes every
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # token instead of its own shard. RMSNorm is per token, so the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # result is the same.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return self.forward_prefill(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states, positions, position_embeddings, attn_metadata
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ── Prefill path ──────────────────────────────────────────────────────

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward_prefill(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.LongTensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_embeddings: tuple[torch.Tensor, torch.Tensor],
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if attn_metadata is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return torch.zeros_like(hidden_states)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = hidden_states.to(self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    tokens, hidden = hidden_states.shape

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Step 1: Fused input RMSNorm + QKV projection + pre-RoPE QK-norm + RoPE
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cos, sin = position_embeddings
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cos_cache = cos.unsqueeze(0)  # [1, T, Dh]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sin_cache = sin.unsqueeze(0)  # [1, T, Dh]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    qkv = NF.qkv_proj(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden=hidden_states.unsqueeze(0),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qkv_weights=self.qkv_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        bias=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        d_head=self.head_dim,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        cos_cache=cos_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sin_cache=sin_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_q_heads=self.num_attention_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_kv_heads=self.num_key_value_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        norm_type=NormType.RMS_NORM,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        norm_weights=self.input_layernorm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        eps=self.input_layernorm.variance_epsilon,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qk_norm_pre_rope_eps=self.q_norm.variance_epsilon,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qk_norm_pre_rope_q_gamma=self.q_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        qk_norm_pre_rope_k_gamma=self.k_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ).squeeze(0)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # TODO: check profile for transpose, QKV kernel can output NBSD directly
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Reshape to per-head layout: [T, Nh, Dh] → [Nh, T, Dh]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        0, 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        0, 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        0, 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Step 3: Update KV cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    layer_name = f"layers.{self.layer_idx}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    slot_mapping = attn_metadata[layer_name]["slot_mapping"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_size = attn_metadata[layer_name]["block_size"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_table = attn_metadata[layer_name]["block_table_tensor"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_indices, position_indices = paged_slot_indices(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        slot_mapping, block_size, self.k_cache.shape[0]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    kv_is_fp8 = self.kv_is_fp8
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if kv_segment_size:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # attention_segmented_cte reads an FP8 K cache only in the packed
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # layout; fail fast on a standard-layout FP8 cache.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        validate_fp8_segmented_supported(kv_is_fp8, self.fp8_packed)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if kv_is_fp8:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_flat = self._quantize_kv(k.reshape(-1, self.head_dim), self.k_scale)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_flat = self._quantize_kv(v.reshape(-1, self.head_dim), self.v_scale)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_flat = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_flat = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    head_indices_for_put = torch.arange(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_key_value_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        dtype=torch.long,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        device=hidden_states.device,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ).repeat_interleave(slot_mapping.shape[0])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_indices_for_put = position_indices.repeat(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.num_key_value_heads_per_rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.fp8_packed:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Packed K [nb, nkh, block_size // 2, Dh, 2]: write whole token pairs.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # The helper merges in the compute dtype; k_flat is already
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # fp8-rounded, so the round trip through bf16 is exact.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        _write_packed_k_pairs(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.k_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            k_flat.to(self.dtype).view(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                self.num_key_value_heads_per_rank, -1, self.head_dim
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            ),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            block_indices,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            position_indices,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.k_cache.index_put_(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            k_flat,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.v_cache.index_put_(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        (block_indices_for_put, head_indices_for_put, position_indices_for_put),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_flat,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Step 4: Attention — no sinks, no sliding window
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if kv_segment_size:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Reads the FP8 cache; the K dequant folds into the softmax scale and
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # the V dequant is undone after attention (no-ops at scale 1.0).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn_output = NF.segmented_attention(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            q,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            k_cache=self.k_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            v_cache=self.v_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            block_tables=block_table,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            prior_tokens=cached_seq_len,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            block_size=block_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            kv_segment_size=kv_segment_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            scale=self.scaling / self.k_scale_float,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            tp_q=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            tp_out=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            fp8_packed=self.fp8_packed,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if self.v_scale_float != 1.0:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn_output = attn_output / self.v_scale_float
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Full prefill attends the local bf16 K/V, not the cache.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        q_flash = q.transpose(1, 2)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_flash = k.transpose(1, 2)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_flash = v

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn_output = NF.flash_attention(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            q_flash,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            k_flash,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            v_flash,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            scale=self.scaling,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            tp_q=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            tp_out=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Step 5: Output projection
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_output = attn_output.unsqueeze(0)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_output = attn_output.squeeze(0)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: Reduce-scatter to SP layout <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return attn_output.contiguous()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ── Decode path ───────────────────────────────────────────────────────

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward_decode(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.LongTensor | None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_embeddings: tuple[torch.Tensor, torch.Tensor],
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attention_mask: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """>>> PARALLELISM: Fused megakernel with TP. <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    <-- MODEL-SPECIFIC: QK-norm via rmsnorm_QK_pre_rope_enabled=True.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    No sinks, no sliding window.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Callers may supply a precomputed mask to reuse this path with custom
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attention semantics (e.g. DFlash bidirectional attention).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    layer_name = f"layers.{self.layer_idx}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    slot_mapping = attn_metadata[layer_name]["slot_mapping"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_size = attn_metadata[layer_name]["block_size"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_table = attn_metadata[layer_name]["block_table_tensor"]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    B = block_table.shape[0]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    tokens, hidden = hidden_states.shape
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    S_decode = tokens // B
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    assert tokens == B * S_decode, (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        f"decode token count ({tokens}) must be a multiple of batch size "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        f"({B}); got S_decode={S_decode}"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = hidden_states.to(self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    S_ctx = max_blocks_per_seq * block_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    nkh = self.num_key_value_heads_per_rank

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    X = hidden_states.view(B, S_decode, hidden)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cos, sin = position_embeddings
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    half_d = self.head_dim // 2
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cos_kernel = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        cos[:, :half_d]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .view(B, S_decode, half_d)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .permute(2, 0, 1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .contiguous()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .to(self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sin_kernel = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sin[:, :half_d]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .view(B, S_decode, half_d)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .permute(2, 0, 1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .contiguous()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        .to(self.dtype)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if attention_mask is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Positions are required to construct the causal attention mask.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        assert positions is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        pos_ids = positions.view(1, B * S_decode)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attention_mask = NF.gen_attention_decode_mask(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            pos_ids=pos_ids.to(torch.float32),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bs=B,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            q_head=self.num_attention_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            s_active=S_decode,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            s_prior=S_ctx,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            start_pos=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            block_len=block_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Fold the pre-attention RMSNorm into the megakernel. The kernel reads
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # gamma as [1, H]. Both call sites below share these 3 arguments.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    rmsnorm_X_gamma = self.input_layernorm.weight.view(1, -1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    rmsnorm_X_eps = self.input_layernorm.variance_epsilon

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # GQA (nkh > 1, e.g. TP=4 on the 8-KV-head model) reaches the fused
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # decode megakernel only via a per-head 3D block table + the 4D cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # kept as-is; a 2D table routes GQA to the slow torch fallback.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # In-kernel KV cache update (update_cache=True): the kernel writes the new
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # K/V into the paged cache, so no scatter runs after it and K/V never go to
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # HBM and back. The kernel then returns the attention output only.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # The slots index the flat num_blocks * kv_heads pool, the same pool that
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the per-head block table targets. The kernel reads uint32.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    kv_cache_update_idx = build_per_head_slot_mapping(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        slot_mapping, block_size, nkh, B
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ).to(torch.uint32)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # FP8 KV: the kernel quantizes the new K/V (x * scale, clamped) when it
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # gets both scales, and writes K in the packed layout when fp8_packed.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # The dequant folds into the softmax scale (K) and W_out (V); at the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # unit scales both stay the bf16 graph's values. All Python-level, so
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # a bf16 cache traces the same graph as before.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    kv_is_fp8 = self.kv_is_fp8
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_scale = self.k_scale if kv_is_fp8 else None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    v_scale = self.v_scale if kv_is_fp8 else None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    softmax_scale = self.scaling
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    W_out = self.o_proj_weight
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if kv_is_fp8 and self.k_scale_float != 1.0:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        softmax_scale = self.scaling / self.k_scale_float
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if kv_is_fp8 and self.v_scale_float != 1.0:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        W_out = self.o_proj_weight / self.v_scale_float

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Packed K needs the kernel's resized block_len >= 2. Every bucket of
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the shipped configs passes (block 32, decode context >= 256). A
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # bucket that fails un-swizzles the cache for the unpacked kernel and
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # re-swizzles its returned K, as gpt_oss does. Static per decode NEFF.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    use_packed_kernel = self.fp8_packed and self._packed_viable(B, S_decode, S_ctx)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    needs_k_back = self.fp8_packed and not use_packed_kernel

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if nkh > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        active_blocks_table = build_per_head_block_table(block_table, nkh).to(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            torch.int32
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        output = NF.attention_decode(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            X=X,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            X_hidden_dim_actual=self.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_X_enabled=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_X_eps=rmsnorm_X_eps,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_X_gamma=rmsnorm_X_gamma,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            W_qkv=self.qkv_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bias_qkv=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_QK_pre_rope_enabled=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rmsnorm_QK_post_rope_enabled=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            cos=cos_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            sin=sin_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rope_contiguous_layout=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            K_cache_transposed=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            active_blocks_table=active_blocks_table,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            K_cache=self.k_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            if not needs_k_back
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            else _unswizzle_packed_k(self.k_cache),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            V_cache=self.v_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attention_mask=attention_mask,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            softmax_scale=softmax_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            sink=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            update_cache=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            kv_cache_update_idx=kv_cache_update_idx,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            k_scale=k_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            v_scale=v_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            fp8_packed=use_packed_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            W_out=W_out,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bias_out=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            transposed_out=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            out_in_sb=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attention_dp=1,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attention_dp_group=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attention_dp_rank=0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            kv_needs_a2a=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            return_updated_kv=needs_k_back,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if needs_k_back:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            output, k_updated, _ = output
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.k_cache.copy_(_swizzle_packed_k(k_updated))

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.tp_group.all_reduce(output)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return output

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # nkh == 1 (TP >= num_kv_heads): the kernel takes a 3D
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # [num_blocks, block_len, d_head] cache + a 2D [B, num_blocks] table.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # A packed K view is [num_blocks, block_len // 2, d_head, 2].
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_cache = self.k_cache_3d if self.k_cache_3d is not None else self.k_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    v_cache = self.v_cache_3d if self.v_cache_3d is not None else self.v_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    k_cache_arg = k_cache if not needs_k_back else _unswizzle_packed_k(k_cache)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    active_blocks_table = block_table.to(torch.int32)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: rmsnorm_QK_pre_rope_enabled=True for QK-norm;
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # must pass learned gamma weights so decode matches prefill.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    output = NF.attention_decode(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        X=X,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        X_hidden_dim_actual=self.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_X_enabled=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_X_eps=rmsnorm_X_eps,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_X_gamma=rmsnorm_X_gamma,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        W_qkv=self.qkv_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        bias_qkv=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_QK_pre_rope_enabled=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rmsnorm_QK_post_rope_enabled=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        cos=cos_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sin=sin_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rope_contiguous_layout=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        K_cache_transposed=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        active_blocks_table=active_blocks_table,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        K_cache=k_cache_arg,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        V_cache=v_cache,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attention_mask=attention_mask,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        softmax_scale=softmax_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sink=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        update_cache=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        kv_cache_update_idx=kv_cache_update_idx,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_scale=k_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_scale=v_scale,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        fp8_packed=use_packed_kernel,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        W_out=W_out,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        bias_out=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        transposed_out=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        out_in_sb=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attention_dp=1,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attention_dp_group=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attention_dp_rank=0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        kv_needs_a2a=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return_updated_kv=needs_k_back,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if needs_k_back:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        output, k_updated, _ = output
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_cache.copy_(_swizzle_packed_k(k_updated))

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: TP all-reduce <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.tp_group.all_reduce(output)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return output


class Qwen3ASRTextMLP(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Dense MLP with TP intermediate sharding.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.>>> PARALLELISM: TP + SP <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- gate_proj, up_proj: hidden → intermediate/TP per rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- down_proj: intermediate/TP → hidden per rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Prefill: all-gather from SP → compute → reduce-scatter to SP
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.- Decode: compute → all-reduce

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.<-- MODEL-SPECIFIC: SiLU activation (gate * silu(up)), no bias.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRTextConfig):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: TP group setup <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.tp_group = get_tp_group()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.world_size = self.tp_group.world_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.hidden_size = config.hidden_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: Intermediate dim sharded across TP <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.intermediate_size_per_rank = config.intermediate_size // self.world_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # The post-attention RMSNorm
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.post_attention_layernorm = Qwen3ASRTextRMSNorm(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.hidden_size, config.rms_norm_eps, config.torch_dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.gate_proj_weight = nn.Parameter(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        torch.empty(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.intermediate_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            dtype=config.torch_dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.up_proj_weight = nn.Parameter(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        torch.empty(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.intermediate_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            dtype=config.torch_dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.down_proj_weight = nn.Parameter(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        torch.empty(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.intermediate_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            dtype=config.torch_dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self._setup_weight_loaders(config)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _setup_weight_loaders(self, config: Qwen3ASRTextConfig):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """>>> PARALLELISM: TP sharding of MLP weights. <<<"""
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    gate_up_loader = sharding_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        shard_dim=1,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        shard_size=self.intermediate_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_storage_transposed=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    down_loader = sharding_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        shard_dim=0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        shard_size=self.intermediate_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_storage_transposed=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(self.gate_proj_weight, gate_up_loader)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(self.up_proj_weight, gate_up_loader)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(self.down_proj_weight, down_loader)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_prefill: bool,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    *,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    fused_add_tensor: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill, all-reduce for decode. <<<

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    This module owns ``post_attention_layernorm``. The caller must not normalize first.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Decode folds that norm into the kernel. ``fused_add_tensor`` also folds the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attention residual in. The return is then ``(output, sum)``. The layer uses the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sum as its next residual. Prefill norms before the gather, which keeps the norm on the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    local shard, and it takes no fused add, because the residual is on the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    shard while the kernel input is gathered.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    TODO: fuse both on prefill after the removal of sequence parallelism.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if is_prefill and fused_add_tensor is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "The prefill path must not fuse the residual add. Give "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "fused_add_tensor on decode only, and add the residual after "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "this call."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if is_prefill:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Normalize on the SP shard, before the gather, then tell the kernel
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # that the activation is already normalized.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = self.post_attention_layernorm(hidden_states)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        norm_type = NormType.NO_NORM
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ln_w = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        norm_type = NormType.RMS_NORM
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ln_w = self.post_attention_layernorm.weight.view(1, -1)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: SiLU gated MLP
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    result = NF.mlp(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.gate_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.up_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.down_proj_weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Turn the column tiling off which introduces gate-up product transpose
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # before the down projection
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        column_tiling=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        eps=self.post_attention_layernorm.variance_epsilon,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ln_w=ln_w,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        norm_type=norm_type,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        fused_add_tensor=fused_add_tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        store_fused_add_result=fused_add_tensor is not None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if fused_add_tensor is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        output, fused_add_result = result
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        output, fused_add_result = result, None

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if is_prefill:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            output = self.tp_group.reduce_scatter(output, dim=0)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.tp_group.all_reduce(output)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if fused_add_result is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return output, fused_add_result
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return output


class Qwen3ASRTextDecoderLayer(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Single Qwen3-ASR decoder layer.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Architecture: hidden → RMSNorm → Attention → residual → RMSNorm → MLP → residual

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.``self_attn`` owns the pre-attention norm and ``mlp`` owns the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.post-attention norm, so this layer calls neither one. Decode folds each norm
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.into the kernel that follows it, and folds the attention residual add into
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.the MLP kernel. Prefill folds the pre-attention norm only.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRTextConfig, layer_idx: int):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.self_attn = Qwen3ASRTextAttention(config, layer_idx=layer_idx)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.mlp = Qwen3ASRTextMLP(config)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.layer_idx = layer_idx

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _is_decode(self, attn_metadata) -> bool:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    layer_name = f"layers.{self.layer_idx}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    max_query_len = attn_metadata[layer_name]["max_query_len"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return max_query_len <= decode_token_threshold

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_embeddings: tuple[torch.Tensor, torch.Tensor],
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_decode = self._is_decode(attn_metadata)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    residual = hidden_states
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = self.self_attn(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states=hidden_states,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        positions=positions,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        position_embeddings=position_embeddings,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn_metadata=attn_metadata,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Decode gives the attention residual to the MLP kernel, which adds it
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # before the norm and returns the sum. That removes 1 add operator and
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the gap between the 2 kernels. The second add stays here.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if is_decode:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states, residual = self.mlp(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states, is_prefill=False, fused_add_tensor=residual
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = residual + hidden_states
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        residual = hidden_states
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = self.mlp(hidden_states, is_prefill=True)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = residual + hidden_states

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return hidden_states


class Qwen3ASRTextModel(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Qwen3-ASR text decoder backbone.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.>>> PARALLELISM: TP + sequence parallelism (SP) on the prefill residual. <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Prefill shards the residual on dim 0. Decode is TP-only. This matches
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.``qwen3/model.py``. SP needs ``T % world_size == 0`` and ``T > world_size``
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.on the prompt length.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRTextConfig):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.config = config

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: TP group <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.tp_group = get_tp_group()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.world_size = self.tp_group.world_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: Vocab-sharded embedding <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.embed_tokens = VocabDimShardedEmbedding(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        vocab_size=config.vocab_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        embed_dim=config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        dtype=config.torch_dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        tp_group=self.tp_group.device_group,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: 28 decoder layers (Qwen3-ASR-1.7B)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.layers = nn.ModuleList(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        [
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            Qwen3ASRTextDecoderLayer(config, layer_idx)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            for layer_idx in range(config.num_hidden_layers)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: Final RMSNorm
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.norm = Qwen3ASRTextRMSNorm(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        config.hidden_size, config.rms_norm_eps, config.torch_dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.rotary_emb = Qwen3ASRTextRotaryEmbedding(config)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    set_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.embed_tokens.weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sharding_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            shard_dim=0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            shard_size=self.embed_tokens.vocab_size_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            is_storage_transposed=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            pad_shard=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    input_ids: torch.LongTensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    rank: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    inputs_embeds: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_token_ids: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- On-device audio encoder cache inputs (prefill-only).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_embedding_buffer: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_block_ids: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_positions: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> torch.Tensor:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    first_layer_name = "layers.0.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    max_query_len = attn_metadata[first_layer_name]["max_query_len"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    decode_token_threshold = attn_metadata[first_layer_name][
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        "decode_token_threshold"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_prefill = max_query_len > decode_token_threshold

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = self.embed_tokens(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        input_ids, scatter_tokens=is_prefill, rank=rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Audio embedding merge (prefill only; decode has no audio inputs). Scatter
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the encoder outputs from the on-device cache blocks into the audio
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # placeholder positions. The merge does the SP coordinate remap internally,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # so it runs on the already-SP-sharded hidden_states.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_prefill
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and encoder_embedding_buffer is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and encoder_block_ids is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and encoder_positions is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = merge_audio_embeddings(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            encoder_embedding_buffer,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            encoder_block_ids,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            encoder_positions,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            rank=self.tp_group.rank_in_group_tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_prefill
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and self.world_size > 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and inputs_embeds is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        and is_token_ids is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        inputs_embeds = shard_select(inputs_embeds, 0, self.tp_group)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_token_ids = shard_select(is_token_ids, 0, self.tp_group)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = NF.merge_prompt_embeds(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states, inputs_embeds, is_token_ids
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    position_embeddings = self.rotary_emb(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        positions, device=hidden_states.device, dtype=hidden_states.dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for decoder_layer in self.layers:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = decoder_layer(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            hidden_states,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            positions=positions,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            position_embeddings=position_embeddings,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn_metadata=attn_metadata,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = self.norm(hidden_states)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: SP — all-gather to reconstruct full sequence <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if is_prefill and self.world_size > 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return hidden_states


# =============================================================================
# Composed model — audio encoder + text decoder + fusion + sampler.
# =============================================================================


class Qwen3ASRForConditionalGeneration(nn.Module):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""Composed Qwen3-ASR model: audio encoder + Qwen3 text decoder.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.Decode-only today (see module docstring). ``SupportsEncoderWarmup`` and the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.audio-fusion path are added in the fusion/bucket follow-up commits.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER."""

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Bare flag, not the SupportsTranscription protocol: the upstream Qwen3-ASR
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# class owns the prompt methods, so this class only advertises the task.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.supports_transcription: bool = True
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.supports_transcription_only: bool = False
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# The forward assembles the decode input_ids on the device, so the runner may send the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# in-graph reorder arguments. The second flag lifts the runner's multimodal exclusion: the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# audio enters only through the prefill graph (encoder cache gather by request), the decode
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# graph has no encoder inputs and the positions are plain 1-D, so the decode reorder does not
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.supports_async_reorder = True
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.supports_async_reorder_multimodal = True

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def __init__(self, config: Qwen3ASRConfig):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    super().__init__()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: EPD role flags (audio-encoder-only vs LM-only pools) <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Read the EPD role flags off the audio encoder's neuron config.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Both default False for a single-pool model.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    audio_neuron_config = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        config.audio_config.neuron_config if config.audio_config else None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.mm_encoder_only = bool(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        audio_neuron_config and audio_neuron_config.mm_encoder_only
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.mm_language_model_only = bool(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        audio_neuron_config and audio_neuron_config.mm_language_model_only
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.config = config
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.text_config = config.text_config

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- MODEL-SPECIFIC: audio tower. Built unless this is the LM-only EPD
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #     pool. dtype is governed by config.audio_config.torch_dtype (bf16).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if config.audio_config is not None and not self.mm_language_model_only:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.audio_tower = Qwen3ASREncoder(config.audio_config)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.audio_tower = None

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if not self.mm_encoder_only:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.language_model = Qwen3ASRTextModel(config.text_config)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.language_model = None

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: text TP group (lm_head + logits gather) <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.tp_group = get_tp_group()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.world_size = self.tp_group.world_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.rank = self.tp_group.rank_in_group

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    nc = config.text_config.neuron_config
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self._gather_logits = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        nc is not None and nc.max_logprobs != 0
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ) or debug_logits_enabled

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if not self.mm_encoder_only:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # >>> PARALLELISM: vocab-dim sharded lm_head (column parallel) <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.lm_head = neuron_nn.ColumnParallelLinear(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            config.text_config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            config.text_config.vocab_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bias=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            dtype=config.text_config.torch_dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            gather_output=not self.on_device_sampling_config,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            tp_group=self.tp_group.device_group,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            weight_layout="in_out",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        transposed = self.lm_head.weight_layout == "in_out"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        set_weight_loader(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.lm_head.weight,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            sharding_weight_loader_with_padding(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                shard_dim=1 if transposed else 0,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                shard_size=self.text_config.vocab_size // self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                num_shards=self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                is_storage_transposed=transposed,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                pad_dim=0 if transposed else 1,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                padded_size=self.text_config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                unpadded_size=self.text_config.hidden_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            ),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # >>> fused lm_head + greedy argmax (TP1 all-greedy only, see get_lmhead_backend) <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self._fused_lmhead = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            get_lmhead_backend() == "fused"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and self.world_size == 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and self.on_device_sampling_config is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and bool(self.on_device_sampling_config.all_greedy)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and not self._gather_logits
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and self.lm_head.weight_layout == "in_out"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            and self.lm_head.bias is None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        logger.info(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "qwen3_asr lm_head backend: %s (requested %s, tp=%d, all_greedy=%s, gather_logits=%s)",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "fused" if self._fused_lmhead else "stock",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            get_lmhead_backend(),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bool(self.on_device_sampling_config is not None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                 and self.on_device_sampling_config.all_greedy),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self._gather_logits,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if self.on_device_sampling_config is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.sampler = Sampler(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                self.on_device_sampling_config,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                process_group=self.tp_group.device_group,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.sampler = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.lm_head = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.sampler = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self._fused_lmhead = False

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Compiler arguments (per-graph neuronx-cc policy)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def compiler_args_for(self, graph_type, example_inputs, base):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Give each graph its own neuronx-cc arguments.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    This is the model's per-graph policy. The runner resolves it one time
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    per graph, before the cache hash. It passes the graph type, the example
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    inputs, and the base argument list. Return the full argument list for
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    that graph.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ``base`` holds the arguments at the caller level. Derive the per-graph
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    arguments from it. These arguments are a default, not an override. An
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    explicit ``--optimization-level`` from the caller wins, so the runner
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    does not call this policy.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Qwen3-ASR compiles the decode graph and the audio-encoder graph at
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ``-O3``. ``-O3`` disables neuronx-cc modular flow. The ``-O1`` default
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    enables it, which adds a wrapper on each collective. ``-O3`` improves
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    the latency of these two graphs. Prefill keeps the base level. The
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    runner tags the audio-encoder graph ``"encoder"``: one tag serves each
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    modality, because every encoder shares one graph path.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Example:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        >>> model.compiler_args_for("decode", None, ["-O1"])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ['-O3']
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        >>> model.compiler_args_for("prefill", None, ["-O1"])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ['-O1']
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    base = list(base or [])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if graph_type in ("decode", "encoder"):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return with_optimization_level(base, 3)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return base

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def _scoped_compiler_args(self, graph_type):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Scope this graph's resolved arguments for an in-process compile.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Read the base arguments that the runner set on the model. Delegate to
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    :meth:`compiler_args_for`. Open a libtorch scope with the result. Open a
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    null scope when the runner did not enable per-graph tuning, so the base
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    arguments apply.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Example:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        >>> with model._scoped_compiler_args("encoder"):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ...     model.audio_tower(...)  # doctest: +SKIP
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    base = getattr(self, "_neuron_base_compiler_args", None)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    args = None if base is None else self.compiler_args_for(graph_type, None, base)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return scoped_compiler_args(args)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Factory
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.@classmethod
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def from_configs(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cls,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hf_config: PretrainedConfig,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    text_neuron_config: NeuronConfig = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_neuron_config: EncoderNeuronConfig = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Build the composed model from a raw HF config + NeuronConfig(s).

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    The keyword ``encoder_neuron_config`` is the runner seam
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    (``neuron_model_runner.py`` calls
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ``from_configs(hf_config=, text_neuron_config=, encoder_neuron_config=)``),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    so we accept that literal name and forward it internally into the audio
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    slot.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    config = Qwen3ASRConfig.from_configs(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hf_config,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        text_neuron_config=text_neuron_config,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        audio_neuron_config=encoder_neuron_config,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return cls(config)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Forward (decode-only: text backbone -> sampling)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.@torch.no_grad()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def forward(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    input_ids: torch.LongTensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions: torch.Tensor,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    attn_metadata: object | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sampling_positions: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sampling_params: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    logit_mask: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    rank: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    inputs_embeds: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_token_ids: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # <-- On-device audio encoder cache inputs (prefill-only), injected by
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #     the runner as mm_kwargs.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_embedding_buffer: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_block_ids: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_positions: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    **kwargs,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Audio-prefill / text-decode step."""
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    positions = positions.to(torch.int32)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    first_layer_name = "layers.0.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    max_query_len = attn_metadata[first_layer_name]["max_query_len"]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    decode_token_threshold = attn_metadata[first_layer_name][
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        "decode_token_threshold"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    is_prefill = max_query_len > decode_token_threshold

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Assemble the decode input_ids on the device.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # The runner gives the device future of the previous step, plus a permutation and a
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # mask that it builds on the host from the request ids. A request that continues takes
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # the reordered future. A new request, a resumed request, a row that still prefills,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # and a padded slot take the host input_ids. The host therefore does not read the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # sampled tokens of the previous step back.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # The arguments are absent on a step that does not use the reorder, and this block is
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # then a no-op. Remove all three arguments in every case, so a later consumer of
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # ``**kwargs`` never sees a partial set.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Keep this block identical in every model that declares supports_async_reorder.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # test_async_reorder.py compares them and fails on a divergence.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    async_reorder_perm = kwargs.pop("async_reorder_perm", None)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    async_reorder_future = kwargs.pop("async_reorder_future", None)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    async_reorder_mask = kwargs.pop("async_reorder_mask", None)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if async_reorder_perm is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        input_ids = torch.where(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            async_reorder_mask,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            input_ids,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            torch.index_select(async_reorder_future, 0, async_reorder_perm),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    T = input_ids.shape[0]

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # >>> PARALLELISM: SP length validation <<<
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"Prompt length {T} is too short for sequence parallelism (world_size={self.world_size}). "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"Use a prompt with length > {self.world_size} and divisible by {self.world_size}."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states = self.language_model(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        input_ids,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        positions,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn_metadata=attn_metadata,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rank=rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        inputs_embeds=inputs_embeds,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        is_token_ids=is_token_ids,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        encoder_embedding_buffer=encoder_embedding_buffer,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        encoder_block_ids=encoder_block_ids,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        encoder_positions=encoder_positions,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    hidden_states_for_logits = torch.index_select(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hidden_states, dim=0, index=sampling_positions
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self._fused_lmhead and logit_mask is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # greedy token ids straight from the hidden states; the [B, V] logits never exist
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        sampled_tokens = lmhead_argmax(hidden_states_for_logits, self.lm_head.weight)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return sampled_tokens, None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    logits = self.lm_head(hidden_states_for_logits)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    gathered_logits = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self._gather_logits:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        gathered_logits = self.tp_group.all_gather(logits, dim=1)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.on_device_sampling_config is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return logits

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    sampled_tokens = self.sampler(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return sampled_tokens, gathered_logits

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# KV cache wiring (text layers)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def get_kv_spec(self) -> KVSpec:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.language_model is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return KVSpec(layers=[])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    layers = []
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for i, layer in enumerate(self.language_model.layers):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        layer_name = f"layers.{i}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        layers.append(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            LayerSpec(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                name=layer_name,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                head_size=layer.self_attn.head_dim,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                dtype=layer.self_attn.dtype,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                sliding_window_size=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                chunk_size=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return KVSpec(layers=layers)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.language_model is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        return
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for i, layer in enumerate(self.language_model.layers):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        layer_name = f"layers.{i}.self_attn"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if layer_name not in kv_caches:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            raise KeyError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                f"KV cache for layer {layer_name} is not initialized. "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                "Call bind_kv_cache() before the first forward pass."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        k_cache = kv_caches[layer_name][0]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        v_cache = kv_caches[layer_name][1]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn = layer.self_attn
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn.k_cache = k_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn.v_cache = v_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Packed FP8 K (neuron_config.fp8_packed_kv + --kv-cache-dtype
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # fp8_e4m3) is [nb, nkh, block // 2, d, 2], one dim above V.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        attn.fp8_packed = k_cache.dim() == v_cache.dim() + 1
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if attn.fp8_packed and k_cache.dtype != torch.float8_e4m3fn:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                f"{layer_name}: packed K cache must be float8_e4m3fn, "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                f"got {k_cache.dtype}"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if attn.kv_is_fp8:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            # Unit scales: the checkpoint has no KV scales.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.k_scale = torch.ones(1, 1, dtype=attn.dtype, device=k_cache.device)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.v_scale = torch.ones(1, 1, dtype=attn.dtype, device=k_cache.device)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.k_scale_float = 1.0
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.v_scale_float = 1.0
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.k_scale = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.v_scale = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Prefill keeps the 4D cache, because its index_put_ gives a head
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # index. Decode at nkh == 1 needs a 3D cache. Make that view here,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # one time, so the decode graph gets a plain input and the in-place
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # cache update needs no copy of the pool. A packed K squeezes to the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # kernel's 4D packed base [nb, block // 2, d, 2].
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        if v_cache.dim() == 4 and v_cache.shape[1] == 1:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.k_cache_3d = k_cache.squeeze(1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.v_cache_3d = v_cache.squeeze(1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.k_cache_3d = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            attn.v_cache_3d = None

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Weight loading
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.@classmethod
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def build_text_weight_mappings(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cls, num_hidden_layers: int, tie_word_embeddings: bool
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> dict[str, str | list[str]]:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Composed-model param name -> HF checkpoint key(s) for the text decoder."""
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    mappings: dict[str, str | list[str]] = {
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        "language_model.embed_tokens.weight": f"{HF_TEXT_PREFIX}.embed_tokens.weight",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        "language_model.norm.weight": f"{HF_TEXT_PREFIX}.norm.weight",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    }

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # tie_word_embeddings=True for Qwen3-ASR-1.7B: lm_head reads the shared
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # embedding weight; otherwise a dedicated thinker.lm_head.weight.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if tie_word_embeddings:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings["lm_head.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings["lm_head.weight"] = HF_LM_HEAD_KEY

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for layer_id in range(num_hidden_layers):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        our = f"language_model.layers.{layer_id}"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        hf = f"{HF_TEXT_PREFIX}.layers.{layer_id}"

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Fused QKV: list value triggers fused_qkv_weight_loader on the param.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.self_attn.qkv_proj_weight"] = [
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{hf}.self_attn.q_proj.weight",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{hf}.self_attn.k_proj.weight",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{hf}.self_attn.v_proj.weight",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Per-head QK-norm (identity name; no TP shard).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.self_attn.q_norm.weight"] = f"{hf}.self_attn.q_norm.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.self_attn.k_norm.weight"] = f"{hf}.self_attn.k_norm.weight"

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # The 2 layer norms live in the modules that fold them in, so our
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # names carry the ``self_attn`` and ``mlp`` prefix. The HF names stay
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # at the layer level.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.self_attn.input_layernorm.weight"] = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{hf}.input_layernorm.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.mlp.post_attention_layernorm.weight"] = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{hf}.post_attention_layernorm.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.mlp.gate_proj_weight"] = f"{hf}.mlp.gate_proj.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.mlp.up_proj_weight"] = f"{hf}.mlp.up_proj.weight"
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings[f"{our}.mlp.down_proj_weight"] = f"{hf}.mlp.down_proj.weight"

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    return mappings

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def load_weights(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Load text-decoder weights (this class) + delegate audio-tower weights.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Text: strip ``thinker.model.`` -> ``language_model.``, fuse q/k/v, tie
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    lm_head to the shared embedding. Audio: delegate to the encoder's own
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    loader (uses the encoder TP group, CPU-side).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if not self.mm_encoder_only:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mappings = self.build_text_weight_mappings(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            len(self.language_model.layers),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.text_config.tie_word_embeddings,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        rank_sharded = checkpoint.load_sharded_pipelined(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.rank,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self.world_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            mappings,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            device,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            strict=False,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ).state_dict

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        # Cast any tensor not already the text dtype (matches VL / qwen3).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        for name, tensor in rank_sharded.items():
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            if tensor.dtype != self.text_config.torch_dtype:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.                rank_sharded[name] = tensor.to(self.text_config.torch_dtype)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.load_state_dict(rank_sharded, strict=False, assign=True)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # Audio tower loads its own thinker.audio_tower.* weights (encoder TP
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # group, CPU-side), independent of the text TP shard.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if self.audio_tower is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.audio_tower.load_weights(checkpoint_path, device="cpu", cpu_mode=True)

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# Audio-fusion path
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.# ------------------------------------------------------------------
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.def embed_multimodal(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    self,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    input_audio_features: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    audio_feature_lengths: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    feature_attention_mask: torch.Tensor | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    encoder_cache=None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    mm_hashes: list[str] | None = None,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    **kwargs,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.) -> None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """Encode audio clips and allocate+write into the on-device cache.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    The runner groups mm_kwargs by modality, so one call carries only audio items.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    It plans the block-packed chunk layout, reserves cache blocks per clip, packs
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    the mel into one whole-chunk stream, and runs the encoder in cache-write mode.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    The encoder scatter-writes into cache.buffer.

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    Args:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        input_audio_features: [num_mel_bins, sum_F] flat mel, concatenated over
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            clips along the frame axis (upstream flat_from_sizes, dim=1),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            unpadded and not chunk-aligned.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        audio_feature_lengths: [num_clips] valid mel frames per clip.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        feature_attention_mask: [num_clips, max_F] mel mask; the fallback source
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            for the per-clip length (matches upstream _parse_and_validate_audio).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        encoder_cache: EncoderCacheBlocks (buffer, allocate, scratch_block_id).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        mm_hashes: one identifier per clip, in clip order.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    """
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    from vllm_neuron.vllm.worker.encoder_cache_blocks import EncoderCacheBlocks

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if input_audio_features is None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "embed_multimodal requires input_audio_features for the audio modality."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cache: EncoderCacheBlocks = encoder_cache
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    ac = self.config.audio_config
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    audio_neuron_config = ac.neuron_config
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    t_start = _time.perf_counter()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 1. Per-clip valid mel lengths. Prefer explicit lengths; else the mask.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if audio_feature_lengths is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        feature_lens = [int(v) for v in audio_feature_lengths.flatten().tolist()]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    elif feature_attention_mask is not None:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        feature_lens = [int(v) for v in feature_attention_mask.sum(-1).tolist()]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    else:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "embed_multimodal needs audio_feature_lengths or "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "feature_attention_mask to derive the per-clip mel length."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    num_clips = len(feature_lens)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if mm_hashes is None or len(mm_hashes) != num_clips:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"embed_multimodal expects one mm_hash per clip: got "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{0 if mm_hashes is None else len(mm_hashes)} hashes for "
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            f"{num_clips} clips."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    block_size = audio_neuron_config.encoder_attention_block_size
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    dp_size = audio_neuron_config.dp_size

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 2. Plan the block-packed layout and 1D-bucket it in one pass. The slot
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    unit is one base chunk. plan packs segments (full windows and clip
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    tails) into blocks, then picks the smallest bucket that fits. Buckets
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    hold block counts. one_item_per_block is required by the cache (one
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    mm_item per block run).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    buckets = audio_neuron_config.num_encoder_input_buckets
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if not buckets:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        raise ValueError(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            "num_encoder_input_buckets must be configured (non-empty) for audio."
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    plan = plan_audio_blocks(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        feature_lens,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        block_size=block_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        window_tokens=ac.window_tokens,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        n_window=ac.n_window,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        n_window_infer=ac.n_window_infer,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        buckets=buckets,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        one_item_per_block=True,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        dp_size=dp_size,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    num_blocks = plan.num_blocks
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    needed_blocks = plan.min_blocks
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    slots_per_block = plan.slots_per_block

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 3. Allocate cache blocks per clip. The block count follows the plan (one
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    clip per block run), NOT dense output tokens, so the cache reserves
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    the same blocks the encoder writes. Real tokens per block sum the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    clip's chunk lengths in that block and skip the pad tail.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    cache_block_map: list[list[int]] = []
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for i in range(num_clips):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        slots = plan.clip_slots[i]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        tokens_per_block = [
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            sum(plan.chunk_lens[s] for s in slots[b : b + slots_per_block])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            for b in range(0, len(slots), slots_per_block)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        ]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        cache_block_map.append(cache.allocate(mm_hashes[i], tokens_per_block))

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 4. CPU preprocessing off the plan.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    chunk_slots = plan.chunk_slots()
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    bound_min, bound_max = plan.block_bounds()

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 5. Pack the mel into a whole-chunk stream, then 1D-bucket it. Split the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    flat mel by valid lengths, pad each clip up to whole base chunks (1s
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    granularity), and concat -> the compact real-chunk stream. Then pad
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    up to num_slots chunks so F = num_slots * num_mel_frames_per_chunk: num_chunks ==
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    num_slots is pinned by the bucketed num_blocks, so the conv and
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    index_select shapes are static. The conv stays block-bucketed (no
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    chunk-count compile dimension), so it also runs on the pad chunks; the
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    bounds mask them and chunk_slots never references them.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    num_mel_frames_per_chunk = 2 * ac.n_window  # mel frames in one base chunk
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    num_mel_bins = input_audio_features.shape[0]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    clip_mels = torch.split(input_audio_features, feature_lens, dim=1)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    parts = []
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    for clip_mel, valid in zip(clip_mels, feature_lens):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        padded = -(-valid // num_mel_frames_per_chunk) * num_mel_frames_per_chunk
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        buf = input_audio_features.new_zeros(num_mel_bins, padded)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        buf[:, :valid] = clip_mel
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        parts.append(buf)
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    input_features = torch.cat(parts, dim=1).unsqueeze(0)  # [1, num_mel_bins, F]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    target_F = (
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        plan.num_slots * num_mel_frames_per_chunk
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )  # num_blocks * slots_per_block * 100
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    if input_features.shape[-1] < target_F:
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        input_features = torch.nn.functional.pad(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            input_features, (0, target_F - input_features.shape[-1])
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 6. write_block_ids: flatten the per-clip cache blocks in clip order; pad
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    the trailing (bucket) blocks with the scratch block id.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    flat_cache_blocks = [b for blocks in cache_block_map for b in blocks]
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    write_block_ids = torch.tensor(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        flat_cache_blocks
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        + [cache.scratch_block_id] * (num_blocks - len(flat_cache_blocks)),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        dtype=torch.int64,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )

 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    # 7. Move to device, dispatch the encoder in cache-write mode (input-output
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    alias on cache.buffer). The last dim is output_dim (post-projector).
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    Scope the "encoder" graph arguments, so the compiled audio-encoder
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    #    graph hashes to the same arguments (-O3) as the capture and the warmup.
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    device = next(self.audio_tower.parameters()).device
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    with self._scoped_compiler_args("encoder"):
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        self.audio_tower(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            input_features.to(dtype=ac.torch_dtype, device=device),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bound_min.to(device),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            bound_max.to(device),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            chunk_slots.to(device),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            cache.buffer,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.            write_block_ids.to(device),
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        )
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    logger.debug(
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        "[embed_multimodal] %d clips, %d blocks (needed %d): %.1fms",
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_clips,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        num_blocks,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        needed_blocks,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.        (_time.perf_counter() - t_start) * 1000,
 touch multimodal bookkeeping. Runtime switch: VLLM_NEURON_MM_ASYNC_REORDER.    )
