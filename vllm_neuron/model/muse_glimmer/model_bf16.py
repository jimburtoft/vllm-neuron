# SPDX-License-Identifier: Apache-2.0
"""
MuseGlimmer Text Decoder (BF16)
===============================

Text decoder of ``MuseGlimmerForConditionalGeneration`` for vLLM-Neuron.

Architecture (per layer, see ``reference/modeling_muse_glimmer.py``):

    residual = x
    x = input_layernorm(x)                        # centered RMSNorm, (1 + w), eps 1e-5
    x = attention(x)                              # gated, QK-normed, SWA+RoPE or full+NoPE
    x = post_attention_layernorm(x)               # centered RMSNorm, eps 1e-8
    x = residual + x
    residual = x
    x = pre_feedforward_layernorm(x)              # centered RMSNorm, eps 1e-5
    x = mlp(x)                                    # SwiGLU
    x = post_feedforward_layernorm(x)             # centered RMSNorm, eps 1e-8
    x = residual + x

Attention:
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    q, k = rmsnorm(q), rmsnorm(k)                 # weightless, per head, eps 1e-5
    q = q * qk_scale_factor                       # folded into the softmax scale here
    q, k = rope(q, k)  (sliding layers only; full layers are NoPE)
    o = softmax(q k^T / sqrt(d)) v                # sliding window on sliding layers
    o = o * sigmoid(x @ Wgate)                    # output gate, before o_proj
    out = o @ Wo

Backbone: embed -> weightless RMSNorm (embed_norm) -> [vision scatter] ->
layers -> weighted RMSNorm. Logits: ``cap * tanh(lm_head(x) * mult / cap)``.

The perception encoder lives in ``vision_encoder_bf16.py`` (``self.visual``);
``embed_multimodal`` packs images on CPU and writes the projected features into
the on-device encoder cache. ``mm_language_model_only`` skips the tower.
"""

from __future__ import annotations

import logging
import math
import os
import time as _time

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.model.qwen3_vl.utils.decode_kv import build_per_head_block_table
from vllm_neuron.model.qwen3_vl.utils.merge_vision_embeds import (
    merge_vision_embeddings,
)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.neuron_utils import can_run_kernel
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from vllm_neuron.model.qwen3_vl.utils.vision_block_packing import (
    ffd_pack_images,
    select_vision_bucket,
)

from .config import MuseGlimmerConfig, MuseGlimmerTextConfig
from .utils.vision_preprocessing import pack_vision_inputs
from .vision_encoder_bf16 import MuseGlimmerVisionModel

logger = logging.getLogger(__name__)

# Escape hatch: route decode attention through the plain-torch reference
# (``_torch_attention_decode_impl``) instead of the fused NKI megakernel.
DECODE_TORCH_ENV = "MUSE_GLIMMER_DECODE_TORCH"


def _decode_torch_forced() -> bool:
    return os.environ.get(DECODE_TORCH_ENV, "0").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Section 1: Normalization
# ---------------------------------------------------------------------------


class MuseGlimmerRMSNorm(nn.Module):
    """Standard RMSNorm (optionally weightless), computed in fp32.

    HF: ``x * pow(mean(x^2) + eps, -0.5) [* w]`` then cast back. rsqrt and
    pow(-0.5) agree to fp32 rounding.
    """

    def __init__(self, dim: int | None, eps: float, dtype: torch.dtype, with_scale: bool):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            xf = xf * self.weight.to(torch.float32)
        return xf.to(in_dtype)


class MuseGlimmerCenteredRMSNorm(nn.Module):
    """Centered RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)``.

    ``w`` is stored as an **fp32** parameter and ``1 + w`` is formed in fp32,
    exactly like HF. Folding ``+1`` into a bf16 weight would round ``1 + w`` to
    8 mantissa bits.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        xf = xf * (1.0 + self.weight)
        return xf.to(in_dtype)


# ---------------------------------------------------------------------------
# Section 2: RoPE (plain 1D, rotate_half, full head_dim; sliding layers only)
# ---------------------------------------------------------------------------


class MuseGlimmerRotaryEmbedding(nn.Module):
    """Default RoPE. inv_freq is computed lazily (meta-tensor rule)."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(cos, sin)`` each ``[T, head_dim // 2]`` in fp32."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0, self.head_dim, 2, dtype=torch.float32, device=positions.device
                )
                / self.head_dim
            )
        )
        freqs = positions.reshape(-1).to(torch.float32)[:, None] * inv_freq[None, :]
        return freqs.cos(), freqs.sin()


# ---------------------------------------------------------------------------
# Section 3: Attention
# ---------------------------------------------------------------------------


class MuseGlimmerAttention(nn.Module):
    """Gated GQA attention with weightless QK-norm.

    Parallelism (kept from qwen3_vl / gpt_oss):
    - Q heads and the gate are column-sharded; o_proj is row-sharded.
    - KV heads < TP -> replicated (``num_kv_replicas``); rank r owns KV head
      ``r // num_kv_replicas``, which matches HF's repeat_kv grouping.
    - Prefill: SP all-gather -> attention -> reduce-scatter.
    - Decode: fused megakernel (W_out=None) -> gate -> o_proj -> all-reduce.
    """

    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qk_norm_eps = config.rms_norm_eps
        # <-- MODEL-SPECIFIC: Q multiplier folded into the softmax scale.
        self.scaling = config.attention_scale
        # <-- MODEL-SPECIFIC: sliding (RoPE) vs full (NoPE) per layer.
        self.is_sliding = config.is_sliding(layer_idx)
        self.apply_rope = config.uses_rope(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # >>> PARALLELISM: TP group + head sharding <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if self.num_attention_heads % self.world_size != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"TP={self.world_size}"
            )
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            if self.world_size % self.num_key_value_heads != 0:
                raise ValueError(
                    f"TP={self.world_size} not a multiple of num_kv_heads="
                    f"{self.num_key_value_heads}"
                )
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            if self.num_key_value_heads % self.world_size != 0:
                raise ValueError(
                    f"num_kv_heads={self.num_key_value_heads} not divisible by "
                    f"TP={self.world_size}"
                )
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=self.dtype)
        )
        # <-- MODEL-SPECIFIC: output gate, sharded exactly like Q.
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(q_size, self.hidden_size, dtype=self.dtype)
        )

        self.k_cache = None
        self.v_cache = None
        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.gate_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.q_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # ── dispatch ────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        if meta["max_query_len"] <= meta["decode_token_threshold"]:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        # >>> PARALLELISM: SP all-gather before attention <<<
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, position_embeddings, attn_metadata)

    def _write_paged_kv(self, k, v, slot_mapping, block_size):
        """k, v: [nkh, T, d] -> 4D paged cache [blocks, nkh, block, d]."""
        nkh = self.num_key_value_heads_per_rank
        n = slot_mapping.shape[0]
        block_idx = (slot_mapping // block_size).repeat(nkh)
        pos_idx = (slot_mapping % block_size).repeat(nkh)
        head_idx = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(n)
        self.k_cache.index_put_(
            (block_idx, head_idx, pos_idx),
            k.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_idx, head_idx, pos_idx),
            v.reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )

    # ── prefill ─────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]

        # <-- MODEL-SPECIFIC: RoPE only on sliding layers (full layers are NoPE).
        cos_cache = sin_cache = None
        if self.apply_rope:
            cos, sin = position_embeddings
            cos_cache = torch.cat((cos, cos), dim=-1).unsqueeze(0).to(self.dtype)
            sin_cache = torch.cat((sin, sin), dim=-1).unsqueeze(0).to(self.dtype)

        # <-- MODEL-SPECIFIC: weightless QK RMSNorm (gamma=None) before RoPE.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            d_head=self.head_dim,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank,
            num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.qk_norm_eps,
            qk_norm_pre_rope_q_gamma=None,
            qk_norm_pre_rope_k_gamma=None,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.reshape(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.reshape(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        self._write_paged_kv(k, v, meta["slot_mapping"], meta["block_size"])

        kv_segment_size = meta.get("kv_segment_size")
        if kv_segment_size:
            # Chunked (segmented) prefill: attend over the paged cache, which
            # now holds prior chunks plus this chunk.
            attn_output = NF.segmented_attention(
                q,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=meta["block_table_tensor"],
                prior_tokens=meta.get("cached_seq_len"),
                block_size=meta["block_size"],
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=True,
                tp_out=True,
            )  # [N, D, T]
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2),  # [N, D, T]
                k.transpose(1, 2),  # [N, D, T]
                v,  # [N, T, D]
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=False,
                tp_out=True,
            )  # [N, D, T]

        # <-- MODEL-SPECIFIC: sigmoid output gate before o_proj.
        gate = torch.sigmoid(hidden_states @ self.gate_proj_weight)  # [T, N*D]
        gate = gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim).permute(1, 2, 0)
        attn_output = (attn_output * gate).to(self.dtype)

        out = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight).squeeze(0)  # [T, H]

        # >>> PARALLELISM: reduce-scatter back to SP layout <<<
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)
        return out.contiguous()

    # ── decode ──────────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        attn_metadata: dict,
    ) -> torch.Tensor:
        meta = attn_metadata[self.layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        assert tokens == B * S
        nkh = self.num_key_value_heads_per_rank
        d = self.head_dim

        hidden_states = hidden_states.to(self.dtype)
        X = hidden_states.view(B, S, hidden)

        cos_k = sin_k = None
        if self.apply_rope:
            cos, sin = position_embeddings  # [T, d/2]
            cos_k = cos.view(B, S, d // 2).permute(2, 0, 1).contiguous().to(self.dtype)
            sin_k = sin.view(B, S, d // 2).permute(2, 0, 1).contiguous().to(self.dtype)

        # In-kernel mask generation: pos_ids (+ SWA window start).
        pos = positions.view(B, S).to(torch.float32)
        swa_start = None
        if self.sliding_window is not None:
            # Runner trims the SWA block table to the window and reports the
            # trimmed-frame offset (= start_block * block_size).
            off = meta.get("swa_kv_pos_offset")
            if off is not None:
                pos = torch.clamp(pos - off.view(B, 1).to(torch.float32), min=0.0)
            # HF window: kv > q - W, i.e. start = pos - W + 1 (inclusive).
            swa_start = torch.clamp(pos - (self.sliding_window - 1), min=0.0)

        k_cache, v_cache = self.k_cache, self.v_cache
        table = block_table
        if nkh == 1:
            # 3D [blocks, block, d] cache + 2D table: kernel-eligible.
            k_cache = k_cache.squeeze(1) if k_cache.dim() == 4 else k_cache
            v_cache = v_cache.squeeze(1) if v_cache.dim() == 4 else v_cache
        elif can_run_kernel(X) and not _decode_torch_forced():
            # GQA reaches the kernel only with a per-head 3D table.
            table = build_per_head_block_table(block_table, nkh)

        decode_kwargs = dict(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_X_enabled=False,  # input norm applied by the decoder layer
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.qk_norm_eps,
            rmsnorm_QK_pre_rope_W_Q=None,
            rmsnorm_QK_pre_rope_W_K=None,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            active_blocks_table=table,
            K_cache=k_cache,
            V_cache=v_cache,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=None,  # gate must be applied before o_proj
        )
        if _decode_torch_forced():
            from vllm_neuron.functional.attention.attention_decode import (
                _torch_attention_decode_impl,
            )

            s_prior = table.shape[-1] * block_size
            mask = NF.gen_attention_decode_mask(
                pos_ids=pos.reshape(1, B * S),
                bs=B,
                q_head=self.num_attention_heads_per_rank,
                s_active=S,
                s_prior=s_prior,
                start_pos=swa_start.reshape(1, B * S) if swa_start is not None else None,
                block_len=block_size,
            )
            output, K_new, V_new = _torch_attention_decode_impl(
                attention_mask=mask, **decode_kwargs
            )
        else:
            output, K_new, V_new = NF.attention_decode(
                K_cache_transposed=False,
                pos_ids=pos,
                swa_start_pos_ids=swa_start,
                transposed_out=False,
                out_in_sb=False,
                **decode_kwargs,
            )

        # output: [B, Nq, d, S] -> [B*S, Nq*d]
        attn = output.permute(0, 3, 1, 2).reshape(B * S, self.q_size)
        gate = torch.sigmoid(hidden_states @ self.gate_proj_weight)
        attn = (attn * gate).to(self.dtype)
        out = attn @ self.o_proj_weight  # [B*S, H]

        # Manual KV scatter. K_new: [d, B*nkh, S] (torch) / [d, B, nkh, S] or
        # [d, B, S] (kernel); V_new: [B, nkh, S, d].
        k_new = K_new.reshape(d, B, nkh, S).permute(2, 1, 3, 0).reshape(nkh, B * S, d)
        v_new = V_new.reshape(B, nkh, S, d).permute(1, 0, 2, 3).reshape(nkh, B * S, d)
        self._write_paged_kv(k_new, v_new, slot_mapping, block_size)

        # >>> PARALLELISM: TP all-reduce of o_proj partials <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Section 4: MLP (SwiGLU)
# ---------------------------------------------------------------------------


class MuseGlimmerMLP(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError("intermediate_size must be divisible by TP")
        inter = config.intermediate_size // self.world_size
        H = config.hidden_size
        dt = config.torch_dtype
        self.gate_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=dt))
        self.up_proj_weight = nn.Parameter(torch.empty(H, inter, dtype=dt))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, H, dtype=dt))
        col = sharding_weight_loader(
            shard_dim=1, shard_size=inter, num_shards=self.world_size, is_storage_transposed=True
        )
        row = sharding_weight_loader(
            shard_dim=0, shard_size=inter, num_shards=self.world_size, is_storage_transposed=True
        )
        set_weight_loader(self.gate_proj_weight, col)
        set_weight_loader(self.up_proj_weight, col)
        set_weight_loader(self.down_proj_weight, row)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        out = NF.mlp(
            hidden_states, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight
        )
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                self.tp_group.all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Section 5: Decoder layer (sandwich norms)
# ---------------------------------------------------------------------------


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        H = config.hidden_size
        self.input_layernorm = MuseGlimmerCenteredRMSNorm(H, config.rms_norm_eps)
        self.post_attention_layernorm = MuseGlimmerCenteredRMSNorm(H, config.post_norm_eps)
        self.pre_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(H, config.rms_norm_eps)
        self.post_feedforward_layernorm = MuseGlimmerCenteredRMSNorm(H, config.post_norm_eps)
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = MuseGlimmerMLP(config)

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata, is_prefill):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, positions, position_embeddings, attn_metadata)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h

        residual = hidden_states
        h = self.pre_feedforward_layernorm(hidden_states)
        h = self.mlp(h, is_prefill=is_prefill)
        h = self.post_feedforward_layernorm(h)
        return residual + h


# ---------------------------------------------------------------------------
# Section 6: Backbone
# ---------------------------------------------------------------------------


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )
        # <-- MODEL-SPECIFIC: weightless norm right after the embedding lookup.
        self.embed_norm = MuseGlimmerRMSNorm(
            None, config.rms_norm_eps, config.torch_dtype, with_scale=False
        )
        self.layers = nn.ModuleList(
            [MuseGlimmerDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MuseGlimmerRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype, with_scale=True
        )
        self.rotary_emb = MuseGlimmerRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        meta0 = attn_metadata["layers.0.self_attn"]
        is_prefill = meta0["max_query_len"] > meta0["decode_token_threshold"]

        # >>> PARALLELISM: embedding reduce-scatters (SP) in prefill <<<
        h = self.embed_tokens(input_ids, scatter_tokens=is_prefill, rank=rank)
        h = self.embed_norm(h)

        # Vision features replace placeholder rows AFTER embed_norm (HF order).
        if is_prefill and vision_embedding_blocks is not None and vision_positions is not None:
            h, _ = merge_vision_embeddings(h, vision_embedding_blocks, vision_positions, rank=self.rank)

        position_embeddings = self.rotary_emb(positions)

        for layer in self.layers:
            h = layer(h, positions, position_embeddings, attn_metadata, is_prefill)

        h = self.norm(h)
        if is_prefill and self.world_size > 1:
            h = self.tp_group.all_gather(h, dim=0)
        return h


# ---------------------------------------------------------------------------
# Section 7: Top-level model (LM head, softcap, KV spec, weight loading)
# ---------------------------------------------------------------------------

HF_TEXT_PREFIX = "model.language_model"


class MuseGlimmerForConditionalGeneration(nn.Module):
    """MuseGlimmer multimodal model: text decoder + perception encoder."""

    def __init__(self, config: MuseGlimmerConfig):
        super().__init__()
        vnc = config.vision_config.neuron_config if config.vision_config else None
        self.mm_encoder_only = bool(vnc and vnc.mm_encoder_only)
        self.mm_language_model_only = bool(vnc and vnc.mm_language_model_only)
        if self.mm_encoder_only:
            raise NotImplementedError(
                "MuseGlimmer: mm_encoder_only (vision-only EPD pool) is not "
                "supported yet; run the monolithic model."
            )
        self.config = config
        self.text_config = config.text_config
        tc = config.text_config

        # Vision tower (skipped on a language-only pool).
        self.visual = None if self.mm_language_model_only else MuseGlimmerVisionModel(config)

        self.language_model = MuseGlimmerTextModel(tc)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = tc.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (nc is not None and nc.max_logprobs != 0) or debug_logits

        if tc.vocab_size % self.world_size != 0:
            raise ValueError(f"vocab_size={tc.vocab_size} not divisible by TP={self.world_size}")
        self.lm_head = neuron_nn.ColumnParallelLinear(
            tc.hidden_size,
            tc.vocab_size,
            bias=False,
            dtype=tc.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=tc.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=tc.hidden_size,
                unpadded_size=tc.hidden_size,
            ),
        )
        self.sampler = (
            Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)
            if self.on_device_sampling_config is not None
            else None
        )
        # <-- MODEL-SPECIFIC: logit multiplier + tanh softcap (Python floats).
        self.logit_softcap = float(tc.final_logit_softcapping)
        self.logit_pre_scale = float(tc.output_multiplier) / self.logit_softcap

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """``cap * tanh(logits * mult / cap)`` in fp32 (elementwise, so it
        commutes with vocab sharding and is applied before sampling and
        before the logprobs gather)."""
        return torch.tanh(logits.to(torch.float32) * self.logit_pre_scale) * self.logit_softcap

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        meta0 = attn_metadata["layers.0.self_attn"]
        is_prefill = meta0["max_query_len"] > meta0["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP.")

        h = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            vision_embedding_blocks=vision_embedding_blocks,
            vision_positions=vision_positions,
        )
        h = torch.index_select(h, dim=0, index=sampling_positions)
        logits = self._softcap(self.lm_head(h))

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled)
        return sampled, gathered_logits

    # ── multimodal ──────────────────────────────────────────────────────

    # no_grad: the runner calls this outside a grad-free context; without it the
    # encoder-cache index_put_ keeps every activation's autograd graph alive (CPU mode
    # leaked ~9 GB per 4096-patch image).
    @torch.no_grad()
    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Encode images and write their features into the on-device encoder cache.

        Flow (as qwen3_vl): allocate cache blocks per item -> pick a vision
        bucket -> one-item-per-block packing -> CPU preprocessing in the
        MuseGlimmer pack order -> VE graph scatter-writes into the cache.
        """
        if self.visual is None:
            raise RuntimeError("MuseGlimmer: vision tower not built (mm_language_model_only)")
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError("MuseGlimmer video input is Task 007.")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("embed_multimodal requires pixel_values and image_grid_thw")

        cache = encoder_cache
        vc = self.config.vision_config
        vnc = vc.neuron_config
        t0 = _time.perf_counter()
        mu = vc.merge_size**2
        block_size = vnc.vision_attention_block_size
        grid_thw = image_grid_thw.to(torch.int64).reshape(-1, 3)
        rows = grid_thw.tolist()
        for t, h, w in rows:
            if t * h * w > block_size:
                raise ValueError(
                    f"vision_attention_block_size={block_size} < image patches {t * h * w}; "
                    "an image must fit in one block. Increase the block size / bucket "
                    "or lower mm_processor_kwargs max_image_tokens."
                )
        tokens = [t * h * w for t, h, w in rows]
        if len(mm_hashes) != len(rows):
            raise ValueError(f"{len(mm_hashes)} mm_hashes for {len(rows)} images")

        block_map = [
            cache.allocate(mm_hashes[i], cache.dense_tokens_per_block(n // mu, block_size // mu))
            for i, n in enumerate(tokens)
        ]
        _bucket, num_blocks = select_vision_bucket(
            sum(tokens), vnc.num_vision_tokens_buckets, block_size, dp_size=vnc.dp_size
        )
        assignment = ffd_pack_images(tokens, block_size, num_blocks, one_item_per_block=True)
        packed = pack_vision_inputs(
            pixel_values.reshape(-1, vc.patch_dim),
            grid_thw,
            assignment,
            head_dim=vc.head_dim,
            rope_theta=vc.rope_theta,
            num_grid_per_side=vc.pos_emb_height,
            merge_size=vc.merge_size,
            rope_dtype=self.visual.dtype,
        )
        flat = [b for blocks in block_map for b in blocks]
        write_ids = torch.tensor(flat + [cache.scratch_block_id] * (num_blocks - len(flat)), dtype=torch.int64)
        t1 = _time.perf_counter()

        device = next(self.visual.parameters()).device
        kw = packed.as_kwargs()
        kw["pixel_values"] = kw["pixel_values"].to(self.visual.dtype)
        kw = {k: v.to(device) for k, v in kw.items()}
        self.visual(**kw, encoder_cache_buffer=cache.buffer, write_block_ids=write_ids.to(device))
        logger.debug(
            "[embed_multimodal] %d images, %d patches, %d blocks: cpu=%.1fms device=%.1fms",
            len(rows), sum(tokens), num_blocks, (t1 - t0) * 1e3, (_time.perf_counter() - t1) * 1e3,
        )

    def build_vision_synthetic_inputs(
        self, bucket: int, vision_neuron_config: VisionNeuronConfig, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Shape-only inputs for ``self.visual`` warmup / tracing (per bucket)."""
        vc = self.config.vision_config
        bs = vision_neuron_config.vision_attention_block_size
        dp = vision_neuron_config.dp_size
        nb = math.ceil(math.ceil(bucket / bs) / dp) * dp
        mu = vc.merge_size**2
        z = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        return {
            "pixel_values": z(nb, bs, vc.patch_dim, dtype=self.visual.dtype),
            "pos_emb_idx": z(4, nb, bs, dtype=torch.int32),
            "pos_emb_weight": z(4, nb, bs, dtype=torch.float32),
            "cos": z(nb, bs, vc.head_dim, dtype=torch.float32),
            "sin": z(nb, bs, vc.head_dim, dtype=torch.float32),
            "bound_min": z(nb, bs, 1, dtype=torch.int32),
            "bound_max": z(nb, bs, 1, dtype=torch.int32),
            "window_bound_min": z(nb, bs, 1, dtype=torch.int32),
            "window_bound_max": z(nb, bs, 1, dtype=torch.int32),
            "merged_order": torch.arange(bs // mu, dtype=torch.int64, device=device).repeat(nb, 1)
            if device != torch.device("meta")
            else z(nb, bs // mu, dtype=torch.int64),
        }

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ):
        config = MuseGlimmerConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    # ── KV cache ────────────────────────────────────────────────────────

    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, layer in enumerate(self.language_model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=attn.num_key_value_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    # <-- MODEL-SPECIFIC: per-layer SWA -> SlidingWindowSpec
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.language_model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise Exception(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    # ── weights ─────────────────────────────────────────────────────────

    def build_text_weight_mappings(self) -> dict[str, str | list[str]]:
        m: dict[str, str | list[str]] = {}
        for i in range(len(self.language_model.layers)):
            hf = f"{HF_TEXT_PREFIX}.layers.{i}"
            ours = f"language_model.layers.{i}"
            m[f"{ours}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ]
            m[f"{ours}.self_attn.gate_proj_weight"] = f"{hf}.self_attn.gate_proj.weight"
            m[f"{ours}.self_attn.o_proj_weight"] = f"{hf}.self_attn.o_proj.weight"
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                m[f"{ours}.{n}.weight"] = f"{hf}.{n}.weight"
            for p in ("gate_proj", "up_proj", "down_proj"):
                m[f"{ours}.mlp.{p}_weight"] = f"{hf}.mlp.{p}.weight"
        m["language_model.embed_tokens.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        m["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"
        m["lm_head.weight"] = "lm_head.weight"
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir: str | None) -> None:
        mappings = self.build_text_weight_mappings()
        params = {n: p for n, p in self.named_parameters() if not n.startswith("visual.")}
        unmapped = sorted(set(params) - set(mappings))
        if unmapped:
            raise RuntimeError(f"MuseGlimmer: parameters without a checkpoint mapping: {unmapped}")

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        visual, self.visual = self.visual, None  # text-TP load must not see vision params
        try:
            sd = checkpoint.load_sharded_pipelined(
                self.rank, self.world_size, self, mappings, device, strict=True
            ).state_dict
        finally:
            self.visual = visual
        # Keep each parameter's own dtype: bf16 weights, fp32 centered-norm w.
        for name, tensor in sd.items():
            want = params[name].dtype
            if tensor.dtype != want:
                sd[name] = tensor.to(want)
        missing = sorted(set(params) - set(sd))
        if missing:
            raise RuntimeError(f"MuseGlimmer: checkpoint did not provide: {missing}")
        self.load_state_dict(sd, strict=False, assign=True)
        # Vision tower (model.vision_tower.*, vision_adapter, vision_projection) on the
        # vision TP group (DP: full copy per rank).
        if self.visual is not None:
            self.visual.load_weights(checkpoint_path, device="cpu", cache_dir=cache_dir)


def per_tower_param_bytes(model) -> dict[str, int]:
    def _bytes(module) -> int:
        return 0 if module is None else sum(p.nbytes for p in module.parameters())

    return {
        "visual_param_bytes": _bytes(getattr(model, "visual", None)),
        "language_model_param_bytes": _bytes(getattr(model, "language_model", None)),
    }
