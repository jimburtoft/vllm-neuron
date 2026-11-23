# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5-MoE (text decoder) for vLLM-Neuron 0.29
================================================

Hybrid decoder: 30 Gated DeltaNet (GDN) layers + 10 full-attention GQA layers
(3:1 pattern from ``layer_types``), every layer followed by a 256-expert top-8
MoE with a sigmoid-gated shared expert. Reference: Qwen/Qwen3.5-35B-A3B.

Hybrid state design (0.29 runner has no mamba/GDN plumbing):
  * Full-attention layers use the runner's paged KV cache (FullAttentionSpec).
  * GDN recurrent + conv state is MODEL-OWNED: one ``register_buffer`` slot pool
    per GDN layer, ``[num_slots, ...]``. The runner gives each live request a
    stable slot (``uses_request_slots``) and passes ``request_slots`` [B] int32
    in the attention metadata. The last slot is a scratch slot for padded rows.
    Writes are ``index_put_`` on dim 0 of the 3/4-D buffer; the runner's
    aliasing FX pass persists them in HBM.
  * Prefill seeds a fresh request from a zero state (prefix caching is off);
    a chunked prefill continues from the slot state (``cached_seq_len > 0``).

Parallelism: TP over heads (attention Q heads, GDN k/v heads), KV heads
replicated when TP > num_kv_heads. Prefill runs sequence-parallel (embedding
reduce-scatters, each mixer all-gathers / reduce-scatters). Decode keeps the
full token set on every rank and all-reduces. MoE: TP-sharded intermediate
(ep_degree=1) or EP over the whole TP group.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   reusable parallelism code
  # <-- MODEL-SPECIFIC: ...    Qwen3.5-specific
"""

from __future__ import annotations

import logging
import math
import os

import nki.language as nl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from .kernels import nkilib_patch as _nkilib_patch

_NKILIB_PATCHED = _nkilib_patch.install()
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from vllm_neuron.functional.moe.router import RouterComputationOrder
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    expert_parallel_tensor_dim_loader,
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from vllm.model_executor.models.interfaces import SupportsEagle3

from vllm_neuron.vllm.spec_decode.decorator import async_speculative_decoding

from .config import Qwen3_5MoeTextConfig

logger = logging.getLogger(__name__)

HF_PREFIX = "model.language_model"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


# =============================================================================
# Section 1: Normalization
# <-- MODEL-SPECIFIC: Qwen3.5 decoder norms are (1 + w) * rms(x) (HF stores w
#     zero-centred). The +1 is folded into the weight at load time so the plain
#     ``w * rms(x)`` form (and any kernel gamma) matches. The GDN output norm is
#     a plain gated RMSNorm (no +1) with fp32 weights.
# =============================================================================


def _plus_one_loader():
    return SafetensorsWeightLoader(transform=lambda s, r: s[0][:].float() + 1.0)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (xf * self.weight.float()).to(in_dtype)


class GatedRMSNorm(nn.Module):
    """HF Qwen3_5MoeRMSNormGated: w * rms(x) (cast to input dtype) * silu(z)."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        y = self.weight * xf.to(in_dtype)
        y = y * F.silu(z.float())
        return y.to(in_dtype)


# =============================================================================
# Section 2: Partial rotary embedding
# <-- MODEL-SPECIFIC: rope_theta=1e7, only the first rotary_dim (=64) of the
#     256-dim head rotates, rotate_half (half-split) pairing. Text-only, so the
#     M-RoPE sections collapse to 1-D positions.
# =============================================================================


class PartialRotary(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.rope_theta = float(config.rope_theta)
        self.rotary_dim = config.rotary_dim
        # Isaac-0.5 quantizes inv_freq to bf16 before the fp32 phase product (checkpoint_bf16_inv_freq_fp32_phase_v1)
        self.inv_freq_bf16 = bool(getattr(config, "null_router", False))

    def forward(self, positions: torch.Tensor):
        d = self.rotary_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, d, 2, dtype=torch.float32, device=positions.device) / d)
        )
        if self.inv_freq_bf16:
            inv_freq = inv_freq.to(torch.bfloat16).float()
        freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, d]
        return emb.cos(), emb.sin()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _apply_partial_rope(x, cos, sin, rotary_dim):
    """x [..., T, Dh]; cos/sin [T, rotary_dim] (fp32)."""
    xr = x[..., :rotary_dim].float()
    xp = x[..., rotary_dim:]
    xr = xr * cos + _rotate_half(xr) * sin
    return torch.cat((xr.to(x.dtype), xp), dim=-1)


# =============================================================================
# Section 3: Full attention (GQA, QK-norm, output gate, partial RoPE, d=256)
# Prefill uses the paged-window torch attention below (d=256 exceeds the
# 128-wide NKI flash kernel). Decode gathers the paged window, splices the live
# K/V at its position, and attends in fp32 (online softmax over 512-key tiles).
# =============================================================================


class Qwen3_5Attention(nn.Module):
    CTX_TILE = 512

    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.scaling = self.head_dim ** -0.5
        self.attn_decode_impl = os.environ.get("QWEN35_ATTN_DECODE", "torch")

        # >>> PARALLELISM: TP head sharding, KV replicated when TP > nkv <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        nq, nkv = config.num_attention_heads, config.num_key_value_heads
        assert nq % self.world_size == 0
        self.nqh = nq // self.world_size
        if self.world_size >= nkv:
            self.nkh = 1
            self.num_kv_replicas = self.world_size // nkv
        else:
            self.nkh = nkv // self.world_size
            self.num_kv_replicas = 1
        self.n_rep = self.nqh // self.nkh

        # <-- MODEL-SPECIFIC: q_proj emits [query | gate] per head (2*Dh)
        q_size = self.nqh * self.head_dim * 2
        kv_size = self.nkh * self.head_dim
        self.q_size, self.kv_size = q_size, kv_size
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, q_size + 2 * kv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.nqh * self.head_dim, self.hidden_size, dtype=self.dtype)
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, torch.float32)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, torch.float32)
        set_weight_loader(self.q_norm.weight, _plus_one_loader())
        set_weight_loader(self.k_norm.weight, _plus_one_loader())

        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.nqh * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        self.k_cache = None
        self.v_cache = None

    # -- shared projection --------------------------------------------------
    def _project(self, x, cos, sin):
        T = x.shape[0]
        qkv = x @ self.qkv_proj_weight  # [T, q+2kv]
        qg, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=-1)
        qg = qg.view(T, self.nqh, 2 * self.head_dim)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        q = self.q_norm(q.reshape(T, self.nqh, self.head_dim)).transpose(0, 1)  # [Nh,T,D]
        k = self.k_norm(k.reshape(T, self.nkh, self.head_dim)).transpose(0, 1)  # [Nk,T,D]
        v = v.reshape(T, self.nkh, self.head_dim).transpose(0, 1)
        q = _apply_partial_rope(q, cos, sin, self.rotary_dim)
        k = _apply_partial_rope(k, cos, sin, self.rotary_dim)
        return q, k, v, gate.reshape(T, self.nqh * self.head_dim)

    def _write_kv(self, k, v, slot_mapping, block_size):
        T = slot_mapping.shape[0]
        bi, pi = paged_slot_indices(slot_mapping, block_size, self.k_cache.shape[0])
        hi = torch.arange(self.nkh, dtype=torch.long, device=k.device).repeat_interleave(T)
        self.k_cache.index_put_(
            (bi.repeat(self.nkh), hi, pi.repeat(self.nkh)),
            k.reshape(-1, self.head_dim).to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (bi.repeat(self.nkh), hi, pi.repeat(self.nkh)),
            v.reshape(-1, self.head_dim).to(self.v_cache.dtype),
        )

    def _gather_window(self, block_table, B, block_size):
        nblk = block_table.shape[1]
        # Unused / padded block-table entries carry the -1 read sentinel. Point them at
        # block 0; the absolute-position causal mask excludes their keys.
        flat = block_table.reshape(-1).long().clamp(min=0)
        Kb = self.k_cache[flat].view(B, nblk, self.nkh, block_size, self.head_dim)
        Vb = self.v_cache[flat].view(B, nblk, self.nkh, block_size, self.head_dim)
        S = nblk * block_size
        if self.nkh == 1:
            return Kb.reshape(B, 1, S, self.head_dim), Vb.reshape(B, 1, S, self.head_dim)
        return (
            Kb.permute(0, 2, 1, 3, 4).reshape(B, self.nkh, S, self.head_dim),
            Vb.permute(0, 2, 1, 3, 4).reshape(B, self.nkh, S, self.head_dim),
        )

    def _attend(self, q, K, V, qpos, qlo=None):
        """q [B,Nh,Sq,D]; K/V [B,Nk,S,D]; qpos [B,1,Sq,1] float absolute positions.
        Causal by absolute position; fp32 online softmax over CTX_TILE keys, QK^T
        contracted in two 128-wide halves (no 256-partition transpose)."""
        B, Nh, Sq, D = q.shape
        S = K.shape[2]
        if self.n_rep > 1:
            K = K.repeat_interleave(self.n_rep, dim=1)
            V = V.repeat_interleave(self.n_rep, dim=1)
        qf = q.float() * self.scaling
        h = D // 2
        neg = -1e30
        m = torch.full((B, Nh, Sq, 1), neg, device=q.device, dtype=torch.float32)
        l = torch.zeros((B, Nh, Sq, 1), device=q.device, dtype=torch.float32)
        acc = torch.zeros((B, Nh, Sq, D), device=q.device, dtype=torch.float32)
        for c0 in range(0, S, self.CTX_TILE):
            c1 = min(c0 + self.CTX_TILE, S)
            Kc = K[:, :, c0:c1].float()
            Vc = V[:, :, c0:c1].float()
            idx = torch.arange(c0, c1, device=q.device, dtype=torch.float32).view(1, 1, 1, -1)
            msk = idx <= qpos
            if qlo is not None:  # packed prefill: key must belong to the query's own request
                msk = msk & (idx >= qlo)
            s = qf[..., :h] @ Kc[..., :h].transpose(-1, -2) + qf[..., h:] @ Kc[..., h:].transpose(-1, -2)
            s = s.masked_fill(~msk, neg)
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new).masked_fill(~msk, 0.0)
            a = torch.exp(m - m_new)
            l = l * a + p.sum(-1, keepdim=True)
            acc = acc * a + p @ Vc
            m = m_new
        return acc / l.clamp_min(1e-30)

    def forward(self, hidden_states, positions, cos, sin, attn_metadata, is_decode, dsp=False, psp=True, pk=None):
        md = attn_metadata[self.layer_name]
        if is_decode:
            if dsp and self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_decode(hidden_states, positions, cos, sin, md, rs_out=dsp)
        if not psp:
            return self.forward_prefill(hidden_states, positions, cos, sin, md, sp=False, ar=True, pk=pk)
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(hidden_states, positions, cos, sin, md, pk=pk)

    def forward_prefill(self, x, positions, cos, sin, md, sp=True, ar=False, pk=None):
        x = x.to(self.dtype)
        T = x.shape[0]
        q, k, v, gate = self._project(x, cos, sin)
        block_size = md["block_size"]
        if pk is not None and md.get("seg"):
            # Action path, batched: B observations packed at a fixed stride seg -> per-observation causal
            # attention as a batch of B (no cross-observation key tiles computed; same _attend math as "fresh").
            seg = md["seg"]
            Bq = T // seg
            kd = self.k_cache.dtype
            # q/k/v are [Nh, T, D] -> [B, Nh, seg, D]
            qb = q.reshape(q.shape[0], Bq, seg, q.shape[2]).transpose(0, 1)
            kb = k.reshape(k.shape[0], Bq, seg, k.shape[2]).transpose(0, 1).to(kd)
            vb = v.reshape(v.shape[0], Bq, seg, v.shape[2]).transpose(0, 1).to(kd)
            qpos = torch.arange(seg, device=x.device, dtype=torch.float32).view(1, 1, seg, 1).expand(Bq, 1, seg, 1)
            out = self._attend(qb, kb, vb, qpos)  # [B, Nh, seg, D]
            out = out.transpose(1, 2).reshape(T, -1).to(self.dtype)  # rows b * seg + i, cols (head, D)
            out = out * torch.sigmoid(gate)
            out = out @ self.o_proj_weight
            if self.world_size > 1:
                if ar:
                    out = self.tp_group.all_reduce(out)
                else:
                    out = self.tp_group.reduce_scatter(out, dim=0)
                    if not sp:
                        out = self.tp_group.all_gather(out, dim=0)
            return out
        if pk is not None:
            # Packed prefill: every packed request starts at position 0 (no cached prefix), so attend over
            # the live keys only, causal and block-diagonal: query i sees keys [i - pos_i, i].
            idx = torch.arange(T, device=x.device, dtype=torch.float32)
            qpos = idx.view(1, 1, T, 1)
            qlo = pk["lo"].view(1, 1, T, 1)
            # Keys padded (zeros, masked) to the cached-window length S, in the cache dtype: the same CTX_TILE
            # partition and operand rounding as the unpacked path, so a 1-request step is bit-identical to K=1.
            kd = self.k_cache.dtype
            Kp, Vp = k.unsqueeze(0).to(kd), v.unsqueeze(0).to(kd)
            S = md["block_table_tensor"].shape[1] * block_size if "block_table_tensor" in md else T
            if _env_flag("QWEN35_PACKED_ATTN_PAD", "0") and S > T:
                Kp, Vp = F.pad(Kp, (0, 0, 0, S - T)), F.pad(Vp, (0, 0, 0, S - T))
            out = self._attend(q.unsqueeze(0), Kp, Vp, qpos, qlo)  # [1,Nh,T,D]
            out = out[0].transpose(0, 1).reshape(T, -1).to(self.dtype)
            out = out * torch.sigmoid(gate)
            # Padding tokens all write the null block; zero them so its contents never depend on the prompts.
            if not md.get("skip_kv_write", False):  # the action path keeps no KV
                rk = pk["real"].to(k.dtype).view(1, T, 1)
                self._write_kv(k * rk, v * rk, md["slot_mapping"], block_size)
            out = out @ self.o_proj_weight
            if self.world_size > 1:
                if ar:
                    out = self.tp_group.all_reduce(out)
                else:
                    out = self.tp_group.reduce_scatter(out, dim=0)
                    if not sp:
                        out = self.tp_group.all_gather(out, dim=0)
            return out
        if md.get("fresh", False):
            # Action path: a prefill with no cached prefix (position 0, one request) attends over the live
            # keys only -- the cached-window gather + K/V scatter (~28 ms of semaphore waits over 10 layers at T=512)
            # is skipped. Same causal-by-position mask; padding queries are discarded downstream.
            qpos = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T, 1)
            kd = self.k_cache.dtype
            out = self._attend(q.unsqueeze(0), k.unsqueeze(0).to(kd), v.unsqueeze(0).to(kd), qpos)
            out = out[0].transpose(0, 1).reshape(T, -1).to(self.dtype)
            out = out * torch.sigmoid(gate)
            if not md.get("skip_kv_write", False):
                self._write_kv(k, v, md["slot_mapping"], block_size)
            out = out @ self.o_proj_weight
            if self.world_size > 1:
                if ar:
                    out = self.tp_group.all_reduce(out)
                else:
                    out = self.tp_group.reduce_scatter(out, dim=0)
                    if not sp:
                        out = self.tp_group.all_gather(out, dim=0)
            return out
        block_table = md["block_table_tensor"][:1]  # one prefill request per step
        # Attend over [cached prior window + live chunk]. The window is read BEFORE
        # this step's write; the live K/V are spliced at sequence-order slots so the
        # right-padding (positions repeat the last real one) never overwrites a real
        # token's slot. Padding queries are discarded downstream.
        K, V = self._gather_window(block_table, 1, block_size)
        S = K.shape[2]
        base = positions[:1].long()
        slot = (base + torch.arange(T, device=x.device, dtype=torch.long)).clamp(min=0, max=S - 1)
        sidx = slot.view(1, 1, T, 1).expand(1, self.nkh, T, self.head_dim)
        K = K.scatter(2, sidx, k.unsqueeze(0).to(K.dtype))
        V = V.scatter(2, sidx, v.unsqueeze(0).to(V.dtype))
        qpos = slot.float().view(1, 1, T, 1)
        out = self._attend(q.unsqueeze(0), K, V, qpos)  # [1,Nh,T,D]
        out = out[0].transpose(0, 1).reshape(T, -1).to(self.dtype)
        out = out * torch.sigmoid(gate)
        self._write_kv(k, v, md["slot_mapping"], block_size)
        out = out @ self.o_proj_weight
        # >>> PARALLELISM: reduce-scatter back to SP layout (all-reduce if not SP) <<<
        if self.world_size > 1:
            if ar:
                out = self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
                if not sp:  # same collectives as the SP path, then restore all tokens
                    out = self.tp_group.all_gather(out, dim=0)
        return out

    def forward_decode(self, x, positions, cos, sin, md, rs_out=False):
        x = x.to(self.dtype)
        block_table = md["block_table_tensor"]
        block_size = md["block_size"]
        B = block_table.shape[0]
        T = x.shape[0]
        Sd = T // B
        q, k, v, gate = self._project(x, cos, sin)
        K, V = self._gather_window(block_table, B, block_size)
        S = K.shape[2]
        # Padded rows (async spec decode, shrinking batch) can carry out-of-range
        # corrected positions: clamp both ends before using them as indices.
        pos = positions.long().clamp(min=0, max=S - 1)
        sidx = pos.view(B, 1, Sd, 1).expand(B, self.nkh, Sd, self.head_dim)
        kb = k.reshape(self.nkh, B, Sd, self.head_dim).permute(1, 0, 2, 3)
        vb = v.reshape(self.nkh, B, Sd, self.head_dim).permute(1, 0, 2, 3)
        K = K.scatter(2, sidx, kb.to(K.dtype))
        V = V.scatter(2, sidx, vb.to(V.dtype))
        qb = q.reshape(self.nqh, B, Sd, self.head_dim).permute(1, 0, 2, 3)
        if self.attn_decode_impl == "d256" and Sd == 1 and S % 128 == 0:
            # attention-decode-d256 flash-decode kernel (GQA heads split over LNC).
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            from vllm_neuron.utils.hardware_config import get_logical_nc_config

            from .kernels.jb_attn_decode_d256 import _nki_attention_decode_kernel

            kv_g = self.nqh // self.nkh
            grid = get_logical_nc_config() if kv_g % get_logical_nc_config() == 0 else 1
            idx = torch.arange(S, device=x.device).view(1, 1, 1, S)
            mask = torch.where(idx <= pos.view(B, 1, 1, 1), 0.0, -1.0e30).to(torch.float32)
            o = wrap_nki(_nki_attention_decode_kernel)[grid](
                q=qb.squeeze(2).contiguous(), k_cache=K.transpose(-1, -2).contiguous(),
                v_cache=V.contiguous(), attn_mask=mask, scale=float(self.scaling),
            )
            out = o.unsqueeze(2).float()
        else:
            out = self._attend(qb, K, V, pos.float().view(B, 1, Sd, 1))
        out = out.permute(0, 2, 1, 3).reshape(T, -1).to(self.dtype)
        out = out * torch.sigmoid(gate)
        self._write_kv(k, v, md["slot_mapping"], block_size)
        out = out @ self.o_proj_weight
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out


# =============================================================================
# Section 4: Gated DeltaNet (linear attention) with model-owned slot state
# =============================================================================


def _permute_tiles(x, order):
    """x.index_select(1, order) for a static order, as slices + cat of contiguous runs
    (a constant-index gather lowers to a slow GpSimd gather)."""
    runs, s = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or order[i] != order[i - 1] + 1:
            runs.append(x[:, order[s] : order[i - 1] + 1])
            s = i
    return runs[0] if len(runs) == 1 else torch.cat(runs, 1)


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _gdn_cte_call(q, k, v, g, beta, state, mask_pack, scale):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.gdn import gdn_cte

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    out = wrap_nki(gdn_cte)[get_logical_nc_config()](
        query=q, key=k, value=v, g_log=g, beta=beta, scale=float(scale),
        algorithm="schur", mask_pack=mask_pack, group_size=8, recurrent_state=state,
    )
    return out[0], out[1]


def _gdn_cte_packed_call(q, k, v, g, beta, mask_pack, chunk_keep, scale):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_neuron.utils.hardware_config import get_logical_nc_config

    from .kernels.gdn_cte_packed import gdn_cte_packed

    out = wrap_nki(gdn_cte_packed)[get_logical_nc_config()](
        query=q, key=k, value=v, g_log=g, beta=beta, mask_pack=mask_pack, chunk_keep=chunk_keep, scale=float(scale),
    )
    return out[0], out[2]


def _chunked_gdn(q, k, v, g, beta, state, chunk=64):
    """Chunked gated delta rule (HF torch_chunk_gated_delta_rule math).

    q,k [B,H,S,Dk] (l2-normed, q pre-scaled), v [B,H,S,Dv], g/beta [B,H,S] fp32,
    state [B,H,Dk,Dv] fp32. S must be a multiple of ``chunk``. Returns
    (out [B,H,S,Dv] fp32, final state). The intra-chunk (I+A)^-1 is formed by
    forward substitution on the (chunk x chunk) strictly-lower block (a static
    Python loop, unrolled at trace time).
    """
    B, H, S, Dk = k.shape
    Dv = v.shape[-1]
    n = S // chunk
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    qc, kc, kbc, vbc = (t.reshape(B, H, n, chunk, t.shape[-1]) for t in (q, k, k_beta, v_beta))
    gc = g.reshape(B, H, n, chunk).cumsum(-1)
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device)
    upper_incl = tri.triu(0)
    upper_strict = tri.triu(1)
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).masked_fill(upper_strict, float("-inf")).exp()
    attn = -((kbc @ kc.transpose(-1, -2)) * decay).masked_fill(upper_incl, 0.0)
    # forward substitution: rows i depend on rows < i
    rows = [attn[..., 0:1, :]]
    for i in range(1, chunk):
        row = attn[..., i : i + 1, :]  # [B,H,n,1,C]
        prev = torch.cat(rows, dim=-2)  # [B,H,n,i,C], rows < i already updated
        rows.append(row + row[..., :i] @ prev)
    attn = torch.cat(rows, dim=-2) + torch.eye(chunk, dtype=attn.dtype, device=q.device)
    value = attn @ vbc
    k_cumdecay = attn @ (kbc * gc.exp().unsqueeze(-1))
    outs = []
    intra_mask = upper_strict
    for i in range(n):
        q_i, k_i, v_i = qc[:, :, i], kc[:, :, i], value[:, :, i]
        a_i = ((q_i @ k_i.transpose(-1, -2)) * decay[:, :, i]).masked_fill(intra_mask, 0.0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        inter = (q_i * gc[:, :, i, :, None].exp()) @ state
        outs.append(inter + a_i @ v_new)
        g_last = gc[:, :, i, -1, None, None]
        state = state * g_last.exp() + (
            k_i * (gc[:, :, i, -1, None] - gc[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    return torch.stack(outs, dim=2).reshape(B, H, S, Dv), state


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, layer_idx: int, num_slots: int, n_ckpt: int = 1):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.Dk = config.linear_key_head_dim
        self.Dv = config.linear_value_head_dim
        self.conv_k = config.linear_conv_kernel_dim
        # >>> PARALLELISM: shard k/v heads over TP <<<
        self.tp_group = get_tp_group()
        self.world_size = tp = self.tp_group.world_size
        Hk, Hv = config.linear_num_key_heads, config.linear_num_value_heads
        assert Hk % tp == 0 and Hv % tp == 0
        self.Hk, self.Hv = Hk // tp, Hv // tp
        self.rep = self.Hv // self.Hk
        self.key_dim = self.Hk * self.Dk
        self.value_dim = self.Hv * self.Dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        full_key, full_val = Hk * self.Dk, Hv * self.Dv

        dt = self.dtype
        self.in_proj_qkv_weight = nn.Parameter(torch.empty(self.hidden_size, self.conv_dim, dtype=dt))
        self.in_proj_z_weight = nn.Parameter(torch.empty(self.hidden_size, self.value_dim, dtype=dt))
        self.in_proj_ba_weight = nn.Parameter(torch.empty(self.hidden_size, 2 * self.Hv, dtype=dt))
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, self.conv_k, dtype=dt))
        self.dt_bias = nn.Parameter(torch.zeros(self.Hv, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.Hv, dtype=torch.float32))
        self.norm = GatedRMSNorm(self.Dv, config.rms_norm_eps)
        self.out_proj_weight = nn.Parameter(torch.empty(self.value_dim, self.hidden_size, dtype=dt))

        # Model-owned state slot pools (fp32 recurrent, bf16 conv). conv keeps the
        # last (k-1) mixed_qkv columns. Each slot holds n_ckpt = 1 + K checkpoints,
        # indexed by position: the state after the token at position p lives in
        # checkpoint p % n_ckpt. A step whose first position is p reads checkpoint
        # (p - 1) % n_ckpt, i.e. the state after the last ACCEPTED token, so
        # speculative-decode rollback needs no host round trip. (No separate base
        # buffer: every write depends on a value read in the same graph, which keeps
        # the device from reordering the write ahead of the read.)
        self.n_ckpt = n_ckpt
        # HBM storage dtype of the recurrent state (compute is always fp32).
        rec_dt = torch.bfloat16 if os.environ.get("QWEN35_GDN_STATE_DTYPE", "fp32") == "bf16" else torch.float32
        self.register_buffer(
            "rec_state",
            torch.zeros(num_slots, n_ckpt, self.Hv, self.Dk, self.Dv, dtype=rec_dt),
            persistent=False,
        )
        # QWEN35_DECODE_MK=1: the decode megakernel keeps the conv state in its native
        # layout, fp32 [slots, ckpt, NC*128*(K-1)*CT] (per slot [NC, 128, K-1, CT], tt012 tile order).
        # _read_state / _write_states convert for the prefill and S>1 paths.
        self.mk_layout = _env_flag("QWEN35_DECODE_MK") and not _env_flag("QWEN35_DECODE_MK_PORTCONV")
        if _env_flag("QWEN35_DECODE_MK_PORTCONV"):  # A/B only (decode MK must be skipped): tt012 params, port conv
            self.decode_impl = "tt012_params"
        conv_shape = (num_slots, n_ckpt, self.conv_dim * (self.conv_k - 1)) if self.mk_layout else (
            num_slots, n_ckpt, self.conv_dim, self.conv_k - 1)
        self.register_buffer(
            "conv_state",
            torch.zeros(*conv_shape, dtype=torch.float32 if self.mk_layout else dt),
            persistent=False,
        )
        if self.mk_layout:  # port-layout copy read by prefill (written by every _write_states)
            self.register_buffer(
                "conv_state_pf", torch.zeros(num_slots, n_ckpt, self.conv_dim, self.conv_k - 1, dtype=dt),
                persistent=False,
            )


        from vllm_neuron import envs as _envs

        self.decode_impl = os.environ.get("QWEN35_GDN_DECODE", "torch")
        self.prefill_impl = os.environ.get("QWEN35_GDN_PREFILL", "nki")
        if _envs.VLLM_NEURON_CPU_MODE:
            self.prefill_impl = "torch"
        self.use_nki_prefill = self.prefill_impl == "nki"
        self.dstair_chunk = int(os.environ.get("QWEN35_DSTAIR_PREFILL_C", "16"))
        kpr, vpr = self.key_dim, self.value_dim

        def qkv_t(s, r):  # HF [conv_dim_full, H] -> [H, conv_dim_local] (q|k|v head slices)
            w = s[0]
            r = r % tp
            parts = [
                w[r * kpr : (r + 1) * kpr, :],
                w[full_key + r * kpr : full_key + (r + 1) * kpr, :],
                w[2 * full_key + r * vpr : 2 * full_key + (r + 1) * vpr, :],
            ]
            return torch.cat(parts, 0).t().contiguous()

        def conv_t(s, r):  # HF [conv_dim_full, 1, K] -> [conv_dim_local, K]
            w = s[0]
            r = r % tp
            parts = [
                w[r * kpr : (r + 1) * kpr, :, :],
                w[full_key + r * kpr : full_key + (r + 1) * kpr, :, :],
                w[2 * full_key + r * vpr : 2 * full_key + (r + 1) * vpr, :, :],
            ]
            return torch.cat(parts, 0)[:, 0, :].contiguous()

        def rows_t(n):  # HF [N*X, H] -> [H, N*X/tp]
            def f(s, r):
                r = r % tp
                return s[0][r * n : (r + 1) * n, :].t().contiguous()
            return f

        def ba_t(s, r):  # [b | a] fused: HF in_proj_b, in_proj_a [Hv_full, H]
            r = r % tp
            b = s[0][r * self.Hv : (r + 1) * self.Hv, :]
            a = s[1][r * self.Hv : (r + 1) * self.Hv, :]
            return torch.cat([b, a], 0).t().contiguous()

        def vec_t(s, r):
            r = r % tp
            return s[0][r * self.Hv : (r + 1) * self.Hv].float()

        def out_t(s, r):  # HF [H, value_full] -> [value_local, H]
            r = r % tp
            return s[0][:, r * vpr : (r + 1) * vpr].t().contiguous()

        set_weight_loader(self.in_proj_qkv_weight, SafetensorsWeightLoader(transform=qkv_t))
        set_weight_loader(self.conv1d_weight, SafetensorsWeightLoader(transform=conv_t))
        set_weight_loader(self.in_proj_z_weight, SafetensorsWeightLoader(transform=rows_t(vpr)))
        set_weight_loader(self.in_proj_ba_weight, SafetensorsWeightLoader(transform=ba_t))
        set_weight_loader(self.dt_bias, SafetensorsWeightLoader(transform=vec_t))
        set_weight_loader(self.A_log, SafetensorsWeightLoader(transform=vec_t))
        set_weight_loader(self.out_proj_weight, SafetensorsWeightLoader(transform=out_t))

        if self.decode_impl in ("tt012", "tt012_params") or self.mk_layout:
            self._init_tt012(full_key, kpr, vpr)

    def _init_tt012(self, full_key, kpr, vpr):
        """GDN decode kernel: per-pcore weight copies in the kernel-native order
        (in_proj columns [q | k | v | z | b | a] of the pcore's heads; conv weights [p, w, t])."""
        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        nc = self.tt_nc = get_logical_nc_config()
        kc, vc, hc = kpr // nc, vpr // nc, self.Hv // nc
        tp, Hv = self.world_size, self.Hv
        ct = (2 * kc + vc) // 128
        self.tt_ct = ct
        nin = 128 * ct + vc + 2 * hc

        def qkv_rows(r, c):
            return [(r * kpr + c * kc, kc), (full_key + r * kpr + c * kc, kc), (2 * full_key + r * vpr + c * vc, vc)]

        def win_t(s, r):  # slices: qkv [Cf, H], z [Vf, H], b [Hvf, H], a [Hvf, H]
            r = r % tp
            per = []
            for c in range(nc):
                parts = [s[0][o : o + n, :] for o, n in qkv_rows(r, c)]
                parts.append(s[1][r * vpr + c * vc : r * vpr + (c + 1) * vc, :])
                parts.append(s[2][r * Hv + c * hc : r * Hv + (c + 1) * hc, :])
                parts.append(s[3][r * Hv + c * hc : r * Hv + (c + 1) * hc, :])
                per.append(torch.cat(parts, 0).t().contiguous())
            return torch.stack(per, 0).reshape(-1)

        def cw_t(s, r):  # HF [Cf, 1, K] -> [nc, 128(p), K(w), ct(t)] fp32 flat
            r = r % tp
            per = []
            for c in range(nc):
                w = torch.cat([s[0][o : o + n, 0, :] for o, n in qkv_rows(r, c)], 0).float()
                per.append(w.reshape(ct, 128, self.conv_k).permute(1, 2, 0))
            return torch.stack(per, 0).reshape(-1).contiguous()

        self.tt_win = nn.Parameter(torch.empty(nc * self.hidden_size * nin, dtype=self.dtype))
        self.tt_cw = nn.Parameter(torch.empty(nc * 128 * self.conv_k * ct, dtype=torch.float32))
        set_weight_loader(self.tt_win, SafetensorsWeightLoader(transform=win_t))
        set_weight_loader(self.tt_cw, SafetensorsWeightLoader(transform=cw_t))
        # conv-channel 128-tile order of the kernel (per pcore [q | k | v]) in local [q | k | v] tiles
        qt, vt = kc // 128, vc // 128
        kq0, v0 = self.key_dim // 128, 2 * self.key_dim // 128
        order = []
        for c in range(nc):
            order += list(range(c * qt, (c + 1) * qt)) + list(range(kq0 + c * qt, kq0 + (c + 1) * qt))
            order += list(range(v0 + c * vt, v0 + (c + 1) * vt))
        inv = [0] * len(order)
        for i, t in enumerate(order):
            inv[t] = i
        self.tt_order, self.tt_inv = order, inv
        self.tt_rowread = os.environ.get("QWEN35_TT012_ROWREAD", "1") == "1"

    # -- state checkpoint access ---------------------------------------------
    def _ckpt_index(self, sl, p_first):
        """Checkpoint holding the state after position p_first - 1 for each row."""
        return torch.remainder(p_first.long() - 1, self.n_ckpt)

    def _read_state(self, sl, j):
        flat = sl * self.n_ckpt + j
        rec = self.rec_state.reshape(-1, self.Hv, self.Dk, self.Dv).index_select(0, flat).float()
        if self.mk_layout:
            return rec, self._conv_from_mk(self.conv_state.reshape(-1, self.conv_state.shape[-1]).index_select(0, flat))
        conv = self.conv_state.reshape(-1, self.conv_dim, self.conv_k - 1).index_select(0, flat)
        return rec, conv

    def _conv_from_mk(self, c):  # [N, NC*128*K1*CT] fp32 -> [N, C, K1] bf16 (local [q|k|v] channel order)
        # Per (pcore, w) slices + cat (a multi-dim permute of this small tensor lowers to ~100 us strided DMAs
        # per layer on the prefill path: TTFT +77 ms).
        N, nc, ct, K1 = c.shape[0], self.tt_nc, self.tt_ct, self.conv_k - 1
        x = c.to(self.dtype).reshape(N, nc, 128, K1, ct)
        cols = []
        for w in range(K1):
            tiles = torch.cat([x[:, p, :, w, :].transpose(1, 2) for p in range(nc)], 1)  # [N, nc*ct, 128]
            cols.append(_permute_tiles(tiles, self.tt_inv).reshape(N, self.conv_dim, 1))
        return torch.cat(cols, 2)

    def _conv_to_mk(self, c):  # [N, C, K1] -> [N, NC*128*K1*CT] fp32
        N, nc, ct, K1 = c.shape[0], self.tt_nc, self.tt_ct, self.conv_k - 1
        x = _permute_tiles(c.reshape(N, nc * ct, 128, K1), self.tt_order).reshape(N, nc, ct, 128, K1)
        parts = []
        for p in range(nc):
            for w in range(K1):
                parts.append(x[:, p, :, :, w].transpose(1, 2).reshape(N, 128 * ct))  # [N, 128(q), ct]
        return torch.stack(parts, 1).reshape(N, nc, K1, 128, ct).transpose(2, 3).reshape(N, -1).float()

    def _write_states(self, sl, recs, convs, p0):
        """recs [B, S, Hv, Dk, Dv], convs [B, S, C, K-1]: the state after the token
        at position p0 + t goes to checkpoint (p0 + t) % n_ckpt."""
        B, S = recs.shape[0], recs.shape[1]
        si = sl.repeat_interleave(S)
        t = torch.arange(S, device=sl.device, dtype=torch.long)
        ci = torch.remainder(p0.long().reshape(B, 1) + t.reshape(1, S), self.n_ckpt).reshape(-1)
        self.rec_state.index_put_(
            (si, ci), recs.reshape(B * S, self.Hv, self.Dk, self.Dv).to(self.rec_state.dtype)
        )
        cv = convs.reshape(B * S, self.conv_dim, self.conv_k - 1)
        if self.mk_layout:
            self.conv_state_pf.index_put_((si, ci), cv.to(self.conv_state_pf.dtype))
            cv = self._conv_to_mk(cv)
            if _env_flag("QWEN35_MK_CONV_FLATW", "1"):
                # flat-row index_copy_ on the [slots*ckpt, X] view (the 2-index index_put_ on the fp32
                # MK-layout pool cost ~+370 ms TTFT once n_ckpt > 1, i.e. with MTP).
                flat = si * self.n_ckpt + ci
                self.conv_state.view(-1, self.conv_state.shape[-1]).index_copy_(0, flat, cv.to(self.conv_state.dtype))
                return
        self.conv_state.index_put_((si, ci), cv.to(self.conv_state.dtype))

    # -- common pieces --------------------------------------------------------
    def _gates(self, ba):
        b, a = ba[..., : self.Hv], ba[..., self.Hv :]
        beta = torch.sigmoid(b.float())
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        return g, beta

    def _split_qkv(self, mixed, repeat=True):  # -> q,k [...,Hv|Hk,Dk], v [...,Hv,Dv]
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        sh = mixed.shape[:-1]
        q = q.reshape(*sh, self.Hk, self.Dk)
        k = k.reshape(*sh, self.Hk, self.Dk)
        v = v.reshape(*sh, self.Hv, self.Dv)
        if repeat and self.rep > 1:
            q = q.repeat_interleave(self.rep, dim=-2)
            k = k.repeat_interleave(self.rep, dim=-2)
        return q, k, v

    def _out(self, core, z):  # core/z [T, Hv, Dv]
        T = core.shape[0]
        y = self.norm(core.to(self.dtype), z)
        return y.reshape(T, self.value_dim) @ self.out_proj_weight

    def forward(self, hidden_states, positions, attn_metadata, is_decode, slots, mask_pack, dsp=False, psp=True,
                pk=None):
        if is_decode:
            if dsp and self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            if self.decode_impl == "nki_block" and hidden_states.shape[0] == slots.shape[0]:
                return self.forward_decode_block(hidden_states, positions, slots, rs_out=dsp)
            if self.decode_impl == "nki_tkg" and hidden_states.shape[0] == slots.shape[0]:
                return self.forward_decode_tkg(hidden_states, positions, slots, rs_out=dsp)
            if self.decode_impl == "dstair" and hidden_states.shape[0] == slots.shape[0]:
                return self.forward_decode_dstair(hidden_states, positions, slots, rs_out=dsp)
            if self.decode_impl == "jb_tkg" and hidden_states.shape[0] == slots.shape[0]:
                return self.forward_decode_jb(hidden_states, positions, slots, rs_out=dsp)
            if self.decode_impl == "tt012" and hidden_states.shape[0] == slots.shape[0]:
                return self.forward_decode_tt012(hidden_states, positions, slots, rs_out=dsp)
            return self.forward_decode(hidden_states, positions, slots, rs_out=dsp)
        if self.world_size > 1 and psp:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        md = attn_metadata[next(iter(attn_metadata))]
        if pk is not None:
            return self.forward_prefill_packed(hidden_states, positions, slots, mask_pack, psp, pk)
        return self.forward_prefill(hidden_states, positions, md, slots, mask_pack, psp)

    def forward_prefill_packed(self, x, positions, slots, mask_pack, psp, pk):
        """several prompts packed in one prefill step (QWEN35_PACKED_PREFILL=1). Each packed request
        starts at position 0 on a 128-token chunk boundary (the scheduler aligns them) and is fully prefilled here.
        ``slots`` [R] / ``pk["e"]`` [R]: state slot and last real token index of each packed request (padded rows:
        scratch slot, repeated index). The conv window and the recurrent state reset at every request start, and
        each request's final recurrent state + conv tail go to its own slot."""
        x = x.to(self.dtype)
        T = x.shape[0]
        real = pk["real"]
        realf = real.to(self.dtype).unsqueeze(-1)
        mixed = (x @ self.in_proj_qkv_weight) * realf  # [T, C]  (pads zeroed)
        z = (x @ self.in_proj_z_weight).reshape(T, self.Hv, self.Dv)
        g, beta = self._gates(x @ self.in_proj_ba_weight)  # [T, Hv]
        realg = real.float().unsqueeze(-1)
        g = g * realg  # pad tokens: decay 1, update 0 -> state unchanged
        beta = beta * realg

        # causal depthwise conv, zero history; tap with delay d only reads inside the request (pos >= d)
        Km1 = self.conv_k - 1
        pos = positions.long()
        seq = torch.cat([torch.zeros(Km1, mixed.shape[1], device=x.device, dtype=mixed.dtype), mixed], dim=0)
        w = self.conv1d_weight  # [C, K]
        conv = torch.zeros_like(mixed, dtype=torch.float32)
        for j in range(self.conv_k):
            d = self.conv_k - 1 - j
            tap = seq[j : j + T].float() * w[:, j].float()
            if d > 0:
                tap = tap * (pos >= d).float().unsqueeze(-1)
            conv = conv + tap
        conv = F.silu(conv).to(self.dtype)
        # per-request conv tail: the last K-1 real inputs (zeros before the request start)
        e = pk["e"]  # [R] long
        R = e.shape[0]
        ar = torch.arange(Km1, device=x.device, dtype=torch.long)
        tidx = (e.view(R, 1) + 1 + ar.view(1, Km1)).reshape(-1)
        pos_e = pos.index_select(0, e)  # [R]
        tval = (ar.view(1, Km1) >= (Km1 - 1 - pos_e).view(R, 1)).to(seq.dtype)  # [R, K-1]
        tail = seq.index_select(0, tidx).reshape(R, Km1, -1) * tval.unsqueeze(-1)
        new_conv = tail.transpose(1, 2)  # [R, C, K-1]

        assert self.use_nki_prefill and T % 128 == 0, "packed prefill needs the nkilib GDN prefill and T % 128 == 0"
        gh = g.transpose(0, 1).unsqueeze(0).contiguous()  # [1,Hv,T]
        bh = beta.transpose(0, 1).unsqueeze(0).contiguous()
        q, k, v = self._split_qkv(conv, repeat=False)
        qh = _l2norm(q.float()).to(self.dtype).transpose(0, 1).unsqueeze(0).contiguous()
        kh = _l2norm(k.float()).to(self.dtype).transpose(0, 1).unsqueeze(0).contiguous()
        vh = v.transpose(0, 1).unsqueeze(0).contiguous()
        core, cst = _gdn_cte_packed_call(qh, kh, vh, gh, bh, mask_pack, pk["chunk_keep"], self.Dk ** -0.5)
        core = core[0].float().transpose(0, 1)  # [T, Hv, Dv]
        # state after request r = the pre-reset state before chunk (e_r // 128) + 1
        recs = cst[0].index_select(1, torch.div(e, 128, rounding_mode="floor") + 1).transpose(0, 1)  # [R,Hv,Dk,Dv]
        if not pk.get("skip_state_write", False):  # the batched action path keeps no GDN state
            self._write_states(slots.long(), recs.float().unsqueeze(1), new_conv.unsqueeze(1), pos_e)

        out = self._out(core, z)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if psp else self.tp_group.all_reduce(out)
        return out

    def forward_prefill(self, x, positions, md, slots, mask_pack, psp=True):
        """Single-request prefill chunk (Neuron schedules one prefill at a time).
        ``slots`` [1]. Continues from the slot state when cached_seq_len > 0."""
        x = x.to(self.dtype)
        T = x.shape[0]
        # Real (non-padding) tokens: the runner right-pads and repeats the last
        # real position, so a token is real iff it is at or before argmax(pos).
        last_real = torch.argmax(positions)
        real = (torch.arange(T, device=x.device) <= last_real)
        realf = real.to(self.dtype).unsqueeze(-1)

        mixed = (x @ self.in_proj_qkv_weight) * realf  # [T, C]  (pads zeroed)
        z = (x @ self.in_proj_z_weight).reshape(T, self.Hv, self.Dv)
        g, beta = self._gates(x @ self.in_proj_ba_weight)  # [T, Hv]
        realg = real.float().unsqueeze(-1)
        g = g * realg  # pad tokens: decay 1, update 0 -> state unchanged
        beta = beta * realg

        # Continuation of a chunked prefill iff the chunk does not start at 0.
        cont = (positions[:1].reshape(()) > 0).to(torch.float32)
        sl = slots.long()
        if self.mk_layout:
            # Prefill reads the port-layout copy (a prefill chunk only ever follows a prefill chunk of the same
            # request: no prefix caching), avoiding the MK-layout shuffle on the prefill critical path.
            flat = sl * self.n_ckpt + self._ckpt_index(sl, positions[:1])
            h0 = self.rec_state.reshape(-1, self.Hv, self.Dk, self.Dv).index_select(0, flat).float()
            c0 = self.conv_state_pf.reshape(-1, self.conv_dim, self.conv_k - 1).index_select(0, flat)
        else:
            h0, c0 = self._read_state(sl, self._ckpt_index(sl, positions[:1]))
        h0 = h0 * cont  # [1,Hv,Dk,Dv]
        c0 = c0[0] * cont.to(self.dtype)  # [C, K-1]

        # causal depthwise conv over [prior (K-1) | chunk]
        Km1 = self.conv_k - 1
        seq = torch.cat([c0.transpose(0, 1), mixed], dim=0)  # [K-1+T, C]
        w = self.conv1d_weight  # [C, K]
        conv = torch.zeros_like(mixed, dtype=torch.float32)
        for j in range(self.conv_k):
            conv = conv + seq[j : j + T].float() * w[:, j].float()
        conv = F.silu(conv).to(self.dtype)
        # new conv state: last K-1 REAL inputs. Real tokens are a prefix, so take
        # the K-1 rows ending at last_real (in the concatenated sequence).
        idx = (last_real + 1 + torch.arange(Km1, device=x.device)).clamp(min=0)
        new_conv = seq.index_select(0, idx).transpose(0, 1)  # [C, K-1]

        gh = g.transpose(0, 1).unsqueeze(0).contiguous()  # [1,Hv,T]
        bh = beta.transpose(0, 1).unsqueeze(0).contiguous()
        if self.prefill_impl == "dstair" and T % self.dstair_chunk == 0:
            # dstair chunked prefill (C=32 doubling/block solve; l2norm + q-scale inside,
            # LNC-sharded over v-heads). Raw GQA-expanded q/k, fp32 operands.
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            from vllm_neuron.utils.hardware_config import get_logical_nc_config

            from .kernels.dstair_gdn_prefill import nki_deltanet_chunked_prefill_v2

            q, k, v = self._split_qkv(conv)  # [T, Hv, D]
            rows = lambda t: t.float().transpose(0, 1).reshape(self.Hv * T, -1).contiguous()
            Cc = self.dstair_chunk
            ones = torch.ones(Cc, Cc, device=x.device, dtype=torch.float32)
            core, hT = wrap_nki(nki_deltanet_chunked_prefill_v2)[get_logical_nc_config()](
                state=h0.reshape(1, self.Hv * self.Dk, self.Dv).float().contiguous(),
                query=rows(q), key=rows(k), value=rows(v),
                g=gh.reshape(self.Hv * T, 1).float().contiguous(),
                beta=bh.reshape(self.Hv * T, 1).float().contiguous(),
                m_incl=torch.tril(ones), m_strict=torch.tril(ones, -1),
                eye=torch.eye(Cc, device=x.device, dtype=torch.float32),
            )
            core = core.reshape(self.Hv, T, self.Dv).transpose(0, 1)  # [T, Hv, Dv]
            hT = hT.reshape(1, self.Hv, self.Dk, self.Dv)
        elif self.use_nki_prefill and T % 128 == 0:
            # nkilib gdn_cte (Schur inverse): GQA native (un-repeated q/k), q/k
            # l2-normed, scale applied on device, fp32 state in/out.
            q, k, v = self._split_qkv(conv, repeat=False)
            qh = _l2norm(q.float()).to(self.dtype).transpose(0, 1).unsqueeze(0).contiguous()
            kh = _l2norm(k.float()).to(self.dtype).transpose(0, 1).unsqueeze(0).contiguous()
            vh = v.transpose(0, 1).unsqueeze(0).contiguous()
            core, hT = _gdn_cte_call(qh, kh, vh, gh, bh, h0.contiguous(), mask_pack, self.Dk ** -0.5)
            core = core[0].float().transpose(0, 1)  # [T, Hv, Dv]
        else:
            q, k, v = self._split_qkv(conv)  # [T, Hv, D]
            q = _l2norm(q.float()) * (self.Dk ** -0.5)
            k = _l2norm(k.float())
            v = v.float()
            qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1,Hv,T,D]
            chunk = 64
            pad = (-T) % chunk
            if pad:
                qh, kh, vh = (F.pad(t, (0, 0, 0, pad)) for t in (qh, kh, vh))
                gh, bh = F.pad(gh, (0, pad)), F.pad(bh, (0, pad))
            core, hT = _chunked_gdn(qh, kh, vh, gh, bh, h0, chunk)
            core = core[0, :, :T].transpose(0, 1)  # [T, Hv, Dv]

        self._write_states(
            sl, hT.to(torch.float32).unsqueeze(1), new_conv.unsqueeze(0).unsqueeze(1),
            positions.index_select(0, last_real.reshape(1)).long(),
        )

        out = self._out(core, z)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if psp else self.tp_group.all_reduce(out)
        return out

    def forward_decode_block(self, x, positions, slots, rs_out=False):
        """S=1 decode through nkilib gdn_block_tkg (in_proj + conv + recurrence in one
        kernel; out_proj stays in torch). nkilib keeps a K-wide conv window and slides
        it by dropping column 0, so the window passed in is [pad | our K-1 columns] and
        the last K-1 output columns are stored back."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.experimental.gdn import gdn_block_tkg

        x = x.to(self.dtype)
        B = x.shape[0]
        sl = slots.long()
        h, cs = self._read_state(sl, self._ckpt_index(sl, positions.reshape(B)))
        win = torch.cat([cs[..., :1], cs], dim=-1).contiguous()  # [B, C, K]
        W_b = self.in_proj_ba_weight[:, : self.Hv].contiguous()
        W_a = self.in_proj_ba_weight[:, self.Hv :].contiguous()
        core, new_win, new_h = wrap_nki(gdn_block_tkg)[1](
            hidden=x.reshape(B, 1, self.hidden_size),
            W_in_qkv=self.in_proj_qkv_weight,
            W_in_z=self.in_proj_z_weight,
            W_in_a=W_a,
            W_in_b=W_b,
            conv_weight=self.conv1d_weight.float(),
            conv_state=win,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            norm_weight=self.norm.weight.float(),
            recurrent_state=h.contiguous(),
        )
        self._write_states(sl, new_h.unsqueeze(1), new_win[..., 1:].unsqueeze(1), positions.reshape(B))
        out = core.reshape(B, self.value_dim).to(self.dtype) @ self.out_proj_weight
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out

    def forward_decode_tkg(self, x, positions, slots, rs_out=False):
        """S=1 decode with the nkilib gdn_tkg recurrence kernel (LNC-sharded over B*Hv;
        folds l2norm, gates, recurrence and the gated RMSNorm). in_proj, conv and
        out_proj stay in torch."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.experimental.gdn import gdn_tkg

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        x = x.to(self.dtype)
        B = x.shape[0]
        sl = slots.long()
        h, cs = self._read_state(sl, self._ckpt_index(sl, positions.reshape(B)))
        mixed = x @ self.in_proj_qkv_weight
        z = (x @ self.in_proj_z_weight).reshape(B * self.Hv, self.Dv)
        ba = x @ self.in_proj_ba_weight
        b, a = ba[:, : self.Hv].float().reshape(-1), ba[:, self.Hv :].float().reshape(-1)
        win = torch.cat([cs, mixed.unsqueeze(-1).to(cs.dtype)], dim=-1)
        conv = F.silu((win.float() * self.conv1d_weight.float().unsqueeze(0)).sum(-1)).to(self.dtype)
        q, k, v = self._split_qkv(conv)  # [B, Hv, D] (GQA-expanded)
        core, new_h = wrap_nki(gdn_tkg)[get_logical_nc_config()](
            q=q.reshape(B * self.Hv, self.Dk).contiguous(),
            k=k.reshape(B * self.Hv, self.Dk).contiguous(),
            v=v.reshape(B * self.Hv, self.Dv).contiguous(),
            b=b.contiguous(), a=a.contiguous(),
            A_log=self.A_log.float(), dt_bias=self.dt_bias.float(),
            z=z.contiguous(), norm_weight=self.norm.weight.float(),
            state_in=h.reshape(B * self.Hv, self.Dk, self.Dv).contiguous(),
        )
        self._write_states(
            sl, new_h.reshape(B, 1, self.Hv, self.Dk, self.Dv), win[..., 1:].unsqueeze(1), positions.reshape(B)
        )
        out = core.reshape(B, self.value_dim).to(self.dtype) @ self.out_proj_weight
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out

    def forward_decode_dstair(self, x, positions, slots, rs_out=False):
        """S=1 decode with the fused dstair DeltaNet kernel (conv+SiLU, l2norm, gates,
        recurrence, gated RMSNorm in one call over all B; LNC-sharded over key heads).
        in_proj and out_proj stay in torch."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .kernels.dstair_gdn_decode import nki_deltanet_full_batched

        x = x.to(self.dtype)
        B = x.shape[0]
        sl = slots.long()
        h, cs = self._read_state(sl, self._ckpt_index(sl, positions.reshape(B)))
        mixed = x @ self.in_proj_qkv_weight  # [B, C]
        z = (x @ self.in_proj_z_weight).reshape(B * self.Hv, self.Dv)
        ba = x @ self.in_proj_ba_weight
        b, a = ba[:, : self.Hv].float().reshape(-1), ba[:, self.Hv :].float().reshape(-1)
        new_h, new_cs, core = wrap_nki(nki_deltanet_full_batched)[get_logical_nc_config()](
            state=h.reshape(B * self.Hv * self.Dk, self.Dv).contiguous(),
            mixed_qkv=mixed.reshape(-1).contiguous(),
            conv_state=cs.reshape(B * self.conv_dim, self.conv_k - 1).to(self.dtype).contiguous(),
            conv_weight=self.conv1d_weight.float().contiguous(),
            conv_bias=torch.zeros(self.conv_dim, dtype=torch.float32, device=x.device),
            a_out=a.contiguous(), b_out=b.contiguous(), z=z.contiguous(),
            A_log=self.A_log.float(), dt_bias=self.dt_bias.float(),
            norm_weight=self.norm.weight.float(),
        )
        self._write_states(
            sl, new_h.reshape(B, 1, self.Hv, self.Dk, self.Dv),
            new_cs.reshape(B, 1, self.conv_dim, self.conv_k - 1), positions.reshape(B),
        )
        out = core.reshape(B, self.value_dim).to(self.dtype) @ self.out_proj_weight
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out

    def forward_decode_jb(self, x, positions, slots, rs_out=False):
        """S=1 decode with deltanet_tkg_batched_bh (all B*Hv items in one call,
        LNC-sharded over items, gated-RMSNorm tail folded). l2norm/gates/conv in torch."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .kernels.jb_tkg_bh import deltanet_tkg_batched_bh

        x = x.to(self.dtype)
        B = x.shape[0]
        BH = B * self.Hv
        sl = slots.long()
        h, cs = self._read_state(sl, self._ckpt_index(sl, positions.reshape(B)))
        mixed = x @ self.in_proj_qkv_weight
        z = (x @ self.in_proj_z_weight).reshape(BH, self.Dv, 1)
        g, beta = self._gates(x @ self.in_proj_ba_weight)  # [B, Hv]
        win = torch.cat([cs, mixed.unsqueeze(-1).to(cs.dtype)], dim=-1)
        conv = F.silu((win.float() * self.conv1d_weight.float().unsqueeze(0)).sum(-1)).to(self.dtype)
        q, k, v = self._split_qkv(conv)
        q = _l2norm(q.float()) * (self.Dk ** -0.5)
        k = _l2norm(k.float())
        bc = lambda t: t.reshape(BH, 1, 1).expand(BH, self.Dk, 1).contiguous()
        core, new_h = wrap_nki(deltanet_tkg_batched_bh)[get_logical_nc_config()](
            q=q.reshape(BH, self.Dk, 1).contiguous(), k=k.reshape(BH, self.Dk, 1).contiguous(),
            v=v.float().reshape(BH, self.Dv, 1).contiguous(), g=bc(g), beta=bc(beta),
            state_in=h.reshape(BH, self.Dk, self.Dv).contiguous(),
            z_in=z.float().contiguous(), norm_weight=self.norm.weight.float().reshape(self.Dv, 1),
            norm_eps=float(self.norm.variance_epsilon),
        )
        self._write_states(
            sl, new_h.reshape(B, 1, self.Hv, self.Dk, self.Dv), win[..., 1:].unsqueeze(1), positions.reshape(B)
        )
        out = core.reshape(B, self.value_dim).to(self.dtype) @ self.out_proj_weight
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out

    def forward_decode_tt012(self, x, positions, slots, rs_out=False):
        """S=1 decode with the fused GDN mixer kernel (in_proj, conv, gates, recurrence,
        gated RMSNorm, out_proj; LNC-split over v-heads). Returns the out_proj partial; the TP
        reduce stays here and the residual is added by the decoder layer."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .kernels.tt012_gdn_decode import tt012_gdn_decode

        x = x.to(self.dtype)
        B = x.shape[0]
        nc, ct, K1 = self.tt_nc, self.tt_ct, self.conv_k - 1
        sl = slots.long()
        # B > 1: a B-row index_select (and a constant tile reorder) feeding the custom call
        # lowers to a slow GpSimd gather (~4 ms/layer at B=8); per-row single-slot reads and
        # slice + cat avoid it. B = 1 keeps index_select (measured faster there).
        flat = sl * self.n_ckpt + self._ckpt_index(sl, positions.reshape(B))
        rec_f = self.rec_state.reshape(-1, self.Hv, self.Dk, self.Dv)
        conv_f = self.conv_state.reshape(-1, self.conv_dim, K1)
        rowread = self.tt_rowread and B > 1
        if rowread:
            h = torch.cat([rec_f.index_select(0, flat[i : i + 1]) for i in range(B)], 0).float()
            cs = torch.cat([conv_f.index_select(0, flat[i : i + 1]) for i in range(B)], 0)
        else:
            h, cs = rec_f.index_select(0, flat).float(), conv_f.index_select(0, flat)
        h_col = x.reshape(B, self.hidden_size // 128, 128).permute(2, 1, 0).reshape(128, -1).contiguous()
        dev = x.device
        ntile = self.conv_dim // 128
        if rowread:
            reorder = _permute_tiles
        else:
            reorder = lambda t, o: t.index_select(1, torch.tensor(o, dtype=torch.long, device=dev))
        cs_k = (
            reorder(cs.reshape(B, ntile, 128, K1), self.tt_order).reshape(B, nc, ct, 128, K1)
            .permute(1, 3, 4, 2, 0).reshape(-1).float().contiguous()
        )
        J = (self.Hv // nc) * B
        ident = torch.eye(128, dtype=self.dtype, device=dev)
        mk = torch.eye(J, dtype=self.dtype, device=dev).repeat_interleave(128, dim=1).contiguous()
        y, cs_o, h_o = wrap_nki(tt012_gdn_decode)[nc](
            h_col=h_col, Win=self.tt_win, Wout=self.out_proj_weight, cw=self.tt_cw, cst_in=cs_k,
            rs_in=h.contiguous(), alog=self.A_log.float(), dtb=self.dt_bias.float(),
            nw=self.norm.weight.float(), ident_in=ident, mk_in=mk,
        )
        new_cs = reorder(
            cs_o.reshape(nc, 128, K1, ct, B).permute(4, 0, 3, 1, 2).reshape(B, ntile, 128, K1), self.tt_inv
        ).reshape(B, 1, self.conv_dim, K1)
        self._write_states(sl, h_o.reshape(B, 1, self.Hv, self.Dk, self.Dv), new_cs, positions.reshape(B))
        out = y.reshape(B, self.hidden_size)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out

    def forward_decode(self, x, positions, slots, rs_out=False):
        """S tokens per request (S = 1 plain decode, S = 1 + K speculative verify).
        x [B*S, H], positions [B*S], slots [B]. The S recurrence steps are unrolled;
        the state after each token is written to its checkpoint."""
        x = x.to(self.dtype)
        T = x.shape[0]
        B = slots.shape[0]
        S = T // B
        sl = slots.long()
        p0 = positions.reshape(B, S)[:, 0]
        h, cs = self._read_state(sl, self._ckpt_index(sl, p0))  # [B,Hv,Dk,Dv], [B,C,K-1]
        mixed = (x @ self.in_proj_qkv_weight).reshape(B, S, self.conv_dim)
        z = (x @ self.in_proj_z_weight).reshape(T, self.Hv, self.Dv)
        g, beta = self._gates(x @ self.in_proj_ba_weight)  # [T, Hv]
        g = g.reshape(B, S, self.Hv)
        beta = beta.reshape(B, S, self.Hv)
        win = torch.cat([cs, mixed.transpose(1, 2).to(cs.dtype)], dim=-1)  # [B, C, K-1+S]
        wf = self.conv1d_weight.float().unsqueeze(0)
        Km1 = self.conv_k - 1
        outs, recs, convs = [], [], []
        for t in range(S):
            w_t = win[..., t : t + self.conv_k]
            conv = F.silu((w_t.float() * wf).sum(-1)).to(self.dtype)  # [B, C]
            q, k, v = self._split_qkv(conv)
            q = _l2norm(q.float()) * (self.Dk ** -0.5)
            k = _l2norm(k.float())
            v = v.float()
            h = h * g[:, t].exp()[..., None, None]
            kv_mem = (h * k.unsqueeze(-1)).sum(-2)
            delta = (v - kv_mem) * beta[:, t].unsqueeze(-1)
            h = h + k.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((h * q.unsqueeze(-1)).sum(-2))  # [B,Hv,Dv]
            recs.append(h)
            convs.append(win[..., t + 1 : t + 1 + Km1])
        self._write_states(sl, torch.stack(recs, 1), torch.stack(convs, 1), p0)
        core = torch.stack(outs, 1).reshape(T, self.Hv, self.Dv)
        out = self._out(core, z)
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0) if rs_out else self.tp_group.all_reduce(out)
        return out


# =============================================================================
# Section 5: MoE (256 experts top-8, softmax -> top-k -> renorm, + shared expert)
# =============================================================================


class Qwen3_5SparseMoe(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        from vllm.config import get_current_vllm_config

        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
        )

        vcfg = get_current_vllm_config()
        self.ep_enabled = bool(vcfg.parallel_config.enable_expert_parallel)
        # >>> PARALLELISM: EP x TP-sub over the TP group (ep_degree from neuron_config) <<<
        self.ep_degree = get_neuron_ep_degree() if self.ep_enabled else 1
        self.ep_rank = get_neuron_ep_rank() if self.ep_enabled else 0
        self.tp_degree = self.world_size // self.ep_degree
        self.E = config.num_experts
        self.E_L = self.E // self.ep_degree
        self.top_k = config.num_experts_per_tok
        self.H = config.hidden_size
        self.I = config.moe_intermediate_size // self.tp_degree
        self.I_sh = config.shared_expert_intermediate_size // self.world_size
        # MoE CTE block (tokens per expert block). 256 is the serving default (an estimated -20 % bytes at 128).
        # The action path (one T=512 prefill) fills only ~5 % of its 269 blocks at 256; smaller blocks cut the
        # memset / per-block DMA work on empty block rows. Must be a multiple of 128 (nkilib TILE_SIZE).
        self.block_size = int(os.environ.get("ISAAC_MOE_BLOCK", "256"))
        assert self.block_size % 128 == 0, self.block_size
        # Decode MoE path: "gather" (hit-list, default) or "dense" (all local experts).
        self.decode_mode = os.environ.get("QWEN35_MOE_DECODE", "gather")
        # router precision for the nki path (kernel needs router_weights in router_mm_dtype)
        self.router_fp32 = os.environ.get("QWEN35_MOE_NKI_ROUTER_FP32", "1") == "1"
        self.gather_max = int(os.environ.get("QWEN35_MOE_GATHER_MAX", "256"))
        self.gather_chunk = int(os.environ.get("QWEN35_MOE_GATHER_CHUNK", "64"))

        dt = self.dtype

        # Isaac-0.5 null-expert router: gate.weight is [E + 1, H] (last row = shared null logit). A real top-k pick
        # survives iff its logit >= the null logit; survivors are renormalized (== the 512-way logical-null
        # reference routing). QWEN35_NULL_ROUTING=0 disables
        # the mask (diagnostic only).
        self.null_router = bool(getattr(config, "null_router", False))
        self.null_mask_on = self.null_router and os.environ.get("QWEN35_NULL_ROUTING", "1") == "1"
        self.E_r = self.E + (1 if self.null_router else 0)
        self.router_weight = nn.Parameter(torch.empty(self.H, self.E_r, dtype=torch.float32))
        # FP8 experts (dstair kernel): official FP8 checkpoint, EP = TP only. The BF16 expert
        # stacks are not created (they do not fit next to the FP8 ones on a 24 GB core).
        self.fp8 = self.decode_mode == "fp8"
        self.fp8_ob = os.environ.get("QWEN35_MOE_FP8_IMPL", "pow2") == "ob"
        if self.fp8:
            assert self.ep_degree == self.world_size and self.I == 512, "fp8 MoE needs EP = TP (full I=512)"
            nt_h, nt_i = self.H // 128, self.I // 128
            self.register_buffer("w8_gate_up", torch.empty(self.E_L, self.H, 2, self.I, dtype=torch.int8), persistent=False)
            self.register_buffer("w8_down", torch.empty(self.E_L, self.I, self.H, dtype=torch.int8), persistent=False)
            self.register_buffer("w8_gate_up_scale", torch.empty(self.E_L, 128, 2 * nt_h * nt_i, dtype=torch.bfloat16), persistent=False)
            self.register_buffer("w8_down_scale", torch.empty(self.E_L, 128, nt_i * nt_h, dtype=torch.bfloat16), persistent=False)
        else:
            self.gate_up_proj_weight = nn.Parameter(torch.empty(self.E_L, self.H, 2 * self.I, dtype=dt))
            self.down_proj_weight = nn.Parameter(torch.empty(self.E_L, self.I, self.H, dtype=dt))
        # shared expert: TP-sharded intermediate over the whole group (always)
        self.shared_gate_up = nn.Parameter(torch.empty(self.H, 2 * self.I_sh, dtype=dt))
        self.shared_down = nn.Parameter(torch.empty(self.I_sh, self.H, dtype=dt))
        self.shared_gate_w = nn.Parameter(torch.empty(self.H, 1, dtype=dt))

        tpd, I, Ish, ws = self.tp_degree, self.I, self.I_sh, self.world_size

        def gate_up_t(s, r):  # HF [E, 2I_full, H] -> [E, H, 2I]
            w = s[0]
            rr = r % tpd
            Ifull = w.get_shape()[1] // 2
            g = w[:, rr * I : (rr + 1) * I, :]
            u = w[:, Ifull + rr * I : Ifull + (rr + 1) * I, :]
            return torch.cat([g, u], 1).transpose(1, 2).contiguous()

        def down_t(s, r):  # HF [E, H, I_full] -> [E, I, H]
            rr = r % tpd
            return s[0][:, :, rr * I : (rr + 1) * I].transpose(1, 2).contiguous()

        gu = SafetensorsWeightLoader(transform=gate_up_t)
        dn = SafetensorsWeightLoader(transform=down_t)
        if self.ep_degree > 1:
            ep_rank = self.ep_rank
            local = list(range(ep_rank * self.E_L, (ep_rank + 1) * self.E_L))
            gu = expert_parallel_tensor_dim_loader(local, gu)
            dn = expert_parallel_tensor_dim_loader(local, dn)
        if not self.fp8:
            set_weight_loader(self.gate_up_proj_weight, gu)
            set_weight_loader(self.down_proj_weight, dn)
        set_weight_loader(
            self.router_weight,
            SafetensorsWeightLoader(transform=lambda s, r: s[0][:].float().t().contiguous()),
        )

        def sh_gu(s, r):  # gate [Ifull,H], up [Ifull,H] -> [H, 2*Ish]
            rr = r % ws
            g = s[0][rr * Ish : (rr + 1) * Ish, :]
            u = s[1][rr * Ish : (rr + 1) * Ish, :]
            return torch.cat([g, u], 0).t().contiguous()

        def sh_dn(s, r):  # [H, Ifull] -> [Ish, H]
            rr = r % ws
            return s[0][:, rr * Ish : (rr + 1) * Ish].t().contiguous()

        set_weight_loader(self.shared_gate_up, SafetensorsWeightLoader(transform=sh_gu))
        set_weight_loader(self.shared_down, SafetensorsWeightLoader(transform=sh_dn))
        set_weight_loader(
            self.shared_gate_w,
            SafetensorsWeightLoader(transform=lambda s, r: s[0][:].t().contiguous()),
        )

    def _ep_rank_tensor(self):
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_group

        return get_neuron_ep_group().rank_in_group_tensor.reshape(1).to(torch.int32)

    def _topk(self, x):
        """x [T,H] -> (topv [T,k] fp32 renormed weights, topi [T,k]). softmax-then-renorm over the top k equals
        exp(l - l_max) / sum over the top k, so the 256-way softmax is skipped. Null router: picks whose logit is
        below the null logit get weight 0 (all-null tokens: all weights 0 -> shared expert only)."""
        logits = x.float() @ self.router_weight
        real = logits[:, : self.E] if self.null_router else logits
        lv, topi = torch.topk(real, self.top_k, dim=-1)
        w = torch.exp(lv - lv[:, :1])
        if self.null_mask_on:
            w = w * (lv >= logits[:, self.E :]).to(w.dtype)
        s = w.sum(-1, keepdim=True)
        return w / torch.where(s > 0, s, torch.ones_like(s)), topi

    def _route(self, x):  # x [T,H] bf16 normed -> affinities [T, E] fp32 (renormed top-k, 0 elsewhere)
        topv, topi = self._topk(x)
        return torch.zeros(x.shape[0], self.E, dtype=torch.float32, device=x.device).scatter(1, topi, topv)

    def _shared(self, x):  # partial over TP (reduced with the routed output)
        gu = x @ self.shared_gate_up
        g, u = gu[..., : self.I_sh], gu[..., self.I_sh :]
        y = (F.silu(g) * u) @ self.shared_down
        return torch.sigmoid((x @ self.shared_gate_w).float()).to(self.dtype) * y

    def forward(self, x, positions, is_decode, dsp=False, psp=True, real=None):
        if not is_decode and not psp:
            return self.forward_prefill(x, positions, sp=False, ar=True, real=real)
        if is_decode:
            if dsp and x.shape[0] * self.world_size * self.top_k <= self.gather_max:
                return self.forward_decode_gather_sp(x)
            if dsp:
                xf = self.tp_group.all_gather(x, dim=0)
                out = self.forward_decode_dense_partial(xf)
                return self.tp_group.reduce_scatter(out, dim=0)
            return self.forward_decode(x)
        return self.forward_prefill(x, positions, real=real)

    def forward_decode(self, x):
        if self.fp8:
            return self.forward_decode_fp8(x)
        if self.decode_mode == "gather":
            # Crossover: at T*k >= gather_max the gather reads (and copies) about as
            # many expert slices as the dense path reads in place.
            if x.shape[0] * self.top_k <= self.gather_max:
                return self.forward_decode_gather(x)
            return self.forward_decode_dense(x)
        if self.decode_mode == "nki" and self.ep_degree == 1:
            return self.forward_decode_nki(x)
        return self.forward_decode_dense(x)

    def forward_decode_fp8(self, x):
        """Decode MoE with the dstair fused FP8 kernel over this rank's E_L experts (EP = TP:
        full I per expert, official FP8 checkpoint requantized offline, see
        tests/t013/prep_fp8_cache.py). The kernel is LNC-sharded over experts and returns
        one partial per program. Router and shared expert in torch; one all-reduce."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .kernels import dstair_moe_fp8 as K8

        x = x.to(self.dtype)
        T0 = x.shape[0]
        T = next(b for b in (32, 64, 128) if b >= T0)  # kernel batch sizes; pad rows get 0 affinity
        aff = self._route(x)  # [T0, E]
        if T != T0:
            x = F.pad(x, (0, 0, 0, T - T0))
            aff = F.pad(aff, (0, 0, 0, T - T0))
        idx = torch.arange(self.E_L, device=x.device, dtype=torch.int32) + self._ep_rank_tensor() * self.E_L
        aff = aff.index_select(1, idx.long()).contiguous()
        kern = K8.nki_moe_fused_w8_fp8_block_coalesced_ob if self.fp8_ob else K8.nki_moe_fused_w8_fp8_block_coalesced
        lnc = get_logical_nc_config()
        part = wrap_nki(kern)[lnc](
            hidden=x.contiguous(), gate_up=self.w8_gate_up, down=self.w8_down,
            gate_up_scales=self.w8_gate_up_scale, down_scales=self.w8_down_scale, affinities=aff,
        )
        out = part.reshape(lnc, T, self.H).sum(0)[:T0].to(self.dtype) + self._shared(x[:T0])
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def forward_decode_nki_raw(self, h_raw, gamma, eps):
        from vllm_neuron.functional.moe.moe_block_tkg import moe_block_tkg

        x_raw = h_raw.to(self.dtype)
        out = moe_block_tkg(
            inp=x_raw.unsqueeze(0),
            gamma=gamma.reshape(1, -1).to(torch.float32),
            router_weights=self.router_weight if self.router_fp32 else self.router_weight.to(self.dtype),
            expert_gate_up_weights=self.gate_up_proj_weight.reshape(self.E_L, self.H, 2, self.I),
            expert_down_weights=self.down_proj_weight,
            rank_id=None,
            top_k=self.top_k,
            eps=eps,
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.float32 if self.router_fp32 else nl.bfloat16,
            skip_router_logits=True,
            is_all_expert=False,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.reshape(x_raw.shape).to(self.dtype)
        # shared expert on the normalized input (torch)
        xf = x_raw.float()
        xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()).to(self.dtype)
        out = out + self._shared(xn)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def forward_decode_gather_sp(self, x):
        """Decode SP: x holds T/TP tokens. Router on local tokens (no duplicate router
        work); all-gather tokens + routing, expert partials on the local I-shard, then
        reduce-scatter back to T/TP (one AG + one RS instead of one AR + the AG/RS the
        mixers already do)."""
        x = x.to(self.dtype)
        topv, topi = self._topk(x)
        xf = self.tp_group.all_gather(x, dim=0)
        topv = self.tp_group.all_gather(topv, dim=0)
        topi = self.tp_group.all_gather(topi.to(torch.int32), dim=0).long()
        out = self._gather_experts(xf, topv, topi) + self._shared(xf)
        return self.tp_group.reduce_scatter(out, dim=0)

    def forward_decode_gather(self, x):
        """Hit-list decode MoE: gather only the T*top_k selected experts' weights.

        Each rank holds every expert (intermediate TP-sharded), so the global top-k
        ids index the local weight stacks directly. HBM traffic per layer drops from
        E_L experts to T*top_k (duplicates included)."""
        x = x.to(self.dtype)
        T = x.shape[0]
        topv, topi = self._topk(x)
        if self.ep_degree > 1:  # this rank holds experts [r*E_L, (r+1)*E_L); foreign picks -> weight 0
            loc = topi.to(torch.int32) - self._ep_rank_tensor() * self.E_L
            mine = (loc >= 0) & (loc < self.E_L)
            topv = topv * mine.to(topv.dtype)
            topi = torch.where(mine, loc, torch.zeros_like(loc)).long()
        out = self._gather_experts(x, topv, topi) + self._shared(x)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def _gather_experts(self, x, topv, topi):
        """Weighted sum of the top-k experts per token via gathered weights. Chunked to
        at most ``gather_chunk`` expert slices per gather (larger gathers stall the
        compiler's ModuleForkPass at 128 slices x 40 layers)."""
        T = x.shape[0]
        idx = topi.reshape(-1)
        xr = x.unsqueeze(1).expand(T, self.top_k, self.H).reshape(T * self.top_k, 1, self.H)
        n = T * self.top_k
        ys = []
        for c0 in range(0, n, self.gather_chunk):
            c1 = min(c0 + self.gather_chunk, n)
            ic = idx[c0:c1]
            gu = torch.bmm(xr[c0:c1], self.gate_up_proj_weight.index_select(0, ic))  # [c,1,2I]
            h = F.silu(gu[..., : self.I]) * gu[..., self.I :]
            ys.append(torch.bmm(h, self.down_proj_weight.index_select(0, ic)))  # [c,1,H]
        y = (torch.cat(ys, 0) if len(ys) > 1 else ys[0]).reshape(T, self.top_k, self.H)
        return (y.float() * topv.unsqueeze(-1)).sum(1).to(self.dtype)

    def forward_decode_dense(self, x):
        out = self.forward_decode_dense_partial(x)
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def forward_decode_dense_partial(self, x):
        """Dense-over-local-experts masked bmm (torch_batched): every local expert
        runs on all B tokens, weighted by its (often zero) affinity. Returns the
        rank-partial sum (caller reduces)."""
        x = x.to(self.dtype)
        aff = self._route(x)  # [T, E]
        if self.ep_degree > 1:
            idx = torch.arange(self.E_L, device=x.device, dtype=torch.int32) + self._ep_rank_tensor() * self.E_L
            aff = aff.index_select(1, idx.long())
        T = x.shape[0]
        xb = x.unsqueeze(0).expand(self.E_L, T, self.H)
        gu = torch.bmm(xb, self.gate_up_proj_weight)  # [E_L, T, 2I]
        h = F.silu(gu[..., : self.I]) * gu[..., self.I :]
        y = torch.bmm(h, self.down_proj_weight)  # [E_L, T, H]
        out = (y.float() * aff.t().unsqueeze(-1)).sum(0).to(self.dtype)
        out = out + self._shared(x)
        return out

    def forward_prefill(self, x, positions, sp=True, ar=False, real=None):
        """SP-local tokens in; blockwise MoE CTE on the all-gathered tokens.
        sp=False: every rank already holds all tokens (all-reduce at the end)."""
        x = x.to(self.dtype)
        aff = self._route(x)  # router on SP-local tokens
        if self.world_size > 1 and sp:
            aff = self.tp_group.all_gather(aff, dim=0)
            xf = self.tp_group.all_gather(x, dim=0)
        else:
            xf = x
        if self.fp8:
            out = self._fp8_prefill_partial(xf, aff) + self._shared(xf)
            return self._prefill_reduce(out, sp, ar)
        Tn = xf.shape[0]
        if real is not None:  # packed prefill: real tokens are not a prefix
            padding_mask = real
        else:
            last_real = torch.argmax(positions)
            padding_mask = torch.arange(Tn, device=x.device) <= last_real
        if os.environ.get("ISAAC_MOE_PREFILL_IMPL", "auto") == "jobs" and self.ep_degree == 1 and Tn % 256 == 0:
            out = self._prefill_jobs(xf, padding_mask) + self._shared(xf)
            return self._prefill_reduce(out, sp, ar)
        if self.ep_degree > 1:
            idx = torch.arange(self.E_L, device=x.device, dtype=torch.int32) + self._ep_rank_tensor() * self.E_L
            aff = NF.get_local_expert_affinities(aff, idx)
        if self.ep_degree > 1 and os.environ.get("ISAAC_MOE_PREFILL_IMPL", "auto") in ("auto", "hybrid"):
            # padding tokens take no routes (build_blockwise_mapping routes them before applying padding_mask)
            aff = aff * padding_mask.unsqueeze(1).to(aff.dtype)
        aff_m, tpid, b2e, cond = NF.build_blockwise_mapping(
            expert_affinities=aff,
            num_local_experts=self.E_L,
            num_experts_per_token=self.top_k,
            block_size=self.block_size,
            moe_group=self.tp_group if self.ep_degree == 1 else _ep_tp_group(),
            tp_degree=self.tp_degree,
            padding_mask=padding_mask,
        )
        tpid = tpid.clamp(min=-1, max=Tn - 1)
        # with EP, bwmm shard_on_block computes every allocated block (sized for all top-k experts of a token
        # on one rank: ~3/4 empty at EP4). shard_on_i_hybrid runs a static loop of NSB blocks + a dynamic loop over the
        # non-empty rest (EP4 TTFT 534 -> 206 ms on a similar MoE). ISAAC_MOE_PREFILL_IMPL=block|hybrid|auto.
        impl_name = os.environ.get("ISAAC_MOE_PREFILL_IMPL", "auto")
        if impl_name == "auto":
            impl_name = "hybrid" if self.ep_degree > 1 else "block"
        extra = {}
        if impl_name == "hybrid":
            nsb = int(os.environ.get("ISAAC_MOE_PREFILL_NSB", "-1"))
            if nsb < 0:
                exp_blocks = -(-Tn * self.top_k // (self.ep_degree * self.block_size))
                nsb = max(2, 2 * (exp_blocks // 2))
            extra["num_static_block"] = nsb
        out = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_i_hybrid if impl_name == "hybrid"
            else MoECTEImplementation.shard_on_block,
            **extra,
            conditions=cond,
            hidden_states=xf,
            expert_affinities_masked=aff_m,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(self.E_L, self.H, 2, self.I),
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=tpid.to(torch.int32),
            block_to_expert=b2e.to(torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            # skip_weight=True shares ONE weight buffer set across blocks, so block i+1's
            # weight DMA waits for block i's matmuls (no prefetch). ISAAC_SKIP_WEIGHT=0 gives fresh per-block buffers.
            skip_weight=os.environ.get("ISAAC_SKIP_WEIGHT", "1") == "1" and impl_name == "block",
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
        ).to(self.dtype)
        out = out + self._shared(xf)
        return self._prefill_reduce(out, sp, ar)

    def _prefill_jobs(self, xf, padding_mask):
        """job-list MoE prefill (kernels/moe_prefill_jobs.py). Every used expert runs once for all of its
        tokens (<= 128 per job; hot experts get overflow jobs), unused experts move no weights. Same routing and the
        same per-route bf16 op order as forward_prefill's bwmm path (score after SwiGLU, before down)."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .kernels.moe_prefill_jobs import _PREP, moe_job_lanes, moe_job_prep, moe_prefill_jobs, num_jobs

        topv, topi = self._topk(xf)
        valid = (topv > 0) & padding_mask.unsqueeze(1)
        J = num_jobs(xf.shape[0], self.E, self.top_k)
        if _PREP == "kernel":  # lane tables built on chip from per-pick (job, lane, weight)
            jl, jexp, zrow, ntail = moe_job_lanes(topv, topi, valid, self.E, J)
            return wrap_nki(moe_prefill_jobs)[get_logical_nc_config()](
                x=xf.to(self.dtype), tokT=zrow, wT=zrow, zidT=zrow, jexp=jexp, zrow=zrow,
                Wgu=self.gate_up_proj_weight, Wd=self.down_proj_weight, jl=jl, ntail=ntail).to(self.dtype)
        tokT, wT, zidT, jexp, zrow = moe_job_prep(topv, topi, valid, self.E, J)
        return wrap_nki(moe_prefill_jobs)[get_logical_nc_config()](
            x=xf.to(self.dtype), tokT=tokT, wT=wT, zidT=zidT, jexp=jexp, zrow=zrow,
            Wgu=self.gate_up_proj_weight, Wd=self.down_proj_weight).to(self.dtype)

    def _prefill_reduce(self, out, sp, ar):
        if self.world_size > 1:
            if ar:
                out = self.tp_group.all_reduce(out)
            else:
                out = self.tp_group.reduce_scatter(out, dim=0)
                if not sp:  # same collectives as the SP path, then restore all tokens
                    out = self.tp_group.all_gather(out, dim=0)
        return out

    def _fp8_prefill_partial(self, xf, aff):
        """FP8 experts on prefill: the decode kernel over 128-token chunks (dense over
        local experts, so cost is routing-independent)."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.utils.hardware_config import get_logical_nc_config

        from .kernels import dstair_moe_fp8 as K8

        kern = K8.nki_moe_fused_w8_fp8_block_coalesced_ob if self.fp8_ob else K8.nki_moe_fused_w8_fp8_block_coalesced
        lnc = get_logical_nc_config()
        idx = torch.arange(self.E_L, device=xf.device, dtype=torch.int32) + self._ep_rank_tensor() * self.E_L
        aff = aff.index_select(1, idx.long())
        Tn = xf.shape[0]
        assert Tn % 128 == 0 or Tn in (32, 64), f"fp8 prefill needs T % 128 == 0, got {Tn}"
        ch = 128 if Tn % 128 == 0 else Tn
        outs = []
        for c0 in range(0, Tn, ch):
            p = wrap_nki(kern)[lnc](
                hidden=xf[c0 : c0 + ch].to(self.dtype).contiguous(), gate_up=self.w8_gate_up, down=self.w8_down,
                gate_up_scales=self.w8_gate_up_scale, down_scales=self.w8_down_scale,
                affinities=aff[c0 : c0 + ch].contiguous(),
            )
            outs.append(p.reshape(lnc, ch, self.H).sum(0))
        return torch.cat(outs, 0).to(self.dtype)


def _ep_tp_group():
    from vllm_neuron.parallel.neuron_parallel_state import get_neuron_ep_tp_group

    return get_neuron_ep_tp_group()


# =============================================================================
# Section 6: Decoder layers, backbone, LM head
# =============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, num_slots, n_ckpt=1):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = config.layer_types[layer_idx] == "full_attention"
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps, torch.float32)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps, torch.float32)
        set_weight_loader(self.input_layernorm.weight, _plus_one_loader())
        set_weight_loader(self.post_attention_layernorm.weight, _plus_one_loader())
        if self.is_full:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, num_slots, n_ckpt)
        self.mlp = Qwen3_5SparseMoe(config)

    def forward(self, h, positions, cos, sin, attn_metadata, is_decode, slots, mask_pack, dsp=False, psp=True,
                pk=None):
        r = h
        x = self.input_layernorm(h)
        if self.is_full:
            x = self.self_attn(x, positions, cos, sin, attn_metadata, is_decode, dsp, psp, pk=pk)
        else:
            x = self.linear_attn(x, positions, attn_metadata, is_decode, slots, mask_pack, dsp, psp, pk=pk)
        h = r + x
        if dsp:
            return h + self.mlp(self.post_attention_layernorm(h), positions, True, dsp=True)
        if is_decode and self.mlp.decode_mode == "nki" and self.mlp.ep_degree == 1:
            # moe_block_tkg fuses the pre-MoE RMSNorm: pass the raw residual stream.
            ln = self.post_attention_layernorm
            return h + self.mlp.forward_decode_nki_raw(h, ln.weight, ln.variance_epsilon)
        r = h
        x = self.post_attention_layernorm(h)
        x = self.mlp(x, positions, is_decode, psp=psp, real=None if pk is None else pk["real"])
        return r + x


@async_speculative_decoding
class Qwen3_5MoeForCausalLM(nn.Module, SupportsEagle3):
    """Neuron Qwen3.5-MoE text model (decoder + LM head + on-device sampling).

    MTP speculative decoding: the target returns its final normed hidden state
    as the (single) auxiliary hidden state for the MTP draft head, and verifies
    1 + K tokens per request with GDN state checkpoints (see Qwen3_5GatedDeltaNet)."""

    uses_request_slots = True

    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        from vllm.config import get_current_vllm_config

        vcfg = get_current_vllm_config()
        max_seqs = vcfg.scheduler_config.max_num_seqs
        self.num_request_slots = max_seqs + 1
        sc = vcfg.speculative_config
        self.num_speculative_tokens = sc.num_speculative_tokens if sc is not None else 0
        self.spec_enabled = self.num_speculative_tokens > 0
        dt = config.torch_dtype

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=dt,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                pad_shard=True,
            ),
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, i, self.num_request_slots, 1 + self.num_speculative_tokens)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, torch.float32)
        set_weight_loader(self.norm.weight, _plus_one_loader())
        self.rotary = PartialRotary(config)

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dt,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_shard=config.vocab_size != (config.vocab_size_ckpt or config.vocab_size),
            ),
        )
        # padded vocab rows (vocab_size > checkpoint vocab): additive -inf-like mask on this rank's logit shard
        self.vocab_padded = config.vocab_size != (config.vocab_size_ckpt or config.vocab_size)
        if self.vocab_padded:
            self.register_buffer(
                "vocab_pad_bias", torch.zeros(config.vocab_size // self.world_size, dtype=dt), persistent=False
            )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )
        self._first_attn = f"layers.{config.full_attention_layers[0]}.self_attn"
        self.decode_sp = os.environ.get("QWEN35_DECODE_SP", "0") == "1"
        # Prefill sequence parallelism (default on). 0 = all-reduce baseline for A/B.
        self.prefill_sp = os.environ.get("QWEN35_PREFILL_SP", "1") == "1"
        self.decode_sp_min_t = int(os.environ.get("QWEN35_DECODE_SP_MIN_T", "4"))
        # all 40 decoder layers of an S=1 decode step in one NKI megakernel.
        self.decode_mk = _env_flag("QWEN35_DECODE_MK")
        if self.decode_mk:
            assert self.world_size == 4 and not self.decode_sp, "QWEN35_DECODE_MK needs TP=4, no decode SP"
        # QWEN35_MK_CONTIG, default 1: MK in slot space (contiguous GDN state DMAs); the runner reads
        # ``mk_contig`` to pad the decode batch to max slot + 1. Not with spec decode (n_ckpt > 1, S > 1 verify
        # stays slot-indexed).
        self.mk_contig = self.decode_mk and _env_flag("QWEN35_MK_CONTIG", "1") and not self.spec_enabled
        # QWEN35_PACKED_PREFILL=1, with VLLM_NEURON_MAX_PREFILLS_PER_BATCH=K: several whole prompts per
        # prefill step. Each starts at position 0 on a 128-token boundary; no chunked-prefill continuation.
        from vllm_neuron.utils.bucket_utils import resolve_max_prefills_per_batch

        _k = resolve_max_prefills_per_batch(vcfg)
        self.packed_prefill = _env_flag("QWEN35_PACKED_PREFILL", "1" if _k > 1 else "0")
        assert self.packed_prefill or _k == 1, "VLLM_NEURON_MAX_PREFILLS_PER_BATCH > 1 needs QWEN35_PACKED_PREFILL=1"
        if self.packed_prefill:
            assert not self.spec_enabled, "QWEN35_PACKED_PREFILL: not supported with MTP spec decode"
        from nkilib.experimental.gdn import N_MASKS

        self.register_buffer(
            "gdn_mask_pack", torch.zeros(N_MASKS, 128, 128, dtype=torch.float32), persistent=False
        )

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        cfg = Qwen3_5MoeTextConfig.from_configs(hf_config, neuron_config)
        logger.info(
            "Qwen3.5-MoE config: layers=%d full=%s rope_theta=%g rotary_dim=%d eps=%g "
            "experts=%d top_k=%d",
            cfg.num_hidden_layers, cfg.full_attention_layers, cfg.rope_theta,
            cfg.rotary_dim, cfg.rms_norm_eps, cfg.num_experts, cfg.num_experts_per_tok,
        )
        from vllm.config import get_current_vllm_config

        cc = get_current_vllm_config().cache_config
        if cc.enable_prefix_caching:
            raise ValueError(
                "Qwen3.5-MoE on Neuron keeps GDN state per request slot, so prefix "
                "caching would skip the GDN prefix. Pass --no-enable-prefix-caching."
            )
        return cls(cfg)

    def _embed(self, input_ids, scatter, rank, is_decode, enc):
        return self.embed_tokens(input_ids, scatter_tokens=scatter, rank=rank)

    def _slots(self, attn_metadata, B, device):
        md = attn_metadata[self._first_attn]
        s = md.get("request_slots")
        if s is None:
            return torch.full((B,), self.num_request_slots - 1, dtype=torch.int32, device=device)
        return s

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        is_token_ids=None,
        attn_metadata=None,
        sampling_positions=None,
        sampling_params=None,
        spec_decode_metadata=None,
        logit_mask=None,
        rank=None,
        **kwargs,
    ):
        md0 = attn_metadata[self._first_attn]
        is_decode = md0["max_query_len"] <= md0["decode_token_threshold"]
        T = input_ids.shape[0]
        B = md0["block_table_tensor"].shape[0]
        slots = self._slots(attn_metadata, B, input_ids.device)
        positions = positions.to(torch.int32)
        sp = (not is_decode) and self.world_size > 1 and self.prefill_sp
        # Decode sequence parallelism: residual stream on T/TP tokens between mixers.
        dsp = (
            is_decode and self.decode_sp and self.world_size > 1
            and T % self.world_size == 0 and T >= self.decode_sp_min_t
        )
        if sp:
            assert T % self.world_size == 0, f"prefill T={T} not divisible by TP"
        h = self._embed(input_ids, sp or dsp, rank, is_decode, getattr(self, "_mm_enc", (None, None, None)))
        rp = getattr(self, "_rot_pos", None)  # VL subclass: [3, T] mRoPE positions (cos/sin only)
        cos, sin = self.rotary(positions if rp is None else rp)
        mk_logits = None
        if self.decode_mk and is_decode and not dsp and T == B and not _env_flag("QWEN35_DECODE_MK_SKIP"):
            h = self._mk_decode(h, positions, cos, sin, md0, slots)
        elif (self.decode_mk and is_decode and not dsp and self.spec_enabled and T > B
              and T == B * (1 + self.num_speculative_tokens) and not _env_flag("QWEN35_DECODE_MK_SKIP_VERIFY")):
            h = self._mk_decode(h, positions, cos, sin, md0, slots, S=T // B)
        else:
            pk = None
            # Mixed mode (MML > largest bucket): a 1-request prefill step (short prompt alone, or one chunk of
            # a long prompt with prefix KV / GDN state) is traced with 1 state-slot row and keeps the unpacked path;
            # packed steps carry K rows (a static shape, so each graph takes exactly one path).
            if self.packed_prefill and not is_decode and slots.shape[0] > 1:
                ti = torch.arange(T, device=input_ids.device, dtype=torch.int32)
                real = md0["slot_mapping"] != 0  # padding tokens write the null block (slot 0)
                start = real & (positions == 0)
                nc = T // 128
                keep = 1.0 - start.reshape(nc, 128)[:, 0].float()
                pk = {
                    "real": real,
                    "lo": (ti - positions).float(),
                    "e": sampling_positions.long(),
                    "chunk_keep": keep.view(nc, 1, 1).expand(nc, 128, 1).contiguous(),
                }
            for layer in self.layers:
                h = layer(h, positions, cos, sin, attn_metadata, is_decode, slots, self.gdn_mask_pack, dsp,
                          sp or is_decode or self.world_size == 1, pk=pk)
        if isinstance(h, tuple):  # MK with in-kernel final norm + lm_head -> (normed h, rank logits)
            h, mk_logits = h
        else:
            h = self.norm(h)
        if sp or dsp:
            h = self.tp_group.all_gather(h, dim=0)
        if is_decode and h.shape[0] == sampling_positions.shape[0]:
            hs = h
            logits = mk_logits
        else:
            hs = torch.index_select(h, 0, sampling_positions)
            logits = None if mk_logits is None else torch.index_select(mk_logits, 0, sampling_positions)
        if logits is None:
            logits = self.lm_head(hs)
        if self.vocab_padded:
            if self.on_device_sampling_config is None:  # gathered full-vocab logits
                logits = logits[..., : self.config.vocab_size_ckpt]
            else:
                logits = logits + self.vocab_pad_bias.to(logits.dtype)
        aux = h if self.spec_enabled else None
        if self.on_device_sampling_config is None:
            return (logits, aux) if aux is not None else logits
        tokens = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, tokens), aux, None
        if aux is not None:
            return tokens, aux, None
        return tokens, None

    def _mk_decode(self, h, positions, cos, sin, md, slots, S=1):
        """the 40 decoder layers of an S=1 decode step as one NKI megakernel call
        (kernels/mk016). Every per-layer tensor is passed as it is (zero-copy): the port's weights,
        the model-owned GDN state pools (slot rows read/written in-kernel) and vLLM's per-layer KV
        caches (new K/V row written in-kernel at slot_mapping). The state/KV outputs alias their
        inputs. In-kernel TP all-reduces over this TP group's ranks.
        S > 1 = MTP verify (S = 1 + K tokens per request, rows b-major): one T=S megakernel
        (mk_t4_step); the state after token t goes to checkpoint (p0 + t) % n_ckpt (ring)."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .kernels.mk016.mk_port_entry import mk_port_step, mk_port_step_lm, mk_t4_step, mk_t4_step_lm

        R = h.shape[0]
        B = R // S
        dev = h.device
        gdn0 = next(L.linear_attn for L in self.layers if not L.is_full)
        n_ckpt, nc = gdn0.n_ckpt, gdn0.tt_nc
        if S == 1 and self.mk_contig and B <= self.num_request_slots:  # contiguous rows 0..B-1 must exist in the pool
            return self._mk_decode_contig(h, positions, cos, sin, md, slots)
        sl = slots.long().reshape(B)
        p = positions.long().reshape(B, S)[:, 0]
        ridx = (sl * n_ckpt + torch.remainder(p - 1, n_ckpt)).to(torch.int32)
        if S == 1:
            widx = (sl * n_ckpt + torch.remainder(p, n_ckpt)).to(torch.int32)
        else:
            tt = torch.arange(S, device=dev, dtype=torch.long).reshape(S, 1)
            widx = (sl.reshape(1, B) * n_ckpt + torch.remainder(p.reshape(1, B) + tt, n_ckpt)).reshape(-1)
            widx = widx.to(torch.int32)
        bt = md["block_table_tensor"].to(torch.int32)
        bs = md["block_size"]
        assert bs == 256, f"QWEN35_DECODE_MK needs --block-size 256 (got {bs})"
        a0 = next(L.self_attn for L in self.layers if L.is_full)
        max_slot = a0.k_cache.shape[0] * bs
        sm = md["slot_mapping"].to(torch.int64).reshape(R)
        sm = torch.where((sm < 0) | (sm >= max_slot), torch.full_like(sm, max_slot - 1), sm).to(torch.int32)
        ctx = bt.shape[1] * bs
        if S == 1:
            pos = positions.reshape(B).clamp(min=0, max=ctx - 1).to(torch.int32)
        else:
            pos = p.clamp(min=0, max=ctx).to(torch.int32)
        J = (gdn0.Hv // nc) * B
        dt = self.config.torch_dtype
        ident = torch.eye(128, dtype=dt, device=dev)
        JM = min(J, 128)  # GDN block mask capped at 128 rows (grouped recurrence above J = 128)
        mkm = torch.eye(JM, dtype=dt, device=dev).repeat_interleave(128, dim=1).contiguous()
        bmask = torch.eye(R, dtype=torch.float32, device=dev).repeat_interleave(8, dim=1).contiguous()
        kw = {}
        ig = ia = 0
        for l, L in enumerate(self.layers):
            m = L.mlp
            kw.update({f"ln1_{l}": L.input_layernorm.weight, f"ln2_{l}": L.post_attention_layernorm.weight,
                       f"Wr_{l}": m.mk_wr, f"Wsgu_{l}": m.mk_wsgu, f"Wsd_{l}": m.shared_down,
                       f"Wsg_{l}": m.mk_wsg, f"gu_{l}": m.gate_up_proj_weight,
                       f"dn_{l}": m.down_proj_weight})
            if L.is_full:
                a = L.self_attn
                kw.update({f"Wqkv_{ia}": a.mk_wqkv, f"Wo_{ia}": a.o_proj_weight,
                           f"qnw_{ia}": a.q_norm.weight, f"knw_{ia}": a.k_norm.weight,
                           f"kc_{ia}": a.k_cache, f"vc_{ia}": a.v_cache})
                ia += 1
            else:
                g = L.linear_attn
                kw.update({f"Win_{ig}": g.tt_win, f"Wout_{ig}": g.out_proj_weight, f"cw_{ig}": g.tt_cw,
                           f"alog_{ig}": g.A_log, f"dtb_{ig}": g.dt_bias, f"nw_{ig}": g.norm.weight,
                           f"cs_{ig}": g.conv_state, f"rs_{ig}": g.rec_state})
                ig += 1
        if S > 1:
            ti = torch.arange(S, device=dev)
            cm = torch.where(ti.reshape(1, S) > ti.reshape(S, 1), -1.0e30, 0.0).to(torch.float32)
            cm = cm.repeat(16 // S, 1).contiguous()
            if self.mk_lm:
                kw.update(lnf=self.norm.weight, lmw=self.mk_lmw, vocab=self.mk_vocab)
            outs = wrap_nki(mk_t4_step_lm if self.mk_lm else mk_t4_step)[nc](
                x_rows=h.to(dt).contiguous(), cosb=cos.float().contiguous(), sinb=sin.float().contiguous(), bt=bt,
                slot=sm, pos0=pos, ridx=ridx, widx=widx, ident_in=ident, mk_in=mkm, bmask_in=bmask, cm_in=cm,
                ranks=tuple(self.tp_group.ranks), **kw,
            )
            return (outs[0], outs[1]) if self.mk_lm else outs[0]
        if self.mk_lm:
            kw.update(lnf=self.norm.weight, lmw=self.mk_lmw, vocab=self.mk_vocab)
        outs = wrap_nki(mk_port_step_lm if self.mk_lm else mk_port_step)[nc](
            x_rows=h.to(dt).contiguous(), cosb=cos.float().contiguous(), sinb=sin.float().contiguous(), bt=bt,
            slot=sm, pos=pos, ridx=ridx, widx=widx, ident_in=ident, mk_in=mkm, bmask_in=bmask,
            ranks=tuple(self.tp_group.ranks), **kw,
        )
        return (outs[0], outs[1]) if self.mk_lm else outs[0]

    def _mk_decode_contig(self, h, positions, cos, sin, md, slots):
        """QWEN35_MK_CONTIG=1, S=1, n_ckpt 1: the MK runs in slot space -- kernel row k is GDN state pool
        row k for k < B, so every GDN state read/write is a static contiguous DMA. Batch rows are permuted into
        slot order here (small per-row tensors only; no state moves) and the outputs gathered back. Kernel rows
        with no request this step (holes / padding) have act 0: their state is kept bit-exactly in-kernel and
        their KV write goes to the scratch KV slot. The runner pads the decode batch to cover max slot + 1."""
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .kernels.mk016.mk_port_entry import mk_port_step, mk_port_step_lm

        B = h.shape[0]
        dev = h.device
        nc = next(L.linear_attn for L in self.layers if not L.is_full).tt_nc
        sl = slots.long().reshape(B)
        ar = torch.arange(B, device=dev, dtype=torch.long)
        valid = (sl != self.num_request_slots - 1) & (sl < B)
        P = (sl.reshape(1, B) == ar.reshape(B, 1)) & valid.reshape(1, B)       # [k, b]
        Pf = P.to(torch.float32)
        act = Pf.sum(dim=1)                                                    # 1 = slot k decodes
        inv = (Pf * ar.reshape(1, B).to(torch.float32)).sum(dim=1).long()                         # batch row of slot k (0 if none)
        gat = torch.where(valid, sl, torch.zeros_like(sl))                      # kernel row of batch row b
        bt = torch.index_select(md["block_table_tensor"], 0, inv).to(torch.int32)
        bs = md["block_size"]
        assert bs == 256, f"QWEN35_DECODE_MK needs --block-size 256 (got {bs})"
        a0 = next(L.self_attn for L in self.layers if L.is_full)
        max_slot = a0.k_cache.shape[0] * bs
        sm = torch.index_select(md["slot_mapping"].to(torch.int64).reshape(B), 0, inv)
        sm = torch.where((sm < 0) | (sm >= max_slot) | (act < 0.5), torch.full_like(sm, max_slot - 1), sm).to(torch.int32)
        ctx = bt.shape[1] * bs
        pos = torch.index_select(positions.reshape(B), 0, inv).clamp(min=0, max=ctx - 1).to(torch.int32)
        xk = torch.index_select(h, 0, inv)
        ck = torch.index_select(cos, 0, inv)
        sk = torch.index_select(sin, 0, inv)
        J = (next(L.linear_attn for L in self.layers if not L.is_full).Hv // nc) * B
        dt = self.config.torch_dtype
        ident = torch.eye(128, dtype=dt, device=dev)
        JM = min(J, 128)
        mkm = torch.eye(JM, dtype=dt, device=dev).repeat_interleave(128, dim=1).contiguous()
        bmask = torch.eye(B, dtype=torch.float32, device=dev).repeat_interleave(8, dim=1).contiguous()
        kw = {}
        ig = ia = 0
        for l, L in enumerate(self.layers):
            m = L.mlp
            kw.update({f"ln1_{l}": L.input_layernorm.weight, f"ln2_{l}": L.post_attention_layernorm.weight,
                       f"Wr_{l}": m.mk_wr, f"Wsgu_{l}": m.mk_wsgu, f"Wsd_{l}": m.shared_down,
                       f"Wsg_{l}": m.mk_wsg, f"gu_{l}": m.gate_up_proj_weight,
                       f"dn_{l}": m.down_proj_weight})
            if L.is_full:
                a = L.self_attn
                kw.update({f"Wqkv_{ia}": a.mk_wqkv, f"Wo_{ia}": a.o_proj_weight,
                           f"qnw_{ia}": a.q_norm.weight, f"knw_{ia}": a.k_norm.weight,
                           f"kc_{ia}": a.k_cache, f"vc_{ia}": a.v_cache})
                ia += 1
            else:
                g = L.linear_attn
                kw.update({f"Win_{ig}": g.tt_win, f"Wout_{ig}": g.out_proj_weight, f"cw_{ig}": g.tt_cw,
                           f"alog_{ig}": g.A_log, f"dtb_{ig}": g.dt_bias, f"nw_{ig}": g.norm.weight,
                           f"cs_{ig}": g.conv_state, f"rs_{ig}": g.rec_state})
                ig += 1
        if self.mk_lm:
            kw.update(lnf=self.norm.weight, lmw=self.mk_lmw, vocab=self.mk_vocab)
        outs = wrap_nki(mk_port_step_lm if self.mk_lm else mk_port_step)[nc](
            x_rows=xk.to(dt).contiguous(), cosb=ck.float().contiguous(), sinb=sk.float().contiguous(), bt=bt,
            slot=sm, pos=pos, ridx=act.contiguous(), widx=inv.to(torch.int32), ident_in=ident, mk_in=mkm,
            bmask_in=bmask, ranks=tuple(self.tp_group.ranks), **kw,
        )
        if self.mk_lm:
            return torch.index_select(outs[0], 0, gat), torch.index_select(outs[1], 0, gat)
        return torch.index_select(outs[0], 0, gat)

    # -- MTP target interface (one aux state: the final normed hidden) --------
    def set_aux_hidden_state_layers(self, layers) -> None:
        pass

    def get_eagle3_aux_hidden_state_layers(self):
        return (self.config.num_hidden_layers - 1,)

    # -- KV cache ---------------------------------------------------------------
    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.layers):
            if layer.is_full:
                a = layer.self_attn
                layers.append(
                    LayerSpec(name=a.layer_name, num_kv_heads=a.nkh, head_size=a.head_dim, dtype=a.dtype)
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches):
        for layer in self.layers:
            if layer.is_full:
                kv = kv_caches[layer.self_attn.layer_name]
                layer.self_attn.k_cache, layer.self_attn.v_cache = kv[0], kv[1]

    # -- weights ----------------------------------------------------------------
    def _mk_prepare(self, device):
        """After weight load (QWEN35_DECODE_MK=1):
        * QWEN35_MK_FOLD=4 (default, T=1 kernel) / QWEN35_MK_FOLD_T4 (default 0, verify kernel): 4 = the RMSNorm
          scale x*rs*w is applied in the PE-transposed column layout (same op order, bit-identical), so the
          transposes start without waiting on the reduce -> rsqrt chain.
        * QWEN35_MK_NORM=fast (default; used where FOLD=0): in-kernel RMSNorm in 3 instructions instead of 6,
          bit-identical output.
        * QWEN35_MK_FOLD=3 (opt-in, faster; greedy differs on near-tie tokens): the RMSNorm reduce/rsqrt chain is taken off the critical path, the weights are
          unchanged: the norm weight multiplies the PE-transposed columns in the PSUM evacuation, the matmul
          outputs are scaled by rsqrt(mean(h^2)+eps). 2 = same with a row multiply (slower). 0 = unfused norm.
        * QWEN35_MK_FOLD=1: fold the input / post-attention RMSNorm weights ((1+w), already +1 here) into
          the MK's copies of the next projections: GDN tt_win (MK-only tensor, folded in place), GQA qkv, router
          (fp32), shared gate_up and shared gate (new MK-only copies; the prefill / S>1 path keeps the originals).
          The kernel then scales the matmul outputs by rsqrt(mean(h^2)+eps) (kernels/mk016, MK_FOLD=1).
        * QWEN35_MK_LM=1 (default): final norm + lm_head in-kernel; lm_head weight copied to the MK tile layout
          [NC, NT, 128, KT, 512] (this rank's vocab shard; the original stays for the MTP draft).
        * QWEN35_MK_PFG (default 2): next GDN layer's first in_proj k-groups prefetched before the MoE all-reduce;
          QWEN35_MK_LMPF (default 4): first lm_head tiles prefetched before the last MoE all-reduce.
        * QWEN35_MK_DEDUP: routed-expert dedup -- each unique expert of the step is loaded once and
          computed dense over all rows (static unique-slot cap + runtime overflow loop). Applies to kernels with
          R >= QWEN35_MK_DEDUP_MIN rows (B*T, default 8); QWEN35_MK_DEDUP_CAP<R> overrides the per-R static cap.
        * QWEN35_MK_FLASH: flash-decode GQA attention (online softmax over chunks of
          QWEN35_MK_FLASH_G KV blocks, default 8). auto (default) = on for decode context buckets above
          QWEN35_MK_FLASH_MIN tokens (default 2048), 1 = always (runtime loop over the request's own
          ceil(pos / (G*256)) chunks), 2 = always (static loop over the bucket), 0 = off (original kernel: the
          whole score row and V window in SBUF, spills to HBM from 32K at B=1 / 16K at B=8).
          QWEN35_MK_FLASH_SPLIT=1 (opt-in, odd B): one request's context split over the 2 pcores + LSE merge.
          QWEN35_MK_FLASH_AUTO: the form auto picks (2 = static, default, faster on HW at full bucket; 1 = fori).
          QWEN35_MK_FLASH_UNI: fori trip count per concurrent request pair (2, default; 0 hangs on HW at B>1).
        The kernel knobs are trace-time env vars, set here for the kernel modules."""
        fold_mode = os.environ.get("QWEN35_MK_FOLD", "4")
        os.environ["MK_FOLD_T4"] = os.environ.get("QWEN35_MK_FOLD_T4", "0")
        os.environ["MK_NORM"] = os.environ.get("QWEN35_MK_NORM", "fast")
        self.mk_fold = fold_mode == "1"
        self.mk_lm = _env_flag("QWEN35_MK_LM", "1") and self.on_device_sampling_config is not None
        os.environ["MK_FOLD"] = fold_mode
        os.environ["MK_PFG"] = os.environ.get("QWEN35_MK_PFG", "2")
        os.environ["MK_LMPF"] = os.environ.get("QWEN35_MK_LMPF", "4")
        os.environ["MK_DEDUP"] = os.environ.get("QWEN35_MK_DEDUP", "1")
        os.environ["MK_DEDUP_MIN"] = os.environ.get("QWEN35_MK_DEDUP_MIN", "8")
        os.environ["MK_FLASH"] = os.environ.get("QWEN35_MK_FLASH", "auto")
        os.environ["MK_FLASH_G"] = os.environ.get("QWEN35_MK_FLASH_G", "8")
        os.environ["MK_FLASH_MIN"] = os.environ.get("QWEN35_MK_FLASH_MIN", "2048")
        os.environ["MK_FLASH_SPLIT"] = os.environ.get("QWEN35_MK_FLASH_SPLIT", "0")
        os.environ["MK_FLASH_AUTO"] = os.environ.get("QWEN35_MK_FLASH_AUTO", "2")
        os.environ["MK_FLASH_UNI"] = os.environ.get("QWEN35_MK_FLASH_UNI", "2")
        for k_, v_ in list(os.environ.items()):
            if k_.startswith("QWEN35_MK_DEDUP_CAP"):
                os.environ[k_[len("QWEN35_"):]] = v_
        H = self.config.hidden_size
        # Folding runs on CPU copies (one-time, at load) and the results go back to the weights' device.
        c = lambda t: t.detach().cpu().float()
        with torch.no_grad():
            for L in self.layers:
                m = L.mlp
                w1, w2 = c(L.input_layernorm.weight), c(L.post_attention_layernorm.weight)
                if not self.mk_fold:
                    m.mk_wr, m.mk_wsgu, m.mk_wsg = m.router_weight, m.shared_gate_up, m.shared_gate_w
                    if L.is_full:
                        L.self_attn.mk_wqkv = L.self_attn.qkv_proj_weight
                    continue
                dev = m.router_weight.device
                m.mk_wr = (c(m.router_weight) * w2[:, None]).contiguous().to(dev)
                m.mk_wsgu = (c(m.shared_gate_up) * w2[:, None]).to(m.shared_gate_up.dtype).contiguous().to(dev)
                m.mk_wsg = (c(m.shared_gate_w) * w2[:, None]).to(m.shared_gate_w.dtype).contiguous().to(dev)
                if L.is_full:
                    a = L.self_attn
                    a.mk_wqkv = (c(a.qkv_proj_weight) * w1[:, None]).to(a.qkv_proj_weight.dtype).contiguous().to(dev)
                else:
                    g = L.linear_attn
                    win = c(g.tt_win).reshape(g.tt_nc, H, -1) * w1[None, :, None]
                    g.tt_win.data = win.reshape(-1).to(g.tt_win.dtype).contiguous().to(g.tt_win.device)
            if self.mk_lm:
                nc = next(L.linear_attn for L in self.layers if not L.is_full).tt_nc
                W = self.lm_head.weight  # [V_rank, H]
                V = W.shape[0]
                assert V % nc == 0
                VC = V // nc
                NT = -(-VC // 512)
                lw = torch.zeros(nc, NT * 512, H, dtype=W.dtype)
                lw[:, :VC] = W.detach().cpu().reshape(nc, VC, H)
                self.mk_lmw = lw.reshape(nc, NT, 512, H // 128, 128).permute(0, 1, 4, 3, 2).contiguous().to(W.device)
                self.mk_vocab = V
                del lw

    def load_weights(self, checkpoint_path, device, cache_dir):
        cfg = self.config
        m = {
            "embed_tokens.weight": f"{HF_PREFIX}.embed_tokens.weight",
            "norm.weight": f"{HF_PREFIX}.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i, layer in enumerate(self.layers):
            p, hp = f"layers.{i}", f"{HF_PREFIX}.layers.{i}"
            m[f"{p}.input_layernorm.weight"] = f"{hp}.input_layernorm.weight"
            m[f"{p}.post_attention_layernorm.weight"] = f"{hp}.post_attention_layernorm.weight"
            m[f"{p}.mlp.router_weight"] = f"{hp}.mlp.gate.weight"
            if not layer.mlp.fp8:
                m[f"{p}.mlp.gate_up_proj_weight"] = f"{hp}.mlp.experts.gate_up_proj"
                m[f"{p}.mlp.down_proj_weight"] = f"{hp}.mlp.experts.down_proj"
            m[f"{p}.mlp.shared_gate_up"] = [
                f"{hp}.mlp.shared_expert.gate_proj.weight",
                f"{hp}.mlp.shared_expert.up_proj.weight",
            ]
            m[f"{p}.mlp.shared_down"] = f"{hp}.mlp.shared_expert.down_proj.weight"
            m[f"{p}.mlp.shared_gate_w"] = f"{hp}.mlp.shared_expert_gate.weight"
            if layer.is_full:
                a = f"{p}.self_attn"
                m[f"{a}.qkv_proj_weight"] = [
                    f"{hp}.self_attn.q_proj.weight",
                    f"{hp}.self_attn.k_proj.weight",
                    f"{hp}.self_attn.v_proj.weight",
                ]
                m[f"{a}.o_proj_weight"] = f"{hp}.self_attn.o_proj.weight"
                m[f"{a}.q_norm.weight"] = f"{hp}.self_attn.q_norm.weight"
                m[f"{a}.k_norm.weight"] = f"{hp}.self_attn.k_norm.weight"
            else:
                a, ha = f"{p}.linear_attn", f"{hp}.linear_attn"
                m[f"{a}.in_proj_qkv_weight"] = f"{ha}.in_proj_qkv.weight"
                m[f"{a}.in_proj_z_weight"] = f"{ha}.in_proj_z.weight"
                m[f"{a}.in_proj_ba_weight"] = [f"{ha}.in_proj_b.weight", f"{ha}.in_proj_a.weight"]
                m[f"{a}.conv1d_weight"] = f"{ha}.conv1d.weight"
                m[f"{a}.dt_bias"] = f"{ha}.dt_bias"
                m[f"{a}.A_log"] = f"{ha}.A_log"
                m[f"{a}.norm.weight"] = f"{ha}.norm.weight"
                m[f"{a}.out_proj_weight"] = f"{ha}.out_proj.weight"
                if layer.linear_attn.decode_impl in ("tt012", "tt012_params") or layer.linear_attn.mk_layout:
                    m[f"{a}.tt_win"] = [f"{ha}.in_proj_qkv.weight", f"{ha}.in_proj_z.weight",
                                        f"{ha}.in_proj_b.weight", f"{ha}.in_proj_a.weight"]
                    m[f"{a}.tt_cw"] = f"{ha}.conv1d.weight"

        ckpt = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # the VL subclass's vision tower loads its own weights (encoder TP group); hide it from the text loader
        vis = self._modules.pop("visual", None)
        try:
            sd = ckpt.load_sharded_pipelined(self.rank, self.world_size, self, m, device).state_dict
            pdt = {n: p.dtype for n, p in self.named_parameters()}
        finally:
            if vis is not None:
                self._modules["visual"] = vis
        for n, t in sd.items():
            if n in pdt and t.dtype != pdt[n]:
                sd[n] = t.to(pdt[n])
        missing = [n for n in pdt if n not in sd]
        assert not missing, f"unloaded parameters: {missing[:8]} (+{len(missing) - 8})"
        self.load_state_dict(sd, strict=False, assign=True)
        # Model-owned buffers were built on meta: materialize them (zeros) here.
        from nkilib.experimental.gdn import build_mask_pack

        for mod in self.modules():
            for name, buf in list(mod._buffers.items()):
                if buf is not None:
                    mod._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
        self.gdn_mask_pack = torch.from_numpy(build_mask_pack()).to(device)
        if self.vocab_padded:
            vr = self.config.vocab_size // self.world_size
            gidx = torch.arange(vr) + self.rank * vr
            bias = torch.where(gidx >= self.config.vocab_size_ckpt, -1e9, 0.0).to(self.vocab_pad_bias.dtype)
            self.vocab_pad_bias = bias.to(device)
        if self.decode_mk:
            self._mk_prepare(device)
        fp8_dir = os.environ.get("QWEN35_MOE_FP8_DIR")
        for i, layer in enumerate(self.layers):
            mlp = layer.mlp
            if mlp.fp8:
                w8 = torch.load(f"{fp8_dir}/l{i}_r{mlp.ep_rank}.pt")
                for n in ("w8_gate_up", "w8_down", "w8_gate_up_scale", "w8_down_scale"):
                    assert w8[n].shape == mlp._buffers[n].shape, (n, w8[n].shape, mlp._buffers[n].shape)
                    mlp._buffers[n] = w8[n].to(device)
