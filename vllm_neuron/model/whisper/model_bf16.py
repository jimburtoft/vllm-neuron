# SPDX-License-Identifier: Apache-2.0
"""
Whisper (large-v3) BF16 implementation for the vllm-neuron NATIVE backend.

Ported from the byte-identical clean-room contrib model
(contrib/whisper/whisper_neuron.py) and adapted to the native-plugin model
contract used by llama3/model.py:

  * top-level ``WhisperForConditionalGeneration`` exposing ``from_configs``,
    ``forward``, ``get_kv_spec``, ``bind_kv_cache``, ``load_weights``,
    ``embed_input_ids``, ``compute_logits``.
  * decoder SELF-attention uses the plugin's BLOCK-MANAGED KV cache (bound via
    ``bind_kv_cache`` and listed in ``get_kv_spec``), consuming the standard
    ``attn_metadata`` contract (slot_mapping / block_table / block_size) exactly
    like llama3.
  * decoder CROSS-attention K/V live as model-owned ``register_buffer``
    (persistent=False) tensors -- Option A (M0 spec + M0.5 de-risk). They are
    invisible to the block manager. For M1 they are allocated + zeroed and read
    read-only on decode; populating them from the audio encoder output is M2
    (see the ``# M2:`` hooks in ``precompute_cross_kv`` and ``forward``).

ANNOTATION GUIDE:
  # >>> PARALLELISM: ...     TP-sharded, reusable.
  # <-- MODEL-SPECIFIC: ...  Whisper-specific.
  # M2: ...                  hook where M2 (cross-KV populate) slots in.
  # M3: ...                  hook where M3 (encoder run-once / audio) slots in.

Whisper specifics (vs llama3):
  * NO RoPE. Positions are learned embeddings added to token embeddings.
  * Attention heads: 20 (decoder + encoder). 20 % 8 != 0 => TP in {1,2,4}.
  * head_dim = 64.
  * q_proj/v_proj have bias, k_proj has NO bias.
  * out_proj / mlp.fc2 keep an fp32 bias (RowParallelLinear rejects non-fp32 bias
    at tp>1) -- RowParallelLinearFP32Bias wrapper.
  * activation is erf-GELU written out (F.gelu graph-breaks under fullgraph).
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.nn.sampler import Sampler
from transformers import PretrainedConfig

from .config import WhisperConfig
from .weight_loaders import (
    build_state,
    load_hf_state,
    resolve_checkpoint_dir,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers (ported from contrib whisper_neuron.py)
# --------------------------------------------------------------------------- #
def gelu(x):
    """Exact (erf) GELU written out so Dynamo traces it (F.gelu is a skipped
    C builtin under libtorch_neuronx_lite and breaks fullgraph=True)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """OpenAI Whisper sinusoidal position embeddings (encoder)."""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def _tp_world() -> int:
    import torch.distributed as dist

    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _tp_rank() -> int:
    import torch.distributed as dist

    if dist.is_initialized():
        return dist.get_rank()
    return 0


class RowParallelLinearFP32Bias(nn.Module):
    """RowParallelLinear (bias=False) + a separate fp32 bias added after the
    all-reduce (contrib whisper_neuron.py:115-133). RowParallelLinear rejects
    non-fp32 bias at tp>1; Whisper out_proj/fc2 have bias.
    """

    def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.rpl = RowParallelLinear(in_features, out_features, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self._out_dtype = dtype

    def forward(self, x):
        y = self.rpl(x)
        # M3 FIX (byte-identical): match the contrib reference EXACTLY. Contrib
        # casts the fp32 bias down to the activation dtype and adds in bf16
        # (whisper_neuron.py:131 ``y = y + self.bias.to(y.dtype)``). The previous
        # plugin path promoted y to fp32, added the fp32 bias, then cast back --
        # a different bf16 rounding that, accumulated across 32 layers x 4 RPLs
        # (self/cross out_proj + mlp fc2), shifted the final hidden state enough
        # to flip a NEAR-TIE at the very first predicted token (503 ' "' at
        # logprob -0.71 vs the reference 2221 ' Mr' at -0.835). Adding in bf16 to
        # mirror contrib removes that drift. This compiles cleanly on device (the
        # contrib model uses this exact add), so the NCC_IVRF100 concern that
        # motivated the fp32 add does not apply here.
        y = y + self.bias.to(y.dtype)
        return y


# --------------------------------------------------------------------------- #
# Encoder (naive single-graph SDPA -- ported verbatim from contrib; used by M3)
# --------------------------------------------------------------------------- #
class EncoderSelfAttention(nn.Module):
    """Encoder non-causal self-attention (contrib SelfAttention, causal=False)."""

    def __init__(self, embed_dim, num_heads, dtype):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        tp = _tp_world()
        assert num_heads % tp == 0, f"heads {num_heads} not divisible by tp {tp}"
        self.num_heads = num_heads // tp

        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.k_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype)

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        bsz, seqlen, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(v.dtype)  # M3: fp32 softmax
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dtype):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, ffn_dim, bias=True, dtype=dtype)
        self.fc2 = RowParallelLinearFP32Bias(ffn_dim, embed_dim, dtype)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        d = cfg.d_model
        self.self_attn = EncoderSelfAttention(d, cfg.encoder_attention_heads, dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.encoder_ffn_dim, dtype)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class WhisperCrossKVEncoder(nn.Module):
    """whisper enc-dec: the compiled ``.visual`` NEFF.

    Wraps the audio encoder AND the per-layer cross-attention K/V projections so
    that "run encoder once + project cross-KV + write the cross-KV buffers"
    happens INSIDE a single compiled graph. This is the robust alternative to
    running ``precompute_cross_kv`` eagerly on the Neuron device (eager
    transpose/matmul/.contiguous() on device tensors raise
    is_contiguous/dtype errors).

    The cross-KV register_buffers are OWNED here (on the compiled module) so the
    plugin's ``aliasing_output_rewrite`` FX pass creates the input_output alias
    for each in-place ``.copy_()`` -- exactly the M0.5-de-risked persistence
    pattern (write in a compiled NEFF, read in the decode NEFF). The decoder
    layers hold references to these SAME tensor objects (set in
    ``WhisperDecoder.bind_cross_kv``) and read them read-only every decode step.

    forward(mel) writes the buffers in place and returns encoder_hidden (so the
    runner's warmup ``visual(**inputs)`` call has a tensor output to trace).
    """

    def __init__(self, encoder: "WhisperEncoder", cross_attns: nn.ModuleList,
                 n_layers: int, n_heads_local: int, head_dim: int, n_ctx: int,
                 dtype):
        super().__init__()
        self.encoder = encoder
        # The cross-attention projection modules live on the decoder layers; we
        # keep a ModuleList reference so their weights load normally under the
        # decoder namespace (weight loader unchanged) -- this ModuleList holds
        # the SAME module objects, so no parameter duplication.
        self.cross_attns = cross_attns
        self.n_layers = n_layers
        for i in range(n_layers):
            self.register_buffer(
                f"cross_k_{i}",
                torch.zeros(1, n_heads_local, n_ctx, head_dim, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"cross_v_{i}",
                torch.zeros(1, n_heads_local, n_ctx, head_dim, dtype=dtype),
                persistent=False,
            )

    def forward(self, input_features, encoder_cache_buffer=None,
                write_block_ids=None, **kwargs):
        # (a) encoder: mel [b,128,3000] -> encoder_hidden [b,1500,d].
        encoder_hidden = self.encoder(input_features)
        # (b) project + write cross-KV buffers in place (compiled -> aliased).
        for i in range(self.n_layers):
            k, v = self.cross_attns[i].project_kv(encoder_hidden)
            getattr(self, f"cross_k_{i}").copy_(k)
            getattr(self, f"cross_v_{i}").copy_(v)
        return encoder_hidden


class WhisperEncoder(nn.Module):
    """Audio encoder. mel[b,128,3000] -> [b,1500,d]. Ported from contrib.

    M3: the encoder NEFF is run once per request via ``embed_multimodal`` /
    ``_execute_mm_encoder``; for M1 it is instantiated (weights load) but not
    driven by the plugin forward.
    """

    def __init__(self, cfg, dtype):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d, dtype=dtype)
        with torch.no_grad():
            # materialized here so the buffer is not meta; overwritten via
            # load_weights if present in the checkpoint.
            if not self.embed_positions.weight.is_meta:
                self.embed_positions.weight.copy_(
                    sinusoids(cfg.max_source_positions, d).to(dtype)
                )
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, dtype) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, input_features, encoder_cache_buffer=None,
                write_block_ids=None, **kwargs):
        # input_features: [b, num_mel_bins=128, 3000] -> encoder_hidden [b, 1500, d]
        #
        # whisper enc-dec: the runner's vision-warmup path calls this as
        # ``visual(input_features=..., encoder_cache_buffer=..., write_block_ids=...)``
        # (neuron_worker.py:1622-1630) -- the vision models scatter-write their
        # output into that block buffer. Whisper does NOT: its encoder output is
        # projected into the decoder cross-KV register_buffers by
        # precompute_cross_kv() instead, so encoder_cache_buffer/write_block_ids
        # are accepted for warmup-signature parity and IGNORED. The encoder just
        # returns encoder_hidden; embed_multimodal() consumes the return value.
        x = gelu(self.conv1(input_features))
        x = gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [b, 1500, d]
        x = x + self.embed_positions.weight[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


# --------------------------------------------------------------------------- #
# Decoder cross-attention (reads model-owned cross-KV register_buffers)
# --------------------------------------------------------------------------- #
class CrossAttention(nn.Module):
    """Decoder cross-attention: Q from decoder tokens, K/V from the encoder
    output. K/V are computed ONCE at prefill (M2) and cached in the owning
    decoder's register_buffers; here we only project Q and read the cache.
    (contrib CrossAttention.attn_cached / project_kv.)
    """

    def __init__(self, embed_dim, num_heads, dtype):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        tp = _tp_world()
        assert num_heads % tp == 0
        self.num_heads = num_heads // tp

        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.k_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype)

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, encoder_hidden_states):
        """M2 hook: project encoder output -> cross K,V once. [b,h,src,d]."""
        bsz, src_len, _ = encoder_hidden_states.shape
        k = self._shape(self.k_proj(encoder_hidden_states), src_len, bsz)
        v = self._shape(self.v_proj(encoder_hidden_states), src_len, bsz)
        return k, v

    def attn_cached(self, hidden_states, cross_k, cross_v):
        """Q from decoder tokens; K/V read from the cached buffers (contrib)."""
        bsz, tgt_len, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        # M3: fp32 softmax (device bf16 softmax accumulation contributes to the
        # first-token near-tie drift vs the OpenAI/HF reference).
        attn = torch.matmul(q, cross_k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(cross_v.dtype)
        ctx = torch.matmul(attn, cross_v)  # [b,h,tgt,d]
        out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
# Decoder self-attention -- BLOCK-MANAGED KV cache (plugin contract)
# --------------------------------------------------------------------------- #
class DecoderSelfAttention(nn.Module):
    """Causal self-attention using the plugin's block-managed paged KV cache.

    The K/V caches are bound externally via ``bind_kv_cache`` (the runner
    allocates them from ``get_kv_spec``). This mirrors llama3's block-managed
    self-attn contract (slot_mapping / block_table / flash_attention) but WITHOUT
    RoPE and WITHOUT the fused QKV megakernel -- Whisper projects q/k/v with the
    standard CPL projections (q/v bias, k no-bias) and writes the paged cache via
    ``index_put_`` exactly like llama3 ``_write_kv_cache``.

    For M1 the goal is a decoder that binds + compiles + runs. The plain flash /
    gathered-cache attention here is correct in structure; a fused / perf path is
    a later optimization, not an M1 requirement.
    """

    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg.d_model
        num_heads = cfg.decoder_attention_heads
        self.total_heads = num_heads
        self.head_dim = d // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dtype = dtype
        tp = _tp_world()
        assert num_heads % tp == 0, (
            f"decoder heads {num_heads} not divisible by tp {tp} "
            f"(whisper-large-v3 supports tp in {{1,2,4}})"
        )
        self.num_heads = num_heads // tp
        # Whisper has no GQA: KV heads == Q heads. Per-rank counts for get_kv_spec.
        self.num_attention_heads_per_rank = self.num_heads
        self.num_key_value_heads_per_rank = self.num_heads

        self.q_proj = ColumnParallelLinear(d, d, bias=True, dtype=dtype)
        self.k_proj = ColumnParallelLinear(d, d, bias=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(d, d, bias=True, dtype=dtype)
        self.out_proj = RowParallelLinearFP32Bias(d, d, dtype)

        # Bound externally via bind_kv_cache (block-managed).
        self.k_cache = None
        self.v_cache = None

    # ---- projections ----
    def _project(self, hidden_states):
        """hidden_states: [T, H] -> q,k,v each [nheads_local, T, head_dim]."""
        tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _write_kv_cache(self, k, v, slot_mapping, block_size):
        """Write K/V into the paged cache (llama3 _write_kv_cache, bf16 path)."""
        blk_idx = slot_mapping // block_size
        pos_idx = slot_mapping % block_size
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank
        k_f = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_f = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        h_idx = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(num_slots)
        self.k_cache.index_put_(
            (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh)), k_f
        )
        self.v_cache.index_put_(
            (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh)), v_f
        )

    def forward_prefill(self, hidden_states, attn_metadata):
        """Full-sequence causal self-attention over the prompt.

        M3: uses the SAME plain SDPA math as the byte-identical contrib reference
        (matmul + explicit upper-triangular causal mask + softmax), NOT the flash
        kernel. The SOT prompt is only ~4 tokens, so there is no perf reason to
        use flash here, and the flash kernel's bf16 numerics diverged from the
        reference just enough to flip the FIRST predicted token (spurious leading
        token, then re-sync). Matching the reference math exactly makes token 0
        byte-identical. q is pre-scaled in ``_project`` (scale applied once).
        """
        layer_name = f"decoder.layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]

        q, k, v = self._project(hidden_states)  # [Nh, T, d]; q pre-scaled
        self._write_kv_cache(k, v, slot_mapping, block_size)

        tokens = hidden_states.shape[0]
        # causal SDPA (contrib forward_prefill math): attn = softmax(q@k^T + mask)@v
        attn = torch.matmul(q, k.transpose(-1, -2))  # [Nh, T, T]; q pre-scaled
        mask = torch.triu(
            torch.full((tokens, tokens), float("-inf"),
                       dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = attn + mask
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(v.dtype)  # M3: fp32 softmax
        ctx = torch.matmul(attn, v)  # [Nh, T, d]
        # [Nh, T, d] -> [T, Nh*d]
        attn_output = ctx.transpose(0, 1).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)

    def forward_decode(self, hidden_states, attn_metadata, positions):
        """Single-token decode: append new K/V to the paged cache and attend over
        the gathered per-sequence cache. Plain (non-fused) for M1 correctness of
        structure; reads the block table to gather the sequence's cache.

        M3: ``positions`` [tokens] carries the absolute target position of each
        decode token. It is required to mask out cache slots the sequence has not
        yet written (the gathered cache spans the FULL block range S_ctx =
        max_blocks * block_size, most of which is still zero). Without this mask
        softmax attends over ~all-zero future slots, corrupting the output
        (the M2 "jumbles token order" symptom). Contrib reference: decode_step
        masks ``arange(S_ctx) <= cur_pos``.
        """
        layer_name = f"decoder.layers.{self.layer_idx}.self_attn"
        meta = attn_metadata[layer_name]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]

        B_local = block_table.shape[0]
        tokens = hidden_states.shape[0]
        S_decode = tokens // B_local

        q, k, v = self._project(hidden_states)  # [Nh, tokens, d]
        self._write_kv_cache(k, v, slot_mapping, block_size)

        # Gather the per-sequence cache [Nh, S_ctx, d] via block table.
        # k_cache: [num_blocks, nkh, block_size, head_dim]
        nkh = self.num_key_value_heads_per_rank
        max_blocks = block_table.shape[1]
        S_ctx = max_blocks * block_size
        # [B_local, max_blocks] -> gather blocks -> [B_local, nkh, S_ctx, d]
        kc = self.k_cache[block_table]  # [B_local, max_blocks, nkh, block_size, d]
        vc = self.v_cache[block_table]
        kc = kc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)
        vc = vc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)

        # q: [Nh, tokens, d] -> [B_local, nkh, S_decode, d]
        qd = q.transpose(0, 1).reshape(B_local, S_decode, nkh, self.head_dim).transpose(1, 2)
        attn = torch.matmul(qd, kc.transpose(-1, -2))  # already scaled q
        # M3 FIX (missing decode mask): the gathered cache spans the FULL block
        # range S_ctx = max_blocks * block_size; only slots [0..cur_pos] have been
        # written for this sequence, the rest are still zero. Softmax over the
        # unwritten zero slots pulls attention mass onto garbage and corrupts the
        # output (M2 "jumbles token order"). Mask slot s for query j to be valid
        # iff s <= absolute_position(j). Gathered slot index == logical sequence
        # position (block_table maps logical block i -> positions i*block_size..),
        # so a flat ``arange(S_ctx) <= pos`` mask is exact. Matches contrib
        # decode_step ``allowed = ar <= cur_pos``.
        pos = positions.to(torch.int32).view(B_local, S_decode)  # [B_local, S_decode]
        ar = torch.arange(S_ctx, device=attn.device, dtype=torch.int32)  # [S_ctx]
        # valid[b, j, s] = s <= pos[b, j]
        valid = ar.view(1, 1, S_ctx) <= pos.view(B_local, S_decode, 1)  # [B,S_decode,S_ctx]
        neg = torch.full((), float("-inf"), dtype=attn.dtype, device=attn.device)
        zero = torch.zeros((), dtype=attn.dtype, device=attn.device)
        add_mask = torch.where(valid, zero, neg)  # [B, S_decode, S_ctx]
        attn = attn + add_mask.unsqueeze(1)  # broadcast over nkh: [B,1,S_decode,S_ctx]
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(vc.dtype)  # M3: fp32 softmax
        ctx = torch.matmul(attn, vc)  # [B_local, nkh, S_decode, d]
        ctx = ctx.transpose(1, 2).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(ctx)


# --------------------------------------------------------------------------- #
# Decoder layer + decoder
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype):
        super().__init__()
        d = cfg.d_model
        self.self_attn = DecoderSelfAttention(cfg, layer_idx, dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.encoder_attn = CrossAttention(d, cfg.decoder_attention_heads, dtype)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.decoder_ffn_dim, dtype)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x, cross_k, cross_v, attn_metadata, is_prefill, positions):
        # x is [T, H] (token-flat, plugin SP layout).
        residual = x
        h = self.self_attn_layer_norm(x)
        if is_prefill:
            sa = self.self_attn.forward_prefill(h, attn_metadata)
        else:
            sa = self.self_attn.forward_decode(h, attn_metadata, positions)
        x = residual + sa

        # cross-attention reads cached cross-K/V (zeroed for M1; M2 populates).
        residual = x
        hc = self.encoder_attn_layer_norm(x)
        # attn_cached expects [b, tgt, H]; treat the token-flat dim as tgt with b=1.
        hc_b = hc.unsqueeze(0)
        ca = self.encoder_attn.attn_cached(hc_b, cross_k, cross_v).squeeze(0)
        x = residual + ca

        residual = x
        hm = self.final_layer_norm(x)
        x = residual + self.mlp(hm)
        return x


class WhisperDecoder(nn.Module):
    def __init__(self, cfg: WhisperConfig, dtype):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        d = cfg.d_model
        tp = _tp_world()
        self.n_heads_local = cfg.decoder_attention_heads // tp
        self.head_dim = d // cfg.decoder_attention_heads
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d, cfg.pad_token_id, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i, dtype) for i in range(cfg.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

        # ---- Option A: cross-KV register_buffers (persistent=False). These are
        # OWNED by the compiled WhisperCrossKVEncoder (.visual) so its NEFF's
        # in-place .copy_() writes go through aliasing_output_rewrite (M0.5).
        # The decoder reads them read-only each decode step by fetching them
        # LIVE from .visual (getattr each forward) rather than caching a tensor
        # reference -- because .to(device)/to_empty replace the buffer storage,
        # a cached reference would go stale. NOT listed in get_kv_spec ->
        # invisible to the block manager.
        self._visual: "WhisperCrossKVEncoder | None" = None

    def bind_cross_kv(self, visual: "WhisperCrossKVEncoder") -> None:
        """whisper enc-dec: point the decoder's cross-KV reads at the compiled
        ``.visual`` module that OWNS the cross-KV buffers. forward() fetches the
        buffers live (getattr) so the same tensor the .visual NEFF writes is the
        one the decode NEFF reads, even after .to(device) swaps storage.

        Stored via object.__setattr__ so nn.Module does NOT register .visual as
        a decoder child (that would create a module-tree cycle decoder ->
        visual -> cross_attns -> decoder layers)."""
        object.__setattr__(self, "_visual", visual)

    def forward(self, input_ids, positions, attn_metadata, is_prefill):
        # input_ids/positions: [T] token-flat. embed + learned position.
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self._visual, f"cross_k_{i}")
            cross_v = getattr(self._visual, f"cross_v_{i}")
            x = layer(x, cross_k, cross_v, attn_metadata, is_prefill, positions)
        return self.layer_norm(x)


# --------------------------------------------------------------------------- #
# Top-level model: the plugin contract
# --------------------------------------------------------------------------- #
class WhisperForConditionalGeneration(nn.Module):
    """Native-plugin Whisper large-v3 (encoder-decoder).

    vLLM generative-model detection requires ``embed_input_ids`` +
    ``compute_logits`` (M0.5 de-risk gotcha #4). The plugin KV/weight/forward
    contract requires ``from_configs``, ``get_kv_spec``, ``bind_kv_cache``,
    ``load_weights``, ``forward``.
    """

    is_text_generation_model = True

    # ── M4: transcription markers on the INSTANCE ────────────────────────────
    # ``from_configs`` returns THIS inner model (factory.py:157 returns the
    # concrete model, not a factory wrapper), so the runner's ``self.model`` is
    # an instance of this class. The Neuron runner's ``get_supported_tasks``
    # inspects ``supports_transcription(model)`` -> ``getattr(model,
    # "supports_transcription", False)`` on that INSTANCE to decide whether to
    # report the "transcription" task. The full serving classmethods
    # (get_generation_prompt, validate_language, ...) live on the REGISTERED
    # factory class (factory.py) which the serving layer resolves via
    # get_model_cls; here we only need the boolean flag + supports_transcription_only
    # so the runner emits ("transcription",). See task016_m4 for the split.
    supports_transcription = True
    supports_transcription_only = True
    supports_multimodal = True

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.world_size = _tp_world()
        self.rank = _tp_rank()

        self.encoder = WhisperEncoder(config, self.dtype)
        self.decoder = WhisperDecoder(config, self.dtype)

        # whisper enc-dec: expose the audio encoder under the ``.visual`` attr
        # so the runner's separate-vision-encoder compile path
        # (neuron_model_runner.py:1398 `hasattr(self.model, "visual")`) picks it
        # up and compiles it as its own NEFF, exactly like qwen3_vl's vision
        # tower. ``.visual`` is a WhisperCrossKVEncoder that runs the encoder AND
        # projects+writes the decoder cross-KV register_buffers IN-GRAPH (one
        # compiled NEFF). It owns the cross-KV buffers; the decoder reads the
        # same tensor objects (bound below). This keeps the cross-KV populate on
        # the compiled path (aliasing-persisted, M0.5) rather than eager-on-device.
        tp = self.world_size
        n_heads_local = config.decoder_attention_heads // tp
        head_dim = config.d_model // config.decoder_attention_heads
        cross_attns = nn.ModuleList(
            [layer.encoder_attn for layer in self.decoder.layers]
        )
        self.visual = WhisperCrossKVEncoder(
            self.encoder,
            cross_attns,
            n_layers=config.decoder_layers,
            n_heads_local=n_heads_local,
            head_dim=head_dim,
            n_ctx=config.max_source_positions,
            dtype=self.dtype,
        )
        # decoder reads the SAME cross-KV tensors the .visual NEFF writes.
        self.decoder.bind_cross_kv(self.visual)

        # ── Task 019: vocab-parallel (sharded) LM head ───────────────────────
        # LM head is tied to decoder.embed_tokens (Whisper proj_out). The
        # baseline did a FULL-vocab tied fp32 F.linear REPLICATED on every TP
        # rank (each rank re-read the whole [vocab, d] weight and computed all
        # 51866 logits). That does not scale with TP and re-reads the full weight
        # every decode step. Task 007 (contrib) showed a vocab-parallel head +
        # on-device global argmax recovers ~633->612 ms AND cuts P99-P50 variance
        # (~39ms -> ~6ms). We port it here using the plugin's INTENDED mechanism
        # (ColumnParallelLinear vocab shard + the plugin Sampler's built-in
        # cross-rank global argmax), NOT the contrib hand-rolled all_gather.
        #
        # Each rank computes only its 1/TP vocab slice. vocab=51866 is not a
        # multiple of TP=4, so we pad to the next multiple (51868) and mask the
        # padded columns to -inf so they can never win the argmax. The weight is
        # tied to embed_tokens (copied per-rank at the end of load_weights).
        d_model = config.d_model
        vocab = config.vocab_size
        tp = self.world_size
        pad = (-vocab) % tp
        self.padded_vocab = vocab + pad
        self.vocab_pad = pad
        self.per_rank = self.padded_vocab // tp
        self.vocab_start = self.rank * self.per_rank
        # count of *valid* (unpadded) vocab logits this rank owns
        self.n_valid = max(0, min(self.per_rank, vocab - self.vocab_start))

        # The LM-head TP group. The plugin always initializes a lm_head TP group
        # (defaults to the full TP group when lm_head_dp_size == 1), and the
        # Sampler's distributed argmax must gather over the SAME group the
        # ColumnParallelLinear shards the vocab across. Fall back to the default
        # WORLD group if the neuron parallel state is not initialized (e.g. TP=1
        # / CPU-mode unit tests), matching every other CPL in this model.
        lm_head_device_group = None
        try:
            from vllm_neuron.parallel.neuron_parallel_state import (
                get_neuron_lm_head_tp_group,
            )

            lm_head_device_group = get_neuron_lm_head_tp_group().device_group
        except Exception:
            lm_head_device_group = None
        self._lm_head_tp_group = lm_head_device_group

        # ColumnParallelLinear shards dim 0 (output = vocab). No bias (Whisper
        # tie). gather_output=False so each rank returns ONLY its local
        # [..., per_rank] logits slice; the sampler gathers across ranks.
        self.lm_head = ColumnParallelLinear(
            d_model,
            self.padded_vocab,
            bias=False,
            gather_output=False,
            dtype=torch.float32,
            tp_group=lm_head_device_group,
        )

        # ── M4: on-device sampling ────────────────────────────────────────────
        # The vllm-neuron SERVING path drives the model through the runner's
        # async/MP executor, which REQUIRES the model's forward to return sampled
        # TOKEN IDS (shape [num_reqs]) -- NOT raw logits -- when on-device
        # sampling is enabled. The runner's _sample() on-device branch does
        # ``[[x] for x in model_output_tensor.tolist()]`` expecting one token id
        # per request. Exactly like llama3/model.py, we own an on-device Sampler
        # and apply it inside forward when configured. Now that the LM head is
        # vocab-SHARDED (each rank has only its slice), the Sampler is given
        # ``process_group=<lm_head TP group>`` so its greedy argmax does the
        # cross-rank global-argmax gather internally (functional/argmax.py). The
        # distributed argmax picks the rank with the highest local max, and on a
        # tie ``torch.argmax`` over the gathered maxes picks the LOWEST rank
        # (which owns the lower vocab indices), and each rank's local argmax is
        # itself lowest-index-on-tie -> byte-identical "lowest global index wins"
        # semantics vs the full-vocab head. When on-device sampling is off (the
        # M3 offline byte-identical driver path), forward all-gathers the sharded
        # logits back to full vocab and returns them so the host can sample --
        # see compute_logits / forward.
        nc = config.neuron_config
        self.on_device_sampling_config = (
            nc.on_device_sampling_config if nc is not None else None
        )
        self.sampler = None
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=lm_head_device_group,
            )

        # ── Task 020: optional Medusa speculative-decoding heads ──────────────
        # Built ONLY when config.medusa_config is set (via additional_config /
        # hf_config). When None, Medusa is disabled -> heads are never
        # constructed -> zero overhead + the existing non-Medusa serving path is
        # untouched. Heads are children of this module (compile INTO the target
        # decode/verify NEFF; their read of the last hidden state is an in-graph
        # zero-copy op). The output projection is the sharded self.lm_head
        # (Medusa-1 tie) -- heads own only the per-head ResBlock params, loaded
        # in load_weights (after the base weights, on CPU). M2 wires the
        # proposer/verify path; M1 only constructs + loads + runs the heads.
        self.medusa_heads = None
        self._medusa_cfg = None
        mc = getattr(config, "medusa_config", None)
        if mc is not None:
            from .medusa_heads import MedusaHeads

            self._medusa_cfg = {
                "num_heads": int(mc.get("num_heads", 5)),
                "medusa_num_layers": int(mc.get("medusa_num_layers", 1)),
                "init": str(mc.get("init", "random")),
                "heads_path": mc.get("heads_path", None),
                "seed": int(mc.get("seed", 0)),
            }
            self.medusa_heads = MedusaHeads(
                num_heads=self._medusa_cfg["num_heads"],
                hidden_size=config.d_model,
                medusa_num_layers=self._medusa_cfg["medusa_num_layers"],
                dtype=self.dtype,
            )

    # ── generative-model detection (vLLM) ────────────────────────────────
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FULL-vocab logits (used by the offline/host-sampling path).

        The sharded head (self.lm_head) returns only this rank's [.., per_rank]
        slice. For the M3 offline byte-identical driver -- which expects
        full-vocab logits and samples on the host -- we all-gather the per-rank
        slices back to the full padded vocab, mask the pad columns to -inf, and
        slice off the padding so the returned width is exactly the real vocab.

        The projection is kept in fp32 (M3 tie-stability: the OpenAI/HF
        reference logits are produced in fp32; doing the final projection in
        fp32 matches the reference's logit precision at the LM head). The
        ColumnParallelLinear weight is fp32; we cast the hidden state to fp32
        before the matmul.
        """
        local = self._logits_local(hidden_states)  # [.., per_rank] this rank
        if self.world_size == 1:
            full = local
        else:
            from torch.distributed._functional_collectives import all_gather_tensor

            group = self._lm_head_tp_group
            full = all_gather_tensor(local, -1, group)  # [.., padded_vocab]
        full = self._mask_pad(full)
        # drop the padded columns so the width matches the real vocab
        return full[..., : self.config.vocab_size]

    def _logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocab-slice logits [.., per_rank], fp32 tied projection."""
        return self.lm_head(hidden_states.to(torch.float32))

    def _mask_pad(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask the padded (>= real vocab) columns of a FULL-vocab logits tensor
        to -inf so they can never win the argmax. No-op when vocab is already a
        multiple of TP (pad == 0)."""
        if self.vocab_pad == 0:
            return logits
        vocab = self.config.vocab_size
        width = logits.shape[-1]
        ar = torch.arange(width, device=logits.device)
        neg = torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device)
        return torch.where(ar < vocab, logits, neg)

    def _mask_pad_local(self, local: torch.Tensor) -> torch.Tensor:
        """Mask this rank's LOCAL [.., per_rank] slice: columns whose global
        vocab index >= real vocab are set to -inf. Only the last rank owns any
        padded columns (n_valid < per_rank)."""
        if self.n_valid >= self.per_rank:
            return local
        per_rank = self.per_rank
        ar = torch.arange(per_rank, device=local.device)
        neg = torch.full((), float("-inf"), dtype=local.dtype, device=local.device)
        return torch.where(ar < self.n_valid, local, neg)

    def _medusa_ondevice_reject(
        self,
        target_argmax: torch.Tensor,
        verify_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Task 023: ON-DEVICE greedy rejection for the Medusa verify step.

        Runs the exact ``RejectionSampler._rejection_greedy_sample`` logic in
        the compiled graph (argmax + compare + cumulative-accept mask), so the
        verify NEFF emits ACCEPTED token ids instead of ``[K+1, vocab]`` logits.
        No ``.to("cpu")`` / ``.tolist()`` / Python loop -- pure on-device tensor
        ops, byte-identical to plain greedy (each accepted position is set to
        the target's own argmax; reject at the first draft != argmax mismatch).

        BS=1 (the served target regime): verify_input_ids is
        ``[anchor, d0..d_{K-1}]`` (K+1 tokens); the drafts being verified are
        ``verify_input_ids[1:K+1]``. The greedy accept for draft ``d_p`` (which
        follows position ``p``) is ``d_p == target_argmax[p]``.

        Args:
            target_argmax: [K+1] int32 target argmax token ids (distributed
                argmax over the sharded LM head).
            verify_input_ids: [K+1] int32 = [anchor, d0..d_{K-1}].

        Returns:
            [1, K+1] int32 accepted token ids, ``-1`` after the first reject --
            the ``_parse_rejection_sampling_output`` contract.
        """
        Kp1 = target_argmax.shape[0]  # K + 1
        K = Kp1 - 1
        # Drafts d_p = verify_input_ids[1 .. K]; compare vs target_argmax[0..K-1].
        # match[p] = (d_p == target_argmax[p]) for p in 0..K-1.
        drafts = verify_input_ids[1:Kp1].to(torch.int32)  # [K]
        tgt_k = target_argmax[:K]  # [K]
        match = (drafts == tgt_k).to(torch.int32)  # [K], 1 = accept
        # n_acc = number of LEADING accepts (== position of the first reject,
        # == K if all K accepted). We compute the leading-accept count WITHOUT
        # torch.cumprod/torch.cumsum -- those lower to a Neuron custom-call the
        # compiler rejects ("Unknown custom-call API version"). Instead: build a
        # lower-triangular ones matrix [K,K] (tri[p,j]=1 iff j<=p) and matmul it
        # against the per-position mismatch vector to get the running mismatch
        # count. accept_prefix[p] = (running_mismatch[p] == 0). This is pure
        # elementwise + matmul, all natively supported.
        mismatch = (1 - match).to(torch.float32)  # [K]
        ar = torch.arange(K, device=target_argmax.device, dtype=torch.int32)
        tri = (ar.unsqueeze(1) >= ar.unsqueeze(0)).to(torch.float32)  # [K,K]
        cummis = torch.matmul(tri, mismatch)  # [K] running mismatch count
        accept_prefix = (cummis == 0).to(torch.int32)  # [K], 1 while accepting
        n_acc = accept_prefix.sum()  # scalar int32 in 0..K
        # Keep positions 0..n_acc inclusive (the accepted prefix + the
        # reject/bonus token at position n_acc); -1 after. keep[p] = p <= n_acc.
        arp1 = torch.arange(Kp1, device=target_argmax.device, dtype=torch.int32)
        keep = arp1 <= n_acc  # [K+1] bool
        neg = torch.full_like(target_argmax, -1)
        accepted = torch.where(keep, target_argmax.to(torch.int32), neg)  # [K+1]
        return accepted.unsqueeze(0)  # [1, K+1]

    def _sharded_argmax(self, head_hidden: torch.Tensor) -> torch.Tensor:
        """Task 023: sharded distributed argmax over head hidden-states.

        Maps [N, d] Medusa head hidden-states -> [N] int32 greedy token ids
        using the model's vocab-sharded LM head + the plugin Sampler's
        cross-rank global argmax (the SAME mechanism greedy decode uses). This
        replaces N per-head full-vocab all-gathers (compute_logits) with a
        single [N, per_rank] distributed argmax. Always greedy (drafts are
        argmax), so sampling_params=None.
        """
        local = self._mask_pad_local(self._logits_local(head_hidden))  # [N, per_rank]
        return self.sampler(local).to(torch.int32)  # [N]

    # ── M2/M3: encoder-run-once + cross-KV populate (audio-mm trigger) ──────
    def embed_multimodal(
        self,
        input_features: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """whisper enc-dec: run the audio encoder ONCE and populate the decoder
        cross-attention K/V register_buffers.

        Driven by the runner's existing MM-encoder trigger
        (``_execute_mm_encoder`` -> this method, neuron_model_runner.py:2265),
        which fires prefill-only, exactly once per request. Unlike the vision
        models this does NOT write into ``encoder_cache`` (EncoderCacheBlocks):
        Whisper's cross-attention needs projected per-layer K/V read on EVERY
        decode step, so the encoder output is projected once (``project_kv``)
        and written into model-owned ``cross_k_{i}``/``cross_v_{i}``
        register_buffers (Option A; M0.5-de-risked HBM persistence). The
        ``encoder_cache`` arg is accepted for interface parity but ignored.

        Args:
            input_features: mel features [b, 128, 3000] (b==1). Supplied by the
                runner's grouped/batched mm_kwargs (audio modality).
            encoder_cache: EncoderCacheBlocks (ignored -- see above).
            mm_hashes: per-item identifiers (unused for cross-KV storage).
        """
        if input_features is None:
            raise ValueError(
                "whisper embed_multimodal requires `input_features` (mel "
                "[b,128,3000]); none supplied by the mm-encoder trigger."
            )
        device = next(self.encoder.parameters()).device
        # Cast dtype on the source device first, THEN move to the Neuron device.
        # A combined device+dtype .to() on Neuron raises
        # "Expected self.dtype() == dst.dtype()".
        mel = input_features.to(dtype=self.dtype).to(device=device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [128,3000] -> [1,128,3000]

        # Run the compiled .visual NEFF: encoder(mel) -> project cross-KV ->
        # in-place write the cross_k_{i}/cross_v_{i} register_buffers. The
        # aliasing_output_rewrite FX pass keeps them HBM-resident across the
        # subsequent decode NEFF calls (M0.5 de-risk verdict: GO). The decoder
        # reads the same tensor objects (bound via decoder.bind_cross_kv).
        self.visual(mel)

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """whisper enc-dec: warmup shape for the encoder NEFF. Whisper's encoder
        input is a fixed mel [1, num_mel_bins, 3000]; there is exactly one shape
        (no bucketing), so ``bucket`` is ignored. Matches ``WhisperEncoder.forward``
        signature (single positional ``input_features``).
        """
        return {
            "input_features": torch.zeros(
                1,
                self.config.num_mel_bins,
                self.config.max_source_positions * 2,  # 1500*2 = 3000 mel frames
                dtype=self.dtype,
                device=device,
            )
        }

    # ── from_configs ──────────────────────────────────────────────────────
    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        vision_neuron_config=None,
        text_neuron_config: NeuronConfig | None = None,
        **kwargs,
    ):
        """Accept both the 2-arg text signature (llama3) and the 3-arg
        multimodal signature (qwen3_vl). M1 is driven via the 2-arg text branch
        (``vision_neuron_config is None``): the decoder loads, binds KV and
        compiles standalone with zeroed cross-KV. M3 will flip the runner to the
        3-arg branch (set ``vision_neuron_config`` via ``additional_config``) to
        wire the audio encoder-run-once trigger; this signature already accepts
        it, so no signature change is needed then.
        """
        # text_neuron_config is the 3-arg name for neuron_config.
        nc = neuron_config if neuron_config is not None else text_neuron_config
        config = WhisperConfig.from_configs(hf_config, nc)

        # Task 020 (M4): surface the Medusa heads config for the SERVED path.
        # The runner constructs the model via the generic from_configs(hf_config,
        # neuron_config) call (neuron_model_runner.py:1190/1210) and never passes
        # a medusa_config -- so we pull it from the active vLLM config's
        # additional_config here. Customer enables Medusa with:
        #   --speculative-config '{"method":"medusa","num_speculative_tokens":5}'
        #   --additional-config  '{"medusa_config":{"init":"zero"}}'
        # (num_speculative_tokens=K is handled by the spec-config shim + runner;
        # medusa_config carries only the heads SOURCE: init=zero|random|load +
        # heads_path when init=load, num_heads defaults to K.) An explicit
        # medusa_config kwarg (e.g. from a test) takes precedence.
        medusa_config = kwargs.get("medusa_config")
        if medusa_config is None:
            try:
                from vllm.config import get_current_vllm_config

                vcfg = get_current_vllm_config()
                add_cfg = getattr(vcfg, "additional_config", None) or {}
                medusa_config = add_cfg.get("medusa_config")
                # Default num_heads (N) to K from the speculative config so the
                # customer only has to set K in ONE place (--speculative-config).
                if medusa_config is not None and "num_heads" not in medusa_config:
                    spec = getattr(vcfg, "speculative_config", None)
                    k = getattr(spec, "num_speculative_tokens", None)
                    if k is not None:
                        medusa_config = {**medusa_config, "num_heads": int(k)}
            except Exception:
                medusa_config = None
        if medusa_config is not None:
            config.medusa_config = medusa_config

        return cls(config)

    # ── KV cache: self-KV only (block-managed). cross-KV is register_buffer ──
    def get_kv_spec(self) -> KVSpec:
        layers = []
        for i, layer in enumerate(self.decoder.layers):
            layer_name = f"decoder.layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        # cross_k_*/cross_v_* intentionally NOT listed (Option A).
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.decoder.layers):
            layer_name = f"decoder.layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── weight loading ──────────────────────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load HF openai/whisper-large-v3 safetensors into the encoder + decoder.

        Meta-tensor materialization (M0.5 de-risk gotcha #2): the model is built
        under ``torch.device("meta")``. We materialize all storage on CPU first
        (``to_empty``), then load real weights into encoder/decoder, then let the
        runner move to device. The cross-KV register_buffers are materialized +
        zeroed here (M2 fills them at prefill).

        TP sharding is delegated to the CPL/RPL ``_load_from_state_dict`` (they
        shard a FULL weight per-rank when the loaded shape != the local shape),
        so ``build_state`` emits FULL (unsharded) tensors.
        """
        # 1) materialize meta tensors to empty CPU storage.
        self.to_empty(device="cpu")

        # to_empty replaces buffer/param tensors with fresh storage, which
        # (a) invalidates the decoder's cross-KV references and (b) may have
        # split the .encoder / .visual.encoder shared-module tensors. Re-point
        # the decoder cross-KV reads at .visual's (now materialized) buffers.
        with torch.no_grad():
            for i in range(self.config.decoder_layers):
                getattr(self.visual, f"cross_k_{i}").zero_()
                getattr(self.visual, f"cross_v_{i}").zero_()
        self.decoder.bind_cross_kv(self.visual)

        # 3) resolve checkpoint dir + load HF state.
        model_dir = resolve_checkpoint_dir(checkpoint_path, cache_dir)
        hf_sd = load_hf_state(model_dir)

        enc_state = build_state(hf_sd, "model.encoder.", self.dtype)
        dec_state = build_state(hf_sd, "model.decoder.", self.dtype)

        # 4) load into submodules. strict=False so we can account for
        #    (a) the tied lm_head (no separate proj_out weight) and
        #    (b) the sinusoidal encoder.embed_positions (present in HF).
        enc_missing, enc_unexpected = self.encoder.load_state_dict(
            enc_state, strict=False, assign=True
        )
        dec_missing, dec_unexpected = self.decoder.load_state_dict(
            dec_state, strict=False, assign=True
        )

        # cross-KV buffers are non-persistent -> not in load_state_dict; they
        # were materialized in step 1/2. Filter them out of "missing".
        def _is_crosskv(name: str) -> bool:
            return name.startswith("cross_k_") or name.startswith("cross_v_")

        dec_missing = [m for m in dec_missing if not _is_crosskv(m)]

        # embed_positions.weight is a sinusoidal constant re-derived in the
        # encoder __init__; if HF omits it, re-materialize it.
        enc_missing_real = []
        for m in enc_missing:
            if m == "embed_positions.weight":
                with torch.no_grad():
                    self.encoder.embed_positions.weight.copy_(
                        sinusoids(
                            self.config.max_source_positions, self.config.d_model
                        ).to(self.dtype)
                    )
            else:
                enc_missing_real.append(m)

        # ── Task 019: tie the sharded LM head to decoder.embed_tokens ────────
        # Whisper's proj_out is tied to decoder.embed_tokens.weight [vocab, d].
        # The vocab-parallel ColumnParallelLinear head (self.lm_head) has a
        # separate [per_rank, d] weight per rank; copy this rank's vocab slice
        # embed_tokens.weight[vocab_start : vocab_start+per_rank] into it, in
        # fp32 (the head projects in fp32). Rows >= real vocab (padding) stay
        # zero; the pad columns are masked to -inf at argmax time so the zero
        # rows never win. Weights are frozen at inference, so copy (vs shared
        # storage) is fine. embed_tokens is on CPU here (post to_empty / pre the
        # runner's device move), so materialize the shard on CPU too.
        self._tie_lm_head()

        # ── Task 020: load Medusa head ResBlock params (if enabled) ───────────
        # After the base weights + lm-head tie, while tensors are on CPU (pre
        # device move). The heads' output projection is the sharded self.lm_head
        # (tie) -- only the ResBlock params are synthesized/loaded here. init
        # "zero"/"random" need NO checkpoint (framework runs untrained); "load"
        # reads a customer checkpoint. MISSING/UNEXPECTED for the head params
        # must be 0 (asserted by the M1 harness).
        self._medusa_load_report = None
        if self.medusa_heads is not None:
            from .medusa_heads import (
                _load_checkpoint_state,
                _normalize_external_state,
                _synthesize_state,
            )

            cfg = self._medusa_cfg
            # materialize head storage on CPU (parity with to_empty above).
            if any(p.is_meta for p in self.medusa_heads.parameters()):
                self.medusa_heads.to_empty(device="cpu")
            if cfg["init"] in ("zero", "random"):
                head_state = _synthesize_state(
                    self.medusa_heads, cfg["init"], seed=cfg["seed"]
                )
            elif cfg["init"] == "load":
                if not cfg["heads_path"]:
                    raise ValueError("medusa_config init='load' requires heads_path")
                raw = _load_checkpoint_state(cfg["heads_path"])
                head_state = _normalize_external_state(raw, self.medusa_heads)
            else:
                raise ValueError(
                    f"medusa_config init must be zero|random|load, got {cfg['init']!r}"
                )
            m_missing, m_unexpected = self.medusa_heads.load_state_dict(
                head_state, strict=False, assign=True
            )
            self._medusa_load_report = {
                "missing": len(m_missing),
                "unexpected": len(m_unexpected),
            }
            logger.info(
                "Medusa heads load: N=%d L=%d init=%s missing=%d unexpected=%d",
                cfg["num_heads"],
                cfg["medusa_num_layers"],
                cfg["init"],
                len(m_missing),
                len(m_unexpected),
            )
            if m_missing or m_unexpected:
                logger.warning(
                    "Medusa heads weight mismatch: missing=%s unexpected=%s",
                    m_missing,
                    m_unexpected,
                )

        n_missing = len(enc_missing_real) + len(dec_missing)
        n_unexpected = len(enc_unexpected) + len(dec_unexpected)
        logger.info(
            "Whisper load_weights: encoder(missing=%d, unexpected=%d) "
            "decoder(missing=%d, unexpected=%d)",
            len(enc_missing_real),
            len(enc_unexpected),
            len(dec_missing),
            len(dec_unexpected),
        )
        if enc_missing_real or enc_unexpected or dec_missing or dec_unexpected:
            logger.warning(
                "Whisper weight mismatch: enc_missing=%s enc_unexpected=%s "
                "dec_missing=%s dec_unexpected=%s",
                enc_missing_real,
                enc_unexpected,
                dec_missing,
                dec_unexpected,
            )
        # Expose counts for the loader verification harness.
        self._weight_load_missing = n_missing
        self._weight_load_unexpected = n_unexpected

    def _tie_lm_head(self) -> None:
        """Copy this rank's vocab slice of decoder.embed_tokens.weight into the
        vocab-parallel ColumnParallelLinear LM head (fp32), padding rows beyond
        the real vocab with zeros. Call after the decoder weights are loaded and
        while tensors are still on CPU (pre device move)."""
        vocab = self.config.vocab_size
        per_rank = self.per_rank
        s = self.vocab_start
        e = s + per_rank
        w_cpu = self.decoder.embed_tokens.weight.detach().to("cpu")  # [vocab, d]
        d = w_cpu.shape[1]
        target_dtype = self.lm_head.weight.dtype
        shard = torch.zeros(per_rank, d, dtype=target_dtype)
        # rows [s:e) mapped from the (unpadded) embed weight; rows >= vocab -> 0.
        valid_e = min(e, vocab)
        if valid_e > s:
            shard[: valid_e - s] = w_cpu[s:valid_e].to(target_dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_(shard.to(self.lm_head.weight.device))

    # ── Task 020 (M1): Medusa draft proposal from a single anchor ──────────
    @torch.no_grad()
    def medusa_propose(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Task 020 (M1): run the Medusa heads on a single anchor hidden state
        and return the K greedy draft token ids [N] int32.

        ``last_hidden`` is the decoder's LAST hidden state (post final LayerNorm)
        at the anchor position -- the SAME tensor compute_logits / self.lm_head
        consume (self.decoder returns self.layer_norm(x); forward computes
        ``hidden_states`` from it). Head j predicts the token at anchor+1+j via
        ResBlock_j(h) -> tied full-vocab lm_head -> argmax.

        The projection uses ``compute_logits`` (all-gathers the sharded per-rank
        vocab slices to full vocab, fp32) so the argmax is a real token id and is
        byte-consistent across TP ranks (each rank's head output projection uses
        its lm_head vocab shard; the all-gather stitches them). This is exactly
        what M2's proposer/verify NEFF calls on the last candidate position to
        produce the next K drafts.
        """
        if self.medusa_heads is None:
            raise RuntimeError(
                "medusa_propose called but Medusa heads are not constructed "
                "(config.medusa_config is None)."
            )
        return self.medusa_heads.propose_from_hidden(
            last_hidden, project_fn=self.compute_logits
        )

    # ── forward (unified prefill/decode; block-managed self-KV) ────────────
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        spec_decode_metadata: object | None = None,
        **kwargs,
    ) -> torch.Tensor:
        positions = positions.to(torch.int32)

        # prefill vs decode from attn_metadata (llama3 model.py:1509-1514).
        first_layer_name = "decoder.layers.0.self_attn"
        meta = attn_metadata[first_layer_name]
        max_query_len = meta["max_query_len"]
        decode_token_threshold = meta["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        # M3: at prefill the audio encoder has already been run and
        # decoder.precompute_cross_kv() has filled the cross-KV buffers via
        # embed_multimodal (_execute_mm_encoder). For M1 the buffers are zeroed;
        # the decoder still reads them (cross-attn contributes zeros).
        hidden_states = self.decoder(
            input_ids, positions, attn_metadata, is_prefill
        )

        # ── Task 020 (M2) / Task 023: Medusa VERIFY-K path ───────────────────
        # When a spec-decode verify step is active AND Medusa heads exist, the
        # runner drives forward over input_ids [1, K+1] = [anchor, d0..d_{K-1}]
        # (the scheduler placed the previous step's drafts into input_ids). The
        # decoder above already wrote the K+1 self-KV slots at the trailing
        # positions (forward_decode's per-position causal mask handles the
        # multi-token window -- M0.5-de-risked). Here we:
        #   (a) run the Medusa heads on the LAST candidate row's hidden state
        #       to produce the next K drafts (heads compile INTO this NEFF);
        #   (b) EITHER
        #       - ON-DEVICE rejection (Task 023, served path when self.sampler
        #         is set): return [bs, K+1] int32 ACCEPTED token ids (-1 padded),
        #         the EAGLE / _parse_rejection_sampling_output contract. This
        #         removes the per-step [K+1, vocab] logits -> CPU copy that made
        #         served Medusa 0.587x in Task 022; OR
        #       - HOST rejection (Lever B, offline byte-identical driver when
        #         self.sampler is None): return the [K+1, vocab] logits 3-tuple
        #         so the host RejectionSampler can index bonus/target logits.
        is_medusa_verify = (
            self.medusa_heads is not None
            and spec_decode_metadata is not None
            and not is_prefill
        )
        if is_medusa_verify:
            # verify rows: the runner passes sampling_positions=logits_indices,
            # which spans all K+1 candidate positions for a spec-decode step.
            if sampling_positions is not None:
                verify_hidden = torch.index_select(
                    hidden_states, dim=0, index=sampling_positions
                )  # [K+1, d_model]
                verify_input_ids = torch.index_select(
                    input_ids, dim=0, index=sampling_positions
                )  # [K+1] = [anchor, d0..d_{K-1}]
            else:
                verify_hidden = hidden_states
                verify_input_ids = input_ids
            # next-K drafts from the LAST candidate position's hidden state.
            last_hidden = verify_hidden[-1:].contiguous()  # [1, d_model]

            if self.sampler is not None:
                # ── Task 023: ON-DEVICE greedy rejection + SHARDED propose ────
                # Reuse the SAME sharded-LM-head distributed argmax already used
                # for greedy decode: feed the local vocab-slice logits for all
                # K+1 rows to self.sampler (all_greedy), which returns [K+1]
                # int32 target argmax ids via the cross-rank global argmax --
                # NO [K+1, vocab] host copy.
                local_logits = self._mask_pad_local(
                    self._logits_local(verify_hidden)
                )  # [K+1, per_rank]
                target_argmax = self.sampler(
                    local_logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
                ).to(torch.int32)  # [K+1]
                accepted = self._medusa_ondevice_reject(
                    target_argmax, verify_input_ids
                )  # [1, K+1] int32, -1 padded
                # Propose the next K drafts via the SHARDED distributed argmax
                # (one [N, per_rank] argmax) instead of N full-vocab all-gathers
                # (propose_from_hidden -> compute_logits per head). This removes
                # N per-step vocab all-gathers from the verify NEFF.
                drafts = self.medusa_heads.propose_from_hidden_sharded(
                    last_hidden, argmax_fn=self._sharded_argmax
                )  # [K] int32
                return accepted, verify_hidden, drafts

            # Offline / host-sampling path: return the [K+1, vocab] logits for
            # the host RejectionSampler (Lever B, byte-identical driver).
            drafts = self.medusa_heads.propose_from_hidden(
                last_hidden, project_fn=self.compute_logits
            )  # [K] int32
            verify_logits = self.compute_logits(verify_hidden)  # [K+1, vocab]
            return verify_logits, verify_hidden, drafts

        # logits for the sampling positions.
        if sampling_positions is not None:
            hidden_states = torch.index_select(
                hidden_states, dim=0, index=sampling_positions
            )

        # ── Task 019: on-device sampling for the serving/async runner path ────
        # When on-device sampling is configured, the runner expects TOKEN IDS
        # (shape [num_reqs]) back from forward (llama3 parity). With the
        # vocab-SHARDED head, each rank computes only its local [.., per_rank]
        # slice; we pad-mask the local slice (only the last rank owns padded
        # columns) and hand it to the Sampler, which -- constructed with
        # process_group=<lm_head TP group> -- runs the plugin's distributed
        # global argmax (functional/argmax.py): gather each rank's (max, argidx),
        # pick the global winner with correct sharding offset, lowest-index-wins
        # on ties. Byte-identical to argmax over the full-vocab head, but each
        # rank only projects/reads 1/TP of the vocab (the perf win).
        #
        # When on-device sampling is OFF (M3 offline byte-identical driver),
        # all-gather the sharded slices to full vocab (compute_logits) and return
        # the full-vocab logits for the host to sample -- unchanged behaviour.
        if self.sampler is not None:
            local_logits = self._mask_pad_local(self._logits_local(hidden_states))
            sampled_tokens = self.sampler(
                local_logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
            )
            return sampled_tokens
        logits = self.compute_logits(hidden_states)
        return logits
