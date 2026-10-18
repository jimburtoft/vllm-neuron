# SPDX-License-Identifier: Apache-2.0
"""Whisper large-v3 (encoder-decoder) in BF16 for the native vLLM Neuron backend.

Design (the same as the 0.21 port):

* The runner sees a decoder-only model with a multimodal (audio) encoder. The
  runner calls ``embed_multimodal`` one time for each request, at prefill.
* ``audio_tower`` is a ``WhisperCrossKVEncoder``. It runs the encoder, projects
  the cross-attention K/V for each decoder layer, and writes them into
  model-owned ``register_buffer`` tensors. All of this occurs in one compiled
  graph. The aliasing FX pass keeps the in-place writes in HBM.
* The decoder self-attention uses the block-managed paged KV cache of the
  runner (``get_kv_spec`` / ``bind_kv_cache``).
* The decoder cross-attention reads the cross-KV buffers. They are not in
  ``get_kv_spec``, so the block manager does not see them.
* The LM head is tied to ``decoder.embed_tokens``. It is vocab-parallel and
  in fp32. The on-device sampler does a distributed greedy argmax.

The cross-KV buffers have one slot for each request (``max_num_seqs`` slots
plus one scratch slot). The runner gives each request a stable slot
(``RequestSlotTable``) and sends the slot of each batch row in
``attn_metadata[layer]["request_slots"]``.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ...     TP-sharded.
  # <-- MODEL-SPECIFIC: ...  Whisper-specific.

Whisper properties that are different from llama3:
  * No RoPE. The model adds learned positions to the token embeddings.
  * 20 attention heads (MHA). 20 % 8 != 0, so TP is 1, 2 or 4.
  * head_dim = 64.
  * q_proj and v_proj have a bias. k_proj has no bias.
  * out_proj and mlp.fc2 keep an fp32 bias. RowParallelLinear accepts only an
    fp32 bias at tp > 1.
  * The activation is erf GELU, written out. ``F.gelu`` breaks the full graph.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.nn.sampler import Sampler

from .config import WhisperConfig
from .weight_loaders import build_state, load_hf_state, resolve_checkpoint_dir

logger = logging.getLogger(__name__)

# Whisper always pads or cuts the audio to a 30 s window: 3000 mel frames, then
# 1500 encoder positions after the stride-2 convolution.
WHISPER_ENCODER_TOKENS = 1500
WHISPER_MEL_FRAMES = 3000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gelu(x):
    """Return the exact (erf) GELU, written out so that Dynamo can trace it."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """Return the OpenAI Whisper sinusoidal position embeddings (encoder)."""
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def tp_info() -> tuple[object | None, int, int]:
    """Return ``(device_group, world_size, rank)`` of the text TP group.

    Return ``(None, 1, 0)`` when the vLLM parallel state is not initialized,
    for example in a CPU unit test.
    """
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
    except (AssertionError, RuntimeError, ImportError):
        return None, 1, 0
    return group.device_group, group.world_size, group.rank_in_group


class RowParallelLinearFP32Bias(nn.Module):
    """A RowParallelLinear without bias, then an fp32 bias add.

    The add occurs after the all-reduce. Cast the bias to the activation dtype
    and add in that dtype. This order matches the byte-identical reference. An
    fp32 add gives a different bf16 rounding, which can change a near-tie at the
    first token.
    """

    def __init__(self, in_features, out_features, dtype, tp_group=None):
        super().__init__()
        self.rpl = RowParallelLinear(
            in_features, out_features, bias=False, dtype=dtype, tp_group=tp_group
        )
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))

    def forward(self, x):
        y = self.rpl(x)
        return y + self.bias.to(y.dtype)


def _cpl(in_f, out_f, bias, dtype, tp_group):
    return ColumnParallelLinear(
        in_f, out_f, bias=bias, gather_output=False, dtype=dtype, tp_group=tp_group
    )


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class EncoderSelfAttention(nn.Module):
    """Non-causal encoder self-attention. The softmax is in fp32."""

    def __init__(self, embed_dim, num_heads, dtype, tp_group, tp):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        if num_heads % tp != 0:
            raise ValueError(f"Encoder heads {num_heads} are not divisible by TP {tp}.")
        self.num_heads = num_heads // tp
        self.q_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.k_proj = _cpl(embed_dim, embed_dim, False, dtype, tp_group)
        self.v_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype, tp_group)

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        bsz, seqlen, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dtype, tp_group):
        super().__init__()
        self.fc1 = _cpl(embed_dim, ffn_dim, True, dtype, tp_group)
        self.fc2 = RowParallelLinearFP32Bias(ffn_dim, embed_dim, dtype, tp_group)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.self_attn = EncoderSelfAttention(
            d, cfg.encoder_attention_heads, dtype, tp_group, tp
        )
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.encoder_ffn_dim, dtype, tp_group)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class WhisperEncoder(nn.Module):
    """Audio encoder: mel ``[b, 128, 3000]`` to hidden ``[b, 1500, d]``."""

    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, dtype, tp_group, tp) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, input_features):
        x = gelu(self.conv1(input_features))
        x = gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class WhisperCrossKVEncoder(nn.Module):
    """The ``audio_tower`` graph: the encoder plus the cross-KV projection.

    The runner compiles this module as a separate graph. ``forward`` runs the
    encoder, projects K/V for each decoder layer, and writes them into the
    ``cross_k_{i}`` / ``cross_v_{i}`` buffers in place. The buffers belong to
    this module, so the aliasing FX pass adds an input-output alias for each
    write. The decode graph reads the same tensors.

    Each buffer is ``[num_slots, heads_local, 1500, head_dim]``. Clip ``j``
    goes to slot ``write_block_ids[j]``. The warmup of the worker sends
    ``write_block_ids`` as zeros, so warmup writes slot 0. Whisper does not use
    ``encoder_cache_buffer``.
    """

    def __init__(
        self,
        encoder: WhisperEncoder,
        cross_attns: nn.ModuleList,
        n_layers: int,
        n_heads_local: int,
        head_dim: int,
        n_ctx: int,
        dtype,
        num_slots: int = 2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.encoder = encoder
        # The same module objects as the decoder cross-attention layers. The
        # weights load under the decoder names. There is no copy.
        self.cross_attns = cross_attns
        self.n_layers = n_layers
        for i in range(n_layers):
            for kind in ("k", "v"):
                self.register_buffer(
                    f"cross_{kind}_{i}",
                    torch.zeros(num_slots, n_heads_local, n_ctx, head_dim, dtype=dtype),
                    persistent=False,
                )

    def forward(self, input_features, encoder_cache_buffer=None, write_block_ids=None):
        encoder_hidden = self.encoder(input_features)
        if write_block_ids is None:
            write_block_ids = torch.zeros(
                input_features.shape[0], dtype=torch.int64, device=input_features.device
            )
        slots = write_block_ids.to(torch.int64)
        for i in range(self.n_layers):
            k, v = self.cross_attns[i].project_kv(encoder_hidden)
            getattr(self, f"cross_k_{i}").index_put_((slots,), k)
            getattr(self, f"cross_v_{i}").index_put_((slots,), v)
        # Return a tensor, so that the warmup trace has an output.
        return encoder_hidden


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class CrossAttention(nn.Module):
    """Decoder cross-attention. Q comes from the decoder. K/V come from the buffers."""

    def __init__(self, embed_dim, num_heads, dtype, tp_group, tp):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_heads = num_heads // tp
        self.q_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.k_proj = _cpl(embed_dim, embed_dim, False, dtype, tp_group)
        self.v_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype, tp_group)

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, encoder_hidden_states):
        """Project the encoder output to cross K and V, ``[b, h, src, d]``."""
        bsz, src_len, _ = encoder_hidden_states.shape
        k = self._shape(self.k_proj(encoder_hidden_states), src_len, bsz)
        v = self._shape(self.v_proj(encoder_hidden_states), src_len, bsz)
        return k, v

    def attn_cached(self, hidden_states, cross_k, cross_v):
        bsz, tgt_len, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        attn = torch.matmul(q, cross_k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(cross_v.dtype)
        ctx = torch.matmul(attn, cross_v)
        out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        return self.out_proj(out)


class DecoderSelfAttention(nn.Module):
    """Causal self-attention on the block-managed paged KV cache.

    The runner allocates the K/V caches from ``get_kv_spec`` and binds them
    with ``bind_kv_cache``. The cache shape is
    ``[num_blocks, num_kv_heads, block_size, head_dim]``.
    """

    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype, tp_group, tp):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"decoder.layers.{layer_idx}.self_attn"
        d = cfg.d_model
        num_heads = cfg.decoder_attention_heads
        self.head_dim = d // num_heads
        self.scaling = self.head_dim**-0.5
        if num_heads % tp != 0:
            raise ValueError(
                f"Decoder heads {num_heads} are not divisible by TP {tp}. "
                "Use TP 1, 2 or 4 for Whisper large-v3."
            )
        self.num_heads = num_heads // tp
        # Whisper is MHA: the KV heads are the Q heads.
        self.num_key_value_heads_per_rank = self.num_heads

        self.q_proj = _cpl(d, d, True, dtype, tp_group)
        self.k_proj = _cpl(d, d, False, dtype, tp_group)
        self.v_proj = _cpl(d, d, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(d, d, dtype, tp_group)

        self.k_cache = None
        self.v_cache = None

    def _project(self, hidden_states):
        """Project ``[T, H]`` to q, k, v, each ``[heads_local, T, head_dim]``."""
        tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _write_kv_cache(self, k, v, slot_mapping, block_size):
        """Write K/V into the paged cache.

        ``paged_slot_indices`` sends a pad slot (-1) to the null block. Thus the
        scatter never gets an out-of-range address.
        """
        num_blocks = self.k_cache.shape[0]
        blk_idx, pos_idx = paged_slot_indices(slot_mapping, block_size, num_blocks)
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank
        k_f = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_f = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        h_idx = torch.arange(nkh, dtype=blk_idx.dtype, device=k.device).repeat_interleave(
            num_slots
        )
        index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
        self.k_cache.index_put_(index, k_f)
        self.v_cache.index_put_(index, v_f)

    def forward_prefill(self, hidden_states, meta):
        """Causal attention over the prompt, with the reference SDPA math.

        The decoder prompt has about 4 tokens, so a flash kernel gives no gain.
        The plain math keeps the first token byte-identical to the reference.
        The pad tokens are after the real tokens, so the causal mask hides them.
        """
        q, k, v = self._project(hidden_states)
        self._write_kv_cache(k, v, meta["slot_mapping"], meta["block_size"])

        tokens = hidden_states.shape[0]
        attn = torch.matmul(q, k.transpose(-1, -2))
        mask = torch.triu(
            torch.full((tokens, tokens), float("-inf"), dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = F.softmax((attn + mask).to(torch.float32), dim=-1).to(v.dtype)
        ctx = torch.matmul(attn, v)
        out = ctx.transpose(0, 1).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(out)

    def forward_decode(self, hidden_states, meta, positions):
        """Decode: write the new K/V, gather the sequence cache, and attend.

        The runner sets an unused block-table entry to -1. Change it to the
        null block (0) before the gather. The position mask hides each slot
        after the position of the query, which includes the null-block slots.
        """
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        block_table = torch.where(
            block_table < 0, torch.zeros_like(block_table), block_table
        )

        B_local = block_table.shape[0]
        tokens = hidden_states.shape[0]
        S_decode = tokens // B_local

        q, k, v = self._project(hidden_states)
        self._write_kv_cache(k, v, meta["slot_mapping"], block_size)

        nkh = self.num_key_value_heads_per_rank
        max_blocks = block_table.shape[1]
        S_ctx = max_blocks * block_size
        kc = self.k_cache[block_table]  # [B, max_blocks, nkh, block_size, d]
        vc = self.v_cache[block_table]
        kc = kc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)
        vc = vc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)

        qd = q.transpose(0, 1).reshape(B_local, S_decode, nkh, self.head_dim).transpose(1, 2)
        attn = torch.matmul(qd, kc.transpose(-1, -2))
        # Slot s is valid for query j only if s <= position(j). The gathered
        # slot index is the logical sequence position.
        pos = positions.to(torch.int32).view(B_local, S_decode)
        ar = torch.arange(S_ctx, device=attn.device, dtype=torch.int32)
        valid = ar.view(1, 1, S_ctx) <= pos.view(B_local, S_decode, 1)
        neg = torch.full((), float("-inf"), dtype=attn.dtype, device=attn.device)
        zero = torch.zeros((), dtype=attn.dtype, device=attn.device)
        attn = attn + torch.where(valid, zero, neg).unsqueeze(1)
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(vc.dtype)
        ctx = torch.matmul(attn, vc)
        ctx = ctx.transpose(1, 2).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(ctx)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.self_attn = DecoderSelfAttention(cfg, layer_idx, dtype, tp_group, tp)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.encoder_attn = CrossAttention(
            d, cfg.decoder_attention_heads, dtype, tp_group, tp
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.decoder_ffn_dim, dtype, tp_group)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x, cross_k, cross_v, meta, is_prefill, positions):
        # x is token-flat, [T, H]. cross_k and cross_v are [B, h, 1500, d], one
        # row for each request in the batch.
        h = self.self_attn_layer_norm(x)
        if is_prefill:
            sa = self.self_attn.forward_prefill(h, meta)
        else:
            sa = self.self_attn.forward_decode(h, meta, positions)
        x = x + sa

        tokens, hidden = x.shape
        num_rows = cross_k.shape[0]
        hc = self.encoder_attn_layer_norm(x).view(num_rows, tokens // num_rows, hidden)
        x = x + self.encoder_attn.attn_cached(hc, cross_k, cross_v).reshape(tokens, hidden)

        x = x + self.mlp(self.final_layer_norm(x))
        return x


class WhisperDecoder(nn.Module):
    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d, cfg.pad_token_id, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i, dtype, tp_group, tp) for i in range(cfg.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)
        self._cross_kv_owner: WhisperCrossKVEncoder | None = None

    def bind_cross_kv(self, owner: WhisperCrossKVEncoder) -> None:
        """Point the cross-KV reads at the module that owns the buffers.

        ``forward`` reads the buffers from the owner at each call. A ``.to()``
        or ``to_empty`` call replaces the buffer tensors, so a stored tensor
        reference becomes stale. ``object.__setattr__`` keeps the owner out of
        the decoder module tree, which prevents a module cycle.
        """
        object.__setattr__(self, "_cross_kv_owner", owner)

    def forward(self, input_ids, positions, attn_metadata, is_prefill):
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        owner = self._cross_kv_owner
        # All decoder layers are in one KV-cache group, so they share one
        # metadata dict. Read it one time: a lookup for each layer adds a
        # Dynamo guard for each layer key on every step.
        meta = attn_metadata[self.layers[0].self_attn.layer_name]
        # With one request slot, read slot 0 directly. A gather of the cross-KV
        # becomes an indirect DMA that copies all of the cross-KV on each step.
        slots = None if owner.num_slots <= 2 else self._request_slots(attn_metadata, x.device)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(owner, f"cross_k_{i}")
            cross_v = getattr(owner, f"cross_v_{i}")
            if slots is None:
                cross_k, cross_v = cross_k[:1], cross_v[:1]
            else:
                cross_k = cross_k.index_select(0, slots)
                cross_v = cross_v.index_select(0, slots)
            x = layer(x, cross_k, cross_v, meta, is_prefill, positions)
        return self.layer_norm(x)

    def _request_slots(self, attn_metadata, device) -> torch.Tensor:
        """Return the cross-KV slot of each batch row, ``[num_rows]``.

        The runner sends the slots in ``attn_metadata[layer]["request_slots"]``.
        A runner that does not send them runs one request at a time, so use
        slot 0.
        """
        meta = attn_metadata[self.layers[0].self_attn.layer_name]
        slots = meta.get("request_slots")
        if slots is None:
            num_rows = meta["block_table_tensor"].shape[0]
            slots = torch.zeros(num_rows, dtype=torch.int32, device=device)
        return slots


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class WhisperForConditionalGeneration(nn.Module):
    """Native Whisper large-v3 (encoder-decoder) for the vLLM Neuron runner.

    The runner reads ``supports_transcription`` and
    ``supports_transcription_only`` on this instance. It then reports only the
    "transcription" task. The serving class methods are on the registered
    factory class.
    """

    is_text_generation_model = True
    supports_transcription = True
    supports_transcription_only = True
    supports_multimodal = True
    # The runner gives each request a stable cross-KV slot and sends it in
    # attn_metadata["request_slots"] and embed_multimodal(request_slots=...).
    uses_request_slots = True

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        tp_group, world, rank = tp_info()
        self.tp_group = tp_group
        self.world_size = world
        self.rank = rank

        enc_cfg = config.encoder_neuron_config
        if enc_cfg is not None and (enc_cfg.tp_size != world or enc_cfg.dp_size != 1):
            raise ValueError(
                f"Whisper encoder_neuron_config has tp_size={enc_cfg.tp_size}, "
                f"dp_size={enc_cfg.dp_size}. Set tp_size={world} and dp_size=1. "
                "The encoder shards the cross-KV heads in the same way as the decoder."
            )

        self.encoder = WhisperEncoder(config, self.dtype, tp_group, world)
        self.decoder = WhisperDecoder(config, self.dtype, tp_group, world)

        # >>> PARALLELISM: the cross-KV heads have the decoder sharding.
        n_heads_local = config.decoder_attention_heads // world
        # One cross-KV slot for each request, plus the scratch slot.
        self.num_request_slots = config.max_num_seqs + 1
        self.audio_tower = WhisperCrossKVEncoder(
            self.encoder,
            nn.ModuleList([layer.encoder_attn for layer in self.decoder.layers]),
            n_layers=config.decoder_layers,
            n_heads_local=n_heads_local,
            head_dim=config.head_dim,
            n_ctx=config.max_source_positions,
            dtype=self.dtype,
            num_slots=self.num_request_slots,
        )
        self.decoder.bind_cross_kv(self.audio_tower)

        # >>> PARALLELISM: vocab-parallel fp32 LM head, tied to embed_tokens.
        # Pad the vocab to a multiple of TP. The pad columns get -inf before
        # the argmax, so they never win.
        vocab = config.vocab_size
        pad = (-vocab) % world
        self.padded_vocab = vocab + pad
        self.vocab_pad = pad
        self.per_rank = self.padded_vocab // world
        self.lm_head = ColumnParallelLinear(
            config.d_model,
            self.padded_vocab,
            bias=False,
            gather_output=False,
            dtype=torch.float32,
            tp_group=tp_group,
        )

        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self.sampler = None
        if self.on_device_sampling_config is not None:
            # The sampler does the cross-rank greedy argmax over the same group
            # that shards the vocab. On a tie, the lowest global index wins,
            # the same as an argmax over the full vocab.
            self.sampler = Sampler(self.on_device_sampling_config, process_group=tp_group)

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Give each graph its own neuronx-cc arguments.

        ``WHISPER_DECODE_OPT_LEVEL`` sets the ``-O`` level of the plain decode
        graph. ``WHISPER_ENCODER_OPT_LEVEL`` sets it for the encoder graph. When
        a variable is not set, the graph keeps the base arguments.

        Example:
            >>> model.compiler_args_for("prefill", None, ["-O2"])  # doctest: +SKIP
            ['-O2']
        """
        import os

        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        level = {
            "decode": os.environ.get("WHISPER_DECODE_OPT_LEVEL"),
            "encoder": os.environ.get("WHISPER_ENCODER_OPT_LEVEL"),
        }.get(graph_type)
        return with_optimization_level(base, int(level)) if level else base

    @property
    def dtype(self) -> torch.dtype:
        """Return the activation dtype.

        A property, not an attribute: Dynamo saves an unguarded module
        attribute as ``_Missing`` by object identity. A ``torch.dtype`` is a
        singleton, so an attribute that holds it also changes the dtype of each
        saved tensor to ``_Missing``, and the warm restore of the compiled
        package fails.
        """
        return self.config.torch_dtype

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
        **overrides,
    ) -> "WhisperForConditionalGeneration":
        """Build the model from the HF config and the neuron configs."""
        config = WhisperConfig.from_configs(
            hf_config, text_neuron_config, encoder_neuron_config, **overrides
        )
        return cls(config)

    # ── vLLM generative-model detection ───────────────────────────────────
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the full-vocab fp32 logits (the host-sampling path)."""
        local = self._logits_local(hidden_states)
        if self.world_size == 1:
            full = local
        else:
            from torch.distributed._functional_collectives import all_gather_tensor

            full = all_gather_tensor(local, local.dim() - 1, self.tp_group)
        return full[..., : self.config.vocab_size]

    def _logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.to(torch.float32))

    def _mask_pad_local(self, local: torch.Tensor, rank) -> torch.Tensor:
        """Set the local logits of the pad columns to -inf.

        SPMD: derive the global vocab index from the runtime ``rank`` tensor,
        so that all ranks trace the same graph.
        """
        if self.vocab_pad == 0:
            return local
        if rank is None:
            rank = torch.tensor(self.rank, dtype=torch.int32)
        offset = rank.to(device=local.device, dtype=torch.int32).reshape(()) * self.per_rank
        idx = torch.arange(self.per_rank, device=local.device, dtype=torch.int32) + offset
        neg = torch.full((), float("-inf"), dtype=local.dtype, device=local.device)
        return torch.where(idx < self.config.vocab_size, local, neg)

    def _pad_logit_mask(self, logit_mask):
        """Pad a full-vocab logit mask to the padded vocab (pad columns False)."""
        if logit_mask is None or logit_mask.shape[-1] == self.padded_vocab:
            return logit_mask
        extra = self.padded_vocab - logit_mask.shape[-1]
        pad = torch.zeros(
            (*logit_mask.shape[:-1], extra), dtype=logit_mask.dtype, device=logit_mask.device
        )
        return torch.cat([logit_mask, pad], dim=-1)

    # ── encoder (SupportsEncoderWarmup) ───────────────────────────────────
    def get_encoder(self) -> nn.Module:
        return self.audio_tower

    def build_encoder_synthetic_inputs(
        self,
        bucket: int,
        encoder_neuron_config: EncoderNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return a zero mel batch for one encoder bucket.

        A bucket counts encoder tokens. One clip has 1500 tokens, so a bucket
        of ``n * 1500`` holds ``n`` clips.
        """
        num_clips = max(1, bucket // WHISPER_ENCODER_TOKENS)
        return {
            "input_features": torch.zeros(
                num_clips,
                self.config.num_mel_bins,
                WHISPER_MEL_FRAMES,
                dtype=self.dtype,
                device=device,
            )
        }

    def embed_multimodal(
        self,
        input_features: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        request_slots: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Run the audio encoder and write the cross-KV slot of each clip.

        Whisper does not write the encoder cache buffer. The runner still reads
        the cache bookkeeping at prefill, so reserve one block (1500 tokens) for
        each clip. Without the reservation, prefill raises
        ``EncoderCacheReadError``.

        Args:
            input_features: Mel features ``[b, 128, 3000]``.
            encoder_cache: The ``EncoderCacheBlocks`` of the runner.
            mm_hashes: One identifier for each clip.
            request_slots: The cross-KV slot of each clip. When the runner
                does not send slots, it runs one request at a time, so use
                slot 0.
        """
        if input_features is None:
            raise ValueError(
                "Whisper embed_multimodal received no input_features. "
                "Send the audio through the Whisper multimodal processor."
            )
        if isinstance(input_features, (list, tuple)):
            input_features = torch.stack(list(input_features))
        mel = input_features
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        num_clips = mel.shape[0]
        if request_slots is None:
            if num_clips != 1 or self.config.max_num_seqs != 1:
                raise ValueError(
                    f"Whisper received {num_clips} clips without request slots "
                    f"(max_num_seqs={self.config.max_num_seqs}). Use a runner that "
                    "sends request_slots, or set max_num_seqs=1."
                )
            request_slots = [0]
        if len(request_slots) != num_clips:
            raise ValueError(
                f"Whisper received {num_clips} clips and {len(request_slots)} "
                "request slots. Send one slot for each clip."
            )
        if mm_hashes is None or len(mm_hashes) != num_clips:
            raise ValueError(
                f"Whisper embed_multimodal expects one mm_hash for each clip: "
                f"{num_clips} clips, {0 if mm_hashes is None else len(mm_hashes)} hashes."
            )
        if encoder_cache is not None:
            for mm_hash in mm_hashes:
                encoder_cache.allocate(mm_hash, [WHISPER_ENCODER_TOKENS])

        device = next(self.encoder.parameters()).device
        # Cast on the source device first. A combined device and dtype .to() on
        # Neuron raises "Expected self.dtype() == dst.dtype()".
        mel = mel.to(dtype=self.dtype).to(device=device)
        slots = torch.tensor(request_slots, dtype=torch.int64).to(device=device)
        self.audio_tower(mel, write_block_ids=slots)

    # ── KV cache ──────────────────────────────────────────────────────────
    def get_kv_spec(self) -> KVSpec:
        layers = [
            LayerSpec(
                name=layer.self_attn.layer_name,
                num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                head_size=layer.self_attn.head_dim,
                dtype=self.dtype,
            )
            for layer in self.decoder.layers
        ]
        # The cross-KV buffers are model-owned and not in the spec.
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for layer in self.decoder.layers:
            name = layer.self_attn.layer_name
            if name not in kv_caches:
                raise KeyError(f"The KV cache for layer {name} is not initialized.")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    # ── weights ───────────────────────────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load the HF safetensors into the encoder and the decoder.

        The runner builds the model on the meta device. Materialize all storage
        on CPU, load the weights, and tie the LM head. The runner then moves the
        model to the device.
        """
        self.to_empty(device="cpu")
        with torch.no_grad():
            for name, buf in self.audio_tower.named_buffers(recurse=False):
                buf.zero_()
        self.decoder.bind_cross_kv(self.audio_tower)

        hf_sd = load_hf_state(resolve_checkpoint_dir(checkpoint_path, cache_dir))
        enc_state = build_state(hf_sd, "model.encoder.", self.dtype)
        dec_state = build_state(hf_sd, "model.decoder.", self.dtype)

        enc_missing, enc_unexpected = self.encoder.load_state_dict(
            enc_state, strict=False, assign=True
        )
        dec_missing, dec_unexpected = self.decoder.load_state_dict(
            dec_state, strict=False, assign=True
        )
        if "embed_positions.weight" in enc_missing:
            enc_missing.remove("embed_positions.weight")
            with torch.no_grad():
                self.encoder.embed_positions.weight.copy_(
                    sinusoids(self.config.max_source_positions, self.config.d_model).to(
                        self.dtype
                    )
                )
        self._tie_lm_head()

        problems = enc_missing + enc_unexpected + dec_missing + dec_unexpected
        logger.info(
            "Whisper load_weights: encoder missing=%d unexpected=%d, "
            "decoder missing=%d unexpected=%d",
            len(enc_missing),
            len(enc_unexpected),
            len(dec_missing),
            len(dec_unexpected),
        )
        if problems:
            raise RuntimeError(
                "Whisper weight names do not match the checkpoint: "
                f"encoder missing={enc_missing} unexpected={enc_unexpected}, "
                f"decoder missing={dec_missing} unexpected={dec_unexpected}."
            )

    def _tie_lm_head(self) -> None:
        """Copy the vocab slice of this rank from embed_tokens into the LM head.

        The rows after the real vocab stay zero. The argmax mask hides them.
        Call this method on CPU, after the decoder weights load.
        """
        vocab = self.config.vocab_size
        start = self.rank * self.per_rank
        end = min(start + self.per_rank, vocab)
        w = self.decoder.embed_tokens.weight.detach().to("cpu")
        shard = torch.zeros(self.per_rank, w.shape[1], dtype=self.lm_head.weight.dtype)
        if end > start:
            shard[: end - start] = w[start:end].to(shard.dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_(shard.to(self.lm_head.weight.device))

    # ── forward ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        # The runner adds these prefill inputs for every multimodal model.
        # Whisper reads the cross-KV buffers instead, so it ignores them.
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        meta = attn_metadata[self.decoder.layers[0].self_attn.layer_name]
        is_prefill = meta["max_query_len"] > meta["decode_token_threshold"]

        hidden_states = self.decoder(input_ids, positions, attn_metadata, is_prefill)
        if sampling_positions is not None:
            hidden_states = torch.index_select(hidden_states, 0, sampling_positions)

        if self.sampler is None:
            return self.compute_logits(hidden_states)

        local = self._mask_pad_local(self._logits_local(hidden_states), rank)
        sampled = self.sampler(
            local,
            sampling_params,
            logit_mask=self._pad_logit_mask(logit_mask),
            tp_rank=rank,
        )
        return sampled, None
