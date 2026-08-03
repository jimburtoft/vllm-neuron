# SPDX-License-Identifier: Apache-2.0
"""
Clean-room Whisper (large-v3) modeling for the vllm-neuron NATIVE backend
(torch.compile(backend="vllm_neuron")).

Task 003 (correctness before speed): self-contained nn.Modules built from the
plugin's ColumnParallelLinear / RowParallelLinear + plain torch attention math;
naive decode (recompute-full-prefix, no KV cache).

Task 004 (KV cache): adds

  * CROSS-attention KV cache -- the encoder output is projected through every
    decoder layer's cross-attn K and V ONCE at prefill and stored as persistent
    HBM buffers (register_buffer(persistent=False) + in-place .copy_(), the
    PyTorch-native HBM-persistence pattern; input_output_aliases is XLA-only and
    unavailable on the torch.compile(backend=...) path). Decode steps compute
    only Q and read the cached cross-K/V -- no per-step recompute, no CPU
    round-trip.

  * SELF-attention KV cache -- a growing cache; each decode step writes the new
    token's K/V at the current position via index_copy_. Decode is restructured
    into (a) a prefill graph that populates both caches for the SOT prompt, and
    (b) a [1,1] decode-step graph that appends one token.

Both caches are module state (register_buffer) so they persist in HBM across the
compiled decode-step graph's repeated calls. The buffers are read/written
in-place; the graph never re-runs the encoder projection.

Not copied from NxDI's modeling_whisper -- referenced for design only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

try:  # on-device collective for the sharded LM-head global argmax
    from torch.distributed._functional_collectives import all_gather_tensor
except Exception:  # pragma: no cover
    all_gather_tensor = None

try:
    from torch.distributed._functional_collectives import all_reduce as _fc_all_reduce
except Exception:  # pragma: no cover
    _fc_all_reduce = None


# --------------------------------------------------------------------------- #
# Lever #2 (Task 007): pre-transposed weights.
#
# The plugin CPL/RPL compute F.linear(x, W) = x @ W.T, and neuronx-cc lowers the
# implicit W.T as a transpose-matmul on the PE array every step. For M=1 decode,
# 79-92% of the decode transpose-matmul time (16.9 ms of TE-matmul) is these
# weight-orientation transposes (measured from the Task 006 decode parquet:
# 48853 128x128 weight tiles = 13.3 ms + te=1048576 = 2.24 ms). We pre-transpose
# the STATIC weights once into a persistent [in, out] buffer and compute
# torch.matmul(x, Wt) directly, removing the runtime transpose.
#
# enable_pretranspose() rebinds the forward of every CPL/RPL under a module AFTER
# weights are loaded and moved to device. Correctness is byte-identical: x @ W.T
# == x @ (W.T) with the same operands, just a different lowering.
# --------------------------------------------------------------------------- #
def _cpl_forward_pt(self, x):
    # ColumnParallelLinear: local = x @ Wt (+bias); optional all-gather.
    local = torch.matmul(x, self._wt)
    if self.bias is not None:
        local = local + self.bias
    if self.tp_size == 1 or not self.gather_output:
        return local
    return all_gather_tensor(local, x.dim() - 1, self.tp_group)


def _rpl_forward_pt(self, x):
    # RowParallelLinear: local = x @ Wt; all-reduce; +bias.
    if self.tp_size == 1:
        out = torch.matmul(x, self._wt)
        if self.bias is not None:
            out = out + self.bias
        return out
    if not self.input_is_parallel:
        x = list(torch.chunk(x, self.tp_size, dim=-1))[self.tp_rank]
    local = torch.matmul(x, self._wt)
    local = _fc_all_reduce(local, reduceOp="sum", group=self.tp_group)
    if self.bias is not None:
        local = local + self.bias
    return local


def enable_pretranspose(module):
    """Rebind CPL/RPL forwards under `module` to use pre-transposed weights.
    Call AFTER load_state_dict + tie + .to(device)."""
    import types
    n = 0
    for m in module.modules():
        if isinstance(m, ColumnParallelLinear):
            with torch.no_grad():
                dev = m.weight.device
                wt = m.weight.detach().to("cpu").t().contiguous().to(dev)
                m._wt = wt
            m.forward = types.MethodType(_cpl_forward_pt, m)
            n += 1
        elif isinstance(m, RowParallelLinear):
            with torch.no_grad():
                dev = m.weight.device
                wt = m.weight.detach().to("cpu").t().contiguous().to(dev)
                m._wt = wt
            m.forward = types.MethodType(_rpl_forward_pt, m)
            n += 1
    return n


class RowParallelLinearFP32Bias(nn.Module):
    """Wrap the plugin RowParallelLinear (bias=False) + a separate fp32 bias.

    The plugin's RowParallelLinear raises NotImplementedError for non-fp32 bias
    when tp_size>1 (XLA constant-inlining issue). Whisper out_proj/fc2 have bias,
    so we keep the bias as fp32, added after the all-reduce, and cast back.
    """

    def __init__(self, in_features, out_features, dtype):
        super().__init__()
        self.rpl = RowParallelLinear(in_features, out_features, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self._out_dtype = dtype

    def forward(self, x):
        y = self.rpl(x)
        y = y + self.bias.to(y.dtype)
        return y


def gelu(x):
    """Exact (erf) GELU written out so Dynamo traces it (F.gelu is a skipped
    C builtin under libtorch_neuronx_lite and breaks fullgraph=True)."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """OpenAI Whisper sinusoidal position embeddings."""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def _tp_world():
    import torch.distributed as dist

    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class WhisperConfigLite:
    """Minimal config carrier so we don't depend on transformers at model build."""

    def __init__(self, hf_cfg):
        self.d_model = hf_cfg.d_model
        self.encoder_layers = hf_cfg.encoder_layers
        self.decoder_layers = hf_cfg.decoder_layers
        self.encoder_attention_heads = hf_cfg.encoder_attention_heads
        self.decoder_attention_heads = hf_cfg.decoder_attention_heads
        self.encoder_ffn_dim = hf_cfg.encoder_ffn_dim
        self.decoder_ffn_dim = hf_cfg.decoder_ffn_dim
        self.num_mel_bins = hf_cfg.num_mel_bins
        self.max_source_positions = hf_cfg.max_source_positions
        self.max_target_positions = hf_cfg.max_target_positions
        self.vocab_size = hf_cfg.vocab_size
        self.pad_token_id = getattr(hf_cfg, "pad_token_id", 0)
        self.activation_function = getattr(hf_cfg, "activation_function", "gelu")
        # Footgun 3: assert large-v3 dims on load.
        assert self.d_model == 1280, f"expected d_model=1280 got {self.d_model}"
        assert self.vocab_size == 51866, (
            f"expected vocab_size=51866 got {self.vocab_size}"
        )
        assert self.encoder_attention_heads == 20, (
            f"expected n_heads=20 got {self.encoder_attention_heads}"
        )


# --------------------------------------------------------------------------- #
# Attention (naive, single-graph friendly) -- Task 003 path, kept for encoder
# and for correctness fallback.
# --------------------------------------------------------------------------- #
class SelfAttention(nn.Module):
    """Encoder (non-causal) or decoder (causal) self-attention.

    QKV projected with ColumnParallelLinear (head-parallel), output with
    RowParallelLinear (all-reduce). Attention math is plain SDPA so the graph
    is a single NEFF.

    Task 004: the decoder self-attn ALSO exposes cache-aware entry points
    (project_kv, attn_with_cache) used by the prefill and decode-step graphs.
    """

    def __init__(self, embed_dim, num_heads, causal, dtype):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.scaling = self.head_dim ** -0.5
        tp = _tp_world()
        assert num_heads % tp == 0
        self.num_heads = num_heads // tp

        # Whisper: q has bias, k has NO bias, v has bias.
        self.q_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        self.k_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.v_proj = ColumnParallelLinear(embed_dim, embed_dim, bias=True, dtype=dtype)
        # RowParallelLinear requires fp32 bias when tp>1; out_proj has bias.
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype)

    def _shape(self, x, seqlen, bsz):
        # [b, s, h_local*d] -> [b, h_local, s, d]
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = self._shape(q, seqlen, bsz)
        k = self._shape(k, seqlen, bsz)
        v = self._shape(v, seqlen, bsz)

        attn = torch.matmul(q, k.transpose(-1, -2))  # already scaled q
        if self.causal:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), dtype=attn.dtype, device=attn.device
            )
            mask = torch.triu(mask, diagonal=1)
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [b, h_local, s, d]
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out

    # ---- Task 004: cache-aware entry points (decoder only) ----
    def project_qkv(self, hidden_states):
        """Project q (scaled), k, v and reshape to [b, h_local, s, d]."""
        bsz, seqlen, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        return q, k, v

    def attn_out(self, ctx, bsz, seqlen):
        """ctx: [b, h_local, s, d] -> out_proj([b, s, h_local*d])."""
        out = ctx.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Decoder cross-attention: Q from decoder, K/V from encoder output.

    Task 004: K/V are computed ONCE from the encoder output (project_kv) and
    stored in persistent HBM buffers by the owning DecoderLayer. Decode reads
    the cached K/V and computes only Q.
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

    def forward(self, hidden_states, encoder_hidden_states):
        # Task 003 naive path (recompute K/V). Kept for reference/fallback.
        bsz, tgt_len, _ = hidden_states.shape
        src_len = encoder_hidden_states.shape[1]
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, src_len, bsz)
        v = self._shape(v, src_len, bsz)

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out

    # ---- Task 004 ----
    def project_kv(self, encoder_hidden_states):
        """Project encoder output -> cross K,V once. [b, h_local, src, d]."""
        bsz, src_len, _ = encoder_hidden_states.shape
        k = self._shape(self.k_proj(encoder_hidden_states), src_len, bsz)
        v = self._shape(self.v_proj(encoder_hidden_states), src_len, bsz)
        return k, v

    def attn_cached(self, hidden_states, cross_k, cross_v):
        """Q from decoder tokens; K/V read from the cached buffers (no recompute)."""
        bsz, tgt_len, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        attn = torch.matmul(q, cross_k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, cross_v)  # [b, h_local, tgt, d]
        out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dtype):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, ffn_dim, bias=True, dtype=dtype)
        self.fc2 = RowParallelLinearFP32Bias(ffn_dim, embed_dim, dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        return x


# --------------------------------------------------------------------------- #
# Lever #1 (Task 007): sharded LM head + on-device greedy argmax.
#
# The LM head weight [vocab, d] is tied to decoder.embed_tokens. In the baseline
# it was read UNSHARDED every decode step (132.8 MB/step) and its full-vocab
# F.linear is a large REGULAR matmul that does not scale with TP. We shard the
# vocab dimension across TP ranks with ColumnParallelLinear (shard_dim=0), so each
# rank computes only its vocab slice's logits, then pick the global argmax ON
# DEVICE via a tiny all-gather of the per-rank (max_value, global_index) pair --
# avoiding both the unsharded weight re-read and the full-vocab CPU gather.
#
# vocab_size=51866 is not divisible by 4, so we pad to a multiple of tp; the
# padded logit columns are masked to -inf on the owning rank so they never win.
# --------------------------------------------------------------------------- #
class ShardedLMHead(nn.Module):
    def __init__(self, vocab_size, d_model, dtype):
        super().__init__()
        self.vocab_size = vocab_size
        tp = _tp_world()
        self.tp = tp
        # pad vocab up to a multiple of tp
        pad = (-vocab_size) % tp
        self.padded_vocab = vocab_size + pad
        self.pad = pad
        self.per_rank = self.padded_vocab // tp
        # ColumnParallelLinear shards dim 0 (output/vocab). No bias (Whisper tie).
        self.proj = ColumnParallelLinear(
            d_model, self.padded_vocab, bias=False, gather_output=False, dtype=dtype
        )
        # rank offset into the global (padded) vocab
        try:
            self.tp_rank = self.proj.tp_rank
        except Exception:
            self.tp_rank = 0
        self.vocab_start = self.tp_rank * self.per_rank
        # per-rank count of *valid* (unpadded) vocab logits this rank owns
        n_valid = max(0, min(self.per_rank, vocab_size - self.vocab_start))
        self.n_valid = n_valid

    def logits_local(self, x):
        """x: [b, s, d] -> local logits [b, s, per_rank] (this rank's vocab slice)."""
        return self.proj(x)

    def greedy_token(self, x):
        """x: [b, 1, d] -> global argmax token id as a 0-d/[1] device long tensor.

        Byte-identical to torch.argmax over the full-vocab logits including tie
        handling (lowest index wins): we compute each rank's local (max, argmax),
        map argmax to the global index, then all-gather and pick the rank with the
        highest max, breaking ties by lowest global index.
        """
        local = self.proj(x)[0, 0]  # [per_rank] logits for this rank's slice
        # mask padded columns on the owning rank so they can never win
        if self.n_valid < self.per_rank:
            neg = torch.full((), float("-inf"), dtype=local.dtype, device=local.device)
            ar = torch.arange(self.per_rank, device=local.device)
            local = torch.where(ar < self.n_valid, local, neg)
        local_f = local.float()
        local_max = torch.max(local_f)                 # scalar
        local_arg = torch.argmax(local_f)              # scalar (lowest idx on tie)
        global_idx = local_arg.to(torch.float32) + float(self.vocab_start)
        if self.tp == 1 or all_gather_tensor is None:
            return local_arg.to(torch.long).view(1) + self.vocab_start
        # pack (max, global_idx) per rank -> all_gather -> pick global argmax
        pair = torch.stack([local_max, global_idx]).view(1, 2)   # [1,2]
        gathered = all_gather_tensor(pair, 0, self.proj.tp_group)  # [tp,2]
        maxes = gathered[:, 0]
        idxs = gathered[:, 1]
        # winner = rank with highest max; tie -> lowest global idx.
        best = torch.max(maxes)
        is_best = maxes >= best  # bf16/fp32 exact-equal ties
        big = torch.full_like(idxs, float("inf"))
        cand = torch.where(is_best, idxs, big)
        tok = torch.min(cand).to(torch.long).view(1)
        return tok


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        d = cfg.d_model
        self.self_attn = SelfAttention(d, cfg.encoder_attention_heads, causal=False, dtype=dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.encoder_ffn_dim, dtype)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class WhisperEncoder(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d, dtype=dtype)
        with torch.no_grad():
            self.embed_positions.weight.copy_(
                sinusoids(cfg.max_source_positions, d).to(dtype)
            )
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, dtype) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, input_features):
        # input_features: [b, num_mel_bins, 3000]
        x = gelu(self.conv1(input_features))
        x = gelu(self.conv2(x))
        x = x.transpose(1, 2)  # [b, 1500, d]
        x = x + self.embed_positions.weight[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        return x


# --------------------------------------------------------------------------- #
# Decoder (Task 004: KV-cache-aware)
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        d = cfg.d_model
        self.self_attn = SelfAttention(d, cfg.decoder_attention_heads, causal=True, dtype=dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.encoder_attn = CrossAttention(d, cfg.decoder_attention_heads, dtype)
        self.encoder_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.decoder_ffn_dim, dtype)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    # ---- Task 003 naive path (kept for fallback) ----
    def forward(self, x, encoder_hidden_states):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_hidden_states)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    # ---- Task 004: prefill (populates self-KV cache slots [0:prompt_len)) ----
    def forward_prefill(self, x, cross_k, cross_v, self_k_cache, self_v_cache):
        """
        x: [1, P, d] SOT prompt hidden states.
        cross_k/cross_v: cached cross-attn K/V [1, h, 1500, d] (already populated).
        self_k_cache/self_v_cache: HBM buffers [1, h, MAXLEN, d]; this returns the
        new K/V for positions [0:P) to be written by the caller (index_copy_).
        Returns (layer_out, new_self_k[1,h,P,d], new_self_v[1,h,P,d]).
        """
        bsz, P, _ = x.shape
        residual = x
        h = self.self_attn_layer_norm(x)
        q, k, v = self.self_attn.project_qkv(h)  # each [1,h,P,d]
        # causal self-attention over the prompt
        attn = torch.matmul(q, k.transpose(-1, -2))
        mask = torch.triu(
            torch.full((P, P), float("-inf"), dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)
        sa = self.self_attn.attn_out(ctx, bsz, P)
        x = residual + sa

        residual = x
        h = self.encoder_attn_layer_norm(x)
        ca = self.encoder_attn.attn_cached(h, cross_k, cross_v)
        x = residual + ca

        residual = x
        h = self.final_layer_norm(x)
        x = residual + self.mlp(h)
        return x, k, v


class WhisperDecoder(nn.Module):
    def __init__(self, cfg, dtype, max_self_len=448):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.max_self_len = max_self_len
        d = cfg.d_model
        tp = _tp_world()
        self.n_heads_local = cfg.decoder_attention_heads // tp
        self.head_dim = d // cfg.decoder_attention_heads
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d, cfg.pad_token_id, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, dtype) for _ in range(cfg.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

        # ---- Lever #1 (Task 007): sharded LM head (vocab-parallel) ----
        # proj_out is tied to embed_tokens in Whisper; we copy the tied weight into
        # the ColumnParallelLinear (its _load_from_state_dict shards dim 0). When
        # tp==1 this is equivalent to F.linear(x, embed_tokens.weight).
        self.use_sharded_lmhead = True
        self.lm_head = ShardedLMHead(cfg.vocab_size, d, dtype)

        # ---- Task 004 persistent HBM caches (register_buffer, persistent=False) ----
        # Cross-attn K/V: [1, h_local, 1500, head_dim] per layer. Populated once.
        # Self-attn K/V: [1, h_local, max_self_len, head_dim] per layer. Grows.
        n_ctx = cfg.max_source_positions  # 1500
        for i in range(cfg.decoder_layers):
            self.register_buffer(
                f"cross_k_{i}",
                torch.zeros(1, self.n_heads_local, n_ctx, self.head_dim, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"cross_v_{i}",
                torch.zeros(1, self.n_heads_local, n_ctx, self.head_dim, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"self_k_{i}",
                torch.zeros(1, self.n_heads_local, max_self_len, self.head_dim, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                f"self_v_{i}",
                torch.zeros(1, self.n_heads_local, max_self_len, self.head_dim, dtype=dtype),
                persistent=False,
            )

    # ---- Task 003 naive path (kept for fallback) ----
    def forward(self, input_ids, positions, encoder_hidden_states):
        x = self.embed_tokens(input_ids)
        pos = self.embed_positions(positions)
        x = x + pos
        for layer in self.layers:
            x = layer(x, encoder_hidden_states)
        x = self.layer_norm(x)
        return x

    # ---- Task 004 ----
    def precompute_cross_kv(self, encoder_hidden_states):
        """Project the encoder output through every layer's cross-attn K/V ONCE
        and store into the persistent buffers via in-place .copy_() (HBM-resident,
        survives across decode-step graph calls). This runs its own compiled graph
        (or eager); the decode-step graph only READS these buffers."""
        for i, layer in enumerate(self.layers):
            k, v = layer.encoder_attn.project_kv(encoder_hidden_states)  # [1,h,1500,d]
            getattr(self, f"cross_k_{i}").copy_(k)
            getattr(self, f"cross_v_{i}").copy_(v)

    def prefill(self, input_ids, positions):
        """Prefill graph. input_ids/positions: [1, P] SOT prompt.
        Writes self-KV cache positions [0:P) and returns logits [1, P, vocab].
        Reads cross-KV buffers (already populated by precompute_cross_kv)."""
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        P = input_ids.shape[1]
        idx = torch.arange(P, device=input_ids.device)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self, f"cross_k_{i}")
            cross_v = getattr(self, f"cross_v_{i}")
            skc = getattr(self, f"self_k_{i}")
            svc = getattr(self, f"self_v_{i}")
            x, new_k, new_v = layer.forward_prefill(x, cross_k, cross_v, skc, svc)
            # write prompt K/V into the growing cache at [0:P)
            skc.index_copy_(2, idx, new_k)
            svc.index_copy_(2, idx, new_v)
        x = self.layer_norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

    def decode_step(self, input_id, position, cur_pos):
        """Decode-step graph. input_id/position: [1,1]. cur_pos: 0-D long tensor
        index into the self-KV cache to write the new token. Returns logits [1,1,vocab].
        Reads cross-KV + self-KV buffers; writes the new token's K/V in-place."""
        x = self.embed_tokens(input_id) + self.embed_positions(position)
        maxlen = self.max_self_len
        # additive mask: allow positions [0..cur_pos], forbid the rest.
        ar = torch.arange(maxlen, device=input_id.device)
        allowed = (ar <= cur_pos)  # [maxlen]
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=self.dtype, device=input_id.device),
            torch.full((), float("-inf"), dtype=self.dtype, device=input_id.device),
        ).view(1, 1, 1, maxlen)
        cur_idx = cur_pos.view(1)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self, f"cross_k_{i}")
            cross_v = getattr(self, f"cross_v_{i}")
            skc = getattr(self, f"self_k_{i}")
            svc = getattr(self, f"self_v_{i}")
            # project this token's K/V and write to cache BEFORE attention
            h = layer.self_attn_layer_norm(x)
            q, k_new, v_new = layer.self_attn.project_qkv(h)  # [1,h,1,d]
            skc.index_copy_(2, cur_idx, k_new)
            svc.index_copy_(2, cur_idx, v_new)
            # self-attention against the full cache
            attn = torch.matmul(q, skc.transpose(-1, -2)) + mask
            attn = F.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, svc)
            sa = layer.self_attn.attn_out(ctx, 1, 1)
            x = x + sa
            # cross-attention (cached)
            hc = layer.encoder_attn_layer_norm(x)
            ca = layer.encoder_attn.attn_cached(hc, cross_k, cross_v)
            x = x + ca
            # mlp
        x = self.layer_norm(x)
        logits = F.linear(x, self.embed_tokens.weight)
        return logits

    # ------------------------------------------------------------------ #
    # Lever #1 (Task 007): tie + greedy variants using the sharded LM head
    # ------------------------------------------------------------------ #
    def tie_lm_head(self):
        """Copy the tied embed_tokens weight into the sharded LM head, padding the
        vocab to a multiple of tp with zero rows (masked to -inf at argmax time).
        Call AFTER load_state_dict + .to(device).

        The shard slice is materialized on CPU (the padded region and the large
        device-side slice+copy trip nrt_tensor_copy on the non-zero ranks), then the
        exact per-rank [per_rank, d] block is copied into the on-device weight."""
        vocab = self.lm_head.vocab_size
        per_rank = self.lm_head.per_rank
        s = self.lm_head.vocab_start
        e = s + per_rank
        d = self.embed_tokens.weight.shape[1]
        w_cpu = self.embed_tokens.weight.detach().to("cpu")  # [vocab, d]
        shard = torch.zeros(per_rank, d, dtype=self.lm_head.proj.weight.dtype)
        # rows [s:e) mapped from the (unpadded) embed weight; rows >= vocab stay 0.
        valid_e = min(e, vocab)
        if valid_e > s:
            shard[: valid_e - s] = w_cpu[s:valid_e].to(shard.dtype)
        with torch.no_grad():
            self.lm_head.proj.weight.copy_(shard.to(self.lm_head.proj.weight.device))

    def _decoder_body_step(self, input_id, position, cur_pos):
        """Shared [1,1] decoder body (everything except the LM head). Returns the
        final normed hidden state x [1,1,d]."""
        x = self.embed_tokens(input_id) + self.embed_positions(position)
        maxlen = self.max_self_len
        ar = torch.arange(maxlen, device=input_id.device)
        allowed = (ar <= cur_pos)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=self.dtype, device=input_id.device),
            torch.full((), float("-inf"), dtype=self.dtype, device=input_id.device),
        ).view(1, 1, 1, maxlen)
        cur_idx = cur_pos.view(1)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self, f"cross_k_{i}")
            cross_v = getattr(self, f"cross_v_{i}")
            skc = getattr(self, f"self_k_{i}")
            svc = getattr(self, f"self_v_{i}")
            h = layer.self_attn_layer_norm(x)
            q, k_new, v_new = layer.self_attn.project_qkv(h)
            skc.index_copy_(2, cur_idx, k_new)
            svc.index_copy_(2, cur_idx, v_new)
            attn = torch.matmul(q, skc.transpose(-1, -2)) + mask
            attn = F.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, svc)
            sa = layer.self_attn.attn_out(ctx, 1, 1)
            x = x + sa
            hc = layer.encoder_attn_layer_norm(x)
            ca = layer.encoder_attn.attn_cached(hc, cross_k, cross_v)
            x = x + ca
            hm = layer.final_layer_norm(x)
            x = x + layer.mlp(hm)
        x = self.layer_norm(x)
        return x

    def decode_step_greedy(self, input_id, position, cur_pos):
        """Lever #1: decode-step that returns the greedy token id ([1] long) chosen
        ON DEVICE from the sharded LM head, instead of the full-vocab logits."""
        x = self._decoder_body_step(input_id, position, cur_pos)
        return self.lm_head.greedy_token(x)

    def decode_step_diag_indep(self, input_id, position, cur_pos):
        """DIAGNOSTIC (NOT byte-identical): run all 32 layers on the SAME input x0,
        summing outputs, so there is NO residual dependency chain between layers.
        Used only to measure the DMA-concurrency CEILING the compiler can reach when
        layer weight-loads are mutually independent. Throwaway — do not ship."""
        x0 = self.embed_tokens(input_id) + self.embed_positions(position)
        maxlen = self.max_self_len
        ar = torch.arange(maxlen, device=input_id.device)
        allowed = (ar <= cur_pos)
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=self.dtype, device=input_id.device),
            torch.full((), float("-inf"), dtype=self.dtype, device=input_id.device),
        ).view(1, 1, 1, maxlen)
        cur_idx = cur_pos.view(1)
        acc = x0
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self, f"cross_k_{i}")
            cross_v = getattr(self, f"cross_v_{i}")
            skc = getattr(self, f"self_k_{i}")
            svc = getattr(self, f"self_v_{i}")
            x = x0  # independent: every layer reads x0, not the prior layer's output
            h = layer.self_attn_layer_norm(x)
            q, k_new, v_new = layer.self_attn.project_qkv(h)
            skc.index_copy_(2, cur_idx, k_new)
            svc.index_copy_(2, cur_idx, v_new)
            attn = torch.matmul(q, skc.transpose(-1, -2)) + mask
            attn = F.softmax(attn, dim=-1)
            ctx = torch.matmul(attn, svc)
            sa = layer.self_attn.attn_out(ctx, 1, 1)
            x = x + sa
            hc = layer.encoder_attn_layer_norm(x)
            ca = layer.encoder_attn.attn_cached(hc, cross_k, cross_v)
            x = x + ca
            hm = layer.final_layer_norm(x)
            x = x + layer.mlp(hm)
            acc = acc + x
        acc = self.layer_norm(acc)
        return F.linear(acc, self.embed_tokens.weight)

    def prefill_greedy(self, input_ids, positions):
        """Lever #1: prefill that returns the greedy token id ([1] long) for the
        LAST prompt position, chosen on device from the sharded LM head."""
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        P = input_ids.shape[1]
        idx = torch.arange(P, device=input_ids.device)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(self, f"cross_k_{i}")
            cross_v = getattr(self, f"cross_v_{i}")
            skc = getattr(self, f"self_k_{i}")
            svc = getattr(self, f"self_v_{i}")
            x, new_k, new_v = layer.forward_prefill(x, cross_k, cross_v, skc, svc)
            skc.index_copy_(2, idx, new_k)
            svc.index_copy_(2, idx, new_v)
        x = self.layer_norm(x)
        last = x[:, -1:, :]  # [1,1,d]
        return self.lm_head.greedy_token(last)


class WhisperNeuron(nn.Module):
    """Top-level container. proj_out tied to decoder.embed_tokens."""

    def __init__(self, cfg, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.encoder = WhisperEncoder(cfg, dtype)
        self.decoder = WhisperDecoder(cfg, dtype)

    def encode(self, input_features):
        return self.encoder(input_features)

    def decode(self, input_ids, positions, encoder_hidden_states):
        h = self.decoder(input_ids, positions, encoder_hidden_states)
        logits = F.linear(h, self.decoder.embed_tokens.weight)
        return logits
