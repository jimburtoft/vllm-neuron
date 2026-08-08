# SPDX-License-Identifier: Apache-2.0
"""
Voxtral audio encoder (Whisper-derived)
========================================

Ported from the whisper-xla plugin integration
(`vllm_neuron.model.whisper.model_bf16`, lines 62-291), simplified for
Voxtral:

  * NO cross-attention -- audio embeddings feed the LLM via an inputs_embeds
    prefix (masked_scatter), not via per-layer cross-KV. So we drop
    `WhisperCrossKVEncoder`, `CrossAttention`, and the register_buffer
    cross-KV plumbing. Encoder outputs [B, T, hidden] and the caller
    packs+projects.

  * The Voxtral audio encoder is Whisper-large-v3 topology (32 layers,
    hidden 1280, 20 heads, head_dim 64, max_source_positions 1500, 128 mel
    bins). Same q/v-with-bias, k-no-bias, FP32-bias-on-out-proj/fc2
    pattern.

Whisper specifics preserved:
  * 20 heads / head_dim=64 => TP in {1, 2, 4, 5, 10, 20}. Assert
    divisibility.
  * q_proj/v_proj have bias, k_proj has NO bias (synthesized zero at
    weight load if missing from a checkpoint).
  * out_proj / mlp.fc2 keep an fp32 bias (RowParallelLinear rejects
    non-fp32 bias at tp>1) via the RowParallelLinearFP32Bias wrapper.
  * Activation is exact erf-GELU written out (F.gelu breaks fullgraph
    under libtorch_neuronx_lite).

SDK 2.31 compiler footgun (NCC_ISAU902): the LayerNorms in this
encoder may be rejected by `neuronx-cc 2.26.6360` without the
`--expand-batch-norm-training` patch documented in
`steering/whisper-nxdi-footguns.md`. Verify at first compile; apply the
`model_wrapper.py` monkey-patch if it fires.

Weight-load key convention (HF Voxtral checkpoint uses `audio_tower.*`):
  audio_tower.conv1.weight/bias
  audio_tower.conv2.weight/bias
  audio_tower.embed_positions.weight    (sinusoidal, but stored as trained)
  audio_tower.layer_norm.weight/bias    (final norm)
  audio_tower.layers.N.self_attn_layer_norm.weight/bias
  audio_tower.layers.N.self_attn.q_proj.weight/bias
  audio_tower.layers.N.self_attn.k_proj.weight             (no bias)
  audio_tower.layers.N.self_attn.v_proj.weight/bias
  audio_tower.layers.N.self_attn.out_proj.weight/bias
  audio_tower.layers.N.final_layer_norm.weight/bias
  audio_tower.layers.N.fc1.weight/bias
  audio_tower.layers.N.fc2.weight/bias
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear

from .config import VoxtralAudioConfig


# --------------------------------------------------------------------------- #
# Helpers (ported from contrib whisper_neuron.py via whisper-xla).
# --------------------------------------------------------------------------- #


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU written out so Dynamo traces it.

    F.gelu is a skipped C builtin under libtorch_neuronx_lite and breaks
    fullgraph=True; writing it out lets the fx-tracing succeed.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> torch.Tensor:
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
    all-reduce.

    Contrib pattern: RowParallelLinear rejects non-fp32 bias at tp>1;
    Whisper (and hence Voxtral audio encoder) out_proj/fc2 have bias, so
    add it in fp32 out-of-band.
    """

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        self.rpl = RowParallelLinear(in_features, out_features, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rpl(x)
        # Cast bias to output dtype for the residual add.
        return y + self.bias.to(y.dtype)


# --------------------------------------------------------------------------- #
# Encoder blocks
# --------------------------------------------------------------------------- #


class EncoderSelfAttention(nn.Module):
    """Encoder non-causal self-attention (Whisper-style).

    Voxtral audio encoder: embed_dim=1280, num_heads=20, head_dim=64.
    Q/V have bias; K does NOT.
    """

    def __init__(self, embed_dim: int, num_heads: int, dtype: torch.dtype):
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

    def _shape(self, x: torch.Tensor, seqlen: int, bsz: int) -> torch.Tensor:
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        attn = torch.matmul(q, k.transpose(-1, -2))
        # fp32 softmax for numerical stability; matches contrib.
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out_proj(out)


class EncoderMLP(nn.Module):
    """Encoder MLP: Linear(embed_dim -> ffn_dim) -> GELU -> Linear(ffn_dim -> embed_dim).

    Voxtral audio encoder: embed_dim=1280, ffn_dim=intermediate_size=5120.
    Both linears have bias (fc2 uses FP32Bias wrapper for TP).
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dtype: torch.dtype):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, ffn_dim, bias=True, dtype=dtype)
        self.fc2 = RowParallelLinearFP32Bias(ffn_dim, embed_dim, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """One Whisper-style encoder layer.

    Attention has residual + pre-LayerNorm (Whisper is pre-norm).
    MLP has the same structure.
    """

    def __init__(self, cfg: VoxtralAudioConfig, dtype: torch.dtype):
        super().__init__()
        d = cfg.hidden_size
        self.self_attn = EncoderSelfAttention(d, cfg.num_attention_heads, dtype)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = EncoderMLP(d, cfg.intermediate_size, dtype)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class VoxtralAudioEncoder(nn.Module):
    """Voxtral audio encoder (Whisper-derived) with folded mel computation.

    Input: raw audio waveform `[B, T_samples]` at 16 kHz float32 (or the
    encoder's compute dtype). Internally computes:
      1. STFT (window_size=400, hop=160)
      2. Mel-filter-bank projection (128 mels, 0-8000 Hz)
      3. log10 + normalize (Whisper convention)
      4. Cast to encoder dtype (bfloat16)
      5. Conv1 + Conv2 (conv2 stride=2 downsample) + positions + N encoder layers

    Output: encoder hidden states `[B, T_out, hidden]` where
      T_out = (T_samples / hop_length - 1) / 2.

    Folding mel into the compiled graph removes the host->device dtype
    coercion issue that Task 004b hit: the runner allocates raw-audio
    synthetic inputs as fp32 (its default), we generate mel on-device in
    fp32, then cast to bf16 inside the graph before conv1. No `.copy_()`
    strict-dtype check trip.
    """

    def __init__(self, cfg: VoxtralAudioConfig, dtype: torch.dtype | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_size
        dtype = dtype if dtype is not None else cfg.torch_dtype
        self.compute_dtype = dtype

        # Mel-filter-bank (precomputed, non-parameter). Registered as a
        # buffer so it moves with the module.
        from mistral_common.audio import mel_filter_bank as _mel_filter_bank

        mel = _mel_filter_bank(
            num_frequency_bins=1 + cfg.window_size // 2,
            num_mel_bins=cfg.num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=cfg.sampling_rate,
        )
        self.register_buffer(
            "mel_filters", torch.as_tensor(mel, dtype=torch.float32), persistent=False,
        )
        # STFT window (Hann), fp32.
        self.register_buffer(
            "stft_window", torch.hann_window(cfg.window_size), persistent=False,
        )
        # Precomputed DFT-basis matrices for matmul-based rFFT (guaranteed
        # XLA-lowerable, unlike torch.fft.rfft which segfaults torch_xla).
        # Forward DFT: X_k = sum_n x_n * exp(-2j*pi*k*n/N)
        #            = sum_n x_n * cos(...) - j * sum_n x_n * sin(...)
        k = torch.arange(cfg.window_size // 2 + 1, dtype=torch.float32)
        n = torch.arange(cfg.window_size, dtype=torch.float32)
        angles = 2 * math.pi * k[:, None] * n[None, :] / cfg.window_size
        self.register_buffer("dft_cos", torch.cos(angles), persistent=False)
        # Negative sin for forward DFT (imag part).
        self.register_buffer("dft_sin", -torch.sin(angles), persistent=False)

        # Mel -> hidden via two conv1d (Whisper conv1 stride=1, conv2 stride=2).
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1, dtype=dtype)

        # Learned positional embeddings.
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d, dtype=dtype)
        with torch.no_grad():
            if not self.embed_positions.weight.is_meta:
                self.embed_positions.weight.copy_(
                    sinusoids(cfg.max_source_positions, d).to(dtype)
                )

        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, dtype) for _ in range(cfg.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

    def compute_mel_spectrogram(self, audio_wave: torch.Tensor) -> torch.Tensor:
        """Whisper-standard log-mel spectrogram on-device.

        Manual STFT via framing + rfft (avoids torch.stft's aten::as_strided
        which the Neuron XLA lowering pass doesn't support).

        Args:
            audio_wave: `[B, T_samples]` waveform at 16 kHz.

        Returns:
            `[B, num_mel_bins, T_frames]` log-mel spectrogram in encoder dtype.
        """
        # Cast to fp32 for numerical stability.
        audio_wave = audio_wave.to(dtype=torch.float32)
        n_fft = self.cfg.window_size
        hop = self.cfg.hop_length

        # Reflect-pad the audio by n_fft/2 on each side to match Whisper's
        # `center=True` STFT convention.
        pad = n_fft // 2
        audio_padded = torch.nn.functional.pad(
            audio_wave, (pad, pad), mode="reflect"
        )
        # Frame the audio into overlapping windows of size n_fft with stride hop.
        # NOTE: DO NOT use `Tensor.unfold(-1, n_fft, hop)`. It uses `as_strided`
        # under the hood and segfaults torch_xla / Neuron runtime (verified
        # with a minimal reproducer in Task 004e-ablation, 2026-08-08).
        # `.expand()` + `.gather()` also fails because expand's non-contiguous
        # view can't be `.contiguous()`d cleanly on the Neuron device.
        # Framing via `torch.nn.functional.conv1d` with identity filters is
        # XLA-safe (uses only conv1d which is fully XLA-lowerable, and
        # produces a contiguous output natively).
        B, T_padded = audio_padded.shape
        num_frames = 1 + (T_padded - n_fft) // hop
        # Build identity filter [n_fft, 1, n_fft]: each output channel
        # picks one sample offset within the window.
        # This filter is data-independent so it could be a buffer, but we
        # build it here to keep the code local. On-device eye construction
        # is cheap.
        eye = torch.eye(n_fft, dtype=audio_padded.dtype, device=audio_padded.device)
        filt = eye.unsqueeze(1)  # [n_fft, 1, n_fft]
        # conv1d: input [B, 1, T_padded] -> [B, n_fft, num_frames].
        frames = torch.nn.functional.conv1d(
            audio_padded.unsqueeze(1), filt, stride=hop
        )
        # Transpose to [B, num_frames, n_fft].
        frames = frames.transpose(-1, -2).contiguous()
        # Apply Hann window.
        frames = frames * self.stft_window  # broadcast [n_fft] over last dim

        # DFT via cos/sin matmul (XLA-lowerable, unlike torch.fft.rfft which
        # segfaults torch_xla execution). Precomputed dft_cos / dft_sin
        # buffers of shape [n_fft//2+1, n_fft].
        real_part = torch.einsum("kn,btn->btk", self.dft_cos, frames)
        imag_part = torch.einsum("kn,btn->btk", self.dft_sin, frames)
        # Magnitude squared: [B, num_frames, n_fft//2+1]
        magnitudes = real_part ** 2 + imag_part ** 2
        # Transpose to [B, freq_bins, num_frames] (Whisper convention).
        magnitudes = magnitudes.transpose(-1, -2)
        # Drop the last frame to match Whisper convention.
        magnitudes = magnitudes[..., :-1]

        # Mel projection: [B, freq_bins, T] -> [B, num_mel_bins, T].
        # mistral_common.mel_filter_bank returns [freq_bins, num_mel_bins].
        mel_spec = torch.einsum(
            "fm,bft->bmt", self.mel_filters, magnitudes
        )

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # Normalize per Whisper: subtract max-8, then shift+scale.
        log_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, log_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        # Cast to encoder compute dtype (bfloat16). Reference an actual
        # parameter's dtype so the cast is fx-traceable.
        return log_spec.type_as(self.conv1.weight)

    def forward(
        self,
        input_features: torch.Tensor,
        encoder_cache_buffer: torch.Tensor | None = None,
        write_block_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode raw audio waveform (or pre-computed mel) to hidden states.

        Args:
            input_features: Either:
              (a) `[B, T_samples]` raw audio at 16 kHz -- computes mel inside.
              (b) `[B, num_mel_bins=128, T_frames]` pre-computed mel -- skips
                  the mel-computation step. Detected via ndim.
            encoder_cache_buffer / write_block_ids: absorbed for warmup-signature
                compatibility (unused in first port -- embed_multimodal handles
                the scatter-write eagerly outside this graph).
            **kwargs: absorb runner-supplied extras.

        Returns:
            `[B, T_out, hidden]` encoder hidden states.
        """
        del encoder_cache_buffer, write_block_ids, kwargs

        # Auto-detect: raw audio has ndim==2 or ndim==1; mel has ndim==3.
        if input_features.dim() == 2:
            # Raw audio: compute mel on-device.
            mel = self.compute_mel_spectrogram(input_features)
        elif input_features.dim() == 3:
            # Pre-computed mel. Cast to encoder weight dtype ONLY at compile
            # time. At runtime, the cast is a no-op (embed_multimodal casts
            # before device transfer) and triggers Neuron runtime's strict
            # `.copy_()` dtype-identity check, so we skip it.
            if torch._dynamo.is_compiling():
                mel = input_features.to(dtype=self.conv1.weight.dtype)
            else:
                mel = input_features
        else:
            raise ValueError(
                f"VoxtralAudioEncoder.forward: input_features has unexpected "
                f"ndim {input_features.dim()} (expected 2 raw-audio or 3 mel-spec)"
            )

        # [B, 128, T] -> [B, d, T]
        x = gelu(self.conv1(mel))
        # [B, d, T] -> [B, d, T/2]
        x = gelu(self.conv2(x))
        # [B, d, T/2] -> [B, T/2, d]
        x = x.transpose(1, 2)
        # Add positional embeddings (broadcast; slice to actual T).
        x = x + self.embed_positions.weight[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)
