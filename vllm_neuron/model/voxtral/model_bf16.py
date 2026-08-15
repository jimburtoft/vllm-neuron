# SPDX-License-Identifier: Apache-2.0
"""
Voxtral BF16 Implementation
============================

Voxtral-Mini-3B on the vllm-neuron native XLA backend.

Composition (matches upstream vLLM voxtral.py structure):
  * self.visual.encoder      -- Whisper-derived audio encoder (mel-in, hidden-out)
                                 (VoxtralAudioEncoder from audio_encoder_bf16.py)
  * self.audio_language_adapter -- Linear(5120->3072) + GELU + Linear(3072->3072)
  * self.language_model       -- vllm_neuron.model.llama3.LlamaForCausalLM
                                 (Ministral-3B = Llama-family GQA 32/8 head_dim=128
                                  30 layers, vocab=131072, RoPE theta=1e8, SwiGLU)

Audio injection pattern (plugin block-cache multimodal):
  1. embed_multimodal(): CPU-side mel-spectrogram + on-device encoder+adapter
     -> scatter-write [num_audio_tokens, 3072] into encoder_cache.buffer[block_ids].
  2. Runner _gather_mm_embeddings() reads block views into
     vision_embedding_blocks + vision_positions kwargs to forward().
  3. forward() scatters audio embeddings into inputs_embeds, delegates to
     self.language_model with inputs_embeds+is_token_ids for merge.

TODO / KNOWN LIMITATIONS (Task 003b remaining):
  * `is_token_ids` mask construction from `input_ids != audio_token_id` may
    need SP alignment. Verify against llama3/model.py L1352-1361.
  * `torch.compile(backend="vllm_neuron")` may not lower `masked_scatter` --
    if it doesn't, use `torch.where(mask, audio_broadcast, ...)` fallback
    (functional/prompt_embeds.py `merge_prompt_embeds` uses torch.where).
  * SDK 2.31 `NCC_ISAU902` LayerNorm compiler rejection may fire on the
    audio encoder -- apply `--expand-batch-norm-training` monkey-patch from
    steering/whisper-nxdi-footguns.md if it does.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.interfaces import SupportsVisionWarmup
from vllm_neuron.model.llama3.config import LlamaConfig
from vllm_neuron.model.llama3.model import LlamaForCausalLM as _LlamaModel
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

from .audio_encoder_bf16 import VoxtralAudioEncoder
from .config import VoxtralConfig, VoxtralTextConfig

logger = logging.getLogger(__name__)


def _voxtral_text_to_llama_config(
    text_config: VoxtralTextConfig,
) -> LlamaConfig:
    """Translate our VoxtralTextConfig into the plugin's LlamaConfig.

    Ministral-3B is Llama-family: same architecture (RMSNorm, GQA, SwiGLU,
    RoPE, tied optional). The plugin's LlamaConfig accepts our dict verbatim
    once we ensure `rope_parameters` is the plugin's dict shape and that
    `tie_word_embeddings` is set explicitly (Voxtral is UNTIED; LlamaConfig
    defaults to tied).
    """
    lc_dict = {
        "vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "intermediate_size": text_config.intermediate_size,
        "num_hidden_layers": text_config.num_hidden_layers,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "head_dim": text_config.head_dim,
        "max_position_embeddings": text_config.max_position_embeddings,
        "rms_norm_eps": text_config.rms_norm_eps,
        "rope_parameters": text_config.rope_parameters,
        "tie_word_embeddings": text_config.tie_word_embeddings,  # False for Voxtral
        "torch_dtype": text_config.torch_dtype,
    }
    return LlamaConfig.from_configs(lc_dict, neuron_config=text_config.neuron_config)


class AudioLanguageAdapter(nn.Module):
    """Two-layer projector from audio-encoder features to LLM hidden dim.

    Matches upstream vLLM voxtral.py AudioLanguageAdapter:
      Linear(audio_hidden * downsample_factor -> text_hidden)
      -> GELU
      -> Linear(text_hidden -> text_hidden)

    For Voxtral-Mini-3B-2507: Linear(5120 -> 3072) -> GELU -> Linear(3072 -> 3072).
    Both linears bias-free.

    HF checkpoint keys:
      multi_modal_projector.linear_1.weight -> w_in.weight
      multi_modal_projector.linear_2.weight -> w_out.weight

    NOTE: uses `torch.nn.functional.gelu` via a manual erf implementation --
    `nn.GELU` module dispatches to `torch._C._nn.gelu` which dynamo can't
    trace. Same pattern as VoxtralAudioEncoder.
    """

    def __init__(self, hidden_size: int, dim: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.w_in = nn.Linear(hidden_size, dim, bias=False, dtype=dtype)
        self.w_out = nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit erf-based GELU (dynamo cannot trace nn.GELU / torch._C._nn.gelu).
        # torch.erf may internally upcast bf16 -> fp32 -- keep the pipe in
        # weight dtype by construction.
        import math
        h = self.w_in(x)
        # Compute GELU in fp32 then cast back (inside the compiled graph the
        # cast is free; outside would trip Neuron's strict dtype-copy check
        # but the whole adapter is inside self.visual which is compiled).
        h_f32 = h.to(torch.float32)
        gelu = h_f32 * 0.5 * (1.0 + torch.erf(h_f32 / math.sqrt(2.0)))
        h = gelu.to(h.dtype)
        return self.w_out(h)


# ---------------------------------------------------------------------------
# Audio processing helpers (CPU-side mel-spectrogram)
# ---------------------------------------------------------------------------


def _compute_mel_spectrogram(
    audio_wave: torch.Tensor,
    num_mel_bins: int = 128,
    window_size: int = 400,
    hop_length: int = 160,
    sampling_rate: int = 16000,
) -> torch.Tensor:
    """Whisper-standard log-mel spectrogram (CPU-side, matches upstream vLLM).

    Args:
        audio_wave: [T_samples] mono waveform at 16 kHz, float32.
        num_mel_bins: Voxtral audio_config.num_mel_bins = 128.
        window_size: Whisper standard = 400 (25 ms at 16 kHz).
        hop_length: Whisper standard = 160 (10 ms at 16 kHz).
        sampling_rate: 16000.

    Returns:
        [num_mel_bins, T_frames] log-mel spectrogram, float32.
        T_frames = ceil(T_samples / hop_length).
    """
    from mistral_common.audio import mel_filter_bank

    device = audio_wave.device
    window = torch.hann_window(window_size, device=device)
    stft = torch.stft(
        audio_wave,
        n_fft=window_size,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    # stft[..., :-1] drops the last frame (Whisper convention -- matches
    # upstream vLLM voxtral.py compute_whisper_melspec).
    magnitudes = stft[..., :-1].abs() ** 2

    mel_filters = torch.tensor(
        mel_filter_bank(
            num_frequency_bins=1 + window_size // 2,
            num_mel_bins=num_mel_bins,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
        ),
        dtype=torch.float32,
        device=device,
    )
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Normalize per upstream Whisper convention.
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class _EncoderPlusAdapter(nn.Module):
    """Composite: encoder + downsample_factor pack + adapter.

    Wrapped as one nn.Module so `torch.compile` treats it as a single graph.
    Keeps the encoder-output-to-adapter dtype cast INSIDE the compiled graph
    (compiler handles it) rather than at runtime host-side (Neuron rejects
    device-side `.to(dtype=)` calls).
    """

    def __init__(self, encoder, adapter, downsample_factor: int):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.downsample_factor = downsample_factor

    def forward(self, input_features, **kwargs):
        # Encoder -> [B, T_out, hidden].
        enc = self.encoder(input_features, **kwargs)
        # Pack downsample_factor frames per projector token.
        df = self.downsample_factor
        B, T_out, D = enc.shape
        pad = (df - (T_out % df)) % df
        if pad:
            enc = torch.nn.functional.pad(enc, (0, 0, 0, pad))
        T_packed = enc.shape[1] // df
        packed = enc.reshape(B, T_packed, D * df)
        # Cast to adapter dtype inside the graph (compiler fuses this).
        adapter_dtype = self.adapter.w_in.weight.dtype
        packed = packed.to(dtype=adapter_dtype)
        out = self.adapter(packed)
        # Force output back to adapter dtype (erf-GELU may have upcast to
        # fp32 internally; cast happens INSIDE the compiled graph so it's
        # not subject to the Neuron runtime strict-dtype check).
        return out.to(dtype=adapter_dtype)


class VoxtralForConditionalGeneration(nn.Module, SupportsVisionWarmup):
    """Voxtral multimodal model (audio -> text) top-level class.

    Implements SupportsVisionWarmup so the plugin's runner drives the audio
    encoder graph capture at warmup. The `vision_neuron_config` slot is
    reused for the audio encoder (plugin convention).

    Carries the class-level `supports_transcription = True` marker so
    vLLM's `supports_transcription(model)` sees it (Protocol check via
    `getattr(model, "supports_transcription", False)`). The transcription
    classmethods are attached at instance level by the factory.
    """

    # SupportsTranscription protocol markers (matches upstream Voxtral).
    supports_transcription: bool = True
    supports_transcription_only: bool = False
    supports_segment_timestamp: bool = False

    def __init__(self, config: VoxtralConfig):
        super().__init__()
        self.config = config
        self.text_config = config.text_config
        self.audio_config = config.audio_config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # Language model: reuse the plugin's Llama implementation.
        # (See _voxtral_text_to_llama_config docstring for rationale.)
        llama_config = _voxtral_text_to_llama_config(config.text_config)
        # Instantiate the concrete Llama model directly. Do NOT go through
        # the factory (which re-parses hf_config as a dict) -- we already
        # have a fully-constructed LlamaConfig.
        self.language_model = _LlamaModel(llama_config)
        logger.info(
            "Voxtral language_model wired: hidden=%d, layers=%d, heads=%d/%d, "
            "head_dim=%d, vocab=%d, rope_theta=%s, tie_word_embeddings=%s",
            llama_config.hidden_size,
            llama_config.num_hidden_layers,
            llama_config.num_attention_heads,
            llama_config.num_key_value_heads,
            llama_config.head_dim,
            llama_config.vocab_size,
            llama_config.rope_parameters.get("rope_theta"),
            llama_config.tie_word_embeddings,
        )

        # Audio-language adapter + Whisper-derived audio encoder, composed as
        # ONE nn.Module (see _EncoderPlusAdapter docstring for rationale).
        #
        # IMPORTANT: only register the composite as `self.visual`. Do NOT also
        # register `self.visual.encoder` or `self.audio_language_adapter` as
        # separate attributes -- that creates duplicate keys in state_dict()
        # and load_state_dict requires both sets to be present. Access encoder
        # via `self.visual.encoder` and adapter via `self.visual.adapter`.
        _encoder = VoxtralAudioEncoder(
            cfg=config.audio_config,
            dtype=config.audio_config.torch_dtype,
        )
        _adapter = AudioLanguageAdapter(
            hidden_size=config.audio_config.hidden_size
            * config.audio_config.downsample_factor,
            dim=config.text_config.hidden_size,
            dtype=config.text_config.torch_dtype,
        )
        self.visual = _EncoderPlusAdapter(
            _encoder,
            _adapter,
            downsample_factor=config.audio_config.downsample_factor,
        )
        logger.info(
            "Voxtral audio encoder wired: hidden=%d, layers=%d, heads=%d, "
            "head_dim=%d, mel_bins=%d, max_src_pos=%d",
            config.audio_config.hidden_size,
            config.audio_config.num_hidden_layers,
            config.audio_config.num_attention_heads,
            config.audio_config.head_dim,
            config.audio_config.num_mel_bins,
            config.audio_config.max_source_positions,
        )

        # Captures for the runner's optional tensor-capture facility.
        self._vision_captures: tuple[torch.Tensor, ...] = ()

        # Optional Medusa speculative-decoding heads (built only when
        # config.medusa_config is set via additional_config). Heads read the
        # Llama decoder's LAST hidden state and predict K future tokens; the
        # output projection is tied to language_model.lm_head (Medusa-1).
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
                hidden_size=config.text_config.hidden_size,
                medusa_num_layers=self._medusa_cfg["medusa_num_layers"],
                dtype=config.text_config.torch_dtype,
            )
            logger.info(
                "Voxtral Medusa heads constructed: N=%d L=%d d=%d init=%s",
                self._medusa_cfg["num_heads"], self._medusa_cfg["medusa_num_layers"],
                config.text_config.hidden_size, self._medusa_cfg["init"],
            )
            # Attach heads + the tied-projection argmax onto the Llama model so
            # its spec-decode verify branch runs the heads in-graph and returns
            # the (accepted, last_hidden, drafts) 3-tuple.
            self.language_model.medusa_heads = self.medusa_heads
            self.language_model._medusa_project_argmax = self._medusa_project_argmax

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ):
        config = VoxtralConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        # Surface the Medusa heads config for the SERVED path. The runner calls
        # from_configs without a medusa_config, so pull it from the active vLLM
        # config's additional_config. Customer enables Medusa with:
        #   --speculative-config '{"method":"medusa","num_speculative_tokens":5}'
        #   --additional-config  '{"medusa_config":{"init":"load","heads_path":"..."}}'
        try:
            from vllm.config import get_current_vllm_config
            vcfg = get_current_vllm_config()
            add_cfg = getattr(vcfg, "additional_config", None) or {}
            medusa_config = add_cfg.get("medusa_config")
            if medusa_config is not None and "num_heads" not in medusa_config:
                spec = getattr(vcfg, "speculative_config", None)
                k = getattr(spec, "num_speculative_tokens", None)
                if k is not None:
                    medusa_config = {**medusa_config, "num_heads": int(k)}
            if medusa_config is not None:
                config.medusa_config = medusa_config
        except Exception:
            pass
        return cls(config)

    # ── KV cache delegation ────────────────────────────────────────────

    def get_kv_spec(self):
        """Delegate to language_model. Voxtral has no cross-attention KV."""
        return self.language_model.get_kv_spec()

    def bind_kv_cache(self, kv_caches):
        return self.language_model.bind_kv_cache(kv_caches)

    # ── Forward ────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        vision_embedding_blocks: tuple[torch.Tensor, ...] | None = None,
        vision_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass with optional audio embedding merge (prefill only).

        Audio embeddings arrive via `vision_embedding_blocks` + `vision_positions`
        (plugin uses the same kwarg names for audio and vision). We build the
        `inputs_embeds` tensor by scattering audio embeddings into audio-token
        slots, then delegate to `self.language_model` with
        `inputs_embeds` + `is_token_ids` mask so `NF.merge_prompt_embeds`
        selects the right source per position.
        """
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        inputs_embeds = None
        is_token_ids = None
        if is_prefill and vision_embedding_blocks is not None and vision_positions is not None:
            # Build a full-sequence inputs_embeds tensor with audio embeddings
            # at their global batch positions, and an is_token_ids mask that
            # is False at audio positions (so merge_prompt_embeds swaps in
            # inputs_embeds there).
            #
            # vision_embedding_blocks: tuple of [block_size, hidden] tensors
            # vision_positions: [max_num_vision_blocks, block_size] with global
            #   batch position or sentinel (= num_tokens).
            T = input_ids.shape[0]
            hidden = self.text_config.hidden_size
            device = input_ids.device
            dtype = self.text_config.torch_dtype

            gathered = torch.stack(vision_embedding_blocks)                  # [B, block_size, hidden]
            flat_embeds = gathered.reshape(-1, gathered.shape[-1])           # [B*block_size, hidden]
            positions_flat = vision_positions.reshape(-1)                    # [B*block_size]

            # Scatter into a [T+1, hidden] tensor (last row is dummy for sentinels).
            # Cast everything to match input_ids-driven embedding dtype (bf16).
            dummy_row = torch.zeros(1, hidden, dtype=dtype, device=device)
            base = torch.zeros(T, hidden, dtype=dtype, device=device)
            with_dummy = torch.cat([base, dummy_row], dim=0)                 # [T+1, hidden]
            # Explicit cast: match with_dummy.dtype exactly (index_put_ is
            # strict about scalar type identity, not just kind).
            with_dummy = with_dummy.index_put((positions_flat,), flat_embeds.to(with_dummy.dtype))
            inputs_embeds = with_dummy[:T]                                   # [T, hidden]

            # is_token_ids: True where the token is a real text token, False
            # where it is an audio placeholder. merge_prompt_embeds swaps in
            # inputs_embeds when is_token_ids is False.
            is_token_ids = input_ids != self.config.audio_token_id           # [T] bool

        # Delegate to the underlying Llama model. LlamaForCausalLM handles LM
        # head, sampler, spec decode, all the rest.
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            attn_metadata=attn_metadata,
            sampling_positions=sampling_positions,
            sampling_params=sampling_params,
            spec_decode_metadata=spec_decode_metadata,
            logit_mask=logit_mask,
            rank=rank,
            **kwargs,
        )

    # ── Multimodal encoder path (embed_multimodal + warmup) ────────────

    def embed_multimodal(
        self,
        audio_arrays: list[torch.Tensor] | torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Encode audio inputs into the on-device encoder cache.

        Args:
            audio_arrays: List of raw audio waveforms (16 kHz mono float32)
                OR a batched tensor. One entry per mm_item.
            encoder_cache: EncoderCacheBlocks instance.
            mm_hashes: Per-item identifiers, same order as audio_arrays.

        For each audio clip:
          1. Compute mel-spectrogram on CPU (STFT + mel bank).
          2. Run whisper_encoder on device -> [1, T_out, 1280].
          3. Pack: [1, T_out, 1280] -> [T_out/4, 5120] (drops batch dim).
          4. Adapter: -> [T_out/4, 3072].
          5. Allocate blocks in encoder_cache and scatter-write embeddings.
        """
        if audio_arrays is None or encoder_cache is None or mm_hashes is None:
            raise ValueError(
                "embed_multimodal requires audio_arrays, encoder_cache, and mm_hashes."
            )

        # Normalize to list-of-tensors.
        if isinstance(audio_arrays, torch.Tensor):
            audio_arrays = list(audio_arrays.unbind(0))

        ac = self.audio_config
        device = next(self.visual.encoder.parameters()).device
        # Use the encoder's actual weight dtype (source of truth), not the
        # config field (which can be None after HF-parse if `torch_dtype` isn't
        # set explicitly in audio_config).
        target_dtype = self.visual.encoder.conv1.weight.dtype
        block_size = encoder_cache.block_size

        for i, audio in enumerate(audio_arrays):
            mm_hash = mm_hashes[i]

            # (1) Compute mel-spectrogram on CPU.
            # Task 004e: computing mel inside the compiled encoder graph
            # segfaults torch_xla (root-caused to Tensor.unfold's as_strided).
            # Workaround: compute mel on CPU eagerly, then feed pre-computed
            # mel to the encoder (the [B, num_mel_bins, T] input path is
            # verified to compile + execute cleanly).
            if audio.device.type != "cpu":
                audio = audio.detach().to(device="cpu")
            if audio.dtype != torch.float32:
                audio = audio.to(dtype=torch.float32)
            mel = _compute_mel_spectrogram(
                audio,
                num_mel_bins=ac.num_mel_bins,
                window_size=ac.window_size,
                hop_length=ac.hop_length,
                sampling_rate=ac.sampling_rate,
            )  # [num_mel_bins, T_frames] fp32
            mel_batched = mel.unsqueeze(0)  # [1, num_mel_bins, T_frames]
            # Cast to encoder compute dtype BEFORE moving to device (Neuron
            # runtime's `.copy_()` fails when host->device transfer requires
            # a dtype change).
            mel_batched = mel_batched.to(dtype=target_dtype)
            mel_batched = mel_batched.to(device=device)

            # (2+3+4) Encoder + pack + adapter as ONE compiled graph
            # (self.visual is _EncoderPlusAdapter, which the runner
            # torch.compile'd at load time). This dodges the runtime
            # device-side dtype cast between encoder fp32 output and
            # adapter bf16 input.
            audio_embeds_2d = self.visual(mel_batched)  # [1, T_packed, 3072]
            print(f"[voxtral-mm] audio_embeds_2d: shape={audio_embeds_2d.shape} dtype={audio_embeds_2d.dtype}", flush=True)
            audio_embeds = audio_embeds_2d.squeeze(0)  # [T_packed, 3072]
            num_audio_tokens = audio_embeds.shape[0]

            # (5) Allocate + scatter-write into the encoder cache buffer.
            tokens_per_block = encoder_cache.dense_tokens_per_block(
                num_audio_tokens, block_size
            )
            block_ids = encoder_cache.allocate(mm_hash, tokens_per_block)

            # Pad audio_embeds up to blocks*block_size and reshape to
            # [num_blocks, block_size, hidden] for scatter-write.
            total_slots = len(block_ids) * block_size
            if audio_embeds.shape[0] < total_slots:
                pad_rows = total_slots - audio_embeds.shape[0]
                audio_embeds = torch.nn.functional.pad(
                    audio_embeds, (0, 0, 0, pad_rows)
                )
            blocked = audio_embeds.reshape(len(block_ids), block_size, -1)

            # Write into cache buffer at block_ids. Runs eagerly here (not in
            # a compiled graph) -- simpler than the qwen3_vl fold-into-VE-graph
            # pattern. Optimization opportunity: fold this write into the
            # encoder graph via input-output aliasing (Task 008).
            #
            # Skip the `.to(encoder_cache.dtype)` cast: Neuron runtime's
            # strict `.copy_()` check rejects device-side `.to(dtype=)` calls
            # (see Task 004e report). audio_embeds should already be in the
            # adapter's bf16 dtype and encoder_cache.buffer dtype should match.
            for j, bid in enumerate(block_ids):
                src = blocked[j]
                if src.dtype != encoder_cache.buffer.dtype:
                    # Neuron runtime rejects device-side `.to(dtype=)`. Use
                    # a CPU roundtrip: neuron -> cpu (same dtype) -> cpu cast
                    # -> back to device (same dtype).
                    src_cpu = src.to(device="cpu")
                    src_cpu = src_cpu.to(dtype=encoder_cache.buffer.dtype)
                    src = src_cpu.to(device=encoder_cache.buffer.device)
                if j == 0:
                    logger.debug(
                        "cache-scatter: src.dtype=%s cache.buffer.dtype=%s",
                        src.dtype, encoder_cache.buffer.dtype,
                    )
                encoder_cache.buffer[bid].copy_(src)

    def build_vision_synthetic_inputs(
        self,
        bucket: int,
        vision_neuron_config: VisionNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Synthetic inputs for warmup-time audio encoder graph capture.

        Task 004e: passing raw audio and computing mel inside the compiled
        encoder graph triggers a torch_xla segfault (root cause:
        Tensor.unfold's as_strided). So the encoder consumes pre-computed
        mel-spectrograms (ndim=3 path), and `embed_multimodal` computes mel
        on CPU before dispatch.

        Bucket = number of post-adapter audio tokens (e.g. 375 for a 30s
        clip after downsample_factor=4 packing). Encoder output frames =
        bucket * downsample_factor. Mel frames = 2x encoder frames (conv2
        stride=2).
        """
        ac = self.audio_config
        t_enc_frames = bucket * ac.downsample_factor
        t_mel_frames = t_enc_frames * 2

        return {
            "input_features": torch.zeros(
                1,
                ac.num_mel_bins,
                t_mel_frames,
                dtype=ac.torch_dtype,
                device=device,
            ),
        }

    # ── Weight loading ──────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from an HF Voxtral checkpoint.

        Two-path strategy (Task 004g):
          A) LLM decoder (`language_model.model.*` + `language_model.lm_head.*`)
             -> `SafetensorsCheckpoint.load_sharded_pipelined`. This mirrors
             llama3.LlamaForCausalLM.load_weights: each parameter has a
             pre-attached loader that handles TP sharding + transpose. Only
             this path handles the plugin's `qkv_proj_weight` fusion +
             `[in, out_per_rank]` transposed layout correctly.
          B) Audio encoder (`audio_tower.*`) + Adapter
             (`multi_modal_projector.linear_{1,2}.*`)
             -> whisper-xla pattern: load full-tensor state, remap keys,
             `load_state_dict(strict=False, assign=True)` and let
             ColumnParallelLinear / RowParallelLinear's
             `_load_from_state_dict` hook do the sharding.

        The LlamaForCausalLM parameters in our tree are named
        `language_model.model.layers.N.self_attn.qkv_proj_weight` (etc.)
        because we composed self.language_model = LlamaForCausalLM. The HF
        Voxtral checkpoint has them under `language_model.model.layers.N.
        self_attn.q_proj.weight` -- so the mapping is just a fused-QKV
        remap with the same `language_model.model.*` prefix on both sides.
        """
        import glob
        import os

        from safetensors.torch import load_file

        from vllm_neuron.utils.checkpoints import (
            SafetensorsCheckpoint,
            _get_checkpoint_source,
        )

        # NOTE: do NOT call `self.to_empty(device="cpu")` here.
        # `to_empty` (via nn.Module._apply) allocates fresh Parameter objects
        # and copies `.data`, but drops the `weight_loader` attribute we
        # attached in `LlamaAttention.__init__` via `set_weight_loader`. Without
        # the loader, `load_sharded_pipelined` falls back to the default
        # identity loader, which can't handle fused-QKV (3 slices) or
        # transposed/sharded storage layout.
        #
        # The pipelined loader itself allocates storage via `load_state_dict(
        # ..., assign=True)` at the end (see llama3.load_weights). The
        # audio-encoder PATH B below also uses `assign=True`, so no pre-alloc
        # is needed there either.

        # Resolve the HF checkpoint directory.
        if os.path.isdir(checkpoint_path):
            model_dir = checkpoint_path
        else:
            source = _get_checkpoint_source(
                checkpoint_path, ".safetensors", cache_dir
            )
            file_names = source.get_file_names()
            for fn in file_names:
                source.download_file(fn)
            model_dir = os.path.dirname(source.get_file_path(file_names[0]))

        # === PATH A: LLM via pipelined loader ================================
        # Build the mappings dict following llama3.load_weights, but keeping
        # the `language_model.` prefix on both sides (that's where the
        # LlamaForCausalLM lives in our module tree, and it's where HF puts
        # the LLM weights in the Voxtral checkpoint).
        llm_mappings: dict[str, str | list[str]] = {}
        num_layers = self.text_config.num_hidden_layers
        for layer_id in range(num_layers):
            prefix = f"language_model.model.layers.{layer_id}"
            llm_mappings[f"{prefix}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            llm_mappings[f"{prefix}.self_attn.o_proj_weight"] = (
                f"{prefix}.self_attn.o_proj.weight"
            )
            llm_mappings[f"{prefix}.input_layernorm.weight"] = (
                f"{prefix}.input_layernorm.weight"
            )
            llm_mappings[f"{prefix}.post_attention_layernorm.weight"] = (
                f"{prefix}.post_attention_layernorm.weight"
            )
            llm_mappings[f"{prefix}.mlp.gate_proj_weight"] = (
                f"{prefix}.mlp.gate_proj.weight"
            )
            llm_mappings[f"{prefix}.mlp.up_proj_weight"] = (
                f"{prefix}.mlp.up_proj.weight"
            )
            llm_mappings[f"{prefix}.mlp.down_proj_weight"] = (
                f"{prefix}.mlp.down_proj.weight"
            )
        # LM top-level + head.
        llm_mappings["language_model.model.embed_tokens.weight"] = (
            "language_model.model.embed_tokens.weight"
        )
        llm_mappings["language_model.model.norm.weight"] = (
            "language_model.model.norm.weight"
        )
        if self.text_config.tie_word_embeddings:
            llm_mappings["language_model.lm_head.weight"] = (
                "language_model.model.embed_tokens.weight"
            )
        else:
            llm_mappings["language_model.lm_head.weight"] = (
                "language_model.lm_head.weight"
            )

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        tp_rank = self.rank
        tp_size = self.world_size
        llm_rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, llm_mappings, device, strict=False,
        ).state_dict

        # Cast to model dtype (bf16). Same convention as llama3.
        target_dtype_text = self.visual.encoder.conv1.weight.dtype  # bf16
        # But actually text config dtype might differ. Use lm_head weight dtype
        # as source of truth for LLM side.
        target_dtype_text = self.language_model.lm_head.weight.dtype
        for name, tensor in list(llm_rank_sharded.items()):
            if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
                if tensor.dtype != target_dtype_text:
                    llm_rank_sharded[name] = tensor.to(target_dtype_text)

        llm_missing, llm_unexpected = self.load_state_dict(
            llm_rank_sharded, strict=False, assign=True,
        )
        logger.info(
            "Voxtral LLM load_weights (pipelined): loaded=%d missing=%d unexpected=%d",
            len(llm_rank_sharded), len(llm_missing), len(llm_unexpected),
        )

        # === PATH B: audio encoder + adapter via full-state-dict ============
        # Load HF safetensors again for the non-LLM keys and remap.
        single = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single):
            hf_sd = load_file(single)
        else:
            hf_sd = {}
            for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
                hf_sd.update(load_file(f))

        remapped: dict[str, torch.Tensor] = {}
        target_dtype_audio = self.visual.encoder.conv1.weight.dtype
        for k, v in hf_sd.items():
            # Skip LLM keys -- already handled by PATH A.
            if k.startswith("language_model."):
                continue
            new_k = self._remap_hf_key(k)
            if new_k is None:
                logger.debug("skipping unmapped HF key: %s", k)
                continue
            # FP32 biases (RowParallelLinearFP32Bias wrapper) stay fp32.
            if new_k.startswith("visual.encoder") and new_k.endswith(".bias") and (
                ".out_proj." in new_k or ".fc2." in new_k
            ):
                remapped[new_k] = v.to(torch.float32)
            else:
                remapped[new_k] = v.to(target_dtype_audio)

        enc_missing, enc_unexpected = self.load_state_dict(
            remapped, strict=False, assign=True,
        )

        # Filter expected-missing: sinusoidal embed_positions may be re-derived.
        real_missing = [
            m for m in enc_missing
            if not m.endswith("visual.encoder.embed_positions.weight")
            and not m.startswith("language_model.")  # LM keys were loaded via PATH A
        ]
        if "visual.encoder.embed_positions.weight" in enc_missing:
            with torch.no_grad():
                from .audio_encoder_bf16 import sinusoids
                self.visual.encoder.embed_positions.weight.copy_(
                    sinusoids(
                        self.audio_config.max_source_positions,
                        self.audio_config.hidden_size,
                    ).to(target_dtype_audio)
                )

        logger.info(
            "Voxtral encoder+adapter load_weights: loaded=%d missing=%d unexpected=%d",
            len(remapped), len(real_missing), len(enc_unexpected),
        )
        if real_missing or enc_unexpected:
            logger.warning(
                "Voxtral load_weights mismatches: missing=%s unexpected=%s",
                real_missing[:20],
                enc_unexpected[:20],
            )

        # === MATERIALIZE any remaining meta tensors ===========================
        # After PATH A + PATH B, any tensor NOT covered by either mapping
        # (mainly non-persistent buffers: mel_filters, stft_window, dft_cos,
        # dft_sin) is still on the meta device. The runner will later call
        # `model.to(device)` which fails on meta tensors. Materialize them
        # here with proper values.
        for name, buf in list(self.named_buffers()):
            if buf.is_meta:
                # Recompute the buffer from the audio_encoder's __init__ logic.
                import math as _math
                enc = self.visual.encoder
                cfg = enc.cfg
                if name.endswith("stft_window"):
                    new_buf = torch.hann_window(cfg.window_size)
                elif name.endswith("mel_filters"):
                    from mistral_common.audio import mel_filter_bank as _mfb
                    mel = _mfb(
                        num_frequency_bins=1 + cfg.window_size // 2,
                        num_mel_bins=cfg.num_mel_bins,
                        min_frequency=0.0, max_frequency=8000.0,
                        sampling_rate=cfg.sampling_rate,
                    )
                    new_buf = torch.as_tensor(mel, dtype=torch.float32)
                elif name.endswith("dft_cos") or name.endswith("dft_sin"):
                    n_fft = cfg.window_size
                    k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
                    n = torch.arange(n_fft, dtype=torch.float32)
                    angles = 2 * _math.pi * k[:, None] * n[None, :] / n_fft
                    if name.endswith("dft_cos"):
                        new_buf = torch.cos(angles)
                    else:
                        new_buf = -torch.sin(angles)
                else:
                    logger.warning("Unhandled meta buffer: %s", name)
                    continue
                # Traverse to the owning module and replace the buffer.
                parts = name.split(".")
                owner = self
                for p in parts[:-1]:
                    owner = getattr(owner, p)
                owner.register_buffer(parts[-1], new_buf.contiguous(), persistent=False)
                logger.debug("Materialized buffer %s shape=%s", name, tuple(new_buf.shape))

        # Load Medusa head ResBlock params (Medusa-1 tie: output projection is
        # language_model.lm_head, so only per-head ResBlocks are loaded).
        if self.medusa_heads is not None:
            from .medusa_heads import load_medusa_heads
            cfg = self._medusa_cfg
            heads, report = load_medusa_heads(
                num_heads=cfg["num_heads"],
                hidden_size=self.config.text_config.hidden_size,
                medusa_num_layers=cfg["medusa_num_layers"],
                dtype=self.config.text_config.torch_dtype,
                init=cfg["init"],
                heads_path=cfg["heads_path"],
                seed=cfg["seed"],
            )
            self.medusa_heads = heads
            # Re-attach the loaded heads onto the Llama model (load replaced the
            # object built in __init__).
            self.language_model.medusa_heads = self.medusa_heads
            self.language_model._medusa_project_argmax = self._medusa_project_argmax
            logger.info("Voxtral Medusa heads loaded: %s", report)

    def _medusa_project_argmax(self, head_hidden: torch.Tensor) -> torch.Tensor:
        """[N, d] head hidden-states -> [N] int32 greedy token ids via the
        language_model's vocab-sharded lm_head + Sampler distributed argmax
        (the same mechanism greedy decode uses). Always greedy."""
        lm = self.language_model
        # Match the Llama forward's lm_head call convention (no fp32 pre-cast;
        # lm_head + sampler handle the head dtype), avoiding a mixed-dtype op.
        local = lm.lm_head(head_hidden.to(lm.lm_head.weight.dtype))
        return lm.sampler(local).to(torch.int32)

    def medusa_propose(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Greedy draft proposal for a SINGLE anchor position. Returns [K] int32.
        Runs the N Medusa heads on the Llama decoder's last hidden state and
        argmaxes each through the tied sharded lm_head."""
        assert self.medusa_heads is not None, "medusa_propose called without heads"
        return self.medusa_heads.propose_from_hidden_sharded(
            last_hidden, self._medusa_project_argmax
        )

    @staticmethod
    def _remap_hf_key(k: str) -> str | None:
        """Remap an HF Voxtral checkpoint key to a plugin module path.

        Returns None for unmapped keys (skip).
        """
        import re

        # === LLM (Ministral-3B) ===
        # HF: language_model.model.layers.N.self_attn.{q,k,v,o}_proj.weight
        # HF: language_model.model.layers.N.mlp.{gate,up,down}_proj.weight
        # HF: language_model.model.layers.N.{input,post_attention}_layernorm.weight
        # HF: language_model.model.{embed_tokens,norm}.weight
        # HF: language_model.lm_head.weight
        # Plugin (via LlamaForCausalLM as self.language_model):
        #   language_model.model.layers.N.self_attn.q_proj.weight (separate) OR
        #   language_model.model.layers.N.self_attn.qkv_proj.weight (fused)
        # Handled by _maybe_fuse_llama_qkv AFTER remap.
        if k.startswith("language_model.model.") or k == "language_model.lm_head.weight":
            return k

        # === AudioLanguageAdapter (multi_modal_projector) ===
        if k == "multi_modal_projector.linear_1.weight":
            return "visual.adapter.w_in.weight"
        if k == "multi_modal_projector.linear_2.weight":
            return "visual.adapter.w_out.weight"

        # === Voxtral audio encoder (audio_tower -> visual.encoder) ===
        if k.startswith("audio_tower."):
            inner = k[len("audio_tower."):]
            # HF flat naming layers.N.fc1/fc2 -> layers.N.mlp.fc1/fc2.
            inner = re.sub(
                r"^(layers\.\d+)\.(fc1|fc2)\.",
                lambda m: m.group(1) + ".mlp." + m.group(2) + ".",
                inner,
            )
            # Now wrap Row-parallel-FP32-bias for out_proj.weight and fc2.weight
            # (bias entries are kept unchanged: FP32Bias wrapper's `.bias`
            # matches the HF `.bias`; only the linear weight moves under `.rpl.weight`).
            inner = re.sub(
                r"^(layers\.\d+\.self_attn\.out_proj)\.weight$",
                r"\1.rpl.weight",
                inner,
            )
            inner = re.sub(
                r"^(layers\.\d+\.mlp\.fc2)\.weight$",
                r"\1.rpl.weight",
                inner,
            )
            return "visual.encoder." + inner

        # Unknown key -- skip.
        return None

    def _maybe_fuse_llama_qkv(
        self, state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse separate q/k/v_proj weights into a single qkv_proj_weight if
        the plugin's LlamaModel uses fused QKV.

        The plugin's naming convention is `qkv_proj_weight` (underscore, not
        `.weight`). Similarly `o_proj_weight`, `gate_proj_weight`, etc.
        Detect via param probe.
        """
        our_params = dict(self.named_parameters())
        probe = "language_model.model.layers.0.self_attn"
        has_fused_underscore = f"{probe}.qkv_proj_weight" in our_params
        has_fused_dot = f"{probe}.qkv_proj.weight" in our_params
        has_separate = f"{probe}.q_proj.weight" in our_params

        if (not has_fused_underscore) and (not has_fused_dot) and has_separate:
            # No fusion needed.
            return state
        if (not has_fused_underscore) and (not has_fused_dot) and (not has_separate):
            sample = [
                n for n in our_params
                if "language_model.model.layers.0.self_attn" in n
            ]
            logger.warning(
                "LlamaAttention q/k/v param naming not recognized. "
                "Sample layer.0 self_attn params: %s",
                sample,
            )
            return state

        # Fusion needed. Concat q/k/v along dim 0 and remap:
        # - Q/K/V separate -> single fused param (name depends on probe result)
        # - o_proj.weight   -> o_proj_weight
        # - mlp.gate_proj.weight -> mlp.gate_proj_weight
        # - mlp.up_proj.weight   -> mlp.up_proj_weight
        # - mlp.down_proj.weight -> mlp.down_proj_weight
        fused_state = {k: v for k, v in state.items()}

        # Determine the target fused name suffix.
        qkv_target = "qkv_proj_weight" if has_fused_underscore else "qkv_proj.weight"
        # Also detect which suffix o_proj / mlp use.
        o_key_target = "o_proj_weight" if f"{probe}.o_proj_weight" in our_params else "o_proj.weight"
        mlp_gate_dot = f"language_model.model.layers.0.mlp.gate_proj.weight" in our_params
        mlp_gate_target = "gate_proj.weight" if mlp_gate_dot else "gate_proj_weight"
        mlp_up_target = "up_proj.weight" if mlp_gate_dot else "up_proj_weight"
        mlp_down_target = "down_proj.weight" if mlp_gate_dot else "down_proj_weight"

        num_layers = self.text_config.num_hidden_layers
        for i in range(num_layers):
            prefix = f"language_model.model.layers.{i}"
            attn = f"{prefix}.self_attn"
            mlp = f"{prefix}.mlp"

            # Fuse QKV.
            q_key = f"{attn}.q_proj.weight"
            k_key = f"{attn}.k_proj.weight"
            v_key = f"{attn}.v_proj.weight"
            fused_key = f"{attn}.{qkv_target}"
            if q_key in fused_state and k_key in fused_state and v_key in fused_state:
                q = fused_state.pop(q_key)
                kw = fused_state.pop(k_key)
                vw = fused_state.pop(v_key)
                fused_state[fused_key] = torch.cat([q, kw, vw], dim=0)

            # Rename o_proj: `.weight` -> configured target (`_weight` on plugin llama3)
            o_src = f"{attn}.o_proj.weight"
            o_dst = f"{attn}.{o_key_target}"
            if o_src in fused_state and o_src != o_dst:
                fused_state[o_dst] = fused_state.pop(o_src)

            # Rename MLP: gate/up/down.
            for src_suffix, dst_suffix in (
                ("gate_proj.weight", mlp_gate_target),
                ("up_proj.weight", mlp_up_target),
                ("down_proj.weight", mlp_down_target),
            ):
                src = f"{mlp}.{src_suffix}"
                dst = f"{mlp}.{dst_suffix}"
                if src in fused_state and src != dst:
                    fused_state[dst] = fused_state.pop(src)

        return fused_state
