# SPDX-License-Identifier: Apache-2.0
"""Factory for the Whisper model.

The plugin registry registers this class for ``WhisperForConditionalGeneration``.
The vLLM serving layer finds the registered class with ``get_model_cls`` and
calls the transcription class methods on it. Thus this class:

* has the ``SupportsTranscription`` and ``SupportsMultiModal`` markers,
* uses the class methods of the upstream Whisper class (language table, prompt,
  speech-to-text config), and
* registers the upstream Whisper audio processor. ``register_processor`` stores
  the processor factory on the decorated class, and the multimodal registry
  reads it from the registered (plugin) class.

``from_configs`` returns the BF16 implementation in ``model_bf16``.
"""

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.whisper import (
    WhisperDummyInputsBuilder,
    WhisperMultiModalProcessor,
    WhisperProcessingInfo,
)
from vllm.model_executor.models.whisper import (
    WhisperForConditionalGeneration as _CoreWhisper,
)
from vllm.model_executor.models.whisper_utils import ISO639_1_SUPPORTED_LANGS
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_neuron.model.interfaces import SupportsEncoderBudget
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

# One Whisper clip gives 1500 encoder tokens (30 s window, stride-2 conv).
_TOKENS_PER_CLIP = 1500


@MULTIMODAL_REGISTRY.register_processor(
    WhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder,
)
class WhisperForConditionalGeneration(
    nn.Module,
    SupportsTranscription,
    SupportsMultiModal,
    SupportsEncoderBudget,
):
    """Factory that selects the Whisper implementation and serves transcription."""

    supported_languages = ISO639_1_SUPPORTED_LANGS
    supports_transcription_only = True
    supports_segment_timestamp = True
    supports_explicit_language_detection = True

    # The upstream class methods use only cls.supported_languages, the tokenizer
    # and the processor. They do not use the model body.
    validate_language = _CoreWhisper.validate_language
    get_generation_prompt = _CoreWhisper.get_generation_prompt
    get_speech_to_text_config = _CoreWhisper.get_speech_to_text_config
    get_num_audio_tokens = _CoreWhisper.get_num_audio_tokens
    get_language_token_ids = _CoreWhisper.get_language_token_ids
    get_language_detection_prompt = _CoreWhisper.get_language_detection_prompt
    parse_language_detection_output = _CoreWhisper.parse_language_detection_output
    get_placeholder_str = _CoreWhisper.get_placeholder_str

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, encoder_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(text_neuron_config, encoder_neuron_config)

        from .model_bf16 import WhisperForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            encoder_neuron_config=encoder_neuron_config,
        )

    @classmethod
    def _validate_config(
        cls,
        text_neuron_config: NeuronConfig | None,
        encoder_neuron_config: EncoderNeuronConfig | None,
    ) -> None:
        quantization = text_neuron_config.quantization if text_neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"Whisper does not support quantization {quantization!r}. "
                "Use None or 'bf16'."
            )
        if encoder_neuron_config is None:
            return
        block = encoder_neuron_config.encoder_attention_block_size
        if block != _TOKENS_PER_CLIP:
            raise ValueError(
                f"Whisper encoder_attention_block_size is {block}. "
                f"Set it to {_TOKENS_PER_CLIP}, so that one cache block holds one clip."
            )
        for bucket in encoder_neuron_config.num_encoder_input_buckets or []:
            if bucket % _TOKENS_PER_CLIP != 0:
                raise ValueError(
                    f"Whisper encoder bucket {bucket} is not a multiple of "
                    f"{_TOKENS_PER_CLIP}. Use buckets of whole clips, e.g. [1500]."
                )

    # ── SupportsEncoderBudget ─────────────────────────────────────────────
    @classmethod
    def get_max_encoder_embed_budget(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int | None:
        # Buckets count encoder tokens in whole clips.
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets)

    @classmethod
    def get_max_encoder_blocks_per_request(
        cls,
        hf_config: PretrainedConfig,
        encoder_neuron_config: EncoderNeuronConfig,
    ) -> int:
        # One request has one clip, and one clip fills one cache block.
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        return 1
