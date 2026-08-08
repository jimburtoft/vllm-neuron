# SPDX-License-Identifier: Apache-2.0
"""Factory for Voxtral model selection based on platform and configuration.

Interface markers: SupportsMultiModal + SupportsTranscription (see upstream
vllm.model_executor.models.voxtral). Ministral-3B language backbone; audio
encoder is Whisper-derived; adapter is Linear-GELU-Linear.
"""

import torch.nn as nn
from transformers import PretrainedConfig

# Import the upstream Voxtral class so we can inherit its transcription
# classmethods verbatim (get_speech_to_text_config, get_generation_prompt,
# get_num_audio_tokens, supported_languages) and its SupportsTranscription
# marker. This is analogous to whisper-xla's factory inheriting
# vllm.model_executor.models.whisper.WhisperForConditionalGeneration
# transcription methods.
from vllm.model_executor.models.voxtral import (
    VoxtralForConditionalGeneration as _CoreVoxtral,
)

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


class VoxtralForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the appropriate Voxtral implementation.

    Carries the interface markers vLLM's OpenAI API server checks to wire
    endpoints:
      * ``supported_languages`` (ISO code -> name dict)
      * ``supports_transcription_only = False`` -- Voxtral serves both chat and transcription
      * transcription classmethods inherited from upstream _CoreVoxtral

    Note: MRO shim rather than direct inheritance to keep the factory
    lightweight; the runner unwraps via `_unwrap_vision_model` etc.
    """

    supported_languages = _CoreVoxtral.supported_languages
    supports_transcription_only = False

    # Delegate the transcription classmethods verbatim.
    get_speech_to_text_config = _CoreVoxtral.get_speech_to_text_config
    get_generation_prompt = _CoreVoxtral.get_generation_prompt
    get_num_audio_tokens = _CoreVoxtral.get_num_audio_tokens

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, text_neuron_config)

        from .model_bf16 import VoxtralForConditionalGeneration as Model

        model = Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        # Propagate the transcription markers onto the actual model instance
        # so `supports_transcription(model)` (which uses runtime-checkable
        # Protocol) sees the required attributes on the returned object.
        model.supported_languages = cls.supported_languages
        model.supports_transcription_only = cls.supports_transcription_only
        # Bind classmethods as attributes so protocol checks see callables.
        model.get_speech_to_text_config = _CoreVoxtral.get_speech_to_text_config
        model.get_generation_prompt = _CoreVoxtral.get_generation_prompt
        model.get_num_audio_tokens = _CoreVoxtral.get_num_audio_tokens
        return model

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
    ) -> None:
        pass

