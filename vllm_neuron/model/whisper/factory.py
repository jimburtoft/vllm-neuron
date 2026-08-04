# SPDX-License-Identifier: Apache-2.0
"""Factory for Whisper model selection.

Whisper is an encoder-decoder (audio) model. Its audio encoder occupies the
"vision"/front-end slot that qwen3_vl uses, so the runner CAN construct it via
the 3-arg multimodal ``from_configs(hf_config, text_neuron_config,
vision_neuron_config)`` signature. For M1 the model is driven via the 2-arg text
branch (``vision_neuron_config is None``): the decoder loads, binds KV and
compiles standalone with zeroed cross-KV. The concrete model's ``from_configs``
accepts BOTH signatures (``**kwargs`` catch-all), so M3 can flip the runner to
the 3-arg branch (by supplying ``vision_neuron_config`` via ``additional_config``)
without changing this factory.

Template: llama3/factory.py (2-arg) + qwen3_vl/factory.py (3-arg).
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

# M4 serving: the OpenAI /v1/audio/transcriptions route is wired only when the
# REGISTERED model class (this factory -- registry.py:25 registers THIS class,
# which the plugin's ModelRegistry.register_model override installs under the
# "WhisperForConditionalGeneration" arch) satisfies vLLM-core's
# SupportsTranscription + SupportsMultiModal interfaces. The serving layer
# (vllm/entrypoints/openai/speech_to_text/speech_to_text.py) resolves the class
# via get_model_cls(model_config) -> get_model_architecture(...) -> THIS factory
# class, then calls the transcription classmethods on it
# (get_speech_to_text_config, get_generation_prompt, validate_language,
# post_process_output, get_num_audio_tokens, supports_segment_timestamp,
# no_space_languages). The Neuron runner's get_supported_tasks() must ALSO
# report "transcription" (it inspects supports_transcription(model) on the
# constructed model instance -- so the marker must travel with BOTH the
# registered class AND the instance, which subclassing the interface provides).
#
# We inherit the concrete transcription/multimodal behaviour directly from
# vLLM-core's upstream Whisper class (the language table, prompt construction,
# audio mm processor) rather than re-implementing it: those classmethods only
# touch cls.supported_languages + the tokenizer/processor and are independent of
# the modeling body, so reusing them keeps us byte-identical to how core routes
# audio and avoids drift.
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


@MULTIMODAL_REGISTRY.register_processor(
    WhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder,
)
class WhisperForConditionalGeneration(
    nn.Module,
    SupportsTranscription,
    SupportsMultiModal,
):
    """Factory that validates config and selects the Whisper implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements; delegates
    forward() to the selected implementation.

    Carries the transcription/multimodal interface markers (M4) so
    ``vllm serve`` routes ``/v1/audio/transcriptions`` here. The concrete
    transcription classmethods are inherited from vLLM-core's upstream Whisper
    class (see the note above); only the audio-mm processor is (re-)registered
    on THIS class because ``register_processor`` stores ``_processor_factory``
    on the decorated class object and the multimodal registry resolves it from
    the *registered* (plugin) class, not core's.
    """

    # ── M4: transcription/multimodal interface markers ──────────────────────
    # SupportsTranscription requires supported_languages; the rest of the
    # transcription classmethods (get_generation_prompt, validate_language,
    # get_speech_to_text_config, get_num_audio_tokens, post_process_output,
    # language-detection helpers) are inherited verbatim from _CoreWhisper below.
    supported_languages = ISO639_1_SUPPORTED_LANGS
    supports_transcription_only = True
    supports_segment_timestamp = True
    supports_explicit_language_detection = True

    # Inherit the concrete transcription classmethods from core Whisper. These
    # are class-body-independent (they only use cls.supported_languages + the
    # tokenizer/processor), so binding them here gives byte-identical routing.
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
        neuron_config: NeuronConfig | None = None,
        vision_neuron_config=None,
        text_neuron_config: NeuronConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config,
            neuron_config=neuron_config,
            vision_neuron_config=vision_neuron_config,
            text_neuron_config=text_neuron_config,
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        vision_neuron_config=None,
        text_neuron_config: NeuronConfig | None = None,
        **kwargs,
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly.

        Accepts both the 2-arg (text) and 3-arg (multimodal) call forms the
        runner uses at neuron_model_runner.py:1178-1190.
        """
        return cls._select_implementation(
            hf_config,
            neuron_config=neuron_config,
            vision_neuron_config=vision_neuron_config,
            text_neuron_config=text_neuron_config,
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        vision_neuron_config=None,
        text_neuron_config: NeuronConfig | None = None,
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model_bf16 import WhisperForConditionalGeneration as Model

        nc = neuron_config if neuron_config is not None else text_neuron_config
        return Model.from_configs(
            hf_config,
            neuron_config=nc,
            vision_neuron_config=vision_neuron_config,
        )

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        # WhisperConfig.__post_init__ asserts large-v3 dims (d_model, vocab,
        # heads). No quantization is supported for Whisper on this path.
        del neuron_config
