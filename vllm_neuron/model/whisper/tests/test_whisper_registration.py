# SPDX-License-Identifier: Apache-2.0
"""No-device import / registration / config unit tests for the native-plugin
Whisper large-v3 model.

These tests run in CI without a Neuron device or model weights. They assert:

1. ``get_models()`` registers ``WhisperForConditionalGeneration``.
2. The factory imports cleanly and carries the transcription / multimodal
   interface markers ``vllm serve`` needs to route ``/v1/audio/transcriptions``.
3. ``WhisperConfig`` asserts the large-v3 architecture (d_model==1280,
   vocab==51866, 20 enc/dec heads) and coerces fp16/fp32 -> bf16.

Device-level correctness (byte-identical greedy vs OpenAI-whisper) lives in
``test_whisper_correctness.py`` behind the ``neuron_device`` marker.
"""

import pytest
import torch


def test_registry_includes_whisper():
    """get_models() must register the Whisper arch under its HF arch name."""
    from vllm_neuron.model.registry import get_models

    models = dict(get_models())
    assert "WhisperForConditionalGeneration" in models, (
        f"Whisper not registered; got {sorted(models)}"
    )


def test_factory_imports_and_carries_transcription_markers():
    """The registered factory class must satisfy the transcription/multimodal
    interfaces so the serving layer routes /v1/audio/transcriptions to it.

    ``SupportsTranscription`` / ``SupportsMultiModal`` are runtime-checkable
    Protocols with non-method members, so ``issubclass()`` raises TypeError; we
    check the interface is in the MRO and the required members are present
    (which is what the serving layer / runner actually inspect)."""
    from vllm.model_executor.models.interfaces import (
        SupportsMultiModal,
        SupportsTranscription,
    )

    from vllm_neuron.model.whisper import WhisperForConditionalGeneration

    # Interface markers are inherited via the MRO (M4 serving wire-up).
    mro = WhisperForConditionalGeneration.__mro__
    assert SupportsTranscription in mro, "must inherit SupportsTranscription"
    assert SupportsMultiModal in mro, "must inherit SupportsMultiModal"

    # supported_languages + the transcription flags the runner's
    # get_supported_tasks() and the serving layer inspect.
    assert getattr(WhisperForConditionalGeneration, "supported_languages", None)
    assert WhisperForConditionalGeneration.supports_transcription_only is True
    assert WhisperForConditionalGeneration.supports_segment_timestamp is True

    # The concrete transcription classmethods inherited from vLLM-core Whisper.
    for name in (
        "validate_language",
        "get_generation_prompt",
        "get_speech_to_text_config",
        "get_num_audio_tokens",
        "get_placeholder_str",
    ):
        assert callable(getattr(WhisperForConditionalGeneration, name)), name


def test_factory_from_configs_is_classmethod():
    """from_configs must accept both the 2-arg (text) and 3-arg (multimodal)
    signatures the runner uses. We only check it is a bound classmethod here
    (constructing the model needs a Neuron device -> device test)."""
    from vllm_neuron.model.whisper import WhisperForConditionalGeneration

    assert hasattr(WhisperForConditionalGeneration, "from_configs")
    import inspect

    sig = inspect.signature(WhisperForConditionalGeneration.from_configs)
    params = sig.parameters
    assert "hf_config" in params
    assert "neuron_config" in params
    assert "vision_neuron_config" in params


def test_config_asserts_large_v3_dims():
    """WhisperConfig must fail fast on non-large-v3 dims (footgun from contrib)."""
    from vllm_neuron.model.whisper import WhisperConfig

    cfg = WhisperConfig()
    assert cfg.d_model == 1280
    assert cfg.vocab_size == 51866
    assert cfg.encoder_attention_heads == 20
    assert cfg.decoder_attention_heads == 20
    assert cfg.head_dim == 64  # 1280 / 20

    with pytest.raises(AssertionError):
        WhisperConfig(d_model=768)
    with pytest.raises(AssertionError):
        WhisperConfig(vocab_size=51865)
    with pytest.raises(AssertionError):
        WhisperConfig(decoder_attention_heads=16)


def test_config_coerces_fp16_and_fp32_to_bf16():
    """whisper-large-v3 ships fp16; the native plugin runs bf16 (fp32-bias
    RowParallel add rejects mixed f32+f16). fp16 and fp32 must coerce to bf16."""
    from vllm_neuron.model.whisper import WhisperConfig

    for src in (torch.float16, torch.float32):
        cfg = WhisperConfig.from_configs({"torch_dtype": src})
        assert cfg.torch_dtype == torch.bfloat16, (
            f"{src} should coerce to bf16, got {cfg.torch_dtype}"
        )

    # bf16 stays bf16.
    cfg = WhisperConfig.from_configs({"torch_dtype": torch.bfloat16})
    assert cfg.torch_dtype == torch.bfloat16


def test_config_from_configs_accepts_string_dtype():
    """HF configs carry torch_dtype as a string ("float16")."""
    from vllm_neuron.model.whisper import WhisperConfig

    cfg = WhisperConfig.from_configs({"torch_dtype": "float16"})
    assert cfg.torch_dtype == torch.bfloat16
