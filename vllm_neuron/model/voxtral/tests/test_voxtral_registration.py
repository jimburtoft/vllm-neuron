# SPDX-License-Identifier: Apache-2.0
"""No-device import / registration / config unit tests for the native-plugin
Voxtral-Mini-3B model.

These tests run in CI without a Neuron device or model weights. They assert:

1. ``get_models()`` registers ``VoxtralForConditionalGeneration``.
2. The factory imports cleanly and carries the transcription / multimodal
   interface markers ``vllm serve`` needs to route ``/v1/audio/transcriptions``.
3. ``VoxtralConfig`` asserts the Mini-3B architecture (Ministral-3B text
   decoder: hidden 3072, 32/8 GQA, head_dim 128, 30 layers, vocab 131072;
   audio encoder: Whisper-derived, hidden 1280, 20 heads, head_dim 64,
   32 layers, 128 mel bins) and coerces fp16/fp32 -> bf16.

Device-level correctness (byte-identical greedy vs CPU HF) lives in
``test_voxtral_correctness.py`` behind the ``neuron_device`` marker.
"""

import pytest
import torch


def test_registry_includes_voxtral():
    """get_models() must register the Voxtral arch under its HF arch name."""
    from vllm_neuron.model.registry import get_models

    models = dict(get_models())
    assert "VoxtralForConditionalGeneration" in models, (
        f"Voxtral not registered; got {sorted(models)}"
    )


def test_factory_imports_and_carries_transcription_markers():
    """The registered factory class must satisfy the transcription/multimodal
    interfaces so the serving layer routes /v1/audio/transcriptions to it.

    ``SupportsTranscription`` is a runtime-checkable Protocol with non-method
    members, so ``issubclass()`` raises TypeError; we check the required
    members are present directly.
    """
    from vllm_neuron.model.voxtral import VoxtralForConditionalGeneration

    # supported_languages + supports_transcription_only are checked by the
    # runner's get_supported_tasks() and the serving layer's route registration.
    assert getattr(VoxtralForConditionalGeneration, "supported_languages", None)
    assert VoxtralForConditionalGeneration.supports_transcription_only is False

    # Concrete transcription classmethods delegated from upstream Voxtral.
    for name in (
        "get_generation_prompt",
        "get_speech_to_text_config",
        "get_num_audio_tokens",
    ):
        assert callable(getattr(VoxtralForConditionalGeneration, name)), name


def test_factory_from_configs_signature():
    """from_configs must accept 3-arg (multimodal) signature the runner uses
    for multimodal models: (hf_config, text_neuron_config, vision_neuron_config)."""
    import inspect

    from vllm_neuron.model.voxtral import VoxtralForConditionalGeneration

    assert hasattr(VoxtralForConditionalGeneration, "from_configs")
    sig = inspect.signature(VoxtralForConditionalGeneration.from_configs)
    params = sig.parameters
    assert "hf_config" in params
    assert "text_neuron_config" in params
    assert "vision_neuron_config" in params


def test_config_asserts_mini_3b_dims():
    """VoxtralConfig must fail fast on non-Mini-3B dims (silent whisper-tiny
    fallback footgun from HF config.json missing keys)."""
    from vllm_neuron.model.voxtral import VoxtralConfig

    # Default (empty) config uses Voxtral-Mini-3B defaults.
    cfg = VoxtralConfig()
    tc = cfg.text_config
    ac = cfg.audio_config
    assert tc.hidden_size == 3072
    assert tc.num_attention_heads == 32
    assert tc.num_key_value_heads == 8  # GQA
    assert tc.head_dim == 128
    assert tc.num_hidden_layers == 30
    assert tc.vocab_size == 131072
    assert ac.hidden_size == 1280
    assert ac.num_attention_heads == 20
    assert ac.head_dim == 64
    assert ac.num_hidden_layers == 32
    assert ac.num_mel_bins == 128
    assert ac.max_source_positions == 1500
    assert cfg.audio_token_id == 24


def test_config_from_hf_config():
    """VoxtralConfig.from_configs must parse a nested HF config
    (text_config + audio_config) and translate rope_theta scalar to
    rope_parameters dict."""
    from vllm_neuron.model.voxtral import VoxtralConfig

    hf = {
        "text_config": {
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": 30,
            "vocab_size": 131072,
            "intermediate_size": 8192,
            "rope_theta": 1e8,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
        },
        "audio_config": {
            "hidden_size": 1280,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "head_dim": 64,
            "num_hidden_layers": 32,
            "num_mel_bins": 128,
            "max_source_positions": 1500,
            "intermediate_size": 5120,
        },
        "audio_token_id": 24,
        "vocab_size": 131072,
        "hidden_size": 3072,
    }
    cfg = VoxtralConfig.from_configs(hf)
    assert cfg.text_config.rope_parameters == {
        "rope_type": "default", "rope_theta": 1e8,
    }


def test_config_coerces_fp16_and_fp32_to_bf16():
    """Voxtral checkpoint ships bf16 but the plugin's config coercion path
    handles fp16 / fp32 checkpoint dtypes (NCC_IVRF100 workaround: mixed-dtype
    stacks broke on 2.30-era compilers)."""
    from vllm_neuron.model.voxtral import VoxtralTextConfig

    for src in (torch.float16, torch.float32):
        cfg = VoxtralTextConfig.from_hf({"torch_dtype": src})
        assert cfg.torch_dtype == torch.bfloat16, (
            f"{src} should coerce to bf16, got {cfg.torch_dtype}"
        )

    # bf16 stays bf16.
    cfg = VoxtralTextConfig.from_hf({"torch_dtype": torch.bfloat16})
    assert cfg.torch_dtype == torch.bfloat16


def test_config_from_hf_accepts_string_dtype():
    """HF configs carry torch_dtype as a string ("bfloat16") or the newer
    "dtype" field."""
    from vllm_neuron.model.voxtral import VoxtralTextConfig

    cfg = VoxtralTextConfig.from_hf({"torch_dtype": "bfloat16"})
    assert cfg.torch_dtype == torch.bfloat16

    cfg = VoxtralTextConfig.from_hf({"dtype": "bfloat16"})
    assert cfg.torch_dtype == torch.bfloat16
