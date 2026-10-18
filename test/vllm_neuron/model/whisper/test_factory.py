# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Whisper registration, factory, config and transcription flags.

These tests run on CPU. They need no Neuron device and no weights.

Run with:
    pytest test/vllm_neuron/model/whisper/test_factory.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_neuron.model.neuron_config import EncoderNeuronConfig

pytestmark = [pytest.mark.fast]


def _enc_cfg(buckets=(1500,), block=1500, tp=1):
    return EncoderNeuronConfig(
        num_encoder_input_buckets=list(buckets),
        encoder_attention_block_size=block,
        tp_size=tp,
        dp_size=1,
    )


class TestRegistration:
    def test_registry_includes_whisper(self):
        """The registry must register the HF architecture name."""
        from vllm_neuron.model.registry import get_models
        from vllm_neuron.model.whisper import WhisperForConditionalGeneration

        models = dict(get_models())
        assert models["WhisperForConditionalGeneration"] is WhisperForConditionalGeneration

    def test_factory_has_transcription_interface(self):
        """The serving layer calls these class methods on the registered class."""
        from vllm.model_executor.models.interfaces import (
            SupportsMultiModal,
            SupportsTranscription,
            supports_transcription,
        )

        from vllm_neuron.model.whisper import WhisperForConditionalGeneration as F

        assert SupportsTranscription in F.__mro__
        assert SupportsMultiModal in F.__mro__
        assert supports_transcription(F)
        assert F.supports_transcription_only is True
        assert "en" in F.supported_languages
        for name in (
            "validate_language",
            "get_generation_prompt",
            "get_speech_to_text_config",
            "get_num_audio_tokens",
            "get_language_token_ids",
            "get_language_detection_prompt",
            "parse_language_detection_output",
            "get_placeholder_str",
        ):
            assert callable(getattr(F, name)), name

    def test_processor_is_registered_on_plugin_class(self):
        """The multimodal registry reads the processor from the registered class."""
        from vllm.model_executor.models.whisper import WhisperMultiModalProcessor

        from vllm_neuron.model.whisper import WhisperForConditionalGeneration as F

        assert "_processor_factory" in F.__dict__
        assert F._processor_factory.processor is WhisperMultiModalProcessor

    def test_instance_class_reports_transcription_only(self):
        """The runner reads the flags on the model instance (the BF16 class)."""
        from vllm.model_executor.models.interfaces import supports_transcription

        from vllm_neuron.model.whisper.model_bf16 import (
            WhisperForConditionalGeneration as M,
        )

        assert supports_transcription(M)
        assert M.supports_transcription_only is True

    def test_encoder_warmup_protocol(self):
        """The worker warms up the encoder only for a SupportsEncoderWarmup model."""
        from vllm_neuron.model.interfaces import SupportsEncoderWarmup
        from vllm_neuron.model.whisper.model_bf16 import (
            WhisperForConditionalGeneration as M,
        )

        assert hasattr(M, "get_encoder")
        assert hasattr(M, "build_encoder_synthetic_inputs")
        assert issubclass(M, SupportsEncoderWarmup)


class TestFactory:
    def test_forwards_configs_to_bf16(self):
        text_cfg = MagicMock()
        text_cfg.quantization = "bf16"
        enc_cfg = _enc_cfg()
        hf_config = MagicMock()
        with patch(
            "vllm_neuron.model.whisper.model_bf16."
            "WhisperForConditionalGeneration.from_configs"
        ) as mock_bf16:
            from vllm_neuron.model.whisper.factory import WhisperForConditionalGeneration

            WhisperForConditionalGeneration.from_configs(
                hf_config, text_neuron_config=text_cfg, encoder_neuron_config=enc_cfg
            )
            mock_bf16.assert_called_once_with(
                hf_config, text_neuron_config=text_cfg, encoder_neuron_config=enc_cfg
            )

    def test_rejects_quantization(self):
        from vllm_neuron.model.whisper.factory import WhisperForConditionalGeneration

        text_cfg = MagicMock()
        text_cfg.quantization = "fp8"
        with pytest.raises(ValueError, match="quantization"):
            WhisperForConditionalGeneration.from_configs(MagicMock(), text_cfg, None)

    @pytest.mark.parametrize("block", [2048, 375])
    def test_rejects_block_size_not_one_clip(self, block):
        from vllm_neuron.model.whisper.factory import WhisperForConditionalGeneration

        with pytest.raises(ValueError, match="encoder_attention_block_size"):
            WhisperForConditionalGeneration.from_configs(
                MagicMock(), None, _enc_cfg(buckets=(4096,), block=block)
            )

    def test_rejects_partial_clip_bucket(self):
        from vllm_neuron.model.whisper.factory import WhisperForConditionalGeneration

        with pytest.raises(ValueError, match="multiple of 1500"):
            WhisperForConditionalGeneration._validate_config(
                None, _enc_cfg(buckets=(1500, 2000))
            )

    def test_encoder_budget(self):
        from vllm_neuron.model.whisper.factory import WhisperForConditionalGeneration as F
        from vllm_neuron.utils.mm_utils import (
            get_max_encoder_blocks_per_request,
            get_max_encoder_embed_budget,
        )

        hf_config = MagicMock()
        hf_config.architectures = ["WhisperForConditionalGeneration"]
        enc = _enc_cfg(buckets=(1500,))
        assert F.get_max_encoder_embed_budget(hf_config, enc) == 1500
        assert F.get_max_encoder_blocks_per_request(hf_config, enc) == 1
        # The runner reads the budget through the registry.
        assert get_max_encoder_embed_budget(hf_config, enc) == 1500
        assert get_max_encoder_blocks_per_request(hf_config, enc) == 1
        assert F.get_max_encoder_embed_budget(hf_config, _enc_cfg(buckets=())) is None


class TestConfig:
    def test_large_v3_defaults(self):
        from vllm_neuron.model.whisper.config import WhisperConfig

        cfg = WhisperConfig()
        assert cfg.head_dim == 64
        assert cfg.torch_dtype == torch.bfloat16

    @pytest.mark.parametrize(
        "field,value", [("d_model", 1024), ("vocab_size", 51865), ("decoder_attention_heads", 16)]
    )
    def test_rejects_other_dims(self, field, value):
        from vllm_neuron.model.whisper.config import WhisperConfig

        with pytest.raises(ValueError):
            WhisperConfig(**{field: value})

    @pytest.mark.parametrize("dtype", ["float16", "float32", torch.float16])
    def test_dtype_becomes_bf16(self, dtype):
        from vllm_neuron.model.whisper.config import WhisperConfig

        cfg = WhisperConfig.from_configs({"dtype": dtype, "d_model": 1280})
        assert cfg.torch_dtype == torch.bfloat16

    def test_attaches_neuron_configs(self):
        from vllm_neuron.model.whisper.config import WhisperConfig

        text_cfg = MagicMock()
        enc = _enc_cfg()
        cfg = WhisperConfig.from_configs({}, text_cfg, enc)
        assert cfg.neuron_config is text_cfg
        assert cfg.encoder_neuron_config is enc


class TestWeightKeys:
    @pytest.mark.parametrize(
        "src,dst",
        [
            ("layers.0.fc1.weight", "layers.0.mlp.fc1.weight"),
            ("layers.0.fc2.weight", "layers.0.mlp.fc2.rpl.weight"),
            ("layers.0.fc2.bias", "layers.0.mlp.fc2.bias"),
            ("layers.7.self_attn.out_proj.weight", "layers.7.self_attn.out_proj.rpl.weight"),
            ("layers.7.encoder_attn.out_proj.weight", "layers.7.encoder_attn.out_proj.rpl.weight"),
            ("layers.7.encoder_attn.k_proj.weight", "layers.7.encoder_attn.k_proj.weight"),
        ],
    )
    def test_remap(self, src, dst):
        from vllm_neuron.model.whisper.weight_loaders import remap_key

        assert remap_key(src) == dst

    def test_rpl_bias_stays_fp32(self):
        from vllm_neuron.model.whisper.weight_loaders import build_state

        sd = {
            "model.decoder.layers.0.fc2.bias": torch.ones(4, dtype=torch.float16),
            "model.decoder.layers.0.fc1.bias": torch.ones(4, dtype=torch.float16),
            "model.encoder.conv1.weight": torch.ones(2, dtype=torch.float16),
        }
        out = build_state(sd, "model.decoder.", torch.bfloat16)
        assert set(out) == {"layers.0.mlp.fc2.bias", "layers.0.mlp.fc1.bias"}
        assert out["layers.0.mlp.fc2.bias"].dtype == torch.float32
        assert out["layers.0.mlp.fc1.bias"].dtype == torch.bfloat16
