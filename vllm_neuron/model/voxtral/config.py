# SPDX-License-Identifier: Apache-2.0
"""
Voxtral Config
==============

Composite multimodal config: top-level VoxtralConfig owns
VoxtralAudioConfig (Whisper-derived audio encoder) and VoxtralTextConfig
(Ministral-3B, Llama-family GQA).

HF checkpoint has nested `audio_config` + `text_config` under a top-level
VoxtralConfig (architectures=["VoxtralForConditionalGeneration"]).

<-- MODEL-SPECIFIC: Fields are Voxtral-specific (Mini-3B-2507).
"""

import json
from dataclasses import dataclass, field, fields

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig


def _from_hf_sub_config(cls, hf_sub_config, neuron_config=None):
    """Shared factory logic for building a sub-config from an HF config sub-object.

    Pattern lifted from vllm_neuron.model.qwen3_vl.config._from_hf_sub_config.
    """
    if isinstance(hf_sub_config, PretrainedConfig):
        config_dict = hf_sub_config.to_dict()
        if (
            hasattr(hf_sub_config, "torch_dtype")
            and hf_sub_config.torch_dtype is not None
        ):
            config_dict["torch_dtype"] = hf_sub_config.torch_dtype
    elif isinstance(hf_sub_config, dict):
        config_dict = hf_sub_config
    else:
        raise TypeError(f"Unsupported config type: {type(hf_sub_config)}")

    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in config_dict.items() if k in field_names}

    # HF config.json uses "dtype" but our dataclass uses "torch_dtype"
    if (
        "torch_dtype" not in filtered
        and "dtype" in config_dict
        and "torch_dtype" in field_names
    ):
        filtered["torch_dtype"] = config_dict["dtype"]

    if "torch_dtype" in filtered and isinstance(filtered["torch_dtype"], str):
        filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

    # NCC_IVRF100 workaround: coerce fp16/fp32 -> bf16 for Neuron BF16 stack.
    if "torch_dtype" in filtered and filtered["torch_dtype"] in (
        torch.float16,
        torch.float32,
    ):
        filtered["torch_dtype"] = torch.bfloat16

    if neuron_config is not None:
        filtered["neuron_config"] = neuron_config

    return cls(**filtered)


@dataclass
class VoxtralTextConfig:
    """Ministral-3B (Llama-family GQA) text decoder config.

    Extracted from hf_config.text_config. Defaults are from Voxtral-Mini-3B-2507.
    """

    # <-- MODEL-SPECIFIC: Ministral-3B dims
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA 32/8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False  # Ministral has separate lm_head.
    # RoPE: Voxtral uses non-scaled RoPE with theta=1e8.
    # Mirrors llama3/config.py's rope_parameters dict shape.
    rope_parameters: dict = field(
        default_factory=lambda: {"rope_type": "default", "rope_theta": 100000000.0}
    )
    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    @classmethod
    def from_hf(cls, hf_sub_config, neuron_config=None):
        return _from_hf_sub_config(cls, hf_sub_config, neuron_config)

    def __post_init__(self):
        # Accept `rope_theta` from HF instead of `rope_parameters` dict.
        # (HF Ministral configs use `rope_theta` scalar + `rope_scaling` dict.)
        pass


@dataclass
class VoxtralAudioConfig:
    """Voxtral audio encoder config (Whisper-derived).

    Extracted from hf_config.audio_config. Defaults are from Voxtral-Mini-3B-2507.
    """

    # <-- MODEL-SPECIFIC: Voxtral audio encoder dims
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_key_value_heads: int = 20  # No GQA in the encoder.
    head_dim: int = 64
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    scale_embedding: bool = False
    activation_function: str = "gelu"
    torch_dtype: torch.dtype = torch.bfloat16

    # Downsample factor from encoder -> projector input: intermediate_size / hidden_size = 4.
    # The projector concatenates 4 encoder frames -> 1 projector token.
    downsample_factor: int = 4

    # Mel filter bank + STFT parameters (Voxtral folds mel into the encoder).
    # Whisper standard values.
    window_size: int = 400
    hop_length: int = 160
    sampling_rate: int = 16000
    # Optional: normalization ceiling from Voxtral processor config.
    global_log_mel_max: float | None = None

    # Framework config (called `vision_neuron_config` per plugin convention
    # even for audio; matches whisper-xla precedent).
    neuron_config: VisionNeuronConfig | None = None

    @classmethod
    def from_hf(cls, hf_sub_config, neuron_config=None):
        return _from_hf_sub_config(cls, hf_sub_config, neuron_config)


@dataclass
class VoxtralConfig:
    """Top-level Voxtral config: composes audio_config + text_config."""

    text_config: VoxtralTextConfig = field(default_factory=VoxtralTextConfig)
    audio_config: VoxtralAudioConfig = field(default_factory=VoxtralAudioConfig)

    # Top-level fields from the HF config.
    audio_token_id: int = 24
    vocab_size: int = 131072  # Mirrors text_config.vocab_size
    hidden_size: int = 3072  # Mirrors text_config.hidden_size
    projector_hidden_act: str = "gelu"
    # Optional Medusa speculative-decoding heads config (from
    # additional_config["medusa_config"]). None -> Medusa disabled.
    medusa_config: dict | None = None

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ):
        """Build VoxtralConfig from HF config + plugin neuron configs.

        The plugin's model runner calls factories with
        `(hf_config, text_neuron_config, vision_neuron_config)` for
        multimodal models (see qwen3_vl/factory.py precedent).
        """
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                cfg_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            cfg_dict = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            cfg_dict = hf_config
        else:
            raise TypeError(f"Unsupported hf_config type: {type(hf_config)}")

        # Handle HF `rope_theta` scalar in text_config -> plugin's dict shape.
        text_cfg = cfg_dict.get("text_config", {})
        if isinstance(text_cfg, dict) and "rope_theta" in text_cfg and "rope_parameters" not in text_cfg:
            text_cfg = dict(text_cfg)  # copy
            text_cfg["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": float(text_cfg["rope_theta"]),
            }

        text_config = VoxtralTextConfig.from_hf(
            text_cfg if text_cfg else cfg_dict, text_neuron_config
        )
        audio_config = VoxtralAudioConfig.from_hf(
            cfg_dict.get("audio_config", {}), vision_neuron_config
        )

        # Top-level field extraction.
        kwargs = {}
        top_field_names = {
            "audio_token_id", "vocab_size", "hidden_size", "projector_hidden_act",
        }
        for k in top_field_names:
            if k in cfg_dict:
                kwargs[k] = cfg_dict[k]

        cfg = cls(text_config=text_config, audio_config=audio_config, **kwargs)
        cfg._assert_expected_dims()
        return cfg

    def _assert_expected_dims(self):
        """Sanity-check against Voxtral-Mini-3B-2507 published dims.

        Guards the "silent whisper-tiny fallback" style footgun called out
        in Task 001 research: if config.json is missing or trimmed, our
        dataclass defaults would silently populate. Assertions surface it.
        """
        tc = self.text_config
        ac = self.audio_config
        # Ministral-3B (text)
        assert tc.hidden_size == 3072, f"text.hidden_size={tc.hidden_size} != 3072"
        assert tc.num_attention_heads == 32, f"text.num_heads={tc.num_attention_heads} != 32"
        assert tc.num_key_value_heads == 8, f"text.num_kv_heads={tc.num_key_value_heads} != 8 (GQA)"
        assert tc.head_dim == 128, f"text.head_dim={tc.head_dim} != 128"
        assert tc.num_hidden_layers == 30, f"text.num_layers={tc.num_hidden_layers} != 30"
        assert tc.vocab_size == 131072, f"text.vocab_size={tc.vocab_size} != 131072"
        # Voxtral audio encoder (Whisper-derived)
        assert ac.hidden_size == 1280, f"audio.hidden_size={ac.hidden_size} != 1280"
        assert ac.num_attention_heads == 20, f"audio.num_heads={ac.num_attention_heads} != 20"
        assert ac.head_dim == 64, f"audio.head_dim={ac.head_dim} != 64"
        assert ac.num_hidden_layers == 32, f"audio.num_layers={ac.num_hidden_layers} != 32"
        assert ac.num_mel_bins == 128, f"audio.num_mel_bins={ac.num_mel_bins} != 128"
        assert ac.max_source_positions == 1500, (
            f"audio.max_source_positions={ac.max_source_positions} != 1500"
        )
        # Adapter shape self-consistency: intermediate_size == hidden_size * downsample_factor
        assert ac.intermediate_size == ac.hidden_size * ac.downsample_factor, (
            f"audio.intermediate_size={ac.intermediate_size} != "
            f"hidden_size ({ac.hidden_size}) * downsample_factor ({ac.downsample_factor})"
        )
        # Top-level mirror check
        assert self.vocab_size == tc.vocab_size
        assert self.hidden_size == tc.hidden_size
