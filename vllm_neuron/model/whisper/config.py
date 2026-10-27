# SPDX-License-Identifier: Apache-2.0
"""Configuration for the native Whisper large-v3 model.

The config holds the Whisper dimensions and the decoder ``NeuronConfig``. The
constructor checks the large-v3 dimensions. A config with different dimensions
fails at construction, before weight loading starts.
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig


@dataclass
class WhisperConfig:
    # <-- MODEL-SPECIFIC: Whisper large-v3 dimensions.
    d_model: int = 1280
    encoder_layers: int = 32
    decoder_layers: int = 32
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    max_target_positions: int = 448
    vocab_size: int = 51866
    pad_token_id: int = 50256
    activation_function: str = "gelu"
    torch_dtype: torch.dtype = torch.bfloat16

    # Decoder neuron config (KV blocks, sampling, TP degree).
    neuron_config: NeuronConfig | None = None
    # Encoder neuron config. The runner runs the audio encoder only when this
    # config is set.
    encoder_neuron_config: EncoderNeuronConfig | None = None

    # The largest number of requests in one batch. The model keeps one
    # cross-KV slot for each request. from_configs reads the value from the
    # vLLM scheduler config.
    max_num_seqs: int = 1

    # Set to False only in unit tests that use small dimensions.
    check_dims: bool = True

    # The prefix of the self-attention layer names (the KV-cache layer names).
    # A speculative-decoding draft uses a different prefix, so that its KV-cache
    # layers do not have the same names as the target layers.
    layer_name_prefix: str = "decoder"

    def __post_init__(self):
        if not self.check_dims:
            return
        # Check the large-v3 dimensions. A wrong config must fail here, not
        # at a shape mismatch in weight loading.
        if self.d_model != 1280:
            raise ValueError(f"Whisper d_model is {self.d_model}. Expected 1280.")
        if self.vocab_size != 51866:
            raise ValueError(
                f"Whisper vocab_size is {self.vocab_size}. Expected 51866 (large-v3)."
            )
        if self.encoder_attention_heads != 20 or self.decoder_attention_heads != 20:
            raise ValueError(
                "Whisper attention heads are "
                f"{self.encoder_attention_heads}/{self.decoder_attention_heads}. "
                "Expected 20/20."
            )

    @property
    def head_dim(self) -> int:
        """Return the attention head width (1280 / 20 = 64)."""
        return self.d_model // self.decoder_attention_heads

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig | dict | str,
        neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
        **overrides,
    ) -> "WhisperConfig":
        """Build the config from an HF config, a dict, or a JSON file path."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            for attr in ("torch_dtype", "dtype"):
                value = getattr(hf_config, attr, None)
                if value is not None:
                    config_dict["torch_dtype"] = value
                    break
        else:
            config_dict = dict(hf_config)

        # transformers 5 writes "dtype" in config.json. The dataclass field is
        # "torch_dtype".
        if "torch_dtype" not in config_dict and config_dict.get("dtype") is not None:
            config_dict["torch_dtype"] = config_dict["dtype"]

        field_names = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in config_dict.items() if k in field_names}
        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        # Whisper large-v3 ships in float16. The model runs in bfloat16.
        # Neuron rejects an fp32 + fp16 add (NCC_IVRF100) in the fp32-bias
        # row-parallel path, so change fp16 and fp32 to bf16.
        if filtered.get("torch_dtype") in (None, torch.float16, torch.float32):
            filtered["torch_dtype"] = torch.bfloat16

        if "max_num_seqs" not in overrides:
            filtered["max_num_seqs"] = _current_max_num_seqs()
        filtered.update(overrides)
        filtered["neuron_config"] = neuron_config
        filtered["encoder_neuron_config"] = encoder_neuron_config
        return cls(**filtered)


def current_max_model_len() -> int | None:
    """Return ``max_model_len`` of the current vLLM config, or None if no config is set."""
    try:
        from vllm.config import get_current_vllm_config

        value = get_current_vllm_config().model_config.max_model_len
    except (AssertionError, AttributeError, RuntimeError, ImportError):
        return None
    return int(value) if value else None


def _current_max_num_seqs() -> int:
    """Return ``max_num_seqs`` of the current vLLM config, or 1 if no config is set."""
    try:
        from vllm.config import get_current_vllm_config

        value = get_current_vllm_config().scheduler_config.max_num_seqs
    except (AssertionError, AttributeError, RuntimeError, ImportError):
        return 1
    return int(value) if value else 1
