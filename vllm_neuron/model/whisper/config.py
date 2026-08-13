# SPDX-License-Identifier: Apache-2.0
"""
Whisper Config
==============

Minimal config carrier for the native-plugin Whisper (large-v3) model. Ported
from the contrib ``WhisperConfigLite`` (contrib/whisper/whisper_neuron.py:160-189)
and adapted to the plugin ``from_configs(hf_config, neuron_config)`` convention
used by the other model packages (llama3/config.py:58-107).

<-- MODEL-SPECIFIC: all fields are Whisper-specific. large-v3 dims are asserted
on construction (footgun 3 from contrib): d_model==1280, vocab==51866, 20 heads.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class WhisperConfig:
    # <-- MODEL-SPECIFIC: Whisper large-v3 architecture parameters.
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

    # Framework config (self-KV block sizing, sampling, TP degree, etc.)
    neuron_config: NeuronConfig | None = None

    # Task 020 Medusa: optional speculative-decoding heads config. None ->
    # Medusa disabled (zero overhead, heads never constructed). When set (a
    # dict, e.g. from additional_config['medusa_config']), the top-level model
    # builds MedusaHeads. Recognized keys:
    #   num_heads:         N = K lookahead heads (default 5)
    #   medusa_num_layers: ResBlocks per head (default 1)
    #   init:              "zero" | "random" | "load" (default "random")
    #   heads_path:        checkpoint path when init == "load" (default None)
    #   seed:              RNG seed for init == "random" (default 0)
    medusa_config: dict | None = None

    def __post_init__(self):
        # Footgun 3 (contrib whisper_neuron.py:177-184): assert large-v3 dims so
        # a mis-sized config fails fast at construction rather than at a shape
        # mismatch deep in weight loading.
        assert self.d_model == 1280, f"expected d_model=1280 got {self.d_model}"
        assert self.vocab_size == 51866, (
            f"expected vocab_size=51866 got {self.vocab_size}"
        )
        assert self.encoder_attention_heads == 20, (
            f"expected encoder n_heads=20 got {self.encoder_attention_heads}"
        )
        assert self.decoder_attention_heads == 20, (
            f"expected decoder n_heads=20 got {self.decoder_attention_heads}"
        )

    @property
    def head_dim(self) -> int:
        # 1280 / 20 = 64
        return self.d_model // self.decoder_attention_heads

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        # Whisper large-v3 ships as float16, but the native plugin runs the
        # model in bfloat16 (the proven contrib dtype; vLLM also casts the
        # engine dtype fp16 -> bf16 on Neuron). Neuron rejects fp32+fp16 mixed
        # adds (NCC_IVRF100) in the fp32-bias RowParallel path, so we coerce
        # fp16 to bf16 here. (fp32 checkpoints stay fp32-cast-to-bf16 too.)
        if filtered_dict.get("torch_dtype") in (torch.float16, torch.float32):
            filtered_dict["torch_dtype"] = torch.bfloat16

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
