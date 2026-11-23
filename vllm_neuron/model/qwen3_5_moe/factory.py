# SPDX-License-Identifier: Apache-2.0
"""Factory for the Qwen3.5-MoE text model (registered for both the multimodal
checkpoint architecture and the text alias; the vision tower is not built)."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5MoeForCausalLM(nn.Module):
    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None):
        super().__init__()
        self._model = self.from_configs(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config, neuron_config=None, **kwargs):
        if neuron_config is None:
            neuron_config = kwargs.get("text_neuron_config")
        q = neuron_config.quantization if neuron_config else None
        if q not in (None, "bf16"):
            raise ValueError(f"quantization={q!r} not supported for Qwen3.5-MoE (BF16 only)")
        from .model import Qwen3_5MoeForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)


class Qwen3_5MoeForConditionalGeneration(nn.Module):
    """Hybrid Qwen3.5-MoE decoder + Qwen3-VL-style vision tower (Isaac-0.5 / Qwen3.5-MoE VL checkpoints).

    The runner calls from_configs(hf_config=, text_neuron_config=, encoder_neuron_config=) for multimodal models
    and looks the encoder-budget classmethods up on this class (mm_utils)."""

    def __init__(self, hf_config, text_neuron_config=None, encoder_neuron_config=None):
        super().__init__()
        self._model = self.from_configs(hf_config, text_neuron_config, encoder_neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config, text_neuron_config=None, encoder_neuron_config=None, **kwargs):
        text_neuron_config = text_neuron_config or kwargs.get("neuron_config")
        q = text_neuron_config.quantization if text_neuron_config else None
        if q not in (None, "bf16"):
            raise ValueError(f"quantization={q!r} not supported for Qwen3.5-MoE (BF16 only)")
        from .model_vl import Qwen3_5MoeVLForConditionalGeneration as Model

        return Model.from_configs(hf_config, text_neuron_config, encoder_neuron_config)


def _bind_vl_classmethods():
    from vllm_neuron.model.qwen3_vl.factory import Qwen3VLForConditionalGeneration as F

    for name in ("get_vision_token_merge_factor", "get_max_encoder_embed_budget", "get_max_encoder_blocks_per_request",
                 "get_max_pixels_token_count", "get_epd_kwargs", "get_placeholder_is_embed"):
        if hasattr(F, name):
            setattr(Qwen3_5MoeForConditionalGeneration, name, classmethod(getattr(F, name).__func__))


_bind_vl_classmethods()
