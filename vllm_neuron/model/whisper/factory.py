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


class WhisperForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the Whisper implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements; delegates
    forward() to the selected implementation.
    """

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
