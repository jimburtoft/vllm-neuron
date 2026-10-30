# SPDX-License-Identifier: Apache-2.0
"""Factory for Gemma3 model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Gemma3ForCausalLM(nn.Module):
    """Factory that validates config and selects the Gemma3 implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements. The factory
    stores the selected implementation and delegates forward() calls to it.

    NOTE: this class shares its name with ``model.Gemma3ForCausalLM``. The
    package ``__init__`` must export THIS one (the factory) -- the registry
    calls ``from_configs`` on whatever it is handed, and only the factory's
    ``from_configs`` runs ``_validate_config``.
    """

    def __init__(
        self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the selected implementation."""
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: object | None = None,
        **kwargs,
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation.

        Must accept BOTH call forms.

        Because this class is registered under ``Gemma3ForConditionalGeneration``
        (what Gemma3 checkpoints, including text-only finetunes, declare
        in config.json), the Neuron model runner classifies the architecture as
        **multimodal** and calls:

            from_configs(hf_config=..., text_neuron_config=..., vision_neuron_config=...)

        ...whereas the text-only path calls the conventional
        ``(hf_config, neuron_config)``. Accepting only the latter fails at
        ``neuron_model_runner.py:1264`` with:

            TypeError: Gemma3ForCausalLM.from_configs() got an unexpected
            keyword argument 'text_neuron_config'

        This port is text-only, so ``vision_neuron_config`` is
        accepted and dropped. ``**kwargs`` absorbs any future runner additions.
        """
        nc = neuron_config if neuron_config is not None else text_neuron_config
        return cls._select_implementation(hf_config, nc)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the implementation based on config.

        Gemma3 ships a single BF16 text-decoder implementation. There is no
        quantized variant and no pooling variant, so this is an unconditional
        dispatch -- unlike qwen3, which splits on ``runner_type``.
        """
        cls._validate_config(hf_config, neuron_config)

        from .model import Gemma3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Reject configurations this port does not implement.

        Failing here gives a clear message instead of a cryptic failure deep in
        weight loading or a silent accuracy bug.
        """
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(
                f"quantization={quantization!r} is not supported for Gemma3. "
                "Only BF16 (quantization=None or 'bf16') is implemented."
            )

        # Parse the config through our own adapter so the checks below see the
        # resolved text-decoder values (the raw hf_config is the multimodal
        # wrapper and nests everything under text_config).
        from .config import Gemma3Config

        config = Gemma3Config.from_configs(hf_config, neuron_config)

        # Gemma3 dropped Gemma2's logit soft-capping and this port does not
        # implement it. A config that sets either value would silently get
        # un-capped logits, so reject it.
        if config.attn_logit_softcapping is not None:
            raise ValueError(
                "attn_logit_softcapping="
                f"{config.attn_logit_softcapping!r} is not implemented for "
                "Gemma3 on Neuron (Gemma3 dropped Gemma2's soft-capping; the "
                "attention kernels have no cap argument)."
            )
        if config.final_logit_softcapping is not None:
            raise ValueError(
                "final_logit_softcapping="
                f"{config.final_logit_softcapping!r} is not implemented for "
                "Gemma3 on Neuron."
            )

        # The MLP is hard-coded to the tanh-approximation GELU (written out
        # explicitly to stay torch.compile-fullgraph clean). Anything else
        # would be silently wrong.
        if config.hidden_activation not in ("gelu_pytorch_tanh",):
            raise ValueError(
                f"hidden_activation={config.hidden_activation!r} is not "
                "supported for Gemma3 on Neuron; only 'gelu_pytorch_tanh' is "
                "implemented."
            )

        # Gemma3 has no attention bias and the port creates no bias params.
        if config.attention_bias:
            raise ValueError(
                "attention_bias=True is not supported for Gemma3 on Neuron; "
                "the port creates no QKV/O bias parameters."
            )
