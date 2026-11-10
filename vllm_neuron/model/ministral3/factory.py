# SPDX-License-Identifier: Apache-2.0
"""Factory for the Ministral 3 text decoder (Llama 3 implementation)."""

import logging

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.llama3.factory import LlamaForCausalLM
from vllm_neuron.model.llama3.model import LlamaForCausalLM as LlamaModelImpl
from vllm_neuron.model.neuron_config import NeuronConfig

from .config import llama4_scaling, ministral3_llama_config, text_config_dict

logger = logging.getLogger(__name__)

# The multimodal checkpoint keeps the decoder under ``language_model.``
# (``language_model.model.layers.N...``, ``language_model.lm_head.weight``).
_TEXT_PREFIX = "language_model."


class Ministral3TextModel(LlamaModelImpl):
    """Llama 3 decoder that reads the ``language_model.*`` checkpoint keys.

    ``vision_tower.*`` and ``multi_modal_projector.*`` are never mapped, so
    they are not loaded.
    """

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(ministral3_llama_config(hf_config, neuron_config))

    def load_weights(self, checkpoint_path, device, cache_dir):
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        names = SafetensorsCheckpoint(checkpoint_path, cache_dir).get_tensor_names()
        self.checkpoint_prefix = (
            _TEXT_PREFIX
            if any(n.startswith(_TEXT_PREFIX) for n in names)
            else ""
        )
        dropped = sum(
            n.startswith(("vision_tower.", "multi_modal_projector.")) for n in names
        )
        logger.info(
            "Ministral3: text prefix=%r, vision/projector tensors not loaded=%d",
            self.checkpoint_prefix,
            dropped,
        )
        return super().load_weights(checkpoint_path, device, cache_dir)


class Ministral3ForCausalLM(LlamaForCausalLM):
    """Factory: validate the Ministral 3 config, then build the text decoder."""

    supports_packed_prefill = True

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: object | None = None,
        **kwargs,
    ) -> nn.Module:
        # A multimodal architecture can be called with text_neuron_config /
        # encoder_neuron_config. This implementation is text-only.
        nc = neuron_config if neuron_config is not None else text_neuron_config
        return cls._select_implementation(hf_config, nc)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)
        return Ministral3TextModel.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        text = text_config_dict(hf_config)
        beta, orig = llama4_scaling(text)
        vllm_config = _current_vllm_config()
        if vllm_config is None:
            return
        model_config = vllm_config.model_config
        max_model_len = getattr(model_config, "max_model_len", None)
        # Llama 4 query scaling: factor 1 + beta*log(1 + floor(pos/orig)) is
        # exactly 1 below orig. It is not implemented, so longer contexts would
        # be silently wrong.
        if beta and orig and max_model_len and max_model_len > orig:
            raise NotImplementedError(
                f"max_model_len={max_model_len} > {orig}: the Llama 4 query "
                "scaling (llama_4_scaling_beta="
                f"{beta}) that applies past position {orig} is not implemented. "
                f"Use --max-model-len <= {orig}."
            )
        mm_config = getattr(model_config, "multimodal_config", None)
        if mm_config is not None:
            if mm_config.get_limit_per_prompt("image") != 0:
                raise NotImplementedError(
                    "Ministral 3 on Neuron serves the text decoder only (the "
                    "Pixtral vision tower is not implemented). Start the server "
                    "with --language-model-only (or "
                    "--limit-mm-per-prompt '{\"image\": 0}')."
                )


def _current_vllm_config():
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no vLLM context (unit tests)
        return None
