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

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Per-graph compiler arguments (opt-in): VLLM_NEURON_DECODE_O3=1 compiles
        the plain decode graph at -O3. Prefill / encoder keep the base level."""
        import os

        base = list(base or [])
        if graph_type == "decode" and os.environ.get("VLLM_NEURON_DECODE_O3") == "1":
            from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

            out = with_optimization_level(base, 3)
            logger.info("ministral3: decode graph compiler args %s", [a for a in out if a.startswith("-O")])
            return out
        if graph_type == "encoder" and os.environ.get("VLLM_NEURON_ENCODER_O3") == "1":
            from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

            return with_optimization_level(base, 3)
        return base

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


def _register_processor(cls):
    from vllm.multimodal import MULTIMODAL_REGISTRY

    from .processing import (
        Mistral3DummyInputsBuilder,
        Mistral3MultiModalProcessor,
        Mistral3ProcessingInfo,
    )

    return MULTIMODAL_REGISTRY.register_processor(
        Mistral3MultiModalProcessor,
        info=Mistral3ProcessingInfo,
        dummy_inputs=Mistral3DummyInputsBuilder,
    )(cls)


@_register_processor
class Ministral3ForCausalLM(LlamaForCausalLM):
    """Factory: validate the Ministral 3 config, then build the text decoder,
    or the text decoder plus the Pixtral tower when images are enabled."""

    supports_packed_prefill = True
    supports_multimodal = True

    # Encoder sizing (used by the runner/platform through the registry class).
    @classmethod
    def get_vision_token_merge_factor(cls, hf_config):
        return hf_config.spatial_merge_size**2

    @classmethod
    def get_max_encoder_embed_budget(cls, hf_config, encoder_neuron_config):
        from .multimodal import Ministral3ForConditionalGeneration as M

        return M.get_max_encoder_embed_budget(hf_config, encoder_neuron_config)

    @classmethod
    def get_encoder_cache_block_size(cls, hf_config, encoder_attention_block_size):
        from .multimodal import Ministral3ForConditionalGeneration as M

        return M.get_encoder_cache_block_size(hf_config, encoder_attention_block_size)

    @classmethod
    def get_max_encoder_blocks_per_request(cls, hf_config, encoder_neuron_config):
        from .multimodal import Ministral3ForConditionalGeneration as M

        return M.get_max_encoder_blocks_per_request(hf_config, encoder_neuron_config)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: object | None = None,
        **kwargs,
    ) -> nn.Module:
        # The runner passes text_neuron_config / encoder_neuron_config for a
        # multimodal architecture. With an encoder config (and image inputs
        # enabled) build the model with the Pixtral tower; otherwise text only.
        nc = neuron_config if neuron_config is not None else text_neuron_config
        if encoder_neuron_config is not None and _images_enabled():
            cls._validate_config(hf_config, nc)
            from .multimodal import Ministral3ForConditionalGeneration

            return Ministral3ForConditionalGeneration.from_configs(
                hf_config, nc, encoder_neuron_config
            )
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
        max_model_len = getattr(getattr(vllm_config, "model_config", None), "max_model_len", None)
        # Llama 4 query scaling (1 + beta*log(1 + floor(pos/orig))) is exactly 1
        # below orig. Past it, the decode attention runs the torch path with a
        # per-token query scale (the fused decode kernel scales Q by a scalar).
        if beta and orig and max_model_len and max_model_len > orig:
            logger.warning(
                "max_model_len=%d > %d: Llama 4 query scaling is on; decode "
                "attention uses the PyTorch path (slower than the fused kernel).",
                max_model_len,
                orig,
            )


def _images_enabled() -> bool:
    vllm_config = _current_vllm_config()
    mm_config = getattr(getattr(vllm_config, "model_config", None), "multimodal_config", None)
    return mm_config is not None and mm_config.get_limit_per_prompt("image") != 0


def _current_vllm_config():
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no vLLM context (unit tests)
        return None
