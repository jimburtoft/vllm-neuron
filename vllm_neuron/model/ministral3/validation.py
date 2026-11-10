# SPDX-License-Identifier: Apache-2.0
"""Validation-time stand-in for ``Mistral3ForConditionalGeneration``.

vLLM inspects the registered architecture class (text generation, multimodal
support, processor) in the frontend before the Neuron worker replaces it with
the plugin implementation. The upstream class cannot be imported with the
transformers release in this environment (see ``processing.py``), so this
module gives an importable class with the same interface flags and the
vendored image processor. It is never instantiated: the worker registers the
Neuron model under the same architecture name.

Keep this module light: the frontend imports it during config validation.
"""

from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY

from .processing import (
    Mistral3DummyInputsBuilder,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    Mistral3MultiModalProcessor,
    info=Mistral3ProcessingInfo,
    dummy_inputs=Mistral3DummyInputsBuilder,
)
class Mistral3ForConditionalGeneration(MistralForCausalLM):
    supports_multimodal = True
    supports_multimodal_raw_input_only = False
    supports_encoder_tp_data = False
    supports_mm_device_do_normalize = False
    requires_raw_input_tokens = False

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")
