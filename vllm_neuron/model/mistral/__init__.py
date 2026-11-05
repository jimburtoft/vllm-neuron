# SPDX-License-Identifier: Apache-2.0
"""Mistral (dense, full attention) on the Llama 3 implementation.

``MistralForCausalLM`` checkpoints with ``sliding_window: null`` (for example
Mistral-7B-v0.3 / Instruct-v0.3) have the same architecture and the same
HuggingFace weight names as Llama: GQA attention without bias, RoPE,
RMSNorm and a SiLU-gated MLP. They run on :mod:`vllm_neuron.model.llama3`.

The Llama 3 attention has no sliding window. A checkpoint that sets
``sliding_window`` (Mistral-7B-v0.1) is rejected here instead of being run
with full attention, which would give wrong results past the window.
"""

from transformers import PretrainedConfig

from vllm_neuron.model.llama3.factory import LlamaForCausalLM
from vllm_neuron.model.neuron_config import NeuronConfig


class MistralForCausalLM(LlamaForCausalLM):
    """Factory: validate the Mistral config, then build the Llama 3 model."""

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        sliding_window = getattr(hf_config, "sliding_window", None)
        max_pos = getattr(hf_config, "max_position_embeddings", None)
        if sliding_window is not None and (max_pos is None or sliding_window < max_pos):
            raise NotImplementedError(
                f"MistralForCausalLM with sliding_window={sliding_window} is not "
                "supported: the Neuron Mistral path uses full attention only. "
                "Use a checkpoint with sliding_window=null (e.g. Mistral-7B-v0.3)."
            )
        super()._validate_config(hf_config, neuron_config)
