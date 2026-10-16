# SPDX-License-Identifier: Apache-2.0
"""Factory for GptOss model selection based on platform and configuration."""

import torch.nn as nn
from transformers import PretrainedConfig

from libtorch_neuronx_lite.compile.platform import get_platform_target
from vllm_neuron.model.neuron_config import NeuronConfig


class GptOssForCausalLM(nn.Module):
    """Factory that validates config and selects the appropriate GptOss implementation.

    This class extends nn.Module to satisfy vLLM's ModelRegistry requirements.
    The factory stores the selected implementation and delegates forward() calls to it.
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
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Create model from configs. Returns the selected implementation directly."""
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        """Select and instantiate the appropriate implementation based on config."""
        cls._validate_config(hf_config, neuron_config)

        # TODO: Below is a preliminary logic and needs to be made more robust
        platform = get_platform_target()
        quantization = neuron_config.quantization if neuron_config else None

        if quantization == "mxfp4":
            from .model_mxfp4 import GptOssForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        if quantization == "bf16":
            from .model_bf16 import GptOssForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        # FP8 expert weights (model_bf16 path). TRN2 has no MX matmul
        # (nki.isa.nc_matmul_mx is NeuronCore-v4 only) but accepts non-MX
        # float8_e4m3 nc_matmul operands, and the nkilib MoE kernels' gen4 check
        # applies only to MX dtypes. The MXFP4 checkpoint is requantized to FP8 ROW
        # with power-of-two scales, which is exact, halving expert-weight HBM and DMA.
        if quantization == "fp8":
            from .model_bf16 import GptOssForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        if platform.startswith("trn3"):
            from .model_mxfp4 import GptOssForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        if quantization is None:
            from .model_bf16 import GptOssForCausalLM as Model

            return Model.from_configs(hf_config, neuron_config)

        # Reject unrecognized values instead of silently falling back to BF16.
        raise ValueError(
            f"Unsupported neuron_config.quantization={quantization!r} for gpt_oss on "
            f"platform={platform}. Supported: 'bf16', 'fp8' (TRN2/TRN3), "
            f"'mxfp4' (TRN3 only). Omit the field for the default."
        )

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        """Validate that the configuration is supported. Add rules as needed."""
        # TODO: Below is one example rule, but we need to add more and make user errors clear

        quantization = neuron_config.quantization if neuron_config else None
        platform = get_platform_target()

        if quantization == "mxfp4" and platform == "trn2":
            raise ValueError(
                "quantization='mxfp4' is not supported on TRN2 (MX compute needs "
                "nki.isa.nc_matmul_mx, a NeuronCore-v4 instruction). Use "
                "quantization='fp8' (exact requantization, half the expert-weight "
                "memory) or quantization='bf16'."
            )
