# SPDX-License-Identifier: Apache-2.0
"""VllmNeuronPlugin module."""

import glob
import warnings


def _register_qwen3_5_moe_config():
    """Register Qwen3.5-MoE config with transformers if not already known.

    The qwen3_5_moe model_type is new and may not be in the installed
    transformers version. We register a minimal PretrainedConfig subclass
    so that AutoConfig.from_pretrained() can load the model's config.json
    without raising ValueError.
    """
    try:
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        if "qwen3_5_moe" not in CONFIG_MAPPING_NAMES:

            class Qwen35MoeTextConfig(PretrainedConfig):
                model_type = "qwen3_5_moe_text"

            class Qwen35MoeConfig(PretrainedConfig):
                model_type = "qwen3_5_moe"
                sub_configs = {"text_config": Qwen35MoeTextConfig}

                def __init__(self, text_config=None, **kwargs):
                    if isinstance(text_config, dict):
                        text_config = Qwen35MoeTextConfig(**text_config)
                    self.text_config = text_config
                    super().__init__(**kwargs)

            AutoConfig.register("qwen3_5_moe", Qwen35MoeConfig)
    except Exception:
        pass  # Non-critical -- only needed for qwen3_5_moe models


_register_qwen3_5_moe_config()


def _is_neuron_dev() -> bool:
    """Detect Neuron device by checking for /dev/neuron* devices."""
    neuron_devices = glob.glob("/dev/neuron*")
    return len(neuron_devices) > 0


def register():
    """Register the Neuron platform if Neuron devices are present, else return None."""
    if not _is_neuron_dev():
        warnings.warn(
            "No Neuron devices found. Skipping Neuron plugin registration.",
            category=UserWarning,
        )
        return None
    return "vllm_neuron.platform.NeuronPlatform"
