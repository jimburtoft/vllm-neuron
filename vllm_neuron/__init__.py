# SPDX-License-Identifier: Apache-2.0
"""VllmNeuronPlugin module."""

import glob
import warnings
from vllm_neuron.utils import set_unique_rt_root_comm_id


def _register_nemotron_h_config():
    """Register NemotronH config with transformers AutoConfig.

    The nemotron_h model type is not in standard transformers, so we register
    a minimal config class at import time so AutoConfig.from_pretrained() can
    deserialize the HF config without trust_remote_code at the vLLM level.
    """
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if "nemotron_h" not in CONFIG_MAPPING_NAMES:
        from transformers import PretrainedConfig

        class NemotronHConfig(PretrainedConfig):
            model_type = "nemotron_h"

        AutoConfig.register("nemotron_h", NemotronHConfig)


_register_nemotron_h_config()


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


# Set unique CCOM bootstrap port to overwrite hard-coded NEURON_RT_ROOT_COMM_ID in XLA based torch_neuronx,
# to prevent port collisions in DP and DI configurations.
set_unique_rt_root_comm_id()
