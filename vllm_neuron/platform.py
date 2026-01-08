# SPDX-License-Identifier: Apache-2.0
import enum
import importlib.metadata
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.utils import FlexibleArgumentParser
else:
    VllmConfig = None
    FlexibleArgumentParser = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NeuronFramework(enum.Enum):
    NEURONX_DISTRIBUTED_INFERENCE = "neuronx-distributed-inference"


class NeuronPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "cpu"
    device_type: str = "cpu"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = ["neuron_quant", "fbgemm_fp8"]
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    # Guard to ensure config overrides are only applied once
    _config_overrides_applied = False

    def __init__(self):
        """Initialize NeuronPlatform and ensure config overrides are applied."""
        super().__init__()
        self._ensure_config_overrides_applied()

    @classmethod
    def _apply_world_size_override(cls, ParallelConfig) -> None:
        """
        Override world_size_across_dp property by replacing it with a method
        that checks for Neuron platform and returns the correct value.
        This approach works at the class level and is timing-independent.
        """
        # Store the original property getter
        original_property = getattr(ParallelConfig, 'world_size_across_dp',
                                    None)
        if original_property is None:
            logger.warning(
                "world_size_across_dp property not found on ParallelConfig")
            return

        # Get the original getter function
        original_getter = original_property.fget if hasattr(
            original_property, 'fget') else None
        if original_getter is None:
            logger.warning(
                "Could not get original getter for world_size_across_dp property"
            )
            return

        def neuron_world_size_across_dp(self) -> int:
            """Neuron-aware world_size_across_dp implementation."""
            from vllm.platforms import current_platform

            # Check if we're on Neuron platform
            if hasattr(current_platform, '__class__') and \
               current_platform.__class__.__name__ == 'NeuronPlatform':
                logger.info(
                    "Using Neuron override: world_size_across_dp returning data_parallel_size"
                )
                return self.data_parallel_size

            # Fall back to original implementation for other platforms
            return original_getter(self)

        # Replace the property with our Neuron-aware version
        ParallelConfig.world_size_across_dp = property(
            neuron_world_size_across_dp)

        logger.debug(
            "Successfully applied world_size_across_dp override via class-level property replacement"
        )

    @classmethod
    def _ensure_config_overrides_applied(cls) -> None:
        """
        Ensure Neuron config overrides are applied in every process.
        This method can be called from multiple places to guarantee
        overrides are available in both main and spawned processes.
        """
        if cls._config_overrides_applied:
            logger.debug("Neuron config overrides already applied, skipping")
            return

        try:
            from vllm.config import ModelConfig, ParallelConfig

            from vllm_neuron import platform_overrides

            logger.info("Applying Neuron config overrides")

            # Apply the overrides
            ModelConfig.verify_with_parallel_config = platform_overrides.skip_verify_with_parallel_config
            ModelConfig._verify_quantization = platform_overrides.skip_verify_quantization
            ModelConfig._verify_cuda_graph = platform_overrides.skip_verify_cuda_graph
            ModelConfig.get_and_verify_max_len = platform_overrides.changed_get_and_verify_max_len

            from vllm.entrypoints.openai.serving_engine import OpenAIServing
            from vllm.entrypoints.renderer import CompletionRenderer

            # for v1/chat/completions
            OpenAIServing._validate_input = platform_overrides.changed_validate_input
            # for v1/completions
            CompletionRenderer._create_tokens_prompt = platform_overrides.changed_create_tokens_prompt

            # Apply chat completion stream generator override for vLLM 0.11.0
            cls._apply_chat_completion_stream_override()
            # Apply the world_size_across_dp override using a different approach
            cls._apply_world_size_override(ParallelConfig)

            cls._config_overrides_applied = True
            logger.info("Neuron config overrides applied successfully")

        except ImportError as e:
            logger.warning(
                f"Could not import vLLM config module for overrides: {e}")
        except Exception as e:
            logger.error(f"Error applying Neuron config overrides: {e}")
            raise

    @classmethod
    def _get_vllm_version(cls) -> str:
        """Get the current vLLM version."""
        try:
            return importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("Could not determine vLLM version")
            return "unknown"

    @classmethod
    def _apply_chat_completion_stream_override(cls) -> None:
        """
        Apply chat completion stream generator override for vLLM 0.11.0.
        This fixes a bug in the harmony parser where delta_text accumulation
        was incorrect during speculation.
        """
        vllm_version = cls._get_vllm_version()

        if vllm_version != "0.11.0":
            logger.debug(
                f"Skipping chat completion stream override - vLLM version {vllm_version} "
                "does not match target version 0.11.0")
            return

        try:
            # Import the fixed streaming function from patches
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

            from vllm_neuron.patches.chat_completion_stream_v0_11_0 import \
                fixed_chat_completion_stream_generator

            # Apply the override by replacing the method
            OpenAIServingChat.chat_completion_stream_generator = fixed_chat_completion_stream_generator

            logger.info(
                f"Applied Neuron chat completion stream generator override for vLLM {vllm_version} "
                "(fixes harmony parser delta_text accumulation bug during speculation)"
            )

        except ImportError as e:
            logger.warning(
                f"Could not import required modules for chat completion stream override: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error applying chat completion stream override: {e}")

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def pre_register_and_update(cls,
                                parser: "FlexibleArgumentParser | None" = None
                                ) -> None:
        # Ensure config overrides are applied
        cls._ensure_config_overrides_applied()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Ensure config overrides are applied in every process
        cls._ensure_config_overrides_applied()

        # Initialize LMCache integration if available
        try:
            from vllm_neuron.lmcache_integration import integrate_lmcache_with_neuron
            integrate_lmcache_with_neuron(vllm_config)
        except Exception as e:
            logger.warning(f"LMCache integration failed: {e}")

        # As of 0.10.2 check_and_update_config is being called every time
        # a VllmConfig object is created, even a default one, to validate params.
        # Our checks are not compatible with an empty VllmConfig. Currently one
        # of the most obvious signs that a VllmConfig is empty is that the model_config
        # is None, however this isn't guaranteed to be true in future vLLM versions.
        # TODO figure out a better way to verify empty VllmConfigs instead of just
        # checking if the VllmConfig.model_config is None.
        model_config = vllm_config.model_config
        if model_config is None:
            return

        disable_scheduler_override = bool(
            int(os.getenv("DISABLE_NEURON_CUSTOM_SCHEDULER", "0")))

        model_config.max_prompt_length = vllm_config.additional_config.get(
            "max_prompt_length", None)

        # Add 1 to num_gpu_blocks_override to account for lazy null block allocation
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.num_gpu_blocks_override is not None \
            and '_neuron_null_block_adjusted' not in cache_config.__dict__:
            logger.info(
                "Adding 1 to num_gpu_blocks_override (%d -> %d) "
                "to account for null block allocation",
                cache_config.num_gpu_blocks_override,
                cache_config.num_gpu_blocks_override + 1)
            cache_config.num_gpu_blocks_override += 1
            cache_config._neuron_null_block_adjusted = True

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm_neuron.worker.neuron_worker.NeuronWorker"

        # Configure executor backend for Neuron devices
        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if disable_scheduler_override:
            logger.warning(
                "The vLLM V1 native scheduler will be used with chunked prefill enabled. "
                "This may lead to suboptimal performance on Neuron devices.")
            assert vllm_config.cache_config.block_size is not None, (
                "When vLLM V1 native scheduler is enabled, block_size must be set."
            )
        else:
            logger.info(
                "The custom Neuron scheduler will disable chunked prefill and schedule requests using "
                "the continuous batching mechanism, prioritizing prefill over decode."
            )
            vllm_config.scheduler_config.scheduler_cls = (
                "vllm_neuron.core.scheduler.ContinuousBatchingNeuronScheduler")
            vllm_config.scheduler_config.chunked_prefill_enabled = False

            sched_cfg = vllm_config.scheduler_config

            # Set default token budget for Neuron to 128k
            sched_cfg.max_num_batched_tokens = 131072
            logger.info(
                "Neuron custom scheduler default: max_num_batched_tokens set to %d. "
                "Override with --max-num-batched-tokens if needed.",
                sched_cfg.max_num_batched_tokens,
            )

            # Set default batch size for Neuron to 32
            if not sched_cfg.max_num_seqs:
                sched_cfg.max_num_seqs = 32
                logger.info(
                    "Neuron custom scheduler default: max_num_seqs set to %d.",
                    sched_cfg.max_num_seqs,
                )

            if not vllm_config.cache_config.enable_prefix_caching:
                # Neuron requires block_size = max_model_len when blockwise KV cache is disabled
                vllm_config.cache_config.block_size = (
                    vllm_config.model_config.max_model_len  # type: ignore
                )
            else:
                assert vllm_config.cache_config.block_size is not None, (
                    "When prefix caching is enabled, block_size must be set.")

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Neuron.")
        return False

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        return True

    @classmethod
    @lru_cache
    def is_neuronx_distributed_inference(cls) -> bool:
        try:
            import neuronx_distributed_inference
        except ImportError:
            neuronx_distributed_inference = None
        return neuronx_distributed_inference is not None

    def get_neuron_framework_to_use(self):
        """Return the specified framework if corresponding installations are
        available.

        If no framework is specified, use neuronx-distributed-inference by
        default.
        If that's unavailable, check and switch to transformers-neuronx.
        """
        if not self.is_neuron():
            raise AssertionError(
                f"Neuron Framework unavailable for platform: {self}")

        nxd_installed = self.is_neuronx_distributed_inference()
        if not nxd_installed:
            raise AssertionError(
                "Unable to import neuronx_distributed_inference. Please verify it is properly installed. "
            )

        return NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE

    def use_neuronx_distributed(self):
        """
        Return True if the framework determined in get_neuron_framework_to_use()
        is NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE, False otherwise. This
        is used to select the Neuron model framework and framework-specific
        configuration to apply during model compilation.
        """
        nxd_framework = NeuronFramework.NEURONX_DISTRIBUTED_INFERENCE
        return self.get_neuron_framework_to_use() == nxd_framework

    @classmethod
    def get_nixl_supported_devices(cls) -> dict:
        """Get mapping of device types to supported kv_buffer_devices for nixl.
        
        Returns:
            dict: Mapping of device types to sets of supported devices.
                  For Neuron platform, CPU is supported for CPU device type.
        """
        return {cls.device_type: {"cpu"}}

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        """Map logical device ID to physical Neuron core ID."""
        if cls.device_control_env_var in os.environ:
            # Parse NEURON_RT_VISIBLE_CORES range (e.g., "0-7")
            core_range = os.environ[cls.device_control_env_var]
            if "-" in core_range:
                start, end = map(int, core_range.split("-"))
                available_cores = list(range(start, end + 1))
            else:
                # Single core or comma-separated list
                available_cores = core_range.split(",")
            logger.debug("device id: %s, physical device id: %s", device_id,
                         available_cores[device_id])
            return available_cores[device_id]
        return device_id

    @classmethod
    def get_device_capability(cls) -> dict:
        """Get Neuron device capabilities and information."""
        try:
            import subprocess
            
            # Use neuron-ls to detect available devices
            result = subprocess.run(['neuron-ls'], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning(f"neuron-ls command failed: {result.stderr}")
                return None
            
            # Parse neuron-ls output to extract device information
            lines = result.stdout.strip().split('\n')
            devices = []
            
            for line in lines:
                # Look for device lines (contain | but not headers)
                if '|' in line and 'NEURON' not in line and '+-' not in line and line.strip():
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) >= 7 and parts[1].isdigit():  # Valid device line
                        device_info = {
                            'device_id': int(parts[1]),
                            'cores': int(parts[2]),
                            'core_ids': parts[3],
                            'memory': parts[4],
                            'pci_bdf': parts[5],
                            'cpu_affinity': parts[6],
                            'numa_node': parts[7] if len(parts) > 7 else None
                        }
                        devices.append(device_info)
            
            if devices:
                capability = {
                    'device_count': len(devices),
                    'devices': devices,
                    'total_cores': sum(device['cores'] for device in devices),
                    'platform': 'neuron'
                }
                logger.info(f"Detected {len(devices)} Neuron device(s) with {capability['total_cores']} total cores")
                return capability
            else:
                logger.warning("No Neuron devices found in neuron-ls output")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("neuron-ls command timed out")
            return None
        except FileNotFoundError:
            logger.error("neuron-ls command not found - ensure Neuron SDK is installed")
            return None
        except Exception as e:
            logger.error(f"Error detecting Neuron devices: {e}")
            return None