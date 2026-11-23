# SPDX-License-Identifier: Apache-2.0
"""Isaac-0.5 robot-action pipeline for vLLM-Omni on Neuron (public SDK 2.33 plugin stack)."""
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

ISAAC05_ACTION_PIPELINE = PipelineConfig(
    model_type="isaac05_action",
    model_arch="IsaacActionPipeline",
    stages=(
        StagePipelineConfig(stage_id=0, model_stage="diffusion", execution_type=StageExecutionType.DIFFUSION,
                            input_sources=(), final_output=True, final_output_type="action",
                            model_arch="IsaacActionPipeline"),
    ),
)


def register() -> None:
    """vllm_omni.general_plugins / vllm.general_plugins entry point (also call it from a launcher before serve)."""
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.diffusion.registry import register_diffusion_model

    register_pipeline(ISAAC05_ACTION_PIPELINE)
    register_diffusion_model(model_arch="IsaacActionPipeline", module_name="isaac05_omni.pipeline_isaac_action",
                             class_name="IsaacActionPipeline")
