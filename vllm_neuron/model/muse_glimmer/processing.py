# SPDX-License-Identifier: Apache-2.0
"""Front-end (vLLM) multimodal processor registration for MuseGlimmer.

vLLM 0.24 has no native MuseGlimmer model. Registering the plugin factory
class with this processor in the front-end process (see
``NeuronPlatform.pre_register_and_update``) makes ``ModelConfig`` resolve the
architecture to the plugin class (``supports_multimodal=True``) instead of the
generic ``TransformersMultiModalForCausalLM``.

Status (Task 003): budget/limits are implemented without touching the HF
processor, so text-only runs (including ``skip_tokenizer_init=True``) work.
The image prompt replacement follows the HF processor
(``<|image_start|> <|patch|> * N <|image_end|>``, only ``<|patch|>`` is
embedded) but is exercised only once the vision tower lands (Task 005).
Video (timestamp interleave) is Task 007.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

IMAGE_TOKEN = "<|patch|>"
IMAGE_START_TOKEN = "<|image_start|>"
IMAGE_END_TOKEN = "<|image_end|>"
VIDEO_TOKEN = "<|video|>"

# HF image/video processor defaults (image_processing_muse_glimmer.py,
# video_processing_muse_glimmer.py).
DEFAULT_MAX_IMAGE_TOKENS = 4096
DEFAULT_MAX_VIDEO_FRAME_TOKENS = 144
DEFAULT_NUM_FRAMES = 96
DEFAULT_TEMPORAL_PATCH = 2


class MuseGlimmerProcessingInfo(BaseProcessingInfo):
    def _mm_kwargs(self) -> Mapping[str, Any]:
        mm_config = self.ctx.model_config.multimodal_config
        return (mm_config.mm_processor_kwargs or {}) if mm_config else {}

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_max_image_tokens(self) -> int:
        return int(self._mm_kwargs().get("max_image_tokens", DEFAULT_MAX_IMAGE_TOKENS))

    def get_max_video_tokens(self) -> int:
        kw = self._mm_kwargs()
        per_group = int(kw.get("max_video_frame_tokens", DEFAULT_MAX_VIDEO_FRAME_TOKENS))
        frames = int(kw.get("num_frames", DEFAULT_NUM_FRAMES))
        groups = -(-frames // DEFAULT_TEMPORAL_PATCH)
        return per_group * groups

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]):
        # Computed from processor caps, so no tokenizer / HF processor needed.
        return {"image": self.get_max_image_tokens(), "video": self.get_max_video_tokens()}


class MuseGlimmerDummyInputsBuilder(BaseDummyInputsBuilder[MuseGlimmerProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0) + VIDEO_TOKEN * mm_counts.get(
            "video", 0
        )

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        num_images = mm_counts.get("image", 0)
        side = 14 * 2 * 32  # 896 px -> 64x64 patches -> 1024 merged tokens
        return {
            "image": self._get_dummy_images(
                width=side,
                height=side,
                num_images=num_images,
                overrides=(mm_options or {}).get("image"),
            )
        }


class MuseGlimmerMultiModalProcessor(BaseMultiModalProcessor[MuseGlimmerProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_thw.prod(-1)
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_thw.prod(-1)
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        if "video" in mm_items:
            raise NotImplementedError("MuseGlimmer video input is Task 007.")
        vocab = self.info.get_tokenizer().get_vocab()
        patch_id = vocab[IMAGE_TOKEN]
        start_id = vocab[IMAGE_START_TOKEN]
        end_id = vocab[IMAGE_END_TOKEN]
        merge = self.info.get_hf_config().vision_config.merge_size ** 2

        def image_replacement(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            n = int(grid.prod()) // merge
            return PromptUpdateDetails.select_token_id(
                [start_id] + [patch_id] * n + [end_id], embed_token_id=patch_id
            )

        return [
            PromptReplacement(
                modality="image", target=[patch_id], replacement=image_replacement
            )
        ]
