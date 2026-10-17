# SPDX-License-Identifier: Apache-2.0
"""Front-end (vLLM) multimodal processor registration for MuseGlimmer.

vLLM 0.24 has no native MuseGlimmer model. Registering the plugin factory
class with this processor in the front-end process (see
``NeuronPlatform.pre_register_and_update``) makes ``ModelConfig`` resolve the
architecture to the plugin class (``supports_multimodal=True``) instead of the
generic ``TransformersMultiModalForCausalLM``.

Image path (Task 005). The HF ``MuseGlimmerProcessor`` performs the prompt
expansion itself: the chat template renders one ``<|patch|>`` per image, and
the processor replaces it with
``<|image_start|> + <|patch|> * (t*h*w / 4) + <|image_end|>``. vLLM then
locates that exact token run (``_get_prompt_updates``) and marks only the
``<|patch|>`` tokens as embeddings. Token ids therefore match HF
``apply_chat_template(..., tokenize=True)`` exactly, for ``LLM.chat`` requests
and for raw prompts holding the HF-rendered chat text (with or without the
leading ``<|begin_of_text|>``: vLLM tokenizes with ``add_special_tokens=True``
and a doubled leading BOS is collapsed, see ``_apply_hf_processor_tokens_only``).

Video (timestamp interleave, ``<|vid_start|>`` / ``<|vid_frame_separator|>``
/ ``<|vid_end|>``, ``is_embed`` only on ``<|video|>``) is Task 007: add
``"video"`` to ``get_supported_mm_limits``, a ``video_replacement`` in
``_get_prompt_updates`` and the video dummy data. Until then video raises.

Budget / limits are computed from processor caps, without the HF processor,
so text-only runs (including ``skip_tokenizer_init=True``) work.
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
        # Video is Task 007 (omitting a modality = unsupported in vLLM).
        return {"image": None}

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
        return {"image": self.get_max_image_tokens()}


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
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-prompt path: a prompt that already starts with BOS (rendered chat text) is
        # tokenized without special tokens, as HF apply_chat_template does.
        bos = self.info.get_tokenizer().bos_token
        if bos and prompt.startswith(bos):
            tok_kwargs = {**tok_kwargs, "add_special_tokens": False}
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _apply_hf_processor_tokens_only(self, prompt_tokens: list[int]) -> list[int]:
        # vLLM 0.24 tokenizes multimodal prompts with add_special_tokens=True for BOTH the
        # chat and the completion path (``info.default_tok_params``). The MuseGlimmer chat
        # template already renders <|begin_of_text|>, and HF apply_chat_template tokenizes
        # without special tokens, so the chat path would start with a doubled BOS. Collapse
        # it so prompt ids equal HF exactly (a completion prompt without BOS is unaffected).
        bos = self.info.get_tokenizer().bos_token_id
        if bos is not None and len(prompt_tokens) >= 2 and prompt_tokens[0] == bos == prompt_tokens[1]:
            return list(prompt_tokens[1:])
        return prompt_tokens

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
            raise NotImplementedError(
                "MuseGlimmer video input is not implemented yet (Task 007)."
            )
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
