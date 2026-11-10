# SPDX-License-Identifier: Apache-2.0
"""Ministral 3 multimodal processor (images), vendored from upstream vLLM.

Upstream ``vllm/model_executor/models/mistral3.py`` imports ``pixtral.py``,
which imports transformers symbols that the transformers release shipped with
vLLM 0.29 no longer has (``PixtralRotaryEmbedding``,
``position_ids_in_meshgrid``). Importing the upstream module therefore fails,
so the processing classes are copied here. They only need the HF
``PixtralProcessor`` and the vLLM multimodal base classes.

Prompt layout of one image (``nrows`` x ``ncols`` merged patches)::

    ([IMG] * ncols + [IMG_BREAK]) * nrows, with the last [IMG_BREAK] -> [IMG_END]

Only ``[IMG]`` positions receive vision embeddings; ``[IMG_BREAK]`` and
``[IMG_END]`` keep their text embeddings.
"""

import math
from collections.abc import Mapping, Sequence

from transformers import BatchFeature, Mistral3Config, PixtralVisionConfig
from transformers.models.pixtral import PixtralProcessor
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

IMAGE_BREAK_TOKEN = "[IMG_BREAK]"
IMAGE_END_TOKEN = "[IMG_END]"

_FAST_TOKENIZERS: dict = {}


def _fast_tokenizer(path: str):
    if path not in _FAST_TOKENIZERS:
        from transformers import PreTrainedTokenizerFast

        _FAST_TOKENIZERS[path] = PreTrainedTokenizerFast.from_pretrained(path)
    return _FAST_TOKENIZERS[path]


def patch_grid(
    image_width: int,
    image_height: int,
    *,
    longest_edge: int,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[int, int, int, int]:
    """Return (ncols, nrows, patch_cols, patch_rows) for one image.

    Matches the HF Pixtral image processor: resize so the longest edge is at
    most ``longest_edge`` (floor), round each side up to ``patch_size``, then
    the projector drops incomplete ``spatial_merge_size`` groups.
    """
    ratio = max(image_width / longest_edge, image_height / longest_edge)
    if ratio > 1:
        image_width = math.floor(image_width / ratio)
        image_height = math.floor(image_height / ratio)
    patch_cols = (image_width - 1) // patch_size + 1
    patch_rows = (image_height - 1) // patch_size + 1
    return (
        patch_cols // spatial_merge_size,
        patch_rows // spatial_merge_size,
        patch_cols,
        patch_rows,
    )


def _tokenizer_expands_images(info) -> bool:
    from vllm.tokenizers.mistral import MistralTokenizer

    return isinstance(info.ctx.tokenizer, MistralTokenizer)


def image_placeholder_ids(info, width, height, mm_kwargs=None) -> list[int]:
    """[IMG] * ncols + [IMG_BREAK] per row, last token [IMG_END]."""
    cfg = info.get_hf_config()
    tok = _fast_tokenizer(info.ctx.model_config.model)
    image_break_id = tok.convert_tokens_to_ids(IMAGE_BREAK_TOKEN)
    image_end_id = tok.convert_tokens_to_ids(IMAGE_END_TOKEN)
    ncols, nrows = info.get_grid(width, height, mm_kwargs)
    tokens = ([cfg.image_token_index] * ncols + [image_break_id]) * nrows
    tokens[-1] = image_end_id
    return tokens


class Mistral3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Mistral3Config:
        return self.ctx.get_hf_config(Mistral3Config)

    def get_hf_processor(self, **kwargs: object):
        # With --tokenizer-mode mistral, vLLM hands the HF processor the
        # mistral-common tokenizer backend, which encodes "[IMG]" as plain text.
        # PixtralProcessor's dummy-text pass then fails its image-token count
        # check. The image placeholders come from _get_prompt_updates (not
        # from this tokenizer), so give the processor the repository's fast
        # tokenizer (tokenizer.json), which knows the special tokens.
        from vllm.tokenizers.mistral import MistralTokenizer

        if isinstance(self.ctx.tokenizer, MistralTokenizer):
            tok = _fast_tokenizer(self.ctx.model_config.model)
            from vllm.transformers_utils.processor import cached_processor_from_config

            merged = self.ctx.get_merged_mm_kwargs(kwargs)
            merged.pop("tokenizer", None)
            return cached_processor_from_config(
                self.ctx.model_config,
                processor_cls=PixtralProcessor,
                tokenizer=tok,
                **merged,
            )
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_longest_edge(
        self, mm_processor_kwargs: Mapping[str, object] | None = None
    ) -> int:
        if _tokenizer_expands_images(self):
            return int(self.ctx.tokenizer.instruct.mm_encoder.mm_config.max_image_size)
        processor = self.get_hf_processor()
        size = processor.image_processor.size
        merged = self.ctx.get_merged_mm_kwargs(mm_processor_kwargs or {})
        if override := merged.get("size"):
            size = size | override
        return int(size["longest_edge"])

    def get_grid(
        self,
        image_width: int,
        image_height: int,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[int, int]:
        if _tokenizer_expands_images(self):
            # mistral-common (the chat template in tekken mode) rounds the
            # merged grid up and resizes the image to exactly that grid.
            from PIL import Image

            enc = self.ctx.tokenizer.instruct.mm_encoder
            return enc._image_to_num_tokens(Image.new("RGB", (image_width, image_height)))
        cfg = self.get_hf_config()
        ncols, nrows, _, _ = patch_grid(
            image_width,
            image_height,
            longest_edge=self.get_longest_edge(mm_processor_kwargs),
            patch_size=cfg.vision_config.patch_size,
            spatial_merge_size=cfg.spatial_merge_size,
        )
        return ncols, nrows

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        ncols, nrows = self.get_grid(image_width, image_height)
        return ncols * nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        edge = self.get_longest_edge()
        return ImageSize(width=edge, height=edge)


class Mistral3DummyInputsBuilder(BaseDummyInputsBuilder[Mistral3ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        processor = self.info.get_hf_processor()
        return processor.image_token * mm_counts.get("image", 0)

    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options, **kwargs):
        # The mistral-common tokenizer encodes the text "[IMG]" as plain text,
        # so build the dummy prompt from the image token id directly.
        from vllm.multimodal.processing import ProcessorInputs

        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        items = self.info.parse_mm_data(dummy_mm_data, validate=False)
        prompt = [1]
        if _tokenizer_expands_images(self.info):
            size = self.info.get_image_size_with_most_features()
            for _ in range(mm_counts.get("image", 0)):
                prompt += image_placeholder_ids(self.info, size.width, size.height)
        else:
            image_id = self.info.get_hf_config().image_token_index
            prompt += [image_id] * mm_counts.get("image", 0)
        return ProcessorInputs(prompt=prompt, mm_data_items=items)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class Mistral3MultiModalProcessor(BaseMultiModalProcessor[Mistral3ProcessingInfo]):
    def _get_hf_processor_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.dummy_inputs.get_dummy_text(mm_counts)

    def _apply_hf_processor_main(self, mm_items, hf_processor_mm_kwargs):
        if _tokenizer_expands_images(self.info):
            return self._apply_mistral_common(mm_items)
        return self._apply_hf(mm_items, hf_processor_mm_kwargs)

    def _apply_mistral_common(self, mm_items):
        """Pixels from the mistral-common image encoder, so the pixel grid
        equals the [IMG] grid the chat template wrote into the prompt."""
        import torch
        from mistral_common.protocol.instruct.messages import ImageChunk

        enc = self.info.ctx.tokenizer.instruct.mm_encoder
        counts = mm_items.get_all_counts()
        pixel_values = []
        if counts.get("image", 0):
            images = mm_items.get_items("image", ImageProcessorItems)
            for i in range(images.get_count()):
                img = images.get(i)
                pixel_values.append(torch.from_numpy(enc(ImageChunk(image=img)).image))
        return BatchFeature({"pixel_values": pixel_values})

    def _apply_hf(self, mm_items, hf_processor_mm_kwargs):
        # The HF Pixtral image processor runs torch ops. If they start torch's
        # intra-op thread pool in the frontend process, the engine core forked
        # after it deadlocks the first time it runs the same processor (the
        # scheduler's encoder-budget profiling). Keep the processor on one thread.
        import torch

        n = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return super()._apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)
        finally:
            torch.set_num_threads(n)

    def _postprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        if not mm_data:
            return processed_data
        pixel_values = processed_data.get("pixel_values")
        if pixel_values is not None:
            # Crop the batch padding so each image is independent (encoder cache).
            image_sizes = processed_data["image_sizes"]
            assert len(pixel_values) == len(image_sizes)
            processed_data["pixel_values"] = [
                p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
            ]
        return processed_data

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _maybe_apply_prompt_updates(
        self, mm_items, prompt_ids, mm_kwargs, mm_prompt_updates
    ):
        # With --tokenizer-mode mistral the chat template (mistral-common) has
        # already expanded every image into its [IMG] / [IMG_BREAK] / [IMG_END]
        # run; only locate the runs (as upstream Pixtral does).
        if not _tokenizer_expands_images(self.info):
            return super()._maybe_apply_prompt_updates(
                mm_items, prompt_ids, mm_kwargs, mm_prompt_updates
            )
        counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, counts)
        self._validate_mm_updates(mm_prompt_updates, counts)
        placeholders = self._find_mm_placeholders(prompt_ids, mm_prompt_updates)
        self._validate_mm_placeholders(placeholders, counts)
        return prompt_ids, placeholders

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index
        assert isinstance(hf_config.vision_config, PixtralVisionConfig)

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            tokens = image_placeholder_ids(
                self.info, size.width, size.height, hf_processor_mm_kwargs
            )
            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image", target=[image_token_id], replacement=get_replacement
            )
        ]
