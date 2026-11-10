# SPDX-License-Identifier: Apache-2.0
"""Ministral 3 with images: Pixtral tower + projector on Neuron, Llama 3 text decoder.

``embed_multimodal`` patchifies each image on the CPU, puts one image per
encoder block (``encoder_attention_block_size`` patches), runs the tower on the
encoder TP / DP groups, and writes the merged rows into the on-device encoder
cache. One cache block holds one image (``block_size / 4`` merged rows). The
decoder merges the cache rows into the ``[IMG]`` positions during prefill
(``[IMG_BREAK]`` / ``[IMG_END]`` keep their text embeddings).
"""

from __future__ import annotations

import logging
import math

import torch
from transformers import PretrainedConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_neuron.model.interfaces import SupportsEncoderBudget, SupportsSpatialMerge
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig

from .config import ministral3_llama_config
from .factory import Ministral3TextModel
from .processing import (
    Mistral3DummyInputsBuilder,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)
from .vision import (
    PixtralVisionTower,
    merge_index,
    patchify,
    pixtral_cos_sin,
)

logger = logging.getLogger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Mistral3MultiModalProcessor,
    info=Mistral3ProcessingInfo,
    dummy_inputs=Mistral3DummyInputsBuilder,
)
class Ministral3ForConditionalGeneration(
    Ministral3TextModel, SupportsSpatialMerge, SupportsEncoderBudget
):
    """Text decoder (llama3) plus the Neuron Pixtral tower as ``visual``."""

    own_weight_prefixes = ("visual.",)
    supports_multimodal = True

    def __init__(self, text_config, hf_config, encoder_neuron_config):
        super().__init__(text_config)
        self.hf_config = hf_config
        self.encoder_neuron_config = encoder_neuron_config
        build = encoder_neuron_config is not None and not getattr(
            encoder_neuron_config, "mm_language_model_only", False
        )
        self.visual = PixtralVisionTower(hf_config, dtype=text_config.torch_dtype) if build else None

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
    ):
        return cls(
            ministral3_llama_config(hf_config, text_neuron_config),
            hf_config,
            encoder_neuron_config,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    # -- encoder sizing ---------------------------------------------------
    @classmethod
    def get_vision_token_merge_factor(cls, hf_config: PretrainedConfig) -> int:
        return hf_config.spatial_merge_size**2

    @classmethod
    def get_max_encoder_embed_budget(cls, hf_config, encoder_neuron_config) -> int | None:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return None
        return max(buckets) // cls.get_vision_token_merge_factor(hf_config)

    @classmethod
    def get_max_encoder_blocks_per_request(cls, hf_config, encoder_neuron_config) -> int:
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        return math.ceil(max(buckets) / encoder_neuron_config.encoder_attention_block_size)

    # -- warmup -----------------------------------------------------------
    def get_encoder(self):
        return self.visual

    def build_encoder_synthetic_inputs(self, bucket, encoder_neuron_config, device):
        vc = self.hf_config.vision_config
        bs = encoder_neuron_config.encoder_attention_block_size
        dp = encoder_neuron_config.dp_size
        nb = math.ceil(math.ceil(bucket / bs) / dp) * dp
        hd = vc.hidden_size // vc.num_attention_heads
        patch_dim = vc.num_channels * vc.patch_size * vc.patch_size
        m = self.get_vision_token_merge_factor(self.hf_config)
        return {
            "pixel_patches": torch.zeros(nb, bs, patch_dim, dtype=self.visual.dtype, device=device),
            "cos": torch.zeros(nb, bs, hd, dtype=torch.float32, device=device),
            "sin": torch.zeros(nb, bs, hd, dtype=torch.float32, device=device),
            "bound_min": torch.zeros(nb, bs, 1, dtype=torch.int32, device=device),
            "bound_max": torch.zeros(nb, bs, 1, dtype=torch.int32, device=device),
            "merge_idx": torch.zeros(nb, bs // m, m, dtype=torch.int64, device=device),
        }

    # -- weights ----------------------------------------------------------
    def load_weights(self, checkpoint_path, device, cache_dir=None):
        super().load_weights(checkpoint_path, device, cache_dir)
        if self.visual is not None:
            self.visual.load_weights(checkpoint_path, torch.device("cpu"))

    # -- encode -----------------------------------------------------------
    def embed_multimodal(
        self,
        pixel_values=None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        **kwargs,
    ) -> None:
        if pixel_values is None:
            raise ValueError("embed_multimodal needs pixel_values (Mistral3 processor)")
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = list(pixel_values.unbind(0)) if pixel_values.dim() == 4 else [pixel_values]
        vc = self.hf_config.vision_config
        enc = self.encoder_neuron_config
        P = vc.patch_size
        M = self.hf_config.spatial_merge_size
        mf = M * M
        bs = enc.encoder_attention_block_size
        hd = vc.hidden_size // vc.num_attention_heads
        theta = vc.rope_parameters["rope_theta"] if getattr(vc, "rope_parameters", None) else vc.rope_theta

        per_image = []
        block_ids = []
        for mm_hash, img in zip(mm_hashes, pixel_values, strict=True):
            img = img.float().cpu()
            rows, cols = img.shape[1] // P, img.shape[2] // P
            n = rows * cols
            if n > bs:
                raise ValueError(
                    f"Image of {rows}x{cols} patches ({n}) exceeds encoder_attention_block_size={bs}. "
                    "Raise the block size or lower mm_processor_kwargs size.longest_edge."
                )
            mi = merge_index(rows, cols, M)
            per_image.append((img, rows, cols, n, mi))
            block_ids.extend(encoder_cache.allocate(mm_hash, [mi.shape[0]]))

        buckets = sorted(enc.num_encoder_input_buckets)
        need = len(per_image) * bs
        bucket = next((b for b in buckets if b >= need), None)
        if bucket is None:
            raise ValueError(f"{len(per_image)} images need {need} tokens; largest bucket {buckets[-1]}")
        dp = enc.dp_size
        nb = math.ceil(math.ceil(bucket / bs) / dp) * dp

        patches = torch.zeros(nb, bs, vc.num_channels * P * P)
        cos = torch.zeros(nb, bs, hd)
        sin = torch.zeros(nb, bs, hd)
        bmin = torch.zeros(nb, bs, 1, dtype=torch.int32)
        bmax = torch.zeros(nb, bs, 1, dtype=torch.int32)
        midx = torch.zeros(nb, bs // mf, mf, dtype=torch.int64)
        for i, (img, rows, cols, n, mi) in enumerate(per_image):
            patches[i, :n] = patchify(img, P, rows, cols)
            c, s = pixtral_cos_sin(rows, cols, hd, theta)
            cos[i, :n], sin[i, :n] = c, s
            bmax[i, :n] = n
            midx[i, : mi.shape[0]] = mi
        write_ids = torch.tensor(
            block_ids + [encoder_cache.scratch_block_id] * (nb - len(block_ids)),
            dtype=torch.int64,
        )
        device = next(self.visual.parameters()).device
        import os
        import time

        timing = os.environ.get("VLLM_NEURON_ENCODER_TIMING") == "1"
        t0 = time.perf_counter()
        self.visual(
            patches.to(self.visual.dtype).to(device),
            cos.to(device),
            sin.to(device),
            bmin.to(device),
            bmax.to(device),
            midx.to(device),
            encoder_cache.buffer,
            write_ids.to(device),
        )
        if timing:
            # Read back one element so the device work is included.
            float(encoder_cache.buffer[write_ids[0], 0, 0].to("cpu"))
            logger.warning(
                "[ministral3 encoder] %d images, %d blocks: %.1f ms",
                len(per_image),
                nb,
                (time.perf_counter() - t0) * 1000,
            )
