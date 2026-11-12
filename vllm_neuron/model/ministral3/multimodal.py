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

import functools
import logging
import math
import os
import time

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

# Graph rows per block for a bucket. A bucket smaller than the encoder attention
# block compiles ONE block of ``bucket`` rows (block = image size, as in the
# muse-glimmer / Qwen3-VL-derived encoder), so a small image does not pay for
# a full block. Requires encoder DP 1 (TP over the encoder ranks).


# Merged rows per on-device encoder-cache block. One image takes ceil(rows / B) blocks
# instead of a whole attention block (pattern from muse-glimmer 5f2834d), so the cache
# costs ~ the image tokens it holds. VLLM_NEURON_ENC_CACHE_BLOCK=0 -> one attention block.
DEFAULT_ENC_CACHE_BLOCK = 256


def resolve_encoder_cache_block_size(attn_block: int, merge_factor: int) -> int:
    if attn_block % merge_factor:
        raise ValueError(f"encoder_attention_block_size ({attn_block}) must be divisible by {merge_factor}")
    full = attn_block // merge_factor
    v = os.environ.get("VLLM_NEURON_ENC_CACHE_BLOCK", "").strip()
    b = int(v) if v else DEFAULT_ENC_CACHE_BLOCK
    if not b or b >= full:
        return full
    if b % 8:
        raise ValueError(f"VLLM_NEURON_ENC_CACHE_BLOCK={b} must be a multiple of 8")
    return b


def graph_block_rows(bucket: int, enc) -> tuple[int, int]:
    """(rows per block, num_blocks) of the encoder graph for ``bucket``."""
    bs, dp = enc.encoder_attention_block_size, enc.dp_size
    if bucket < bs and dp == 1:
        if bucket % 128:
            raise ValueError(f"encoder bucket {bucket} must be a multiple of 128")
        return bucket, 1
    return bs, math.ceil(math.ceil(bucket / bs) / dp) * dp


def pack_slots() -> int:
    """VLLM_NEURON_PIXTRAL_PACK=k (k >= 2): a request with several images packs up to
    k images into ONE encoder block of k equal slots (slot = block / k patches), with
    the cross-image attention tiles skipped at trace time (attention_cte
    ``segment_cu_seqlens``). 0 / unset: one image per block."""
    v = os.environ.get("VLLM_NEURON_PIXTRAL_PACK", "").strip()
    return int(v) if v else 0


def slot_layout(rows_blk: int, k: int) -> tuple:
    """Canonical cumulative layout of k equal slots in a block (128-aligned)."""
    slot = rows_blk // k
    assert slot % 128 == 0 and slot * k == rows_blk, (rows_blk, k)
    return tuple(i * slot for i in range(k + 1))


@functools.lru_cache(maxsize=256)
def _image_tables(rows: int, cols: int, hd: int, theta: float, merge: int):
    """cos, sin [rows*cols, hd] and the merge index for one image grid (cached)."""
    c, s = pixtral_cos_sin(rows, cols, hd, theta)
    return c, s, merge_index(rows, cols, merge)


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
    def get_encoder_cache_block_size(cls, hf_config, encoder_attention_block_size) -> int:
        attn = getattr(encoder_attention_block_size, "encoder_attention_block_size", encoder_attention_block_size)
        return resolve_encoder_cache_block_size(attn, cls.get_vision_token_merge_factor(hf_config))

    @classmethod
    def get_max_encoder_blocks_per_request(cls, hf_config, encoder_neuron_config) -> int:
        """Cache blocks one request can use: per image, ceil(largest image rows / B);
        times the per-request image limit (vLLM --limit-mm-per-prompt)."""
        buckets = encoder_neuron_config.num_encoder_input_buckets
        if not buckets:
            return 0
        mu = cls.get_vision_token_merge_factor(hf_config)
        attn = encoder_neuron_config.encoder_attention_block_size
        b = cls.get_encoder_cache_block_size(hf_config, attn)
        per_image = math.ceil(min(max(buckets), attn) // mu / b)
        by_bucket = math.ceil(max(buckets) / attn) * math.ceil((attn // mu) / b)
        limit = 1
        try:
            from vllm.config import get_current_vllm_config

            mm = get_current_vllm_config().model_config.multimodal_config
            limit = max(1, int(mm.get_limit_per_prompt("image")))
        except Exception:  # noqa: BLE001 - no vLLM context
            pass
        return max(by_bucket, limit * per_image)

    # -- warmup -----------------------------------------------------------
    def get_encoder(self):
        return self.visual

    def build_encoder_synthetic_inputs(self, bucket, encoder_neuron_config, device):
        vc = self.hf_config.vision_config
        bs, nb = graph_block_rows(bucket, encoder_neuron_config)
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
            **self._cache_write_ids_input(bs // m, nb, encoder_neuron_config, device),
        }

    def _cache_write_ids_input(self, merged_rows, nb, enc, device):
        cb = self.get_encoder_cache_block_size(self.hf_config, enc.encoder_attention_block_size)
        k = math.ceil(merged_rows / cb)
        if k <= 1 and cb >= merged_rows:
            return {}
        return {"cache_write_ids": torch.zeros(nb * k, dtype=torch.int64, device=device)}

    def _run_visual(self, *args):
        return self.visual(*args)

    def _scoped_compiler_args(self, graph_type):
        """Scope the per-graph compiler args (compiler_args_for) around the encoder call,
        as the worker does for the encoder warmup."""
        import contextlib

        base = getattr(self, "_neuron_base_compiler_args", None)
        if base is None:
            return contextlib.nullcontext()
        try:
            from libtorch_neuronx_lite.compile.compiler_args import scoped_compiler_args
        except ImportError:  # CPU-only environments
            return contextlib.nullcontext()
        return scoped_compiler_args(self.compiler_args_for(graph_type, None, base))

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

        timing = os.environ.get("VLLM_NEURON_ENCODER_TIMING") == "1"
        t_host = time.perf_counter()
        per_image = []
        block_ids = []
        for mm_hash, img in zip(mm_hashes, pixel_values, strict=True):
            img = img.cpu()
            rows, cols = img.shape[1] // P, img.shape[2] // P
            n = rows * cols
            if n > bs:
                raise ValueError(
                    f"Image of {rows}x{cols} patches ({n}) exceeds encoder_attention_block_size={bs}. "
                    "Raise the block size (the largest image is 110x110 = 12100 patches at 1540 px)."
                )
            c, s_, mi = _image_tables(rows, cols, hd, float(theta), M)
            per_image.append((img, rows, cols, n, c, s_, mi))
            cb = encoder_cache.block_size
            block_ids.append(
                encoder_cache.allocate(mm_hash, encoder_cache.dense_tokens_per_block(mi.shape[0], cb))
            )

        buckets = sorted(enc.num_encoder_input_buckets)
        k_pack = pack_slots()
        if enc.dp_size == 1 and k_pack >= 2 and len(per_image) >= 2:
            # Pack up to k images per block: slot = smallest bucket / k that holds the
            # largest image, block = k slots (must be a compiled bucket).
            need = max(p[3] for p in per_image)
            bucket = next((b for b in buckets if b % k_pack == 0 and b // k_pack >= need
                           and (b // k_pack) % 128 == 0), None)
            if bucket is not None:
                rows_blk, nb = graph_block_rows(bucket, enc)
                if nb == 1:
                    for i in range(0, len(per_image), k_pack):
                        grp = list(range(i, min(i + k_pack, len(per_image))))
                        self._encode_packed(
                            [per_image[j] for j in grp], [block_ids[j] for j in grp],
                            bucket, rows_blk, k_pack, encoder_cache, timing, t_host,
                        )
                        t_host = time.perf_counter()
                    return
        if enc.dp_size == 1:
            # One image per block of the smallest bucket that holds the largest
            # image (block = image size); several images run that graph per image.
            need = max(p[3] for p in per_image)
            bucket = next((b for b in buckets if b >= need), None)
            if bucket is None:
                raise ValueError(f"image of {need} patches > largest bucket {buckets[-1]}")
            rows_blk, nb = graph_block_rows(bucket, enc)
            groups = [list(range(i, min(i + nb, len(per_image)))) for i in range(0, len(per_image), nb)]
        else:
            need = len(per_image) * bs
            bucket = next((b for b in buckets if b >= need), None)
            if bucket is None:
                raise ValueError(f"{len(per_image)} images need {need} tokens; largest bucket {buckets[-1]}")
            rows_blk, nb = graph_block_rows(bucket, enc)
            groups = [list(range(len(per_image)))]
        for group in groups:
            self._encode_group(
                [per_image[i] for i in group],
                [block_ids[i] for i in group],
                bucket, rows_blk, nb, encoder_cache, timing, t_host,
            )
            t_host = time.perf_counter()

    def _encode_packed(self, per_image, block_ids, bucket, rows_blk, k, encoder_cache, timing, t_host):
        """Several images in ONE graph block, image i at slot i (offset i * rows_blk / k).

        Attention: per-image [start, end) bounds, plus the slot layout as a trace-time
        segment_cu_seqlens so cross-slot tiles are skipped. Projector: merge index rows of
        image i go to its own run of cache-block-aligned output rows, so each image's merged
        rows fill whole cache blocks of its own (the prefill gather stays block-aligned).
        """
        vc = self.hf_config.vision_config
        P = vc.patch_size
        mf = self.hf_config.spatial_merge_size ** 2
        hd = vc.hidden_size // vc.num_attention_heads
        dtype = self.visual.dtype
        slot = rows_blk // k
        cb = encoder_cache.block_size
        k_cache_per_slot = math.ceil((slot // mf) / cb)
        out_rows = k * k_cache_per_slot * cb  # merged output rows of the packed graph
        patches = torch.zeros(1, rows_blk, vc.num_channels * P * P, dtype=dtype)
        cos = torch.zeros(1, rows_blk, hd)
        sin = torch.zeros(1, rows_blk, hd)
        bmin = torch.zeros(1, rows_blk, 1, dtype=torch.int32)
        bmax = torch.zeros(1, rows_blk, 1, dtype=torch.int32)
        midx = torch.zeros(1, out_rows, mf, dtype=torch.int64)
        ids = []
        for i in range(k):
            off = i * slot
            if i < len(per_image):
                img, rows, cols, n, c, s_, mi = per_image[i]
                patches[0, off : off + n] = patchify(img.to(dtype), P, rows, cols)
                cos[0, off : off + n], sin[0, off : off + n] = c, s_
                bmin[0, off : off + slot] = off
                bmax[0, off : off + slot] = off + n
                o = i * k_cache_per_slot * cb
                midx[0, o : o + mi.shape[0]] = mi + off
                mine = list(block_ids[i])
            else:
                bmin[0, off : off + slot] = off
                bmax[0, off : off + slot] = off + 1
                mine = []
            ids.extend(mine + [encoder_cache.scratch_block_id] * (k_cache_per_slot - len(mine)))
        cache_write_ids = torch.tensor(ids, dtype=torch.int64)
        device = next(self.visual.parameters()).device
        host_ms = (time.perf_counter() - t_host) * 1000
        t0 = time.perf_counter()
        # VLLM_NEURON_PIXTRAL_PACK_NOPRUNE=1: packed bounds without tile pruning (A/B control).
        self.visual.segment_cu_seqlens = (
            None if os.environ.get("VLLM_NEURON_PIXTRAL_PACK_NOPRUNE") == "1" else slot_layout(rows_blk, k)
        )
        try:
            with self._scoped_compiler_args("encoder"):
                self._run_visual(
                    patches.to(device),
                    cos.to(device),
                    sin.to(device),
                    bmin.to(device),
                    bmax.to(device),
                    midx.to(device),
                    encoder_cache.buffer,
                    cache_write_ids[:1].to(device),
                    cache_write_ids.to(device),
                )
        finally:
            self.visual.segment_cu_seqlens = None
        if timing:
            float(encoder_cache.buffer[int(cache_write_ids[0]), 0, 0].to("cpu"))
            logger.warning(
                "[ministral3 encoder] %d images packed, bucket %d (%d slots x %d rows): host %.1f ms, device %.1f ms",
                len(per_image), bucket, k, slot, host_ms, (time.perf_counter() - t0) * 1000,
            )

    def _encode_group(self, per_image, block_ids, bucket, rows_blk, nb, encoder_cache, timing, t_host):
        vc = self.hf_config.vision_config
        P = vc.patch_size
        M = self.hf_config.spatial_merge_size
        mf = M * M
        hd = vc.hidden_size // vc.num_attention_heads

        dtype = self.visual.dtype
        patches = torch.zeros(nb, rows_blk, vc.num_channels * P * P, dtype=dtype)
        cos = torch.zeros(nb, rows_blk, hd)
        sin = torch.zeros(nb, rows_blk, hd)
        bmin = torch.zeros(nb, rows_blk, 1, dtype=torch.int32)
        bmax = torch.zeros(nb, rows_blk, 1, dtype=torch.int32)
        midx = torch.zeros(nb, rows_blk // mf, mf, dtype=torch.int64)
        for i, (img, rows, cols, n, c, s_, mi) in enumerate(per_image):
            patches[i, :n] = patchify(img.to(dtype), P, rows, cols)
            cos[i, :n], sin[i, :n] = c, s_
            bmax[i, :n] = n
            midx[i, : mi.shape[0]] = mi
        cb = encoder_cache.block_size
        k = math.ceil((rows_blk // mf) / cb)
        ids = []
        for j in range(nb):
            mine = block_ids[j] if j < len(block_ids) else []
            ids.extend(list(mine) + [encoder_cache.scratch_block_id] * (k - len(mine)))
        cache_write_ids = torch.tensor(ids, dtype=torch.int64)
        write_ids = cache_write_ids[::k].contiguous()  # one id per graph block (unused when split)
        split = bool(self._cache_write_ids_input(rows_blk // mf, nb, self.encoder_neuron_config, "cpu"))
        device = next(self.visual.parameters()).device
        host_ms = (time.perf_counter() - t_host) * 1000
        t0 = time.perf_counter()
        with self._scoped_compiler_args("encoder"):
            self._run_visual(
                patches.to(device),
                cos.to(device),
                sin.to(device),
                bmin.to(device),
                bmax.to(device),
                midx.to(device),
                encoder_cache.buffer,
                write_ids.to(device),
                cache_write_ids.to(device) if split else None,
            )
        if timing:
            # Read back one element so the device work is included.
            float(encoder_cache.buffer[int(write_ids[0]), 0, 0].to("cpu"))
            logger.warning(
                "[ministral3 encoder] %d images, bucket %d (%d x %d rows): host %.1f ms, device %.1f ms",
                len(per_image),
                bucket,
                nb,
                rows_blk,
                host_ms,
                (time.perf_counter() - t0) * 1000,
            )
