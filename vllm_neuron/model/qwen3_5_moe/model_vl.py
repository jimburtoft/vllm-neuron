# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE hybrid decoder (GDN + gated GQA + MoE) plus the Qwen3-VL-style vision tower.

Used for checkpoints whose config has a vision_config and a multimodal architecture
(Qwen3_5MoeForConditionalGeneration, Isaac-0.5). Text-only serving keeps using
model.Qwen3_5MoeForCausalLM directly.

What is added on top of the text model:
- ``visual``: vllm-neuron's Qwen3VLVisionModel (same ViT: patch16 / temporal 2 / merge 2, 2D vision rotary,
  learned pos-embed interpolation, merger), with deepstack_visual_indexes = [] and out_hidden = text hidden.
- ``embed_multimodal`` / ``get_encoder`` / ``build_encoder_synthetic_inputs`` / ``get_mrope_input_positions``
  reuse Qwen3VLGenerationMixin's implementations: they only read ``self.config.vision_config`` (+ neuron_config),
  the four vision token ids and ``self.visual``.
- 3-D interleaved mRoPE: ``rotary_position_ids`` [3, T] drive cos/sin only; 1-D ``positions`` stay the KV / GDN
  / mask positions. mrope_section [11, 11, 10] over the 32 rotary frequency pairs (rotary_dim 64).
- Prefill embedding merge of the on-device encoder cache (merge_multimodal_embeddings).
"""

import os
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.functional.merge_multimodal_embeddings import merge_multimodal_embeddings
from vllm_neuron.model.interfaces import SupportsEncoderWarmup, SupportsMRoPE
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.model.qwen3_vl.config import Qwen3VLVisionConfig
from vllm_neuron.model.qwen3_vl.model_common import Qwen3VLGenerationMixin
from vllm_neuron.model.qwen3_vl.vision_encoder_bf16 import Qwen3VLVisionModel

from .config import Qwen3_5MoeTextConfig
from .model import Qwen3_5MoeForCausalLM, PartialRotary


@dataclass
class Qwen3_5MoeVLConfig:
    """The attribute surface Qwen3VLGenerationMixin's vision helpers read."""

    text_config: Qwen3_5MoeTextConfig
    vision_config: Qwen3VLVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int


class MRotary(PartialRotary):
    """PartialRotary with 3-D interleaved mRoPE. positions: [T] (text, all axes equal) or [3, T]."""

    def __init__(self, config: Qwen3_5MoeTextConfig, mrope_section):
        super().__init__(config)
        self.mrope_section = list(mrope_section)
        assert sum(self.mrope_section) == self.rotary_dim // 2, (self.mrope_section, self.rotary_dim)

    def _axis_onehot(self, n, device):
        a = torch.zeros(n, dtype=torch.long)  # axis per frequency slot (Isaac05Qwen36TextRotaryEmbedding layout)
        for axis, off in enumerate((1, 2), start=1):
            a[off: self.mrope_section[axis] * 3: 3] = axis
        return torch.nn.functional.one_hot(a, 3).t().to(torch.float32).to(device)

    def forward(self, positions: torch.Tensor):
        if positions.ndim == 1:
            return super().forward(positions)
        d = self.rotary_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=positions.device) / d))
        if self.inv_freq_bf16:
            inv_freq = inv_freq.to(torch.bfloat16).float()
        # Interleaved mRoPE as elementwise arithmetic. No torch.where / select: neuronx-cc 2.28 fails to legalize the
        # tensor-select of compute_interleaved_mrope here (NCC_ILSA902). No matmul either: an fp32 matmul may run
        # at reduced precision on the PE and positions > 256 are not exact in bf16. Slot j takes axis a(j) (T/H/W):
        # pos_sel[t, j] = sum_a onehot[a, j] * pos[a, t] (0/1 masks: exact), freqs = pos_sel * inv_freq.
        if os.environ.get("QWEN35_MROPE_IMPL") == "where":  # reproducer path (NCC_ILSA902 on neuronx-cc 2.28)
            from vllm_neuron.model.qwen3_vl.model_bf16 import Qwen3VLTextRotaryEmbedding as R

            f3 = positions.float()[:, :, None] * inv_freq[None, None, :]
            freqs = R.compute_interleaved_mrope(f3, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        oh = self._axis_onehot(inv_freq.shape[0], positions.device)  # [3, d/2] fp32 0/1
        p = positions.float()
        pos_sel = p[0][:, None] * oh[0][None, :] + p[1][:, None] * oh[1][None, :] + p[2][:, None] * oh[2][None, :]
        freqs = pos_sel * inv_freq[None, :]  # [T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Qwen3_5MoeVLForConditionalGeneration(Qwen3_5MoeForCausalLM, SupportsEncoderWarmup, SupportsMRoPE):
    # vision helpers from the Qwen3-VL mixin (bound as plain functions; they only touch self.config.vision_config,
    # the token ids, self.visual and self._scoped_compiler_args)
    get_mrope_input_positions = Qwen3VLGenerationMixin.get_mrope_input_positions
    embed_multimodal = Qwen3VLGenerationMixin.embed_multimodal
    get_encoder = Qwen3VLGenerationMixin.get_encoder
    build_encoder_synthetic_inputs = Qwen3VLGenerationMixin.build_encoder_synthetic_inputs
    _scoped_compiler_args = Qwen3VLGenerationMixin._scoped_compiler_args

    def __init__(self, text_config: Qwen3_5MoeTextConfig, vl_config: Qwen3_5MoeVLConfig, mrope_section):
        # The mixin's vision helpers read self.config.vision_config and the four vision token ids; the text model
        # reads self.config as the text config. Attach the VL fields to the (plain dataclass) text config.
        for k in ("vision_config", "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            setattr(text_config, k, getattr(vl_config, k))
        super().__init__(text_config)
        self.vl_config = vl_config
        self.rotary = MRotary(text_config, mrope_section)
        enc_nc = vl_config.vision_config.neuron_config
        self.mm_encoder_only = bool(enc_nc and enc_nc.mm_encoder_only)
        self.mm_language_model_only = bool(enc_nc and enc_nc.mm_language_model_only)
        assert not self.mm_encoder_only, "encoder-only (EPD) pool not supported for the hybrid model"
        self.visual = None if self.mm_language_model_only else Qwen3VLVisionModel(vl_config.vision_config,
                                                                                  dtype=torch.bfloat16)

    def compiler_args_for(self, graph_type, example_inputs, base):
        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        if graph_type == "encoder":
            return with_optimization_level(base, 3)
        return base

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, text_neuron_config: NeuronConfig | None = None,
                     encoder_neuron_config: EncoderNeuronConfig | None = None, neuron_config=None):
        text_neuron_config = text_neuron_config or neuron_config
        tcfg = Qwen3_5MoeTextConfig.from_configs(hf_config, text_neuron_config)
        top = hf_config.to_dict() if isinstance(hf_config, PretrainedConfig) else dict(hf_config)
        hv = hf_config.vision_config if isinstance(hf_config, PretrainedConfig) else top["vision_config"]
        vcfg = Qwen3VLVisionConfig.from_hf_config(hv, encoder_neuron_config)
        assert vcfg.deepstack_visual_indexes == [], vcfg.deepstack_visual_indexes
        assert vcfg.out_hidden_size == tcfg.hidden_size, (vcfg.out_hidden_size, tcfg.hidden_size)
        text = top.get("text_config") or {}
        if hasattr(text, "to_dict"):
            text = text.to_dict()
        rp = text.get("rope_parameters") or text.get("rope_scaling") or {}
        assert rp.get("mrope_interleaved", True), "only interleaved mRoPE is implemented"
        vl = Qwen3_5MoeVLConfig(
            text_config=tcfg, vision_config=vcfg,
            image_token_id=top.get("image_token_id"), video_token_id=top.get("video_token_id"),
            vision_start_token_id=top.get("vision_start_token_id"), vision_end_token_id=top.get("vision_end_token_id"),
        )
        from vllm.config import get_current_vllm_config

        if get_current_vllm_config().cache_config.enable_prefix_caching:
            raise ValueError("Qwen3.5-MoE on Neuron keeps GDN state per request slot: pass --no-enable-prefix-caching.")
        return cls(tcfg, vl, rp.get("mrope_section", [11, 11, 10]))

    def _embed(self, input_ids, scatter, rank, is_decode, enc):
        h = self.embed_tokens(input_ids, scatter_tokens=scatter, rank=rank)
        buf, ids, pos = enc
        if (not is_decode) and buf is not None and ids is not None and pos is not None:
            # SP prefill: h holds this rank's T/TP rows; merge remaps global token positions to the local slice
            r = self.tp_group.rank_in_group_tensor if scatter else torch.zeros(1, dtype=torch.int32, device=h.device)
            h, _ = merge_multimodal_embeddings(h, buf, ids, pos, rank=r)
        return h

    def forward(self, input_ids, positions, *args, rotary_position_ids=None, encoder_embedding_buffer=None,
                encoder_block_ids=None, encoder_positions=None, **kwargs):
        self._mm_enc = (encoder_embedding_buffer, encoder_block_ids, encoder_positions)
        self._rot_pos = rotary_position_ids
        try:
            return super().forward(input_ids, positions, *args, **kwargs)
        finally:
            self._mm_enc = (None, None, None)
            self._rot_pos = None

    def load_weights(self, checkpoint_path, device, cache_dir):
        super().load_weights(checkpoint_path, device, cache_dir)
        if self.visual is not None:
            self.visual.load_weights(checkpoint_path, device="cpu", cpu_mode=True)
