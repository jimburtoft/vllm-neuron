# SPDX-License-Identifier: Apache-2.0
"""DFlash drafter for MuseGlimmer (``MuseGlimmerAssistantModel`` checkpoints).

The assistant is the Qwen3-shaped DFlash drafter of ``qwen3/dflash_model.py``
(5 layers, 32 Q / 8 KV heads, weighted RMSNorm + q/k norm, SwiGLU, non-causal
block of 16 = bonus + 15 masks, context K/V from ``hidden_norm(fc(aux))``).
Differences handled here:

* checkpoint names ``encoder.fc`` / ``encoder.output_norm_enc`` (z-lab: ``fc`` /
  ``hidden_norm``);
* RoPE theta comes from ``rope_parameters`` (5e5); the Qwen3 dataclass default
  is 1e6 and would be used silently;
* every layer is ``sliding_attention`` with window 2048. HF builds the drafter
  mask with ``create_bidirectional_sliding_window_mask``: key position
  ``k >= q - W`` (inclusive, ``|q - k| <= W``), which is one wider than the
  target's causal SWA (``k > q - W``). Passed as the per-query ``start_pos`` of
  the decode mask. The draft KV cache stays full length (no SWA trimming);
  the mask alone enforces the window;
* the embedding and LM head are borrowed from ``language_model.embed_tokens``
  (the raw table, before ``embed_norm``: HF bypasses it for the noise block)
  and ``lm_head``. Draft logits are the raw ``lm_head`` output, as in HF and in
  vLLM's ``qwen3_dflash``; the target's ``output_multiplier`` + tanh softcap is
  strictly monotone, so the draft argmax is the same either way.
"""

from __future__ import annotations

import os

import torch

from vllm_neuron.functional.attention.attention_decode_mask import (
    gen_attention_decode_mask,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3.config import DFlashConfig
from vllm_neuron.model.qwen3.dflash_model import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

ARCHITECTURE = "DFlashMuseGlimmerAssistantModel"
MUSE_GLIMMER_VOCAB = 202048


def _draft_config(hf_config, neuron_config: NeuronConfig | None) -> DFlashConfig:
    d = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    if getattr(hf_config, "torch_dtype", None) is not None:
        d["torch_dtype"] = hf_config.torch_dtype
    rp = d.get("rope_parameters") or {}
    if "rope_theta" not in d or d.get("rope_theta") is None:
        if "rope_theta" not in rp:
            raise ValueError("MuseGlimmer drafter: rope_parameters.rope_theta missing")
        d["rope_theta"] = float(rp["rope_theta"])
    if rp.get("rope_type", "default") != "default":
        raise ValueError(f"MuseGlimmer drafter: unsupported rope_type {rp.get('rope_type')}")
    # The checkpoint omits vocab_size; vLLM's MuseGlimmerAssistantConfig fills 202048.
    d.setdefault("vocab_size", MUSE_GLIMMER_VOCAB)
    cfg = DFlashConfig.from_configs(d, neuron_config)
    # Follow the target's CPU-mode compute-dtype knob (fp32 structural checks).
    if os.environ.get("MUSE_GLIMMER_COMPUTE_DTYPE", "bfloat16") == "float32":
        cfg.torch_dtype = torch.float32
    layer_types = d.get("layer_types") or []
    window = d.get("sliding_window")
    if layer_types and any(t == "sliding_attention" for t in layer_types):
        if not all(t == "sliding_attention" for t in layer_types):
            raise NotImplementedError("MuseGlimmer drafter: mixed sliding/full layers")
        if window is None:
            raise ValueError("MuseGlimmer drafter: sliding layers without sliding_window")
        cfg.sliding_window = int(window)
    else:
        cfg.sliding_window = None
    return cfg


class MuseGlimmerDFlashModel(DFlashQwen3Model):
    """Backbone with the drafter's inclusive bidirectional sliding window."""

    @torch.no_grad()
    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping):
        # Warmup feeds bf16 synthetic states; match the drafter dtype (fp32 CPU checks).
        context_states = context_states.to(self.config.torch_dtype)
        return super().precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata):
        W = getattr(self.config, "sliding_window", None)
        attn0 = self.layers[0].self_attn
        layer0_meta = attn_metadata[f"layers.{attn0.layer_idx}.self_attn"]
        if W is not None and layer0_meta.get("attention_mask", None) is None:
            bs = layer0_meta["block_table_tensor"].shape[0]
            block_size = layer0_meta["block_size"]
            s_prior = layer0_meta["max_blocks_per_seq"] * block_size
            s_active = positions.shape[0] // bs
            pos = positions.view(1, bs * s_active).to(torch.float32)
            mask = gen_attention_decode_mask(
                pos_ids=pos,
                bs=bs,
                q_head=attn0.num_attention_heads_per_rank,
                s_active=s_active,
                s_prior=s_prior,
                start_pos=torch.clamp(pos - float(W), min=0.0),
                block_len=block_size,
                is_causal=False,
            )
            attn_metadata = dict(attn_metadata)
            attn_metadata[f"layers.{attn0.layer_idx}.self_attn"] = dict(
                layer0_meta, attention_mask=mask
            )
        return super().forward(input_ids, positions, attn_metadata)


class MuseGlimmerDFlashForCausalLM(DFlashQwen3ForCausalLM):
    ckpt_fc_weight_name = "encoder.fc.weight"
    ckpt_hidden_norm_weight_name = "encoder.output_norm_enc.weight"

    model_cls = MuseGlimmerDFlashModel

    @classmethod
    def from_configs(
        cls,
        config,
        start_layer_idx: int = 0,
        neuron_config: NeuronConfig | None = None,
    ):
        return cls(_draft_config(config, neuron_config), start_layer_idx=start_layer_idx)

    def alias_target_head(self, target_model) -> None:
        target = getattr(target_model, "_orig_mod", target_model)
        if not hasattr(target, "model") and hasattr(target, "language_model"):

            class _View:
                pass

            view = _View()
            view.model = target.language_model  # .embed_tokens: raw table, no embed_norm
            view.lm_head = target.lm_head
            target = view
        super().alias_target_head(target)
