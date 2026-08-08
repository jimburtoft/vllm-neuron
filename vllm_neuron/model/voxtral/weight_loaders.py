# SPDX-License-Identifier: Apache-2.0
"""Weight loaders / HF-key mappings for the Voxtral multimodal model.

Voxtral has three weight namespaces (in the HF checkpoint from
`mistralai/Voxtral-Mini-3B-2507`):

  - ``language_model.model.*``       -- Ministral-3B decoder (Llama-family)
  - ``language_model.lm_head.weight`` -- Separate LM head (Voxtral is untied)
  - ``multi_modal_projector.linear_{1,2}.weight`` -- AudioLanguageAdapter
  - ``audio_tower.*``                -- Whisper-derived audio encoder

Our plugin-side parameter names:

  - ``language_model.model.layers.N.self_attn.{qkv_proj_weight,o_proj_weight}``
  - ``language_model.model.layers.N.mlp.{gate,up,down}_proj_weight``
  - ``language_model.model.layers.N.{input_layernorm,post_attention_layernorm}.weight``
  - ``language_model.model.embed_tokens.weight``, ``.norm.weight``
  - ``language_model.lm_head.weight`` (untied for Voxtral)
  - ``audio_language_adapter.{w_in,w_out}.weight``
  - ``whisper_encoder.conv{1,2}.{weight,bias}``, ``.embed_positions.weight``,
    ``.layer_norm.{weight,bias}``, ``.layers.N.self_attn_layer_norm.{weight,bias}``,
    ``.layers.N.self_attn.{q_proj,v_proj}.{weight,bias}``,
    ``.layers.N.self_attn.k_proj.weight`` (no bias -- checkpoint matches),
    ``.layers.N.self_attn.out_proj.rpl.weight``,
    ``.layers.N.self_attn.out_proj.bias`` (FP32 bias wrapper),
    ``.layers.N.mlp.fc1.{weight,bias}``,
    ``.layers.N.mlp.fc2.rpl.weight``, ``.layers.N.mlp.fc2.bias``,
    ``.layers.N.final_layer_norm.{weight,bias}``
"""

from __future__ import annotations


HF_TEXT_MODEL_PREFIX = "language_model.model"   # embed_tokens, norm, layers.N.*
HF_LM_HEAD_KEY = "language_model.lm_head.weight"
HF_ADAPTER_PREFIX = "multi_modal_projector"      # linear_1, linear_2
HF_AUDIO_PREFIX = "audio_tower"                  # Whisper-derived encoder


def build_hf_mappings(
    *,
    num_hidden_layers: int,
    num_audio_layers: int,
    tie_word_embeddings: bool,
) -> dict[str, str | list[str]]:
    """Construct the parameter-name -> HF-key mapping dict.

    Returned in the plugin's ``SafetensorsCheckpoint.load_sharded_pipelined``
    convention: ``{"our.param.name": "hf.key.name"}`` for 1:1 mappings, or
    ``{"our.param.name": [hf_k, hf_v, hf_v]}`` for fused (QKV) params.

    Args:
        num_hidden_layers: LLM decoder layer count (Ministral-3B = 30).
        num_audio_layers:  Audio encoder layer count (Voxtral audio = 32).
        tie_word_embeddings: Whether the LM head shares weight with
            ``embed_tokens``. For Voxtral this is False.
    """
    m: dict[str, str | list[str]] = {}

    # === Ministral-3B LLM decoder ===
    for i in range(num_hidden_layers):
        hf_prefix = f"{HF_TEXT_MODEL_PREFIX}.layers.{i}"
        our_prefix = f"language_model.model.layers.{i}"

        # Fused QKV: plugin holds `qkv_proj_weight`; HF has separate q/k/v_proj.
        m[f"{our_prefix}.self_attn.qkv_proj_weight"] = [
            f"{hf_prefix}.self_attn.q_proj.weight",
            f"{hf_prefix}.self_attn.k_proj.weight",
            f"{hf_prefix}.self_attn.v_proj.weight",
        ]
        m[f"{our_prefix}.self_attn.o_proj_weight"] = (
            f"{hf_prefix}.self_attn.o_proj.weight"
        )

        # LayerNorms.
        m[f"{our_prefix}.input_layernorm.weight"] = (
            f"{hf_prefix}.input_layernorm.weight"
        )
        m[f"{our_prefix}.post_attention_layernorm.weight"] = (
            f"{hf_prefix}.post_attention_layernorm.weight"
        )

        # MLP (SwiGLU: gate + up + down).
        m[f"{our_prefix}.mlp.gate_proj_weight"] = f"{hf_prefix}.mlp.gate_proj.weight"
        m[f"{our_prefix}.mlp.up_proj_weight"] = f"{hf_prefix}.mlp.up_proj.weight"
        m[f"{our_prefix}.mlp.down_proj_weight"] = f"{hf_prefix}.mlp.down_proj.weight"

    # LM top: embed_tokens, final norm, LM head.
    m["language_model.model.embed_tokens.weight"] = (
        f"{HF_TEXT_MODEL_PREFIX}.embed_tokens.weight"
    )
    m["language_model.model.norm.weight"] = f"{HF_TEXT_MODEL_PREFIX}.norm.weight"

    if tie_word_embeddings:
        # Untied for Voxtral by default, but keep the branch for symmetry.
        m["language_model.lm_head.weight"] = (
            f"{HF_TEXT_MODEL_PREFIX}.embed_tokens.weight"
        )
    else:
        m["language_model.lm_head.weight"] = HF_LM_HEAD_KEY

    # === AudioLanguageAdapter (multi_modal_projector) ===
    m["audio_language_adapter.w_in.weight"] = f"{HF_ADAPTER_PREFIX}.linear_1.weight"
    m["audio_language_adapter.w_out.weight"] = f"{HF_ADAPTER_PREFIX}.linear_2.weight"

    # === Voxtral audio encoder (Whisper-derived) ===
    for i in range(num_audio_layers):
        hf_prefix = f"{HF_AUDIO_PREFIX}.layers.{i}"
        our_prefix = f"whisper_encoder.layers.{i}"

        # Self-attention. q_proj/v_proj biased; k_proj NO bias; out_proj uses
        # RowParallelLinearFP32Bias wrapper (rpl inner + separate fp32 bias).
        m[f"{our_prefix}.self_attn.q_proj.weight"] = (
            f"{hf_prefix}.self_attn.q_proj.weight"
        )
        m[f"{our_prefix}.self_attn.q_proj.bias"] = (
            f"{hf_prefix}.self_attn.q_proj.bias"
        )
        m[f"{our_prefix}.self_attn.k_proj.weight"] = (
            f"{hf_prefix}.self_attn.k_proj.weight"
        )
        # k_proj.bias intentionally NOT mapped -- module has bias=False.
        m[f"{our_prefix}.self_attn.v_proj.weight"] = (
            f"{hf_prefix}.self_attn.v_proj.weight"
        )
        m[f"{our_prefix}.self_attn.v_proj.bias"] = (
            f"{hf_prefix}.self_attn.v_proj.bias"
        )
        # RowParallelLinearFP32Bias: `.rpl.weight` holds the linear, `.bias`
        # is the separate fp32 additive bias.
        m[f"{our_prefix}.self_attn.out_proj.rpl.weight"] = (
            f"{hf_prefix}.self_attn.out_proj.weight"
        )
        m[f"{our_prefix}.self_attn.out_proj.bias"] = (
            f"{hf_prefix}.self_attn.out_proj.bias"
        )

        # Attention pre-norm.
        m[f"{our_prefix}.self_attn_layer_norm.weight"] = (
            f"{hf_prefix}.self_attn_layer_norm.weight"
        )
        m[f"{our_prefix}.self_attn_layer_norm.bias"] = (
            f"{hf_prefix}.self_attn_layer_norm.bias"
        )

        # MLP. fc1 biased, fc2 uses FP32Bias wrapper.
        m[f"{our_prefix}.mlp.fc1.weight"] = f"{hf_prefix}.fc1.weight"
        m[f"{our_prefix}.mlp.fc1.bias"] = f"{hf_prefix}.fc1.bias"
        m[f"{our_prefix}.mlp.fc2.rpl.weight"] = f"{hf_prefix}.fc2.weight"
        m[f"{our_prefix}.mlp.fc2.bias"] = f"{hf_prefix}.fc2.bias"

        # Final layer norm (MLP pre-norm).
        m[f"{our_prefix}.final_layer_norm.weight"] = (
            f"{hf_prefix}.final_layer_norm.weight"
        )
        m[f"{our_prefix}.final_layer_norm.bias"] = (
            f"{hf_prefix}.final_layer_norm.bias"
        )

    # Encoder top-level.
    m["whisper_encoder.conv1.weight"] = f"{HF_AUDIO_PREFIX}.conv1.weight"
    m["whisper_encoder.conv1.bias"] = f"{HF_AUDIO_PREFIX}.conv1.bias"
    m["whisper_encoder.conv2.weight"] = f"{HF_AUDIO_PREFIX}.conv2.weight"
    m["whisper_encoder.conv2.bias"] = f"{HF_AUDIO_PREFIX}.conv2.bias"
    m["whisper_encoder.embed_positions.weight"] = (
        f"{HF_AUDIO_PREFIX}.embed_positions.weight"
    )
    m["whisper_encoder.layer_norm.weight"] = f"{HF_AUDIO_PREFIX}.layer_norm.weight"
    m["whisper_encoder.layer_norm.bias"] = f"{HF_AUDIO_PREFIX}.layer_norm.bias"

    return m
