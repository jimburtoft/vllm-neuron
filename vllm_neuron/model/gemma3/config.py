# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Config (text decoder)
============================

<-- MODEL-SPECIFIC: All fields in this config are Gemma3-text-decoder specific.

Checkpoints: ``google/gemma-3-12b-it`` / ``google/gemma-3-4b-it``. They ship the **multimodal**
``Gemma3ForConditionalGeneration`` config, which nests the text decoder under
``text_config`` (``model_type: gemma3_text``) and a SigLIP tower under
``vision_config``. This port is text-only, so ``from_configs()``:

  * extracts the nested ``text_config`` (falling back to a flat
    ``gemma3_text`` config if the checkpoint was re-exported text-only),
  * **drops ``vision_config`` entirely**,
  * normalizes the two RoPE layouts (flat ``rope_local_base_freq`` /
    ``rope_theta`` / ``rope_scaling`` used by transformers < 4.53, and the
    per-layer-type ``rope_parameters`` dict used by >= 4.53),
  * normalizes ``sliding_window_pattern`` (old) vs ``layer_types`` (new).

Architecture summary (verified against the checkpoint, not assumed):

  hidden 2560 / intermediate 10240 / 34 layers / 8 Q heads / 4 KV heads (GQA
  2:1) / head_dim 256 (note 8*256 = 2048 != 2560, so QKV is NON-SQUARE) /
  vocab 262208 / max_pos 131072 / rms_norm_eps 1e-6 / sliding_window 1024 with
  ``sliding_window_pattern`` 6 (5 local : 1 global) / dual RoPE (local theta
  1e4, global theta 1e6 with linear x8 scaling) / ``query_pre_attn_scalar`` 256
  / ``gelu_pytorch_tanh`` gated MLP / no attention bias / no logit softcapping
  / tied LM head.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Gemma3Config:
    # <-- MODEL-SPECIFIC: Gemma3-4B text-decoder architecture parameters
    vocab_size: int = 262208
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    # head_dim must be int|None: HF configs may pass null, and the
    # __post_init__ fallback only works if the field can hold None.
    # Gemma3 sets it EXPLICITLY to 256 -- it is NOT hidden_size //
    # num_attention_heads (2560 // 8 = 320), so never derive it silently for
    # this architecture.
    head_dim: int | None = 256
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6

    # <-- MODEL-SPECIFIC: hybrid sliding/global attention schedule.
    # HF makes every ``sliding_window_pattern``-th layer FULL/global and all
    # others sliding: layer i is sliding iff ``(i + 1) % pattern != 0``.
    # With pattern=6 and 34 layers that is 5 local : 1 global, global at
    # layer indices 5, 11, 17, 23, 29.
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    # transformers >= 4.53 materializes the schedule as an explicit list of
    # "sliding_attention" / "full_attention". When present it WINS over
    # ``sliding_window_pattern`` (it is what HF's own attention uses).
    layer_types: list[str] | None = None

    # <-- MODEL-SPECIFIC: DUAL RoPE. Sliding (local) layers and global (full)
    # layers use different bases, and the linear position scaling applies to
    # the GLOBAL layers only.
    rope_local_base_freq: float = 10000.0  # theta for sliding-window layers
    rope_theta: float = 1000000.0  # theta for global (full-attention) layers
    rope_scaling: dict | None = field(
        default_factory=lambda: {"rope_type": "linear", "factor": 8.0}
    )

    # <-- MODEL-SPECIFIC: Gemma's softmax scale is query_pre_attn_scalar**-0.5,
    # NOT head_dim**-0.5. For Gemma3-4B they happen to both be 256, but the
    # fields are independent in the config and must not be conflated.
    query_pre_attn_scalar: float = 256.0

    # <-- MODEL-SPECIFIC: Gemma3 DROPPED Gemma2's logit soft-capping. Both are
    # null in the checkpoint; kept as fields so a config that sets them fails
    # loudly in _validate_config rather than being silently ignored.
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None

    attention_bias: bool = False
    # Gemma3's text config names this ``hidden_activation`` (not ``hidden_act``).
    hidden_activation: str = "gelu_pytorch_tanh"

    # <-- MODEL-SPECIFIC: Gemma3 TIES the LM head to the input embedding; the
    # checkpoint has no ``lm_head.weight``.
    tie_word_embeddings: bool = True

    torch_dtype: torch.dtype = torch.bfloat16

    # Framework config
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Assert the dims this port was written against. These are cheap and
        # catch a mis-parsed nested config immediately instead of producing a
        # model that runs and emits fluent garbage.
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be "
            f"divisible by num_key_value_heads ({self.num_key_value_heads})"
        )
        assert self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) must be even for rotate_half RoPE"
        )
        assert self.sliding_window_pattern >= 1, (
            f"sliding_window_pattern ({self.sliding_window_pattern}) must be >= 1"
        )
        if self.layer_types is not None:
            assert len(self.layer_types) == self.num_hidden_layers, (
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
            unknown = set(self.layer_types) - {"sliding_attention", "full_attention"}
            assert not unknown, f"unknown layer_types values: {sorted(unknown)}"

    # ── Hybrid attention schedule ────────────────────────────────────────

    def is_sliding(self, layer_idx: int) -> bool:
        """True if ``layer_idx`` uses sliding-window attention (and local RoPE).

        Prefers an explicit ``layer_types`` list (transformers >= 4.53) and
        falls back to the ``sliding_window_pattern`` arithmetic that HF itself
        uses to build that list:

            "sliding_attention" if (i + 1) % pattern else "full_attention"

        i.e. every ``pattern``-th layer (1-indexed) is GLOBAL. Do not copy
        GPT-OSS's ``layer_idx % 2 == 0`` -- that is GPT-OSS's own schedule.
        """
        if self.layer_types is not None:
            return self.layer_types[layer_idx] == "sliding_attention"
        return bool((layer_idx + 1) % self.sliding_window_pattern != 0)

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None
    ):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if hasattr(hf_config, "torch_dtype") and hf_config.torch_dtype is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = dict(hf_config)

        top_dtype = config_dict.get("torch_dtype")
        top_tie = config_dict.get("tie_word_embeddings")

        # <-- MODEL-SPECIFIC: extract the nested text_config, DROP vision.
        text_cfg = config_dict.get("text_config")
        if isinstance(text_cfg, PretrainedConfig):
            text_dict = text_cfg.to_dict()
        elif isinstance(text_cfg, dict):
            text_dict = dict(text_cfg)
        else:
            # Already a flat gemma3_text config.
            text_dict = dict(config_dict)

        # The SigLIP tower is never ported. Popping it here means it can never
        # leak into a field lookup, and the weight mapping in model.py never
        # references vision_tower.* / multi_modal_projector.* so those tensors
        # are simply not loaded.
        text_dict.pop("vision_config", None)

        text_dict = cls._normalize_rope(text_dict)
        text_dict = cls._normalize_layer_schedule(text_dict)

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in text_dict.items() if k in field_names}

        # Only keep rope_scaling when it is a real scaling spec; a null/empty
        # value must fall through to the dataclass default (linear x8) rather
        # than disabling the global-layer scaling.
        if not filtered.get("rope_scaling"):
            filtered.pop("rope_scaling", None)

        # dtype: prefer the text config's, fall back to the top level.
        dtype = filtered.get("torch_dtype", top_dtype)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None:
            filtered["torch_dtype"] = dtype

        # tie_word_embeddings: Gemma3's text stack ties by default; honor an
        # explicit setting from either level when present.
        if "tie_word_embeddings" in text_dict:
            filtered["tie_word_embeddings"] = text_dict["tie_word_embeddings"]
        elif top_tie is not None:
            filtered["tie_word_embeddings"] = top_tie

        # Gemma3 names the activation ``hidden_activation``; tolerate a config
        # that only carries the generic ``hidden_act``.
        if "hidden_activation" not in filtered and "hidden_act" in text_dict:
            filtered["hidden_activation"] = text_dict["hidden_act"]

        if neuron_config is not None:
            filtered["neuron_config"] = neuron_config

        return cls(**filtered)

    # ── from_configs helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_rope(text_dict: dict) -> dict:
        """Normalize both HF RoPE layouts into the flat dual-theta fields.

        transformers < 4.53 (what older Gemma3 checkpoints ship)::

            rope_local_base_freq: 10000.0
            rope_theta:           1000000.0
            rope_scaling:         {"rope_type": "linear", "factor": 8.0}

        transformers >= 4.53 unifies them per layer type::

            rope_parameters: {
              "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
              "full_attention":    {"rope_type": "linear", "factor": 8.0,
                                    "rope_theta": 1e6},
            }
        """
        out = dict(text_dict)
        rope_params = out.get("rope_parameters")
        if not isinstance(rope_params, dict):
            return out

        local_p = rope_params.get("sliding_attention") or {}
        global_p = rope_params.get("full_attention") or {}

        if local_p.get("rope_theta") is not None:
            out["rope_local_base_freq"] = float(local_p["rope_theta"])
        if global_p.get("rope_theta") is not None:
            out["rope_theta"] = float(global_p["rope_theta"])

        # The scaling spec lives on the GLOBAL layers only.
        if global_p.get("rope_type") not in (None, "default"):
            scaling = {k: v for k, v in global_p.items() if k != "rope_theta"}
            out["rope_scaling"] = scaling
        return out

    @staticmethod
    def _normalize_layer_schedule(text_dict: dict) -> dict:
        """Derive ``sliding_window_pattern`` from ``layer_types`` if only the
        list is present, so both fields agree no matter which HF version wrote
        the config. ``layer_types`` remains authoritative in ``is_sliding()``.
        """
        out = dict(text_dict)
        layer_types = out.get("layer_types")
        if out.get("sliding_window_pattern") is None and isinstance(layer_types, list):
            for i, lt in enumerate(layer_types):
                if lt == "full_attention":
                    out["sliding_window_pattern"] = i + 1
                    break
        return out
