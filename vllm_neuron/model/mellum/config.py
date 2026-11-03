# SPDX-License-Identifier: Apache-2.0
"""
Mellum Config
=============

Checkpoint: ``JetBrains/Mellum2.1-12B-A2.5B-Thinking`` (and the Mellum2 family):
``model_type: mellum``, ``architectures: ["MellumForCausalLM"]``.

Architecture (verified against config.json and transformers' ``modular_mellum.py``):

  hidden 2304 / 28 layers / 32 Q heads / 4 KV heads / head_dim 128 / vocab 98304 /
  RMSNorm eps 1e-6 (gamma * norm(x)) / per-head QK-norm before RoPE / no bias /
  untied LM head.
  Attention schedule: ``layer_types`` = [sliding x3, full] x7, sliding window 1024.
  RoPE per layer type (``rope_parameters`` is NESTED by layer type):
    sliding_attention: default, theta 5e5
    full_attention:    yarn, theta 5e5, factor 16, original_max_position_embeddings 8192,
                       beta_fast 32, beta_slow 1, attention_factor 1.2773
  MLP: sparse MoE in every layer (``mlp_layer_types``): 64 experts, top-8, softmax router
  in fp32, renormalized top-k (``norm_topk_prob``), SiLU GLU, moe_intermediate_size 896,
  no shared expert.

Config parsing traps handled here:
  * ``rope_parameters`` is a dict of dicts keyed by layer type. A flat parse (or vLLM's
    ``MellumConfig(Qwen3MoeConfig)`` shim) injects a flat ``rope_theta`` of 1e4 and drops the
    YaRN parameters. ``from_configs`` reads both sub-dicts explicitly and ``validate`` asserts
    the result.
  * ``num_experts`` is stored as ``num_local_experts`` by transformers' attribute map.
"""

import json
import math
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

_SLIDING = "sliding_attention"
_FULL = "full_attention"


@dataclass
class MellumConfig:
    # <-- MODEL-SPECIFIC: architecture
    vocab_size: int = 98304
    hidden_size: int = 2304
    intermediate_size: int = 7168  # dense MLP size (unused: every layer is sparse)
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int | None = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    # Hybrid attention schedule
    sliding_window: int = 1024
    layer_types: list[str] | None = None

    # MoE
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 896
    norm_topk_prob: bool = True
    mlp_layer_types: list[str] | None = None

    # RoPE per layer type (parsed from the nested rope_parameters)
    rope_parameters: dict = field(default_factory=dict)
    sliding_rope_theta: float = 500000.0
    full_rope_theta: float = 500000.0
    full_rope_type: str = "default"
    full_rope_factor: float = 1.0
    full_rope_original_max_position: int | None = None
    full_rope_beta_fast: float = 32.0
    full_rope_beta_slow: float = 1.0
    full_rope_attention_factor: float = 1.0
    full_rope_truncate: bool = True

    torch_dtype: torch.dtype = torch.bfloat16
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [_FULL] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers
        self.validate()

    # Attributes the shared Gemma3 attention module reads.
    @property
    def query_pre_attn_scalar(self) -> float:
        # Mellum uses the standard head_dim ** -0.5 softmax scale.
        return float(self.head_dim)

    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == _SLIDING

    def validate(self) -> None:
        """Load-time config assert: fail on a mis-parsed config instead of running wrong."""
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.head_dim % 2 == 0
        assert len(self.layer_types) == self.num_hidden_layers, (
            f"layer_types has {len(self.layer_types)} entries, num_hidden_layers={self.num_hidden_layers}"
        )
        unknown = set(self.layer_types) - {_SLIDING, _FULL}
        assert not unknown, f"unknown layer_types values: {sorted(unknown)}"
        assert len(self.mlp_layer_types) == self.num_hidden_layers
        if any(t != "sparse" for t in self.mlp_layer_types):
            raise NotImplementedError(
                "Mellum on Neuron implements sparse MoE layers only; the checkpoint has "
                f"mlp_layer_types={sorted(set(self.mlp_layer_types))}."
            )
        assert self.num_experts_per_tok <= self.num_experts
        assert self.full_rope_type in ("default", "yarn"), (
            f"full_attention rope_type {self.full_rope_type!r} is not implemented"
        )
        if self.full_rope_type == "yarn":
            assert self.full_rope_factor > 1.0 and self.full_rope_original_max_position, (
                "yarn RoPE needs factor > 1 and original_max_position_embeddings"
            )
        if self.attention_bias:
            raise NotImplementedError("attention_bias=True is not implemented for Mellum on Neuron")
        if self.hidden_act != "silu":
            raise NotImplementedError(f"hidden_act={self.hidden_act!r} is not implemented")

    def summary(self) -> str:
        return (
            f"layers={self.num_hidden_layers} sliding={sum(self.is_sliding(i) for i in range(self.num_hidden_layers))} "
            f"window={self.sliding_window} heads={self.num_attention_heads}/{self.num_key_value_heads} d={self.head_dim} "
            f"experts={self.num_experts} top{self.num_experts_per_tok} I={self.moe_intermediate_size} "
            f"norm_topk={self.norm_topk_prob} rope sliding theta={self.sliding_rope_theta} full {self.full_rope_type} "
            f"theta={self.full_rope_theta} factor={self.full_rope_factor} orig={self.full_rope_original_max_position} "
            f"attn_factor={self.full_rope_attention_factor:.6f}"
        )

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                d = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            d = hf_config.to_dict()
            dt = getattr(hf_config, "torch_dtype", None) or getattr(hf_config, "dtype", None)
            if dt is not None:
                d["torch_dtype"] = dt
        else:
            d = dict(hf_config)

        d = dict(d)
        if "num_experts" not in d and "num_local_experts" in d:
            d["num_experts"] = d["num_local_experts"]
        if d.get("torch_dtype") is None and d.get("dtype") is not None:
            d["torch_dtype"] = d["dtype"]

        d.update(cls._parse_rope(d))
        # Reduced-layer bring-up (hf_overrides num_hidden_layers < checkpoint): keep the schedule prefix.
        n = d.get("num_hidden_layers")
        for key in ("layer_types", "mlp_layer_types"):
            if isinstance(d.get(key), list) and n is not None and len(d[key]) > n:
                d[key] = d[key][:n]

        names = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in d.items() if k in names}
        if isinstance(kw.get("torch_dtype"), str):
            kw["torch_dtype"] = getattr(torch, kw["torch_dtype"])
        if kw.get("torch_dtype") is None:
            kw.pop("torch_dtype", None)
        kw["neuron_config"] = neuron_config
        return cls(**kw)

    @staticmethod
    def _parse_rope(d: dict) -> dict:
        """Read the nested per-layer-type rope_parameters into flat fields."""
        rp = d.get("rope_parameters")
        if not isinstance(rp, dict) or not rp:
            raise ValueError("Mellum config has no rope_parameters")
        if _SLIDING not in rp and _FULL not in rp:
            raise ValueError(
                "Mellum rope_parameters must be nested by layer type "
                f"({_SLIDING!r}, {_FULL!r}); got keys {sorted(rp)}"
            )
        s = rp.get(_SLIDING) or rp.get(_FULL)
        f = rp.get(_FULL) or rp.get(_SLIDING)
        if s.get("rope_type", "default") != "default":
            raise NotImplementedError(f"sliding rope_type {s.get('rope_type')!r} is not implemented")
        out = {
            "sliding_rope_theta": float(s["rope_theta"]),
            "full_rope_theta": float(f["rope_theta"]),
            "full_rope_type": f.get("rope_type", "default"),
        }
        if out["full_rope_type"] == "yarn":
            factor = float(f["factor"])
            out.update(
                full_rope_factor=factor,
                full_rope_original_max_position=int(f["original_max_position_embeddings"]),
                full_rope_beta_fast=float(f.get("beta_fast") or 32.0),
                full_rope_beta_slow=float(f.get("beta_slow") or 1.0),
                full_rope_truncate=bool(f.get("truncate", True)),
            )
            af = f.get("attention_factor")
            if af is None:
                # transformers' default (no mscale fields): 0.1 * ln(factor) + 1
                af = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
            out["full_rope_attention_factor"] = float(af)
        return out
