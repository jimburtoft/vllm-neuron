# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE text config (hybrid Gated DeltaNet + GQA + 256-expert MoE).

Reference checkpoint: Qwen/Qwen3.5-35B-A3B (also loads Qwen3.6-35B-A3B).
40 layers, layer_types gives the 3:1 linear/full pattern.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3_5MoeTextConfig:
    vocab_size: int = 248320
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    layer_types: list = field(default_factory=list)
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    torch_dtype: torch.dtype = torch.bfloat16

    # Full attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # MoE
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    # Isaac-0.5: router gate has one extra "null" row (text_config.isaac05_moe); see model.py Qwen3_5MoE.
    null_router: bool = False
    # checkpoint vocab; vocab_size is padded up to a multiple of 128 (Isaac-0.5: 256279 -> 256384) so the
    # lm_head / MK lm-head tiles shard evenly; padded logits are masked to -inf.
    vocab_size_ckpt: int = 0

    neuron_config: NeuronConfig | None = None

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @classmethod
    def from_configs(cls, hf_config, neuron_config: NeuronConfig | None = None):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                d = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            d = hf_config.to_dict()
        else:
            d = dict(hf_config)
        # Composite (multimodal) checkpoint: the decoder lives in text_config.
        text = d.get("text_config") or d
        if hasattr(text, "to_dict"):
            text = text.to_dict()
        text = dict(text)

        # transformers 5 keeps rope_theta / partial_rotary_factor only inside
        # rope_parameters. Read them explicitly so the dataclass default never
        # silently wins.
        rp = text.get("rope_parameters") or text.get("rope_scaling") or {}
        if "rope_theta" in rp:
            text["rope_theta"] = rp["rope_theta"]
        if "partial_rotary_factor" in rp:
            text["partial_rotary_factor"] = rp["partial_rotary_factor"]

        names = {f.name for f in cls.__dataclass_fields__.values()}
        kw = {k: v for k, v in text.items() if k in names}
        kw.pop("torch_dtype", None)
        kw["torch_dtype"] = torch.bfloat16
        if neuron_config is not None:
            kw["neuron_config"] = neuron_config
        moe_meta = text.get("isaac05_moe") or d.get("isaac05_moe")
        if moe_meta:
            assert moe_meta.get("router_contract") == [kw.get("num_experts", 256), moe_meta.get("num_null_experts")], moe_meta
            assert moe_meta.get("score_func") == "softmax" and moe_meta.get("route_norm") and moe_meta.get("route_scale") == 1.0
            kw["null_router"] = True
        v = int(kw.get("vocab_size", cls.vocab_size))
        kw["vocab_size_ckpt"] = v
        kw["vocab_size"] = -(-v // 128) * 128
        cfg = cls(**kw)
        if not cfg.layer_types:
            cfg.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        return cfg
