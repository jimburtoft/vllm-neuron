# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS FP8 on the CPU-compile path: load_weights_lite must declare the same expert
parameters (names, shapes, dtypes) that load_weights creates, on the meta device."""

import types

import pytest
import torch
from torch import nn

from vllm_neuron.model.gpt_oss import model_bf16
from vllm_neuron.model.gpt_oss.model_bf16 import GptOssForCausalLM
from vllm_neuron.model.gpt_oss.quantization_fp8 import quantize_expert_state_dict

pytestmark = pytest.mark.fast

E, H, I = 4, 64, 48


def _model(fp8: bool, num_layers: int = 2):
    """Stand-in with just what the CPU-compile helper reads: layers[*].mlp.experts."""
    layers = []
    for _ in range(num_layers):
        ex = nn.Module()
        ex.fp8_experts = fp8
        ex.intermediate_size_per_rank = I
        with torch.device("meta"):
            ex.gate_up_proj_weight = nn.Parameter(
                torch.empty(E, H, 2 * I, dtype=torch.bfloat16)
            )
            ex.down_proj_weight = nn.Parameter(
                torch.empty(E, I, H, dtype=torch.bfloat16)
            )
        layers.append(types.SimpleNamespace(mlp=types.SimpleNamespace(experts=ex)))
    m = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=layers),
        config=types.SimpleNamespace(neuron_config={}),
    )
    m._fp8_keep_bf16_prefill_copy = types.MethodType(
        GptOssForCausalLM._fp8_keep_bf16_prefill_copy, m
    )
    return m


def _signature(params):
    return {n.split(".")[-1]: (tuple(p.shape), p.dtype) for n, p in params}


def _real_load_signature(keep_bf16: bool):
    # Values exactly representable in FP8 under a power-of-two scale, as MXFP4 weights are.
    prefix = "model.layers.0.mlp.experts."
    sd = {
        prefix + "gate_up_proj_weight": torch.randint(-6, 7, (E, H, 2 * I)).float()
        * 0.5,
        prefix + "down_proj_weight": torch.randint(-6, 7, (E, I, H)).float() * 0.5,
    }
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    out, extra = quantize_expert_state_dict(sd, keep_bf16=keep_bf16, gate_up_major=True)
    return _signature({**out, **extra}.items())


@pytest.mark.parametrize("keep_bf16", [False, True])
def test_matches_real_load(monkeypatch, keep_bf16):
    monkeypatch.setattr(model_bf16, "_GPTOSS_HYBRID_PREFILL", keep_bf16)
    m = _model(fp8=True)
    GptOssForCausalLM._declare_fp8_experts_for_cpu_compile(m)
    expected = _real_load_signature(keep_bf16)
    for layer in m.model.layers:
        ex = layer.mlp.experts
        assert _signature(ex.named_parameters()) == expected
        assert all(p.is_meta for p in ex.parameters())
        # The forward path picks the prefill weights by this attribute.
        assert hasattr(ex, "gate_up_proj_weight_bf16") is keep_bf16


def test_idempotent(monkeypatch):
    monkeypatch.setattr(model_bf16, "_GPTOSS_HYBRID_PREFILL", False)
    m = _model(fp8=True)
    GptOssForCausalLM._declare_fp8_experts_for_cpu_compile(m)
    before = _signature(m.model.layers[0].mlp.experts.named_parameters())
    GptOssForCausalLM._declare_fp8_experts_for_cpu_compile(m)
    assert _signature(m.model.layers[0].mlp.experts.named_parameters()) == before


def test_bf16_model_untouched():
    m = _model(fp8=False)
    GptOssForCausalLM._declare_fp8_experts_for_cpu_compile(m)
    sig = _signature(m.model.layers[0].mlp.experts.named_parameters())
    assert sig == {
        "gate_up_proj_weight": ((E, H, 2 * I), torch.bfloat16),
        "down_proj_weight": ((E, I, H), torch.bfloat16),
    }
