# SPDX-License-Identifier: Apache-2.0
"""Decode MoE kernel selection: selective-expert vs all-expert by batch size."""

import pytest

from vllm_neuron.model.gpt_oss.model_bf16 import use_all_experts_decode

pytestmark = pytest.mark.fast

TOP_K = 4


@pytest.mark.parametrize(
    "num_tokens, expected_all_expert",
    [
        (1, False),
        (2, False),
        (4, False),
        (8, False),
        (16, True),
        (32, True),
        (64, True),
    ],
)
def test_20b_tp4(num_tokens, expected_all_expert):
    # GPT-OSS 20B at TP4: 32 local experts. Selective is faster up to 8 tokens on trn2.
    assert use_all_experts_decode(num_tokens, TOP_K, 32) is expected_all_expert


@pytest.mark.parametrize(
    "num_tokens, expected_all_expert",
    [(8, False), (16, False), (32, False), (64, True), (128, True)],
)
def test_120b_tp4(num_tokens, expected_all_expert):
    # GPT-OSS 120B at TP4: 128 local experts. Selective is faster up to 32 tokens on trn2.
    assert use_all_experts_decode(num_tokens, TOP_K, 128) is expected_all_expert


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_expert_parallel_always_all_expert(num_tokens):
    assert use_all_experts_decode(num_tokens, TOP_K, 32, ep_degree=2) is True
