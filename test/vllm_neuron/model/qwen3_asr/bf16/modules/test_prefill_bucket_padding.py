# SPDX-License-Identifier: Apache-2.0
"""CPU tests: Qwen3-ASR full prefill at a prefill bucket that is not a multiple of 128.

A 30 s clip is a 403-token prompt (390 audio + 13 text tokens). The bucket
``[128, 256, 416, 512, 1024]`` pads it to 416 instead of 512, which makes the
prefill graph ~11 % faster on trn2 (35.0 vs 39.2 ms). Nothing on the TP1
full-prefill path requires a multiple of 128 (the bucket validator, the QKV /
o_proj / MLP CTE kernels, the causal attention_cte, the per-token KV write).

These tests check the part that needs no device: the bucket config resolves,
and one decoder attention layer gives the same outputs on the real tokens and
the same KV cache when the prompt is padded to 416 instead of 512, the way the
runner pads it (pad tokens after the real ones, pad positions repeat the last
real position, pad slots go to the null block 0). The pad tokens are causally
masked, so the real rows cannot see them.

    VLLM_NEURON_CPU_MODE=1 pytest \
    test/vllm_neuron/model/qwen3_asr/bf16/modules/test_prefill_bucket_padding.py -q
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_asr.config import Qwen3ASRTextConfig
from vllm_neuron.model.qwen3_asr.model_bf16 import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRTextAttention,
    Qwen3ASRTextRotaryEmbedding,
)
from vllm_neuron.utils.bucket_utils import (
    get_bucket_for_count,
    resolve_num_batched_tokens_buckets,
)
from vllm_neuron.utils.executor import MPExecutor

pytestmark = [pytest.mark.fast]

BLOCK = 32
HEAD_DIM = 128
HIDDEN = 256
NUM_BLOCKS = 40  # block 0 is the null block; 16 blocks hold 512 tokens
BUCKETS = [128, 256, 416, 512, 1024]


def test_bucket_config_resolves_and_selects_416():
    buckets = resolve_num_batched_tokens_buckets(
        1024, configured_buckets=BUCKETS, use_configured_buckets=True
    )
    assert buckets == BUCKETS
    assert get_bucket_for_count(403, buckets) == 416  # 30 s clip
    assert get_bucket_for_count(117, buckets) == 128  # 8 s clip
    assert get_bucket_for_count(417, buckets) == 512


def _text_config():
    return Qwen3ASRTextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        num_hidden_layers=1,
        torch_dtype=torch.bfloat16,
        neuron_config=NeuronConfig(),
    )


def _model_load():
    cfg = _text_config()
    attn = Qwen3ASRTextAttention(cfg, layer_idx=0)
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        attn.qkv_proj_weight = torch.nn.Parameter(
            (torch.randn(attn.qkv_proj_weight.shape, generator=g) * 0.08).to(
                torch.bfloat16
            )
        )
        attn.o_proj_weight = torch.nn.Parameter(
            (torch.randn(attn.o_proj_weight.shape, generator=g) * 0.05).to(
                torch.bfloat16
            )
        )
        for norm in (attn.input_layernorm, attn.q_norm, attn.k_norm):
            norm.weight.copy_(1.0 + 0.1 * torch.randn(norm.weight.shape, generator=g))
    k = torch.zeros(NUM_BLOCKS, 1, BLOCK, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(NUM_BLOCKS, 1, BLOCK, HEAD_DIM, dtype=torch.bfloat16)
    holder = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=attn)]
        )
    )
    Qwen3ASRForConditionalGeneration.bind_kv_cache(
        holder, {"layers.0.self_attn": [k, v]}
    )
    rope = Qwen3ASRTextRotaryEmbedding(cfg)

    class Runner(torch.nn.Module):
        def forward(self, hidden, positions, md):
            with torch.no_grad():
                attn.k_cache.zero_()
                attn.v_cache.zero_()
                pe = rope(positions, hidden.device, torch.bfloat16)
                out = attn(hidden, positions, pe, md)
            return out.float(), attn.k_cache.float(), attn.v_cache.float()

    return Runner()


def _padded_inputs(h_real, table, padded_len):
    """Pad like the runner: zeros after the real tokens, last position repeated, null-block slots."""
    real = h_real.shape[0]
    pad = padded_len - real
    h = torch.cat([h_real, torch.zeros(pad, HIDDEN, dtype=h_real.dtype)])
    pos = torch.cat([torch.arange(real), torch.full((pad,), real - 1)])
    p = torch.arange(real)
    slots = torch.cat(
        [table[p // BLOCK] * BLOCK + p % BLOCK, torch.zeros(pad, dtype=torch.long)]
    )
    md = {
        "layers.0.self_attn": {
            "block_table_tensor": table[None].to(torch.int32),
            "slot_mapping": slots.to(torch.long),
            "max_query_len": padded_len,
            "block_size": BLOCK,
            "max_blocks_per_seq": table.shape[0],
            "decode_token_threshold": 1,
        }
    }
    return h, pos, md


@pytest.mark.parametrize("real", [403, 390])
def test_prefill_416_matches_512_on_real_tokens(real):
    table = torch.arange(1, 17)  # 16 blocks x 32 = 512 slots
    g = torch.Generator().manual_seed(7)
    h_real = torch.randn(real, HIDDEN, generator=g).to(torch.bfloat16)
    ex = MPExecutor(world_size=1, model_load=_model_load, device="cpu")
    try:
        res = {}
        for T in (416, 512):
            ex.dispatch(*_padded_inputs(h_real, table, T))
            res[T] = ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()
    (o416, k416, v416), (o512, k512, v512) = res[416], res[512]
    assert o416.shape[0] == 416 and o512.shape[0] == 512
    torch.testing.assert_close(o416[:real], o512[:real], atol=2e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(
        o416[:real].flatten(), o512[:real].flatten(), dim=0
    )
    assert cos > 0.9999, f"cos={cos}"
    # The real tokens' K/V land in the same slots; pads only touch the null block 0.
    torch.testing.assert_close(k416[1:], k512[1:], atol=0, rtol=0)
    torch.testing.assert_close(v416[1:], v512[1:], atol=0, rtol=0)
    assert k416[1:].abs().sum() > 0
