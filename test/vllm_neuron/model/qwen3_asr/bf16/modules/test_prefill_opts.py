# SPDX-License-Identifier: Apache-2.0
"""CPU tests: Qwen3-ASR prefill graph options (VLLM_NEURON_QWEN3_ASR_PREFILL_OPT).

- ``mlpfuse``: at TP1 the prefill MLP kernel adds the attention residual and applies the
  post-attention RMSNorm itself (as decode does) instead of a torch add + torch RMSNorm.
- ``gqa``: prefill attention passes the 8 KV heads to attention_cte (native GQA) instead
  of a repeat_interleave copy to 16 heads.

On CPU the NKI calls run their torch fallbacks, so this checks the wiring and the math
(one decoder layer, same weights, stock vs option): outputs match within bf16 rounding and
the KV cache is bit-identical. The kernels themselves are checked on trn2.

    VLLM_NEURON_CPU_MODE=1 pytest \
    test/vllm_neuron/model/qwen3_asr/bf16/modules/test_prefill_opts.py -q
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_asr import model_bf16 as mb
from vllm_neuron.model.qwen3_asr.config import Qwen3ASRTextConfig
from vllm_neuron.utils.executor import MPExecutor

pytestmark = [pytest.mark.fast]

BLOCK = 32
HEAD_DIM = 128
HIDDEN = 256
NUM_BLOCKS = 20


def test_env_parsing(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_PREFILL_OPT", raising=False)
    assert mb.get_prefill_opts() == frozenset()
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_PREFILL_OPT", " MLPFuse , gqa ")
    assert mb.get_prefill_opts() == frozenset({"mlpfuse", "gqa"})
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_PREFILL_OPT", "gqa,bogus")
    with pytest.raises(ValueError, match="bogus"):
        mb.get_prefill_opts()


def _text_config():
    return Qwen3ASRTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        num_hidden_layers=1,
        torch_dtype=torch.bfloat16,
        neuron_config=NeuronConfig(),
    )


def _make_load(opts: str):
    def _model_load():
        import os

        os.environ["VLLM_NEURON_QWEN3_ASR_PREFILL_OPT"] = opts
        cfg = _text_config()
        layer = mb.Qwen3ASRTextDecoderLayer(cfg, layer_idx=0)
        g = torch.Generator().manual_seed(4321)
        with torch.no_grad():
            for name, p in layer.named_parameters():
                if name.endswith("layernorm.weight") or "norm.weight" in name:
                    p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
                else:
                    p.copy_((torch.randn(p.shape, generator=g) * 0.06).to(p.dtype))
        k = torch.zeros(NUM_BLOCKS, 2, BLOCK, HEAD_DIM, dtype=torch.bfloat16)
        v = torch.zeros(NUM_BLOCKS, 2, BLOCK, HEAD_DIM, dtype=torch.bfloat16)
        holder = types.SimpleNamespace(
            language_model=types.SimpleNamespace(
                layers=[types.SimpleNamespace(self_attn=layer.self_attn)]
            )
        )
        mb.Qwen3ASRForConditionalGeneration.bind_kv_cache(
            holder, {"layers.0.self_attn": [k, v]}
        )
        rope = mb.Qwen3ASRTextRotaryEmbedding(cfg)
        flags = (layer.mlp.prefill_fuse_ok, layer.self_attn.prefill_native_gqa)

        class Runner(torch.nn.Module):
            def forward(self, hidden, positions, md):
                with torch.no_grad():
                    layer.self_attn.k_cache.zero_()
                    layer.self_attn.v_cache.zero_()
                    pe = rope(positions, hidden.device, torch.bfloat16)
                    out = layer(hidden, positions, pe, md)
                return (
                    out.float(),
                    layer.self_attn.k_cache.float(),
                    layer.self_attn.v_cache.float(),
                    torch.tensor(flags),
                )

        return Runner()

    return _model_load


def _inputs(real, padded):
    g = torch.Generator().manual_seed(11)
    h = torch.cat(
        [torch.randn(real, HIDDEN, generator=g), torch.zeros(padded - real, HIDDEN)]
    ).to(torch.bfloat16)
    pos = torch.cat([torch.arange(real), torch.full((padded - real,), real - 1)])
    table = torch.arange(1, 1 + (padded + BLOCK - 1) // BLOCK)
    p = torch.arange(real)
    slots = torch.cat(
        [table[p // BLOCK] * BLOCK + p % BLOCK, torch.zeros(padded - real, dtype=torch.long)]
    )
    md = {
        "layers.0.self_attn": {
            "block_table_tensor": table[None].to(torch.int32),
            "slot_mapping": slots,
            "max_query_len": padded,
            "block_size": BLOCK,
            "max_blocks_per_seq": table.shape[0],
            "decode_token_threshold": 1,
        }
    }
    return h, pos, md


def _run(opts, args):
    ex = MPExecutor(world_size=1, model_load=_make_load(opts), device="cpu")
    try:
        ex.dispatch(*args)
        return ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()


@pytest.mark.parametrize("opts", ["mlpfuse", "gqa", "mlpfuse,gqa"])
def test_prefill_opts_match_stock(opts):
    real = 150
    args = _inputs(real, 160)
    o0, k0, v0, f0 = _run("", args)
    o1, k1, v1, f1 = _run(opts, args)
    assert f0.tolist() == [False, False]
    assert f1.tolist() == ["mlpfuse" in opts, "gqa" in opts]
    cos = torch.nn.functional.cosine_similarity(
        o0[:real].flatten().double(), o1[:real].flatten().double(), dim=0
    )
    assert cos > 0.9999, f"cos={cos}"
    torch.testing.assert_close(o1[:real], o0[:real], atol=3e-2, rtol=3e-2)
    if opts == "gqa":
        # GQA only changes which K/V tensor attention reads: bit-identical on CPU.
        torch.testing.assert_close(o1, o0, atol=0, rtol=0)
    # KV write precedes both options: identical cache.
    torch.testing.assert_close(k1, k0, atol=0, rtol=0)
    torch.testing.assert_close(v1, v0, atol=0, rtol=0)
    assert k0[1:].abs().sum() > 0
