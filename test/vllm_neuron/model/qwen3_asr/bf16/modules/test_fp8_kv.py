# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the packed FP8 KV cache of the Qwen3-ASR text decoder.

These cover the parts that need no Neuron device:

* ``_write_packed_k_pairs`` (the prefill K write into the packed cache) gives
  the same cache as the reference un-swizzle -> scatter -> re-swizzle, for
  odd prompt lengths, prompts that start mid-pair, several sequences in one
  call, padding slots, and more than one KV head.
* ``bind_kv_cache`` detects the packed layout, makes the unit scales, and
  makes the 3D views for one KV head per rank.
* A prefill -> decode round trip of ``Qwen3ASRTextAttention`` on the torch
  fallback, at one KV head per rank (the 3D decode path), with a packed FP8
  cache vs a bf16 cache: the FP8 cache holds exactly the fp8-rounded bf16
  cache, and the decode outputs agree to FP8 precision. Run for a bucket where
  the packed kernel applies and one where it does not (un-swizzle fallback).

The decode megakernel CPU fallback differs from HF under QK-norm (see
``test_text_attention.py``), so this test compares FP8 with bf16 on the same
fallback, not with HF. Device accuracy comes from the e2e WER run.

    VLLM_NEURON_CPU_MODE=1 pytest \
    test/vllm_neuron/model/qwen3_asr/bf16/modules/test_fp8_kv.py -q
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
)
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_asr.config import Qwen3ASRTextConfig
from vllm_neuron.model.qwen3_asr.model_bf16 import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRTextAttention,
    Qwen3ASRTextRotaryEmbedding,
    _write_packed_k_pairs,
)
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.executor import MPExecutor

pytestmark = [pytest.mark.fast]

FP8 = torch.float8_e4m3fn
BLOCK = 32
HEAD_DIM = 128


def _fp8_round(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(FP8)


# ---------------------------------------------------------------------------
# _write_packed_k_pairs vs the whole-cache reference
# ---------------------------------------------------------------------------


def _slots_for(block_table, start, length):
    """Slots of tokens [start, start + length) of one sequence."""
    pos = torch.arange(start, start + length)
    return block_table[pos // BLOCK] * BLOCK + pos % BLOCK


@pytest.mark.parametrize("nkh", [1, 2])
@pytest.mark.parametrize(
    "segments",
    [
        [(0, 37)],  # odd prompt length, ends mid-pair
        [(0, 64)],  # block aligned
        [(5, 20)],  # starts mid-pair (chunk after an odd prefix)
        [(0, 1)],  # one token
        [(0, 33), (3, 30), (0, 2)],  # several sequences in one call
    ],
)
@pytest.mark.parametrize("pad", [0, 3])
def test_write_packed_k_pairs_matches_reference(nkh, segments, pad):
    torch.manual_seed(0)
    nb, d = 16, 8
    # Pre-existing cache contents (the partner lane of a mid-pair start must
    # keep them).
    base = _fp8_round(torch.randn(nb, nkh, BLOCK, d) * 4)
    tables = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8])]
    slots = torch.cat(
        [_slots_for(tables[i], s, n) for i, (s, n) in enumerate(segments)]
        + [torch.full((pad,), -1)]
    )
    T = slots.shape[0]
    k_new = _fp8_round(torch.randn(nkh, T, d) * 4)

    safe = torch.where(slots < 0, torch.zeros_like(slots), slots)
    blk, pos = safe // BLOCK, safe % BLOCK

    # Reference: un-swizzle, scatter (real rows only), re-swizzle.
    ref = base.clone().to(torch.bfloat16)
    real = slots >= 0
    for h in range(nkh):
        ref[blk[real], h, pos[real]] = k_new[h][real].to(torch.bfloat16)
    ref_packed = _swizzle_packed_k(ref.to(FP8))

    cache = _swizzle_packed_k(base.clone())
    _write_packed_k_pairs(cache, k_new.to(torch.bfloat16), blk, pos)

    # Block 0 is the null block (padding writes land there).
    got = cache[1:].view(torch.uint8)
    want = ref_packed[1:].view(torch.uint8)
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# bind_kv_cache
# ---------------------------------------------------------------------------


class _Attn(types.SimpleNamespace):
    """Just the attributes bind_kv_cache touches."""

    @property
    def kv_is_fp8(self):
        return self.k_cache is not None and self.k_cache.dtype == FP8


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("nkh", [1, 2])
def test_bind_kv_cache_detects_packed(packed, nkh):
    layers = [
        types.SimpleNamespace(
            self_attn=_Attn(fp8_packed=False, dtype=torch.bfloat16, k_cache=None)
        )
        for _ in range(2)
    ]
    stand_in = types.SimpleNamespace(
        language_model=types.SimpleNamespace(layers=layers)
    )
    kv = {}
    for i in range(2):
        k = (
            torch.zeros(4, nkh, BLOCK // 2, HEAD_DIM, 2, dtype=FP8)
            if packed
            else torch.zeros(4, nkh, BLOCK, HEAD_DIM, dtype=FP8)
        )
        v = torch.zeros(4, nkh, BLOCK, HEAD_DIM, dtype=FP8)
        kv[f"layers.{i}.self_attn"] = [k, v]

    Qwen3ASRForConditionalGeneration.bind_kv_cache(stand_in, kv)
    for layer in layers:
        a = layer.self_attn
        assert a.fp8_packed is packed
        assert a.k_scale is not None and float(a.k_scale) == 1.0
        assert a.v_scale is not None and float(a.v_scale) == 1.0
        if nkh == 1:
            assert a.v_cache_3d.dim() == 3
            assert a.k_cache_3d.dim() == (4 if packed else 3)
            # The views alias the bound cache.
            assert a.k_cache_3d.data_ptr() == a.k_cache.data_ptr()
        else:
            assert a.k_cache_3d is None and a.v_cache_3d is None


# ---------------------------------------------------------------------------
# Module round trip: prefill -> decode, bf16 cache vs packed FP8 cache
# ---------------------------------------------------------------------------

HIDDEN = 256
NUM_BLOCKS = 24  # block 0 is the null block


def _text_config(nkh):
    return Qwen3ASRTextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2 * nkh,
        num_key_value_heads=nkh,
        head_dim=HEAD_DIM,
        num_hidden_layers=1,
        torch_dtype=torch.bfloat16,
        neuron_config=NeuronConfig(),
    )


def _make_attention(nkh, fp8):
    """Build the attention module with seeded random weights and bind a cache."""
    cfg = _text_config(nkh)
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

    dtype = FP8 if fp8 else torch.bfloat16
    k = torch.zeros(NUM_BLOCKS, nkh, BLOCK, HEAD_DIM, dtype=dtype)
    v = torch.zeros(NUM_BLOCKS, nkh, BLOCK, HEAD_DIM, dtype=dtype)
    if fp8:
        k = k.view(NUM_BLOCKS, nkh, BLOCK // 2, HEAD_DIM, 2)
    holder = types.SimpleNamespace(
        language_model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=attn)]
        )
    )
    Qwen3ASRForConditionalGeneration.bind_kv_cache(
        holder, {"layers.0.self_attn": [k, v]}
    )
    return attn, cfg


def _round_trip_load(nkh, fp8):
    def model_load():
        attn, cfg = _make_attention(nkh, fp8)
        rope = Qwen3ASRTextRotaryEmbedding(cfg)

        class Runner(torch.nn.Module):
            def forward(self, hidden, positions, md):
                pe = rope(positions, hidden.device, torch.bfloat16)
                out = attn(hidden, positions, pe, md)
                k = attn.k_cache
                if attn.fp8_packed:
                    k = _unswizzle_packed_k(k)
                # fp8 is not picklable through the executor; widen exactly.
                return (
                    out.float(),
                    k.to(torch.float32),
                    attn.v_cache.to(torch.float32),
                )

        return Runner()

    return model_load


def _md(block_table, slot_mapping, max_query_len):
    return {
        "layers.0.self_attn": {
            "block_table_tensor": block_table.to(torch.int32),
            "slot_mapping": slot_mapping.to(torch.long),
            "max_query_len": max_query_len,
            "block_size": BLOCK,
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": 1,
        }
    }


def _run_round_trip(nkh, fp8, prompt_lens, n_blocks_per_seq):
    torch.manual_seed(7)
    ex = MPExecutor(world_size=1, model_load=_round_trip_load(nkh, fp8), device="cpu")
    try:
        tables = [
            torch.arange(1 + i * n_blocks_per_seq, 1 + (i + 1) * n_blocks_per_seq)
            for i in range(len(prompt_lens))
        ]
        # Prefill each prompt (one sequence per call, as the runner does).
        prefill_out = []
        for table, L in zip(tables, prompt_lens):
            h = torch.randn(L, HIDDEN, dtype=torch.bfloat16)
            slots = _slots_for(table, 0, L)
            ex.dispatch(h, torch.arange(L), _md(table[None], slots, L))
            prefill_out.append(ex.collect(timeout=600)[0])
        # One decode step for all sequences.
        B = len(prompt_lens)
        h = torch.randn(B, HIDDEN, dtype=torch.bfloat16)
        pos = torch.tensor(prompt_lens)
        slots = torch.stack(
            [_slots_for(t, L, 1)[0] for t, L in zip(tables, prompt_lens)]
        )
        ex.dispatch(h, pos, _md(torch.stack(tables), slots, 1))
        dec = ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()
    return prefill_out, dec


@pytest.mark.parametrize(
    "prompt_lens, n_blocks_per_seq",
    [
        # S_ctx 256, B 2: the packed kernel applies (resized block_len 2).
        ([37, 50], 8),
        # S_ctx 256, B 1 at LNC 2: s_prior sharding resizes block_len to 1, so
        # decode un-swizzles, runs unpacked, and re-swizzles the result.
        ([61], 8),
    ],
)
def test_round_trip_fp8_vs_bf16(prompt_lens, n_blocks_per_seq):
    nkh = 1
    pre_bf, dec_bf = _run_round_trip(nkh, False, prompt_lens, n_blocks_per_seq)
    pre_f8, dec_f8 = _run_round_trip(nkh, True, prompt_lens, n_blocks_per_seq)

    # Prefill output attends local bf16 K/V: identical with either cache.
    for a, b in zip(pre_bf, pre_f8):
        assert torch.equal(a[0], b[0])

    # The FP8 cache holds exactly the fp8-rounded bf16 cache, prefill and the
    # decode token alike (both paths project the same K/V).
    _, k_bf, v_bf = dec_bf
    _, k_f8, v_f8 = dec_f8
    assert torch.equal(_fp8_round(k_bf).float(), k_f8)
    assert torch.equal(_fp8_round(v_bf).float(), v_f8)
    assert k_f8.abs().sum() > 0

    # Decode output: FP8 vs bf16 within FP8 precision.
    out_bf, out_f8 = dec_bf[0], dec_f8[0]
    cos = F.cosine_similarity(out_bf.flatten(), out_f8.flatten(), dim=0)
    rel = (out_bf - out_f8).norm() / out_bf.norm()
    assert cos > 0.995, f"cos={cos}"
    assert rel < 0.08, f"rel={rel}"


def test_prefill_write_gqa_two_heads():
    """Two KV heads per rank (TP4 layout): the packed prefill write per head."""

    def run(fp8):
        def model_load():
            attn, cfg = _make_attention(2, fp8)
            rope = Qwen3ASRTextRotaryEmbedding(cfg)

            class Runner(torch.nn.Module):
                def forward(self, hidden, positions, md):
                    pe = rope(positions, hidden.device, torch.bfloat16)
                    out = attn(hidden, positions, pe, md)
                    k = (
                        _unswizzle_packed_k(attn.k_cache)
                        if attn.fp8_packed
                        else attn.k_cache
                    )
                    return out.float(), k.float(), attn.v_cache.float()

            return Runner()

        ex = MPExecutor(world_size=1, model_load=model_load, device="cpu")
        try:
            table = torch.arange(1, 9)
            L = 45
            g = torch.Generator().manual_seed(11)
            h = torch.randn(L, HIDDEN, generator=g).to(torch.bfloat16)
            ex.dispatch(
                h, torch.arange(L), _md(table[None], _slots_for(table, 0, L), L)
            )
            return ex.collect(timeout=600)[0]
        finally:
            ex.shutdown()

    out_bf, k_bf, v_bf = run(False)
    out_f8, k_f8, v_f8 = run(True)
    assert torch.equal(out_bf, out_f8)
    assert torch.equal(_fp8_round(k_bf).float(), k_f8)
    assert torch.equal(_fp8_round(v_bf).float(), v_f8)
