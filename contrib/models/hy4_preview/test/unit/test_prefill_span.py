# SPDX-License-Identifier: Apache-2.0
"""Prefill attention bounded by ``kv_segment_size`` == full-span prefill attention.

``prefill_span_bound`` drops keys past the prefix implied by the graph's
``kv_segment_size`` (multi-entry ``kv_segment_size_buckets``); the result must
be bitwise identical. Also: ``HY4_SEL_GATHER_DTYPE=uint8`` selects the same
keys as bf16.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from hy4_stubs import Group, model, set_group

bucket_utils = pytest.importorskip("vllm_neuron.utils.bucket_utils")

BLOCK, MAX_LEN, CHUNK = 32, 8192, 128
SEG_BUCKETS = [512, 2048, 8192]


class FakeTP(Group):
    """world > 1 with the collectives faked as rank 0 replicated."""

    def __init__(self, world):
        self.world_size, self.rank_in_group = world, 0

    def all_gather(self, t, dim=0):
        if t.numel() == 1:  # the rank gather: block j comes from rank j
            return torch.arange(self.world_size, dtype=t.dtype)
        return torch.cat([t] * self.world_size, dim=dim)


def make_config(seg_buckets):
    c = SimpleNamespace(
        torch_dtype=torch.bfloat16, hidden_size=64, num_attention_heads=2, q_lora_rank=32,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        index_topk=64, index_n_heads=4, index_head_dim=16, indexer_types=("full",),
        indexer_rope_first=False, rms_norm_eps=1e-6, gated_mla=True, learnable_sink=True,
        rope_theta=10000.0, neuron_config=SimpleNamespace(kv_segment_size_buckets=seg_buckets),
    )
    c.qk_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
    c.kv_latent_dim = c.kv_lora_rank + c.qk_rope_head_dim
    return c


def build(seg_buckets=None, world=1, seed=0):
    set_group(FakeTP(world) if world > 1 else Group())
    cfg = make_config(seg_buckets)
    torch.manual_seed(seed)
    attn = model.HYV4Attention(cfg, 0)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif "bias" in name:
                p.zero_()
            else:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.2)
    attn.kv_pages = torch.randn(MAX_LEN // BLOCK + 8, 1, BLOCK, attn.entry_dim).to(torch.bfloat16)
    return attn, model.HYV4RotaryEmbedding(cfg)


@pytest.fixture(scope="module")
def table():
    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(MAX_LEN // BLOCK + 7, generator=g)[: MAX_LEN // BLOCK] + 1
    return perm.view(1, -1).to(torch.int32)


@torch.no_grad()
def run(attn, rope, table, cached, seg=0, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(CHUNK, 64, generator=g).to(torch.bfloat16)
    pos = torch.arange(cached, cached + CHUNK)
    slots = table[0, pos // BLOCK].long() * BLOCK + pos % BLOCK
    meta = {
        "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32),
        "block_table_tensor": table,
        "slot_mapping": slots,
        "max_query_len": CHUNK, "decode_token_threshold": 1, "kv_segment_size": seg,
    }
    saved = attn.kv_pages.clone()
    out, sel = attn.forward_prefill(x, rope(pos), meta, None, torch.tensor(0))
    attn.kv_pages = saved
    return out.float(), sel


def test_bound_rules():
    assert model.check_prefill_span_bound(SEG_BUCKETS)
    assert model.check_prefill_span_bound([2048, 4096, 8192])
    assert model.prefill_span_bound(2048, [2048]) is None  # one bucket: inert
    assert model.prefill_span_bound(8192, SEG_BUCKETS) is None  # largest bounds nothing
    assert model.prefill_span_bound(512, SEG_BUCKETS) == model.ATTN_KEY_BLOCK


@pytest.mark.parametrize("cached", [0, 300, 512, 513, 2048, 2049, 5000])
def test_kv_segment_bound_is_exact(table, cached):
    seg = bucket_utils.pick_kv_segment_size(cached, SEG_BUCKETS)
    bound = model.prefill_span_bound(seg, SEG_BUCKETS)
    assert bound is None or (bound >= cached and bound % model.ATTN_KEY_BLOCK == 0)
    attn_n, rope = build(None)
    attn_b, _ = build(SEG_BUCKETS)
    full, _ = run(attn_n, rope, table, cached, seg)
    out, _ = run(attn_b, rope, table, cached, seg)
    assert torch.equal(out, full)


def test_uint8_selection_gather(monkeypatch, table):
    res = {}
    for dt in ("bf16", "uint8"):
        monkeypatch.setattr(model, "SEL_GATHER_DTYPE", dt)
        attn, rope = build(world=2)
        res[dt] = run(attn, rope, table, 1500)
    assert torch.equal(res["bf16"][0], res["uint8"][0])
    assert torch.equal(res["bf16"][1], res["uint8"][1])
