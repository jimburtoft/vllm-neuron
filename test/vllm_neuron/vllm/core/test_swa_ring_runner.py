# SPDX-License-Identifier: Apache-2.0
"""VLLM_NEURON_SWA_RING through the real runner helpers and the real Gemma3 mask.

CPU only (torch mask fallback). The test drives:
* ``NeuronModelRunner._apply_swa_ring_slot_mapping`` for the prefill and decode
  KV writes of one SWA group (and checks that a full group is left unchanged);
* ``NeuronModelRunner._compute_swa_num_blocks`` / ``_compute_swa_decode_tensors``
  for the decode table and the position offset;
* ``gemma3.model._precompute_decode_attn_masks`` for the sliding-window mask;
and reconstructs, for each decode step, the positions whose K/V the attention
reads (valid prior slots + the active token stitched at slot s_prior - 1). The
set must equal [p - 1023, p] at every step, through the ring wrap.
"""

import functools
import types

import numpy as np
import pytest
import torch

W = 1024
BS = 32
ML = 4096


def _runner_cls():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    return NeuronModelRunner


def _fake_runner(table, ctx, swa_spec, full_spec):
    R = _runner_cls()
    fake = types.SimpleNamespace()
    fake._swa_ring = True
    fake._dcp_size = 1
    fake.max_model_len = ML
    fake.device = torch.device("cpu")
    fake._swa_groups = {}
    fake._raw_block_table = R._raw_block_table
    fake._gather_swa_columns = R._gather_swa_columns
    fake._compute_swa_num_blocks = functools.partial(R._compute_swa_num_blocks, fake)
    mk = lambda: types.SimpleNamespace(
        block_table=types.SimpleNamespace(np=table, cpu=torch.from_numpy(table)),
        slot_mapping=types.SimpleNamespace(np=np.zeros(8192, dtype=np.int64)),
        block_size=BS,
    )
    fake.input_batch = types.SimpleNamespace(
        num_computed_tokens_cpu=ctx,
        block_table=types.SimpleNamespace(block_tables=[mk(), mk()]),
    )
    fake.kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=[
            types.SimpleNamespace(kv_cache_spec=swa_spec),
            types.SimpleNamespace(kv_cache_spec=full_spec),
        ]
    )
    return fake


def _specs(window=W):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    common = dict(block_size=BS, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    return SlidingWindowSpec(sliding_window=window, **common), FullAttentionSpec(**common)


def _mask_to_slots(mask, B, S_ctx):
    """[S_ctx, B, q, 1] gen_mask layout -> [B, S_ctx] in table (block, offset) order."""
    from vllm_neuron.functional.attention.attention_decode import (
        _MASK_P_MAX,
        _mask_resize_block_len,
    )

    q = mask.shape[2]
    bl = _mask_resize_block_len(BS, B, q, 1, S_ctx)
    m = mask.permute(1, 2, 3, 0)
    if bl > 0:
        m = m.reshape(-1, bl, _MASK_P_MAX).swapaxes(-1, -2).reshape(B, q, 1, S_ctx)
    return m[:, 0, 0, :]


def _swa_decode_mask(sliding_window, md, positions, q_heads, layer):
    """SWA decode mask in the trimmed frame, as the model builds it.

    Same arithmetic as gemma3 ``_precompute_decode_attn_masks`` and MuseGlimmer
    ``forward_decode``: end = clamp(pos - offset, 0), start = clamp(end - W + 1, 0).
    """
    from vllm_neuron.functional.attention.attention_decode_mask import (
        gen_attention_decode_mask,
    )

    m = md[layer]
    B = m["block_table_tensor"].shape[0]
    pos = positions.reshape(1, B)
    pos = torch.clamp(pos - m["swa_kv_pos_offset"].reshape(1, B), min=0)
    start = torch.clamp(pos - sliding_window + 1, min=0)
    return gen_attention_decode_mask(
        pos_ids=pos.to(torch.float32), bs=B, q_head=q_heads, s_active=1,
        s_prior=m["max_blocks_per_seq"] * m["block_size"],
        start_pos=start.to(torch.float32), block_len=m["block_size"],
    )


@pytest.mark.parametrize(
    "W, table_tokens", [(1024, 0), (2048, 0), (2048, 4096)], ids=["w1024", "w2048", "w2048_t4096"]
)
@pytest.mark.parametrize("L", [100, 1023, 1024, 1025, 2048])
@pytest.mark.parametrize("B", [2, 3])
def test_runner_ring_decode_attention_set(monkeypatch, L, B, W, table_tokens):
    if table_tokens:
        monkeypatch.setenv("VLLM_NEURON_SWA_RING_TABLE_TOKENS", str(table_tokens))
    else:
        monkeypatch.delenv("VLLM_NEURON_SWA_RING_TABLE_TOKENS", raising=False)
    ML = 8192 if W > 1024 else globals()["ML"]
    monkeypatch.setitem(globals(), "ML", ML)
    R_cls = _runner_cls()
    swa_spec, full_spec = _specs(W)
    nb_exp = max(W, table_tokens) // BS
    rng = np.random.default_rng(L + B)
    width = ML // BS
    table = rng.permutation(np.arange(1, 1 + B * width)).reshape(B, width).astype(np.int32)
    ctx = np.zeros(B, dtype=np.int32)
    fake = _fake_runner(table, ctx, swa_spec, full_spec)
    cache = np.full((B * width + 1) * BS, -1, dtype=np.int64)
    full_before = None

    def write(positions, req):
        n = len(positions)
        # The full group starts with the linear slots; the ring call must not touch it.
        lin = table[req, positions // BS].astype(np.int64) * BS + positions % BS
        fake.input_batch.block_table.block_tables[1].slot_mapping.np[:n] = lin
        counts = np.bincount(req, minlength=B)
        R_cls._apply_swa_ring_slot_mapping(fake, positions, req, counts)
        assert np.array_equal(fake.input_batch.block_table.block_tables[1].slot_mapping.np[:n], lin)
        sl = fake.input_batch.block_table.block_tables[0].slot_mapping.np[:n].copy()
        keep = sl >= BS
        assert len(np.unique(sl[keep])) == keep.sum()
        cache[sl[keep]] = positions[keep]

    # Prefill each row with prompt L + 7 * row (one row per step, as single-shot).
    lens = np.array([L + 7 * b for b in range(B)])
    for b in range(B):
        write(np.arange(lens[b]), np.full(lens[b], b, dtype=np.int64))
    ctx[:] = lens
    blk = types.SimpleNamespace(block_table=types.SimpleNamespace(np=table, cpu=torch.from_numpy(table)))
    for step in range(W + 76):
        p = ctx.astype(np.int64).copy()
        nb = fake._compute_swa_num_blocks(W, BS, B)
        assert nb == nb_exp
        bt, off = R_cls._compute_swa_decode_tensors(fake, blk, B, W, BS, 0)
        assert tuple(bt.shape) == (B, nb_exp)
        md = {
            "layers.0.self_attn": dict(block_size=BS, block_table_tensor=bt, max_blocks_per_seq=nb,
                                       swa_kv_pos_offset=off),
            "layers.5.self_attn": dict(block_size=BS, block_table_tensor=torch.from_numpy(table),
                                       max_blocks_per_seq=width),
        }
        sm = _swa_decode_mask(W, md, torch.from_numpy(p), 2, "layers.0.self_attn")
        valid = _mask_to_slots(sm, B, nb * BS).numpy() > 0
        for b in range(B):
            assert valid[b, nb * BS - 1], "the active slot must be valid"
            slots = (bt[b].numpy().astype(np.int64)[:, None] * BS + np.arange(BS)).reshape(-1)
            prior = valid[b].copy()
            prior[-1] = False
            got = set(cache[slots[prior]].tolist()) | {int(p[b])}
            exp = set(range(max(0, int(p[b]) - W + 1), int(p[b]) + 1))
            assert got == exp, (L, B, b, int(p[b]), sorted(got ^ exp)[:6])
        # Write after the read (in-kernel update / deferred scatter order).
        write(p, np.arange(B, dtype=np.int64))
        ctx += 1


def test_ring_refuses_prefill_continuation():
    R_cls = _runner_cls()
    swa_spec, full_spec = _specs()
    table = np.arange(1, 1 + 128).reshape(1, 128).astype(np.int32)
    fake = _fake_runner(table, np.zeros(1, dtype=np.int32), swa_spec, full_spec)
    pos = np.arange(512, 1024)
    with pytest.raises(RuntimeError, match="does not start at position 0"):
        R_cls._apply_swa_ring_slot_mapping(fake, pos, np.zeros(len(pos), dtype=np.int64),
                                           np.array([len(pos)]))


def test_ring_off_swa_num_blocks_unchanged(monkeypatch):
    """Flag off: _compute_swa_num_blocks keeps the SWA_ALIGN 1024 width (64 blocks)."""
    R_cls = _runner_cls()
    monkeypatch.setenv("VLLM_NEURON_SWA_ALIGN_TOKENS", "1024")
    fake = types.SimpleNamespace(_dcp_size=1, max_model_len=ML, _decode_token_threshold=lambda: 1)
    assert R_cls._compute_swa_num_blocks(fake, W, BS, 155) == 64
    fake._swa_ring = True
    assert R_cls._compute_swa_num_blocks(fake, W, BS, 155) == 32
