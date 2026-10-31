# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill, then decode, on a DCP-sharded cache == the replicated one-rank path.

A prompt of three chunks is prefilled chunk by chunk and then decoded a few
steps, twice: on one rank with a replicated paged cache (``forward_prefill`` /
``forward_decode_split``), and on W threaded ranks with ``dcp == W``, each rank's
cache written only through the slot mappings the runner would build
(``_compute_slot_mapping_cpu``, interleave = block size). Both write their
caches as they go, so later chunks and the decode steps read what the earlier
ones wrote. Outputs (fp32) and the DSA selections (exactly) must match.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from hy4_stubs import model
from test_decode_cp import HIDDEN, Hub, build_rank, make_config, rel
from test_decode_dcp import BS, NULL, build_reference

STEPS = 3  # decode steps after the prompt


def slots_dcp(table_row: np.ndarray, pos: np.ndarray, world: int, rank: int) -> np.ndarray:
    """``_compute_slot_mapping_cpu`` for one request's tokens at ``pos``."""
    vbs = BS * world
    blocks = table_row[pos // vbs]
    voff = pos % vbs
    local = (voff // BS) % world == rank
    slot = blocks * BS + voff // (world * BS) * BS + voff % BS
    return np.where(local, slot, NULL)


class Seq:
    """One request: its reference block ids and its DCP (virtual) block ids."""

    def __init__(self, total, world, g):
        self.ref_w = -(-total // BS)
        self.ref_ids = (torch.randperm(self.ref_w + 3, generator=g)[: self.ref_w] + 1).to(torch.int32)
        self.dw = -(-total // (BS * world))
        self.ids = (torch.randperm(self.dw + 3, generator=g)[: self.dw] + 1).to(torch.int32)
        self.world = world

    def ref_meta(self, pos, prior):
        p = torch.as_tensor(pos, dtype=torch.long)
        return {
            "block_table_tensor": self.ref_ids.view(1, -1),
            "slot_mapping": self.ref_ids[p // BS].long() * BS + p % BS,
            "cached_seq_len": torch.tensor([[prior]], dtype=torch.int32),
            "max_query_len": len(p), "decode_token_threshold": 1,
        }

    def dcp_meta(self, pos, prior, rank):
        p = np.asarray(pos, dtype=np.int64)
        return {
            "block_table_tensor": self.ids.view(1, -1),
            "slot_mapping": torch.from_numpy(slots_dcp(self.ids.numpy(), p, self.world, rank)),
            "cached_seq_len": torch.tensor([[prior]], dtype=torch.int32),
            "max_query_len": len(p), "decode_token_threshold": 1,
        }


def run_ranks(ranks, rope, x, pos, metas):
    """Every rank through ``forward`` over both layers; caches keep their writes."""
    world = len(ranks)
    res, errs = [None] * world, []

    def body(r):
        try:
            outs, sels, selected = [], [], None
            for a in ranks[r]:
                rk = torch.tensor(r, dtype=torch.int32)
                out, selected = a.forward(x, pos, rope(pos), {a.layer_name: metas[r]}, selected, rk)
                outs.append(out.float())
                sels.append(selected)
            res[r] = (outs, sels)
        except Exception as e:  # pragma: no cover
            errs.append(e)
            ranks[r][0].tp_group.hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise errs[0]
    return res


def assemble_prefill_sel(res, li, world, chunk, ref_span, prior):
    """Per-rank ``(sel_ctx, sel_own)`` -> the reference's ``[C, span + C]`` mask."""
    sel = torch.zeros(chunk, ref_span + chunk, dtype=torch.bool)
    for r in range(world):
        sel_ctx, sel_own = res[r][1][li]
        kp = model.HYV4Attention._dcp_key_pos
        # positions as the model computes them (bs and world from a stand-in)
        stub = type("S", (), {"_block_size": lambda self: BS, "world_size": world})()
        rk = torch.tensor([r])
        gp = kp(stub, sel_ctx.shape[1], rk, "cpu")
        ot = kp(stub, sel_own.shape[1], rk, "cpu")
        assert not sel_ctx[:, gp >= prior].any(), "selected past the prefix"
        keep = gp < ref_span
        sel[:, gp[keep]] |= sel_ctx[:, keep]
        ok = ot < chunk
        assert not sel_own[:, ~ok].any()
        sel[:, ref_span + ot[ok]] |= sel_own[:, ok]
    return sel


@pytest.mark.parametrize("world,order_seed,chunk_blocks,topk,qblock", [
    (4, 2, 2, 20, 256),  # chunk 32
    (4, 3, 2, 20, 8),  # query blocking
    (8, 1, 1, 24, 256),  # chunk 32
    (8, 4, 2, 200, 16),  # top-k beyond the keys: everything visible
])
@torch.no_grad()
def test_prefill_then_decode_dcp(monkeypatch, world, order_seed, chunk_blocks, topk, qblock):
    torch.set_num_threads(2)
    cfg = make_config(torch.float32)
    cfg.index_topk = topk
    chunk = chunk_blocks * BS * world
    n_chunks = 3
    total = n_chunks * chunk + STEPS + 1
    g = torch.Generator().manual_seed(1000 + world + order_seed)
    seq = Seq(total, world, g)

    monkeypatch.setattr(model, "DEC_CP", False)
    monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", False)
    monkeypatch.setattr(model, "DCP_PREFILL_QBLOCK", qblock)
    ref_layers = build_reference(cfg)
    for a in ref_layers:
        a.kv_pages = (torch.randn(seq.ref_w + 4, 1, BS, a.entry_dim, generator=g) * 1.5)
    rope = model.HYV4RotaryEmbedding(cfg)
    order = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed)).tolist()
    rs_slot = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed + 50)).tolist()
    hub = Hub(world, order, rs_slot=rs_slot)
    monkeypatch.setattr(model, "_dcp_world", lambda: world)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    monkeypatch.setattr(model, "_dcp_world", lambda: 1)
    for rk in ranks:
        for a in rk:
            a.kv_pages = torch.randn(seq.dw + 4, 1, BS, a.entry_dim, generator=g) * 1.5

    ref_span = seq.ref_w * BS
    for c in range(n_chunks):
        prior = c * chunk
        pos = torch.arange(prior, prior + chunk)
        x = torch.randn(chunk, HIDDEN, generator=g)
        ref_meta = seq.ref_meta(pos, prior)
        ref_out, ref_sel, selected = [], [], None
        for a in ref_layers:
            out, selected = a.forward(x, pos, rope(pos), {a.layer_name: ref_meta}, selected)
            ref_out.append(out.float())
            ref_sel.append(selected)
        metas = [seq.dcp_meta(pos, prior, r) for r in range(world)]
        res = run_ranks(ranks, rope, x, pos, metas)
        for li in range(2):
            sel = assemble_prefill_sel(res, li, world, chunk, ref_span, prior)
            assert torch.equal(sel, ref_sel[li]), (c, li)
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                assert rel(res[r][0][li], ref_out[li]) < 2e-5, (c, li)

    for step in range(STEPS):
        p = n_chunks * chunk + step
        pos = torch.tensor([p])
        x = torch.randn(1, HIDDEN, generator=g)
        ref_meta = seq.ref_meta(pos, p)
        ref_out, selected = [], None
        for a in ref_layers:
            out, selected = a.forward(x, pos, rope(pos), {a.layer_name: ref_meta}, selected)
            ref_out.append(out.float())
        metas = [seq.dcp_meta(pos, p, r) for r in range(world)]
        res = run_ranks(ranks, rope, x, pos, metas)
        for li in range(2):
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                assert rel(res[r][0][li], ref_out[li]) < 2e-5, (step, li)


def test_topk_threshold_dist_matches_single():
    """Disjoint key shards over W ranks give ``topk_threshold`` of their union, bit for bit."""
    world, rows, n1, n2, k = 4, 5, 24, 8, 30
    g = torch.Generator().manual_seed(3)
    s1 = [torch.randn(rows, n1, generator=g) for _ in range(world)]
    s2 = [torch.randn(rows, n2, generator=g) for _ in range(world)]
    v1 = [torch.rand(rows, n1, generator=g) < 0.7 for _ in range(world)]
    v2 = [torch.rand(rows, n2, generator=g) < 0.7 for _ in range(world)]
    s1[0][0, 0] = s1[1][0, 3] = 0.25  # a tie across ranks
    s_all = torch.cat(s1 + s2, dim=1)
    v_all = torch.cat(v1 + v2, dim=1)
    want = model.topk_threshold(s_all, v_all, k)
    from test_decode_cp import HubGroup

    hub = Hub(world, [2, 0, 3, 1])
    out = [None] * world

    def body(r):
        out[r] = model.topk_threshold_dist(s1[r], v1[r], s2[r], v2[r], k, HubGroup(world, r, hub), world)

    ts = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for r in range(world):
        assert torch.equal(out[r], want)
