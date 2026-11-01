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
        sel_ctx, sel_own = res[r][1][li][:2]
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


# Output tolerance per reduce-scatter payload dtype. fp32: the merge is exact up
# to summation order. bf16: the summed partials (unit-scale after the global-max
# rescale) are rounded once per rank to 8 mantissa bits (2^-9 relative), and the
# harness sums the bf16 values rank by rank in bf16; the measured worst case is
# printed by the test (-s) and stays well under this.
TOL = {"fp32": 2e-5, "bf16": 1.5e-2}
WORST: dict = {}


@pytest.mark.parametrize("mode,wire,pack", [
    ("keysplit", "fp32", 576), ("keysplit", "bf16", 576), ("keysplit", "bf16", 513),
    ("keysplit", "bf16", 0), ("keysplit", "fp32", 0), ("gather", "fp32", 576),
])
@pytest.mark.parametrize("world,order_seed,chunk_blocks,topk,qblock", [
    (4, 2, 2, 20, 256),  # chunk 32
    (4, 3, 2, 20, 8),  # query blocking
    (8, 1, 1, 24, 256),  # chunk 32
    (8, 4, 2, 200, 16),  # top-k beyond the keys: everything visible
])
@torch.no_grad()
def test_prefill_then_decode_dcp(
    monkeypatch, world, order_seed, chunk_blocks, topk, qblock, mode, wire, pack
):
    torch.set_num_threads(2)
    cfg = make_config(torch.float32)
    cfg.index_topk = topk
    chunk = chunk_blocks * BS * world
    n_chunks = 3
    # A whole number of virtual blocks, so the replicated table spans exactly the
    # positions the DCP tables do (same key count on both sides).
    vbs = BS * world
    total = -(-(n_chunks * chunk + STEPS + 1) // vbs) * vbs
    g = torch.Generator().manual_seed(1000 + world + order_seed)
    seq = Seq(total, world, g)

    monkeypatch.setattr(model, "DEC_CP", False)
    monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", False)
    monkeypatch.setattr(model, "DCP_PREFILL_QBLOCK", qblock)
    monkeypatch.setattr(model, "DCP_RS_DTYPE", wire)
    monkeypatch.setattr(model, "DCP_RS_PACK", pack)
    monkeypatch.setattr(model, "DCP_PREFILL", mode)
    tol = TOL[wire]
    worst = 0.0
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
    rep = None
    if mode == "gather":
        # The replicated path on the same W ranks (dcp = 1, each a full copy of
        # the reference cache): what the gathered prefill must equal bit for bit.
        rep_hub = Hub(world, order, rs_slot=rs_slot)
        rep = [[build_rank(cfg, a, world, r, rep_hub) for a in ref_layers] for r in range(world)]

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
        if rep is not None:
            rep_res = run_ranks(rep, rope, x, pos, [ref_meta] * world)
        for li in range(2):
            if mode == "gather":  # the replicated path itself
                for r in range(world):
                    assert torch.equal(res[r][1][li], ref_sel[li]), (c, li)
                    # Same keys, same code, same ranks: bit-identical.
                    assert torch.equal(res[r][1][li], rep_res[r][1][li]), (c, li)
                    assert torch.equal(res[r][0][li], rep_res[r][0][li]), (c, li)
            else:
                sel = assemble_prefill_sel(res, li, world, chunk, ref_span, prior)
                assert torch.equal(sel, ref_sel[li]), (c, li)
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                e = rel(res[r][0][li], ref_out[li])
                worst = max(worst, e)
                assert e < tol, (c, li, e)

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
                e = rel(res[r][0][li], ref_out[li])
                worst = max(worst, e)
                assert e < tol, (step, li, e)
    key = (mode, wire, pack)
    WORST[key] = max(WORST.get(key, 0.0), worst)
    print(f"\nDCP prefill+decode {mode} {wire}/{pack} W={world}: max rel err {worst:.3e}")


@pytest.mark.parametrize("bits,qblock", [(1, 0), (5, 2), (8, 0), (8, 3), (7, 0)])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_threshold_dist_matches_single(bits, qblock, ties):
    """Disjoint key shards over W ranks give ``topk_threshold`` of their union, bit
    for bit, for any number of bisection steps per all-reduce (24 = 24, 3x8, 5x5-ish
    rounds); with heavy ties the selected set (all keys >= lo) is the same too."""
    world, rows, n1, n2, k = 4, 5, 24, 8, 30
    g = torch.Generator().manual_seed(3 + bits)
    if ties:  # few distinct values, ties at the k-th score
        s1 = [torch.randint(0, 6, (rows, n1), generator=g).float() * 0.5 for _ in range(world)]
        s2 = [torch.randint(0, 6, (rows, n2), generator=g).float() * 0.5 for _ in range(world)]
    else:
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
        out[r] = model.topk_threshold_dist(
            s1[r], v1[r], s2[r], v2[r], k, HubGroup(world, r, hub), world,
            bits=bits, qblock=qblock,
        )

    ts = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    want_sel = v_all & (s_all >= want)
    for r in range(world):
        assert torch.equal(out[r], want)
        got_sel = torch.cat(
            [v1[q] & (s1[q] >= out[r]) for q in range(world)]
            + [v2[q] & (s2[q] >= out[r]) for q in range(world)], dim=1,
        )
        assert torch.equal(got_sel, want_sel)
    # One all-gather for the bracket, one all-reduce per round.
    assert hub.calls == 1 + -(-model.TOPK_BISECTION_STEPS // bits)


@pytest.mark.parametrize("pack,want_shared", [(576, 4), (0, 5)])
@torch.no_grad()
def test_dcp_prefill_collective_count(monkeypatch, pack, want_shared):
    """Collectives per layer: full = bracket + 24 / bits rounds + 2 route + q + m +
    reduce-scatter(s); shared = q + m + reduce-scatter(s); each plus the o_proj
    all-reduce every path has; independent of the query blocking."""
    world, chunk = 4, 2 * BS * 4
    cfg = make_config(torch.float32)
    cfg.index_topk = 20
    g = torch.Generator().manual_seed(5)
    seq = Seq(3 * chunk, world, g)
    monkeypatch.setattr(model, "DCP_PREFILL", "keysplit")
    monkeypatch.setattr(model, "DCP_PREFILL_QBLOCK", 8)
    monkeypatch.setattr(model, "DCP_RS_PACK", pack)
    ref_layers = build_reference(cfg)
    hub = Hub(world, [1, 3, 0, 2], rs_slot=[2, 0, 3, 1])
    monkeypatch.setattr(model, "_dcp_world", lambda: world)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    for rk in ranks:
        for a in rk:
            a.kv_pages = torch.randn(seq.dw + 4, 1, BS, a.entry_dim, generator=g)
    rope = model.HYV4RotaryEmbedding(cfg)
    prior = chunk
    pos = torch.arange(prior, prior + chunk)
    x = torch.randn(chunk, HIDDEN, generator=g)
    metas = [seq.dcp_meta(pos, prior, r) for r in range(world)]
    rs = 1 if pack else 2
    rounds = -(-model.TOPK_BISECTION_STEPS // model.DCP_TOPK_BITS)
    for li, want in ((0, 1 + rounds + 2 + 2 + rs + 1), (1, want_shared)):
        hub.calls = 0
        res = [None] * world

        def body(r, li=li):
            a = ranks[r][li]
            sel = None if li == 0 else body.sel[r]
            out, sel = a.forward(x, pos, rope(pos), {a.layer_name: metas[r]}, sel, torch.tensor(r))
            res[r] = sel

        if li == 1:
            body.sel = body_sel
        ts = [threading.Thread(target=body, args=(r,)) for r in range(world)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        body_sel = res
        assert hub.calls == want, (li, hub.calls, want)


@pytest.mark.parametrize("backend", ["v1", "cte"])
@pytest.mark.parametrize("world,order_seed", [(4, 6), (8, 7)])
@torch.no_grad()
def test_gather_mode_sparse_prefill(monkeypatch, backend, world, order_seed):
    """``HY4_DCP_PREFILL=gather`` through the query-split sparse prefill (v1 and
    cte backends, all-to-all in shuffled order): bit-identical to the same ranks
    running it on a replicated cache, chunk after chunk, then decode."""
    from test_decode_cp import hub_all_to_all

    torch.set_num_threads(2)
    cfg = make_config(torch.float32)
    cfg.index_topk = 40
    chunk = 128
    n_chunks = 3
    # Spans in whole 128-key tiles (the sparse path's requirement).
    total = -(-(n_chunks * chunk + STEPS + 1) // 128) * 128
    g = torch.Generator().manual_seed(77 + world)
    seq = Seq(total, world, g)
    monkeypatch.setattr(model, "DEC_CP", False)
    monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", True)
    monkeypatch.setattr(model, "PREFILL_SPARSE_MIN_SPAN", 0)
    monkeypatch.setattr(model, "PREFILL_KSEL", 128)
    monkeypatch.setattr(model, "DCP_PREFILL", "gather")
    monkeypatch.setattr(model, "_all_to_all", hub_all_to_all)
    ref_layers = build_reference(cfg)
    for a in ref_layers:
        a.kv_pages = torch.randn(seq.ref_w + 4, 1, BS, a.entry_dim, generator=g)
    rope = model.HYV4RotaryEmbedding(cfg)
    gen = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(world, generator=gen).tolist()
    a2a = (torch.randperm(world, generator=gen).tolist(), torch.randperm(world, generator=gen).tolist())
    hub = Hub(world, order, *a2a)
    rep_hub = Hub(world, order, *a2a)
    rep = [[build_rank(cfg, a, world, r, rep_hub) for a in ref_layers] for r in range(world)]
    monkeypatch.setattr(model, "_dcp_world", lambda: world)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    monkeypatch.setattr(model, "_dcp_world", lambda: 1)
    for rk, rp in zip(ranks, rep):
        for a, b in zip(rk, rp):
            a.kv_pages = torch.randn(seq.dw + 4, 1, BS, a.entry_dim, generator=g)
            a.prefill_backend = b.prefill_backend = backend
    for c in range(n_chunks):
        prior = c * chunk
        pos = torch.arange(prior, prior + chunk)
        x = torch.randn(chunk, HIDDEN, generator=g)
        got = run_ranks(ranks, rope, x, pos, [seq.dcp_meta(pos, prior, r) for r in range(world)])
        want = run_ranks(rep, rope, x, pos, [seq.ref_meta(pos, prior)] * world)
        for r in range(world):
            for li in range(2):
                assert torch.equal(got[r][0][li], want[r][0][li]), (c, li, r)
                assert isinstance(got[r][1][li], tuple), "sparse path not taken"
                for u, v in zip(got[r][1][li], want[r][1][li]):
                    assert u is None and v is None or torch.equal(u, v)
    monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", False)
    for step in range(STEPS):
        p = n_chunks * chunk + step
        pos = torch.tensor([p])
        x = torch.randn(1, HIDDEN, generator=g)
        got = run_ranks(ranks, rope, x, pos, [seq.dcp_meta(pos, p, r) for r in range(world)])
        want = run_ranks(rep, rope, x, pos, [seq.ref_meta(pos, p)] * world)
        for r in range(world):
            for li in range(2):
                assert rel(got[r][0][li], want[r][0][li]) < 2e-5, (step, li)
