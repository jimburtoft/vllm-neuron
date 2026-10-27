# SPDX-License-Identifier: Apache-2.0
"""Query-split sparse prefill (``HY4_PREFILL_SPARSE_NKI``) == head-split dense prefill.

W ranks run as threads (the ``test_decode_cp`` harness: shuffled all-gather
block order, summing all-reduce). Two layers -- "full" then "shared" -- for a
first chunk (``cached_seq_len = 0``) and later chunks. The sparse path's row
lists must select exactly the dense path's keys, and the outputs must agree
with the dense head-split run and with the one-rank reference.
"""

from __future__ import annotations

import threading

import pytest
import torch

from hy4_stubs import Group, model, set_group
from test_decode_cp import BLOCK, HIDDEN, PAGES, WIDTH, Hub, build_rank, build_reference, make_config, rel

CHUNK = 128  # keys = WIDTH * BLOCK + CHUNK = 640, a multiple of 128


def inputs(cfg, cached, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(CHUNK, HIDDEN, generator=g).to(cfg.torch_dtype)
    table = (torch.randperm(PAGES - 1, generator=g)[:WIDTH] + 1).view(1, -1).to(torch.int32)
    pos = torch.arange(cached, cached + CHUNK)
    slots = table[0, pos // BLOCK].long() * BLOCK + pos % BLOCK
    meta = {
        "cached_seq_len": torch.tensor([[cached]], dtype=torch.int32),
        "block_table_tensor": table, "slot_mapping": slots,
        "max_query_len": CHUNK, "decode_token_threshold": 1,
    }
    return x, pos, meta


def run(ranks, rope, x, pos, meta):
    world = len(ranks)
    res, errs = [None] * world, []

    def body(r):
        try:
            outs, sels, selected = [], [], None
            for a in ranks[r]:
                saved = a.kv_pages.clone()
                rk = torch.tensor(r, dtype=torch.int32)
                out, selected = a.forward(x, pos, rope(pos), {a.layer_name: meta}, selected, rk)
                a.kv_pages = saved
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


def rows_to_mask(rows, n_keys):
    q = rows.shape[0]
    mask = torch.zeros(q, n_keys + 1, dtype=torch.bool)
    mask[torch.arange(q).view(-1, 1), rows.long().clamp(max=n_keys)] = True
    return mask[:, :n_keys]


@pytest.mark.parametrize("backend", ["v1", "cte-skip", "cte-clamp"])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("world,order_seed", [(4, 2), (8, 1)])
@torch.no_grad()
def test_prefill_sparse_matches_dense(monkeypatch, dtype, tol, world, order_seed, backend):
    torch.set_num_threads(2)
    monkeypatch.setattr(model, "PREFILL_KSEL", 128)  # TOPK = 64 plus ties
    cfg = make_config(dtype)
    ref_layers = build_reference(cfg)
    rope = model.HYV4RotaryEmbedding(cfg)
    order = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed)).tolist()
    hub = Hub(world, order)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    for layers in ranks:
        for a in layers:
            a.prefill_backend = backend.split("-")[0]
            a.prefill_cte_oob = backend.split("-")[-1]
    per = CHUNK // world
    n_keys = WIDTH * BLOCK + CHUNK
    for ci, cached in enumerate([0, 40, 300, WIDTH * BLOCK - CHUNK]):
        x, pos, meta = inputs(cfg, cached, seed=200 + ci)
        monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", False)
        set_group(Group())
        ref = [None, None]
        selected = None
        for li, a in enumerate(ref_layers):
            saved = a.kv_pages.clone()
            out, selected = a.forward_prefill(x, rope(pos), meta, selected, torch.tensor(0))
            a.kv_pages = saved
            ref[li] = (out.float(), selected)
        dense = run(ranks, rope, x, pos, meta)
        monkeypatch.setattr(model, "PREFILL_SPARSE_NKI", True)
        monkeypatch.setattr(model, "PREFILL_SPARSE_MIN_SPAN", 0)
        sparse = run(ranks, rope, x, pos, meta)
        for li in range(2):
            for r in range(world):
                rows, bound, _ = sparse[r][1][li]
                assert rows.shape == (per, 128) and rows.dtype == torch.int32
                want = dense[r][1][li][r * per : (r + 1) * per]
                assert torch.equal(want, ref[li][1][r * per : (r + 1) * per])
                assert torch.equal(rows_to_mask(rows, n_keys), want), (cached, li, r)
                assert torch.equal(bound > 0, rows < n_keys)
                assert torch.equal(sparse[r][0][li], sparse[0][0][li]), "ranks disagree"
                assert rel(sparse[r][0][li], dense[r][0][li]) < tol, (cached, li)
                assert rel(sparse[r][0][li], ref[li][0]) < tol, (cached, li)


def test_prefill_rows_exact():
    """Row lists: every selected key, ascending, then the out-of-range pad."""
    set_group(Group())
    a = model.HYV4Attention(make_config(torch.float32), 0)
    g = torch.Generator().manual_seed(0)
    q, n_keys = 6, 1024
    sel = torch.rand(q, n_keys, generator=g) < 0.15
    sel[1] = False
    sel[2, :] = False
    sel[2, -1] = True
    sel[3, :256] = True  # more than the slots: the first ksel kept
    old = model.PREFILL_KSEL
    try:
        model.PREFILL_KSEL = 200  # -> 256 slots
        rows, bound = a._prefill_rows(sel[:, :768], sel[:, 768:])
    finally:
        model.PREFILL_KSEL = old
    assert rows.shape == (q, 256)
    for i in range(q):
        want = torch.nonzero(sel[i]).flatten()[:256].to(torch.int32)
        n = want.numel()
        assert torch.equal(rows[i, :n], want), i
        assert (rows[i, n:] == n_keys).all()
        assert (bound[i, :n] > 0).all() and (bound[i, n:] < 0).all()


class _FakeSelectRows:
    """``select_to_rows``'s contract in torch: ``[Q, ksel + chunk]`` rows, pad ``K``."""

    def __getitem__(self, grid):
        def run(sel, sel2, ksel, chunk=2048):
            sel = torch.cat((sel, sel2), dim=1)
            q, k = sel.shape
            out = torch.full((q, ksel + min(chunk, k)), k, dtype=torch.int32)
            for i in range(q):
                nz = torch.nonzero(sel[i]).flatten()[:ksel]
                out[i, : nz.numel()] = nz.to(torch.int32)
            return out

        return run


@torch.no_grad()
def test_prefill_rows_kernel_dispatch_same_set(monkeypatch):
    """The device branch of ``_prefill_rows`` (kernel output sliced, bound from
    ``rows < K``) selects the same set per query as the torch reference."""
    set_group(Group())
    a = model.HYV4Attention(make_config(torch.float32), 0)
    a._select_rows_k = _FakeSelectRows()
    g = torch.Generator().manual_seed(3)
    q, n_keys = 8, 2048
    s = torch.round(torch.randn(q, n_keys, generator=g) * 2) / 2  # ties
    vis = torch.rand(q, n_keys, generator=g) < 0.8
    vis[0] = False
    vis[0, 5] = True
    sel = model.select_topk(s, vis, 300)
    monkeypatch.setattr(model, "PREFILL_KSEL", 384)
    ref_rows, ref_bound = a._prefill_rows_ref(sel)
    monkeypatch.setattr(model, "_use_sparse_kernels", lambda t: True)
    rows, bound = a._prefill_rows(sel[:, :1536], sel[:, 1536:])
    assert rows.shape == ref_rows.shape and rows.dtype == torch.int32
    for i in range(q):
        assert set(rows[i][bound[i] > 0].tolist()) == set(ref_rows[i][ref_bound[i] > 0].tolist())
    assert torch.equal(bound > 0, ref_bound > 0)


@pytest.mark.parametrize("style", ["interleaved", "neox"])
def test_apply_rope_nocat_bitwise(monkeypatch, style):
    monkeypatch.setattr(model, "ROPE_STYLE", style)
    g = torch.Generator().manual_seed(1)
    for shape in [(7, 64), (7, 3, 64)]:
        for dt in (torch.float32, torch.bfloat16):
            x = torch.randn(*shape, generator=g).to(dt)
            ang = torch.randn(7, 32, generator=g) * 3
            cos, sin = ang.cos(), ang.sin()
            assert torch.equal(model.apply_rope_nocat(x, cos, sin), model.apply_rope(x, cos, sin))


@pytest.mark.parametrize("oob", ["skip", "clamp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_prefill_cte_matches_v1(oob, dtype):
    """``_prefill_attn_cte`` (flat per-slot rows from the position tables, then the
    ``sparse_prefill_attn`` contract) == ``sparse_prefill_ref`` (position mode):
    prefix rows past the cached length, chunk rows, pads, a query with nothing
    selected and one with chunk keys only."""
    set_group(Group())
    a = model.HYV4Attention(make_config(dtype), 0)
    a.prefill_cte_oob = oob
    g = torch.Generator().manual_seed(5)
    a.kv_pages = torch.randn(PAGES, 1, BLOCK, a.entry_dim, generator=g).to(dtype)
    pages, bs, width = a._num_pages(), a._block_size(), a.kv_pages.shape[-1]
    span, chunk, q, ksel, nh = 4 * bs, 128, 6, 128, a.num_heads
    table = (torch.randperm(pages - 1, generator=g)[: span // bs] + 1).view(1, -1).to(torch.int32)
    prior = torch.tensor([[span - 7]])
    pos_map = a._prefill_pos_map(table, prior)
    n_keys = span + chunk
    rows = torch.full((q, ksel), n_keys, dtype=torch.int32)
    for i in range(q):
        n = [0, 40, 128, 9, 77, 100][i]
        pool = torch.arange(span, n_keys) if i == 4 else torch.arange(n_keys)
        if oob == "clamp":  # the model never selects an unmapped prefix position (vis_ctx)
            pool = pool[(pool < prior.item()) | (pool >= span)]
        pick = pool[torch.randperm(pool.numel(), generator=g)[:n]].sort().values
        rows[i, :n] = pick.to(torch.int32)
    # Skip mode: prefix positions at or past ``prior`` may be selected (unmapped -> zeros, as v1).
    bound = torch.where(rows < n_keys, torch.tensor(1e30), torch.tensor(-1e30)).float()
    lat = a.kv_lora_rank
    ql = torch.randn(q, nh, lat, generator=g)
    qr = torch.randn(q, nh, a.rope_dim, generator=g)
    own_lat = torch.randn(chunk, lat, generator=g).to(a.kv_pages.dtype)
    own_rope = torch.randn(chunk, a.rope_dim, generator=g).to(a.kv_pages.dtype)
    col = a.latent_slice.start
    want = model.sparse_prefill_ref(
        ql, qr, a.kv_pages, pos_map, own_lat, own_rope, rows, bound, a.scaling, col, a.dtype
    )
    got = a._prefill_attn_cte(ql, qr, pos_map, own_lat, own_rope, rows, bound, col)
    sink = torch.randn(1, nh, generator=g)

    def fold(m, den, num):  # forward_prefill_sparse's sink fold
        m_f = torch.maximum(m, sink)
        sc = torch.exp(m - m_f)
        return num * sc.unsqueeze(-1) / (den * sc + torch.exp(sink - m_f)).unsqueeze(-1)

    # Raw m/den/num agree wherever a query has a selected key; a query with none
    # (all -BIG) differs raw in clamp mode (it sums row 0) but folds to zero.
    live = (bound > 0).any(dim=1)
    for w, x in zip(want, got):
        assert torch.allclose(w[live], x[live], rtol=1e-5, atol=1e-5)
    assert torch.allclose(fold(*want), fold(*got), rtol=1e-5, atol=1e-6)
    assert not live[0] and fold(*got)[0].abs().max() == 0
    assert width >= col + lat + a.rope_dim
