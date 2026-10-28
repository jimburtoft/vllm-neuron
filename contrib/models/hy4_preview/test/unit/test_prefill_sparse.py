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
from test_decode_cp import BLOCK, HIDDEN, PAGES, TOPK, WIDTH, Hub, hub_all_to_all, build_rank, build_reference, make_config, rel

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


@pytest.mark.parametrize("backend", ["v1", "cte"])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("world,order_seed", [(4, 2), (8, 1)])
@torch.no_grad()
def test_prefill_sparse_matches_dense(monkeypatch, dtype, tol, world, order_seed, backend):
    n_exact = 0
    torch.set_num_threads(2)
    monkeypatch.setattr(model, "PREFILL_KSEL", 128)  # TOPK = 64 plus ties
    cfg = make_config(dtype)
    ref_layers = build_reference(cfg)
    rope = model.HYV4RotaryEmbedding(cfg)
    gen = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(world, generator=gen).tolist()
    # Independent shuffles for the all-to-all's block order and slot assignment.
    hub = Hub(world, order, torch.randperm(world, generator=gen).tolist(),
              torch.randperm(world, generator=gen).tolist())
    monkeypatch.setattr(model, "_all_to_all", hub_all_to_all)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    for layers in ranks:
        for a in layers:
            a.prefill_backend = backend
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
        tok_ok = torch.ones(CHUNK, dtype=torch.bool)  # token's selection exact in both layers
        for li in range(2):
            for r in range(world):
                rows, bound, _ = sparse[r][1][li]
                assert rows.shape == (per, 128) and rows.dtype == torch.int32
                # cte: rank r attends the query block of its all-to-all slot.
                b = hub.a2a_slot[r] if backend == "cte" else r
                want = dense[r][1][li][b * per : (b + 1) * per]
                assert torch.equal(want, ref[li][1][b * per : (b + 1) * per])
                if backend == "v1":
                    assert torch.equal(rows_to_mask(rows, n_keys), want), (cached, li, r)
                    assert torch.equal(bound > 0, rows < n_keys)
                else:
                    # Exactly min(TOPK, visible) keys, all inside the dense
                    # selection (which also keeps every key tied with the k-th).
                    live = torch.where(bound > 0, rows, torch.full_like(rows, n_keys))
                    got = rows_to_mask(live, n_keys)
                    assert not (got & ~want).any(), (cached, li, r)
                    n_vis = want.sum(dim=1).clamp(max=TOPK)
                    assert torch.equal(got.sum(dim=1), n_vis), (cached, li, r)
                    tok_ok[b * per : (b + 1) * per] &= (got == want).all(dim=1)
                assert torch.equal(sparse[r][0][li], sparse[0][0][li]), "ranks disagree"
        # Attention is per query: tokens whose selection is exact must match. A
        # token that dropped keys tied with the k-th index score (cte: exactly
        # k kept) attends a different set -- its selection is checked above
        # (subset, count), not its output. The toy indexer ties often (relu
        # zeros); realistic fp32 scores rarely do (test_prefill_topk_exact_no_ties).
        n_exact += int(tok_ok.sum())
        ok = tok_ok
        for li in range(2):
            for r in range(world):
                out = sparse[r][0][li]
                assert rel(out[ok], dense[r][0][li][ok]) < tol, (cached, li)
                assert rel(out[ok], ref[li][0][ok]) < tol, (cached, li)
    if backend == "v1":
        assert n_exact == 4 * CHUNK
    else:
        assert n_exact > 2 * CHUNK, "the exact check must cover most tokens"


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_prefill_cte_matches_v1(dtype):
    """``_prefill_attn_cte`` (flat per-slot rows from the position tables, then the
    ``sparse_prefill_attn`` contract) == ``sparse_prefill_ref`` (position mode):
    prefix rows past the cached length, chunk rows, pads, a query with nothing
    selected and one with chunk keys only."""
    set_group(Group())
    a = model.HYV4Attention(make_config(dtype), 0)
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
        pick = pool[torch.randperm(pool.numel(), generator=g)[:n]].sort().values
        rows[i, :n] = pick.to(torch.int32)
    # Skip mode: prefix positions at or past ``prior`` may be selected (unmapped -> zeros, as v1).
    bound = torch.where(rows < n_keys, torch.tensor(1e30), torch.tensor(-1e30)).float()
    lat = a.kv_lora_rank
    ql = torch.randn(q, nh, lat, generator=g)
    qr = torch.randn(q, nh, a.rope_dim, generator=g)
    # cte rounds the query to the cache dtype (bf16 scores); give v1 the same query.
    ql, qr = ql.to(dtype).float(), qr.to(dtype).float()
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
    # (all -BIG) is compared after the fold only.
    live = (bound > 0).any(dim=1)
    for w, x in zip(want, got):
        assert torch.allclose(w[live], x[live], rtol=1e-5, atol=1e-5)
    assert torch.allclose(fold(*want), fold(*got), rtol=1e-5, atol=1e-6)
    assert not live[0] and fold(*got)[0].abs().max() == 0
    assert width >= col + lat + a.rope_dim


@pytest.mark.parametrize("n_vis_min", [10, 3000])
@torch.no_grad()
def test_prefill_topk_exact_no_ties(monkeypatch, n_vis_min):
    """cte ``_prefill_topk`` on tie-free scores == the threshold selection
    (``select_topk``) exactly, as positions with bound ``+BIG``; pads and
    invisible picks carry ``-BIG``; a query with fewer than k visible keys
    keeps all of them."""
    set_group(Group())
    cfg = make_config(torch.float32)
    a = model.HYV4Attention(cfg, 0)
    a.index_topk = 2000  # -> 2048 slots, 48 pads
    g = torch.Generator().manual_seed(7)
    q, p, c = 8, 6144, 2048
    sc_ctx = torch.randn(q, p, generator=g)
    sc_own = torch.randn(q, c, generator=g)
    vis_ctx = (torch.arange(p) < 4500).view(1, p).expand(q, p).clone()
    vis_own = torch.arange(c).view(1, c) <= (100 + torch.arange(q)).view(q, 1)
    vis_ctx[0] = False
    vis_ctx[0, :n_vis_min] = True
    rows, bound = a._prefill_topk(sc_ctx, vis_ctx, sc_own, vis_own)
    assert rows.shape == (q, 2048) and rows.dtype == torch.int32
    want = model.select_topk(torch.cat((sc_ctx, sc_own), 1), torch.cat((vis_ctx, vis_own), 1), 2000)
    live = torch.where(bound > 0, rows, torch.full_like(rows, p + c))
    assert torch.equal(rows_to_mask(live, p + c), want)
    assert (rows[:, 2000:] == p + c).all() and (bound[:, 2000:] < 0).all()
