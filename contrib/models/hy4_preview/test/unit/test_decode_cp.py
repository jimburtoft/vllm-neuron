# SPDX-License-Identifier: Apache-2.0
"""Key-sharded decode attention (``HY4_DEC_CP``) == ``forward_decode_split``.

W ranks run as threads, each with its own ``HYV4Attention`` holding its heads'
weights and its own (replicated) copy of the paged cache. The fake group's
all_gather returns blocks in a shuffled rank order (as on Trn2) and all_reduce
sums. Two layers -- a "full" indexer layer, then a "shared" one reusing its
selection -- are compared against the unsharded one-rank reference: output,
DSA selection (reassembled from the ranks' local masks), and rank agreement.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from hy4_stubs import Group, model, set_group

HEADS, HIDDEN, BLOCK, WIDTH, TOPK = 8, 64, 16, 32, 64
SPAN = BLOCK * WIDTH  # 512 keys
PAGES = WIDTH * 4 + 8

CASES = [  # (positions per row, dead rows)
    ([300], [False]),
    ([10], [False]),  # fewer visible keys than k
    ([63], [False]),
    ([64], [False]),
    ([65], [False]),  # shard boundaries at W=8 (64 keys per shard)
    ([127, 128, 511, 5], [False] * 4),
    ([200, 0, 450, 1], [False, True, False, True]),  # dead rows
]


class Hub:
    def __init__(self, world, order):
        self.order = order  # order[j] = rank at all-gather block j
        self.slots = [None] * world
        self.barrier = threading.Barrier(world)

    def exchange(self, rank, t):
        self.slots[rank] = t
        self.barrier.wait()
        got = list(self.slots)
        self.barrier.wait()
        return got


class HubGroup(Group):
    def __init__(self, world, rank, hub):
        self.world_size, self.rank_in_group, self.hub = world, rank, hub

    def all_reduce(self, t):
        got = self.hub.exchange(self.rank_in_group, t)
        return torch.stack([g.float() for g in got]).sum(0).to(t.dtype)

    def all_gather(self, t, dim=0):
        got = self.hub.exchange(self.rank_in_group, t)
        return torch.cat([got[j] for j in self.hub.order], dim=dim)


def make_config(dtype):
    c = SimpleNamespace(
        torch_dtype=dtype, hidden_size=HIDDEN, num_attention_heads=HEADS, q_lora_rank=32,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        index_topk=TOPK, index_n_heads=4, index_head_dim=16, indexer_types=("full", "shared"),
        indexer_rope_first=False, rms_norm_eps=1e-6, gated_mla=True, learnable_sink=True,
        rope_theta=10000.0, neuron_config=None,
    )
    c.qk_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim
    c.kv_latent_dim = c.kv_lora_rank + c.qk_rope_head_dim
    return c


def build_reference(cfg, seed=7):
    set_group(Group())
    torch.manual_seed(seed)
    layers = []
    for li in range(2):
        a = model.HYV4Attention(cfg, li)
        with torch.no_grad():
            for name, p in a.named_parameters():
                if "norm" in name and "weight" in name:
                    p.fill_(1.0)
                elif "bias" in name:
                    p.zero_()
                elif name == "sinks":
                    p.copy_(torch.randn_like(p) * 2.0)
                else:
                    p.copy_(torch.randn(p.shape) * 0.25)
        a.kv_pages = (torch.randn(PAGES, 1, BLOCK, a.entry_dim) * 1.5).to(cfg.torch_dtype)
        layers.append(a)
    return layers


SHARDED = {  # parameter -> (dim, per-head size)
    "q_b_proj_weight": (1, "qk"), "gate_proj_weight": (1, "v"), "o_proj_weight": (0, "v"),
    "w_uk": (0, 1), "w_uv": (0, 1), "sinks": (0, 1),
}


def build_rank(cfg, ref, world, rank, hub):
    set_group(HubGroup(world, rank, hub))
    a = model.HYV4Attention(cfg, ref.layer_idx)
    nh = HEADS // world
    sizes = {"qk": cfg.qk_head_dim, "v": cfg.v_head_dim, 1: 1}
    src = dict(ref.named_parameters())
    with torch.no_grad():
        for name, p in a.named_parameters():
            full = src[name]
            if name in SHARDED:
                dim, per = SHARDED[name]
                n = nh * sizes[per]
                full = full.narrow(dim, rank * n, n)
            p.copy_(full)
    a.kv_pages = ref.kv_pages.clone()
    return a


def inputs(cfg, positions, dead, seed):
    reqs = len(positions)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(reqs, HIDDEN, generator=g).to(cfg.torch_dtype)
    table = torch.stack(
        [torch.randperm(PAGES - 1, generator=g)[:WIDTH] + 1 for _ in range(reqs)]
    ).to(torch.int32)
    pos = torch.tensor(positions, dtype=torch.int32)
    slots = table[torch.arange(reqs), (pos // BLOCK).long()].long() * BLOCK + (pos % BLOCK).long()
    slots = torch.where(torch.tensor(dead), torch.full_like(slots, -1), slots)
    meta = {
        "block_table_tensor": table, "slot_mapping": slots,
        "max_query_len": 1, "decode_token_threshold": 1,
    }
    return x, pos, meta


def run_reference(layers, rope, x, pos, meta):
    outs, sels, selected = [], [], None
    for a in layers:
        saved = a.kv_pages.clone()
        out, selected = a.forward_decode_split(x, pos, rope(pos), meta, selected)
        a.kv_pages = saved
        outs.append(out.float())
        sels.append(selected)
    return outs, sels


def run_cp(ranks, rope, x, pos, meta, use_forward):
    world = len(ranks)
    res, errs = [None] * world, []

    def body(r):
        try:
            outs, sels, selected = [], [], None
            for a in ranks[r]:
                saved = a.kv_pages.clone()
                rk = torch.tensor(r, dtype=torch.int32)
                if use_forward:  # dispatch through forward() under DEC_CP
                    out, selected = a.forward(x, pos, rope(pos), {a.layer_name: meta}, selected, rk)
                else:
                    out, selected = a.forward_decode_cp(x, pos, rope(pos), meta, selected, rk)
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


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("world,order_seed", [(8, 1), (4, 2), (2, 3)])
@torch.no_grad()
def test_decode_cp_matches_split(monkeypatch, dtype, tol, world, order_seed):
    torch.set_num_threads(2)
    cfg = make_config(dtype)
    ref_layers = build_reference(cfg)
    rope = model.HYV4RotaryEmbedding(cfg)
    order = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed)).tolist()
    hub = Hub(world, order)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    assert ranks[0][0].cp_supported(WIDTH)
    for ci, (positions, dead) in enumerate(CASES):
        x, pos, meta = inputs(cfg, positions, dead, seed=100 + ci)
        monkeypatch.setattr(model, "DEC_CP", False)
        ref_out, ref_sel = run_reference(ref_layers, rope, x, pos, meta)
        monkeypatch.setattr(model, "DEC_CP", True)
        res = run_cp(ranks, rope, x, pos, meta, use_forward=(ci == 0))
        for li in range(2):
            local = [res[r][1][li] for r in range(world)]
            for r in range(1, world):
                assert torch.equal(local[r][1], local[0][1]), "own-token bit differs"
            sel = torch.cat([local[r][0] for r in range(world)] + [local[0][1]], dim=1)
            assert torch.equal(sel, ref_sel[li]), (positions, li)
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                assert rel(res[r][0][li], ref_out[li]) < tol, (positions, li)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize(
    "world,order_seed,kloc,pad",
    [(8, 1, 0, "oob"), (4, 2, 0, "oob"), (2, 3, 0, "null"), (2, 4, 128, "oob"), (4, 5, 64, "null")],
)
@torch.no_grad()
def test_decode_cp_sparse_matches(monkeypatch, dtype, tol, world, order_seed, kloc, pad):
    """``HY4_DEC_SPARSE_NKI`` (torch references of the kernels) == dense CP == split.

    Same selection bit for bit; output against the split reference at ``tol`` and
    against the dense key-sharded path. ``kloc`` > 0 exercises the compaction
    (exact here: at most ``TOPK`` keys are selected, so no rank exceeds 64)."""
    torch.set_num_threads(2)
    cfg = make_config(dtype)
    ref_layers = build_reference(cfg)
    rope = model.HYV4RotaryEmbedding(cfg)
    order = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed)).tolist()
    hub = Hub(world, order)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    monkeypatch.setattr(model, "SPARSE_KLOC", kloc)
    monkeypatch.setattr(model, "SPARSE_PAD_ROW", pad)
    for ci, (positions, dead) in enumerate(CASES):
        x, pos, meta = inputs(cfg, positions, dead, seed=100 + ci)
        monkeypatch.setattr(model, "DEC_CP", False)
        ref_out, ref_sel = run_reference(ref_layers, rope, x, pos, meta)
        monkeypatch.setattr(model, "DEC_CP", True)
        monkeypatch.setattr(model, "DEC_SPARSE_NKI", False)
        dense = run_cp(ranks, rope, x, pos, meta, use_forward=False)
        monkeypatch.setattr(model, "DEC_SPARSE_NKI", True)
        res = run_cp(ranks, rope, x, pos, meta, use_forward=(ci == 0))
        for li in range(2):
            for r in range(world):
                assert torch.equal(res[r][1][li][0], dense[r][1][li][0]), "selection differs"
                assert torch.equal(res[r][1][li][1], dense[r][1][li][1]), "own bit differs"
            local = [res[r][1][li] for r in range(world)]
            sel = torch.cat([local[r][0] for r in range(world)] + [local[0][1]], dim=1)
            assert torch.equal(sel, ref_sel[li]), (positions, li)
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                assert rel(res[r][0][li], ref_out[li]) < tol, (positions, li)
                assert rel(res[r][0][li], dense[r][0][li]) < tol, (positions, li)


def test_sparse_rows_compaction_exact():
    """The one-hot compaction keeps every selected row, in order, and pads the rest."""
    set_group(Group())
    cfg = make_config(torch.float32)
    a = model.HYV4Attention(cfg, 0)
    a.kv_pages = torch.zeros(PAGES, 1, BLOCK, a.entry_dim)
    g = torch.Generator().manual_seed(0)
    reqs, shard = 3, 256
    rows = torch.randint(0, PAGES * BLOCK, (reqs, shard), generator=g).to(torch.int32)
    sel = torch.rand(reqs, shard, generator=g) < 0.3
    sel[2] = False  # a request with nothing selected on this rank
    old = model.SPARSE_KLOC
    try:
        model.SPARSE_KLOC = 100  # -> 128 slots
        out, bound = a._cp_sparse_rows(rows, sel)
    finally:
        model.SPARSE_KLOC = old
    assert out.shape == (reqs, 128)
    for i in range(reqs):
        want = rows[i][sel[i]]
        n = want.numel()
        assert n <= 128
        assert torch.equal(out[i, :n], want)
        assert (out[i, n:] == PAGES * BLOCK).all()
        assert (bound[i, :n] > 0).all() and (bound[i, n:] < 0).all()
