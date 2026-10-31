# SPDX-License-Identifier: Apache-2.0
"""Decode with a DCP-sharded latent cache (``decode_context_parallel_size == tp``).

Each of W threaded ranks holds only its own key shard, laid out as the plugin's
runner lays it out (``_compute_slot_mapping_cpu`` with interleave = block size:
token ``t`` is owned by rank ``(t // bs) % W`` and stored in that rank's copy of
block ``table[t // (bs * W)]`` at offset ``t % bs``; the block table is shared and
sized per virtual block of ``bs * W`` tokens; non-owned writes go to the null
block). The output and DSA selection must equal the unsharded one-rank
``forward_decode_split`` over a replicated cache holding the same tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hy4_stubs import Group, model, set_group
from test_decode_cp import HEADS, HIDDEN, Hub, build_rank, make_config, rel, run_cp

BS = 4
SPAN = 256  # keys per request (the reference block table covers it)
NULL = 0  # the plugin's NULL_BLOCK_ID

CASES = [  # (positions per row, dead rows)
    ([255], [False]),
    ([37], [False]),  # fewer visible keys than k; not a multiple of W * bs
    ([16, 33], [False, False]),
    ([130, 0, 201, 75], [False, False, False, True]),  # position 0 and a padded row
    ([3, 250, 64, 1], [False, False, True, False]),
]


def slot_mapping_dcp(table: np.ndarray, pos: np.ndarray, world: int, rank: int) -> np.ndarray:
    """The ownership arithmetic of the runner's ``_compute_slot_mapping_cpu``
    (``cp_kv_cache_interleave_size = block_size``), one decode token per row."""
    vbs = BS * world
    blocks = table[np.arange(len(pos)), pos // vbs]
    voff = pos % vbs
    local = (voff // BS) % world == rank
    slot = blocks * BS + voff // (world * BS) * BS + voff % BS
    return np.where(local, slot, NULL)


def build_case(cfg, ref_layers, world, positions, dead, seed):
    """-> (x, pos, ref meta, per-rank metas, per-rank caches per layer, key positions)."""
    reqs = len(positions)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(reqs, HIDDEN, generator=g).to(cfg.torch_dtype)
    pos = torch.tensor(positions, dtype=torch.int64)
    entry_dim = ref_layers[0].entry_dim
    # Every request's tokens (the cached history), in global order.
    tokens = [
        [(torch.randn(SPAN, entry_dim, generator=g) * 1.5).to(cfg.torch_dtype) for _ in range(reqs)]
        for _ in ref_layers
    ]

    # Reference: replicated cache, one block per bs tokens, disjoint block ids.
    ref_w = SPAN // BS
    ref_pages = reqs * ref_w + 1
    ids = (torch.randperm(ref_pages - 1, generator=g) + 1).view(reqs, ref_w).to(torch.int32)
    ref_table = torch.full((reqs, ref_w), -1, dtype=torch.int32)
    for i in range(reqs):
        if not dead[i]:
            ref_table[i, : positions[i] // BS + 1] = ids[i, : positions[i] // BS + 1]
    for li, a in enumerate(ref_layers):
        pages = (torch.randn(ref_pages, 1, BS, entry_dim, generator=g) * 1.5).to(cfg.torch_dtype)
        for i in range(reqs):
            for t in range(positions[i]):
                if not dead[i]:
                    pages[ids[i, t // BS], 0, t % BS] = tokens[li][i][t]
        a.kv_pages = pages
    ref_slots = torch.where(
        torch.tensor(dead),
        torch.full((reqs,), -1, dtype=torch.long),
        ids[torch.arange(reqs), pos // BS].long() * BS + pos % BS,
    )
    ref_meta = {
        "block_table_tensor": ref_table, "slot_mapping": ref_slots,
        "max_query_len": 1, "decode_token_threshold": 1,
    }

    # DCP: one shared table of virtual blocks; each rank's cache holds its slice.
    dw = SPAN // (BS * world)
    pages_d = reqs * dw + 1
    vids = (torch.randperm(pages_d - 1, generator=g) + 1).view(reqs, dw).to(torch.int32)
    table = torch.full((reqs, dw), -1, dtype=torch.int32)
    for i in range(reqs):
        if not dead[i]:
            n = positions[i] // (BS * world) + 1
            table[i, :n] = vids[i, :n]
    null_entry = [a.kv_pages[NULL].clone() for a in ref_layers]
    caches, metas = [], []
    for r in range(world):
        per_layer = []
        for li in range(len(ref_layers)):
            pages = (torch.randn(pages_d, 1, BS, entry_dim, generator=g) * 1.5).to(cfg.torch_dtype)
            pages[NULL] = null_entry[li]
            for i in range(reqs):
                for t in range(positions[i]):
                    if dead[i] or (t // BS) % world != r:
                        continue
                    blk = vids[i, t // (BS * world)]
                    off = (t % (BS * world)) // (world * BS) * BS + t % BS
                    pages[blk, 0, off] = tokens[li][i][t]
            per_layer.append(pages)
        caches.append(per_layer)
        tnp = torch.where(table < 0, torch.zeros_like(table), table).numpy()
        slots = torch.from_numpy(slot_mapping_dcp(tnp, pos.numpy(), world, r))
        slots = torch.where(torch.tensor(dead), torch.full_like(slots, NULL), slots)
        metas.append({
            "block_table_tensor": table, "slot_mapping": slots,
            "max_query_len": 1, "decode_token_threshold": 1,
        })
    return x, pos, ref_meta, metas, caches


def key_positions(world, rank, width):
    j = torch.arange(width * BS)
    return (j // BS) * (world * BS) + rank * BS + j % BS


def run_dcp(ranks, rope, x, pos, metas, use_forward):
    """``run_cp`` with a per-rank metadata dict."""
    world = len(ranks)
    import threading

    res, errs = [None] * world, []

    def body(r):
        try:
            outs, sels, selected = [], [], None
            for a in ranks[r]:
                saved = a.kv_pages.clone()
                rk = torch.tensor(r, dtype=torch.int32)
                if use_forward:
                    out, selected = a.forward(
                        x, pos, rope(pos), {a.layer_name: metas[r]}, selected, rk
                    )
                else:
                    out, selected = a.forward_decode_cp(x, pos, rope(pos), metas[r], selected, rk)
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
                elif name == "sinks":
                    p.copy_(torch.randn_like(p) * 2.0)
                else:
                    p.copy_(torch.randn(p.shape) * 0.25)
        a.kv_pages = torch.zeros(1, 1, BS, a.entry_dim, dtype=cfg.torch_dtype)
        layers.append(a)
    return layers


@pytest.mark.parametrize("sparse,pad", [(False, "oob"), (True, "oob"), (True, "null")])
@pytest.mark.parametrize("world,order_seed", [(4, 2), (8, 1)])
@torch.no_grad()
def test_decode_dcp_sharded_matches_split(monkeypatch, world, order_seed, sparse, pad):
    torch.set_num_threads(2)
    cfg = make_config(torch.float32)
    ref_layers = build_reference(cfg)
    rope = model.HYV4RotaryEmbedding(cfg)
    order = torch.randperm(world, generator=torch.Generator().manual_seed(order_seed)).tolist()
    hub = Hub(world, order)
    monkeypatch.setattr(model, "_dcp_world", lambda: world)
    ranks = [[build_rank(cfg, a, world, r, hub) for a in ref_layers] for r in range(world)]
    monkeypatch.setattr(model, "_dcp_world", lambda: 1)
    assert all(a.dcp == world for rk in ranks for a in rk)
    monkeypatch.setattr(model, "DEC_SPARSE_NKI", sparse)
    monkeypatch.setattr(model, "SPARSE_PAD_ROW", pad)
    for ci, (positions, dead) in enumerate(CASES):
        x, pos, ref_meta, metas, caches = build_case(
            cfg, ref_layers, world, positions, dead, seed=200 + ci
        )
        monkeypatch.setattr(model, "DEC_CP", False)
        ref_out, ref_sel = [], []
        selected = None
        for a in ref_layers:
            saved = a.kv_pages.clone()
            out, selected = a.forward_decode_split(x, pos, rope(pos), ref_meta, selected)
            a.kv_pages = saved
            ref_out.append(out.float())
            ref_sel.append(selected)
        for r in range(world):
            for li, a in enumerate(ranks[r]):
                a.kv_pages = caches[r][li]
        # DEC_CP stays off: dcp > 1 alone must route decode to the sharded path.
        res = run_dcp(ranks, rope, x, pos, metas, use_forward=(ci % 2 == 0))
        live = ~torch.tensor(dead)
        width = metas[0]["block_table_tensor"].shape[1]
        for li in range(2):
            sel = torch.zeros_like(ref_sel[li])
            for r in range(world):
                sel_ctx, sel_own, _ = res[r][1][li]
                assert torch.equal(sel_own, res[0][1][li][1]), "own-token bit differs"
                kp = key_positions(world, r, width)
                sel[:, kp] = sel_ctx
            sel[:, -1:] = res[0][1][li][1]
            assert torch.equal(sel[live], ref_sel[li][live]), (positions, li)
            for r in range(world):
                assert torch.equal(res[r][0][li], res[0][0][li]), "ranks disagree"
                assert rel(res[r][0][li][live], ref_out[li][live]) < 2e-5, (positions, li)


def test_slot_mapping_matches_runner_formula():
    """``slot_mapping_dcp`` is the runner's formula; check it against a direct
    per-token enumeration of ownership (rank (t // bs) % W, local slot)."""
    world = 4
    table = np.array([[5, 9, 2, 7]])
    for t in range(4 * BS * world):
        for r in range(world):
            s = slot_mapping_dcp(table, np.array([t]), world, r)[0]
            if (t // BS) % world == r:
                assert s == table[0, t // (BS * world)] * BS + t % BS
            else:
                assert s == NULL


def test_dcp_must_equal_tp(monkeypatch):
    cfg = make_config(torch.float32)
    hub = Hub(4, list(range(4)))
    ref = build_reference(cfg)[0]
    monkeypatch.setattr(model, "_dcp_world", lambda: 2)
    with pytest.raises(ValueError, match="decode_context_parallel_size=2"):
        build_rank(cfg, ref, 4, 0, hub)
