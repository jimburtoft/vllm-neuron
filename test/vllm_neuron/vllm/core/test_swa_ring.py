# SPDX-License-Identifier: Apache-2.0
"""VLLM_NEURON_SWA_RING: ring slot mapping, ring table and decode attention set.

Pure numpy. The simulation models a paged cache of one SWA group, the runner's
ring slot mapping (prefill + decode), the fixed ring decode table, the decode
mask (prior slot s valid iff s < min(p, R)) and the kernel's active-token stitch
at table slot s_prior - 1. Each cache slot stores the position that wrote it.
For every decode step the test reconstructs the set of positions that the
attention reads and compares it with the Gemma3 SWA set [p - W + 1, p].
"""

import numpy as np
import pytest

from vllm_neuron.vllm.core import swa_ring as sr

W = 1024
BS = 32
NB = W // BS  # 32 ring blocks
R = sr.ring_modulus(W, BS)  # 1023
S_PRIOR = sr.ring_slots(W, BS)  # 1024


def test_geometry():
    assert (sr.ring_blocks(W, BS), S_PRIOR, R, sr.hold_blocks(W, BS)) == (32, 1024, 1023, 33)
    sr.check_ring_geometry(W, BS)
    sr.check_ring_geometry(1000, 32)  # 32 blocks = 1024 slots, R 1023 >= 999


def _table(rows, width=128, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, 1 + rows * width)).reshape(rows, width)
    return ids.astype(np.int32)


class Cache:
    """One SWA group of a paged cache: slot -> position that wrote it (-1 empty)."""

    def __init__(self, num_blocks):
        self.pos = np.full(num_blocks * BS, -1, dtype=np.int64)

    def write(self, slots, positions):
        keep = slots >= BS  # block 0 is the null block (discard)
        # Disjointness of one step's writes (the coalesced scatter needs it).
        s = slots[keep]
        assert len(np.unique(s)) == len(s), "two tokens of one step share a slot"
        self.pos[s] = positions[keep]


def _decode_attended(cache, table_row, p):
    """Positions one decode step at position p attends (prior via mask + active)."""
    ring = table_row[:NB].astype(np.int64)
    slots = (ring[:, None] * BS + np.arange(BS)[None, :]).reshape(-1)  # table order
    prior = cache.pos[slots]
    s = np.arange(S_PRIOR)
    valid = s < min(p, R)
    valid[S_PRIOR - 1] = False  # last slot is the active slot (kernel stitch)
    out = set(prior[valid].tolist())
    assert -1 not in out, f"p={p}: a valid slot is empty"
    out.add(p)  # the active token, stitched at s_prior - 1
    return out


def _expected(p):
    return set(range(max(0, p - W + 1), p + 1))


@pytest.mark.parametrize("L", [100, 1023, 1024, 1025, 2048])
def test_prefill_slot_mapping(L):
    bt = _table(1)
    pos = np.arange(L)
    slots = sr.ring_slot_mapping(pos, np.zeros(L, dtype=np.int64), bt, BS, W)
    kept = slots != 0
    # Only the last R positions are written, each to its ring slot.
    assert kept.sum() == min(L, R)
    assert np.array_equal(np.flatnonzero(kept), np.arange(max(0, L - R), L))
    r = pos[kept] % R
    assert np.array_equal(slots[kept], bt[0, r // BS].astype(np.int64) * BS + r % BS)
    # Ring slot 1023 (block 31, offset 31) is never written.
    assert int(bt[0, NB - 1]) * BS + BS - 1 not in set(slots.tolist())
    # Every written slot is inside the first 32 table blocks.
    assert set((slots[kept] // BS).tolist()) <= set(bt[0, :NB].tolist())


@pytest.mark.parametrize("p", [0, 1, 1022, 1023, 1024, 2047, 2048, 4095])
def test_decode_slot_and_mask(p):
    bt = _table(2, seed=1)
    slots = sr.ring_slot_mapping(np.array([p, p + 7]), np.array([0, 1]), bt, BS, W)
    for row, q in ((0, p), (1, p + 7)):
        r = q % R
        assert slots[row] == int(bt[row, r // BS]) * BS + r % BS
        assert r != S_PRIOR - 1
    assert sr.ring_mask_pos(np.array([p]), W, BS)[0] == min(p, R)
    assert sr.ring_table(bt, W, BS).shape == (2, NB)
    assert np.array_equal(sr.ring_table(bt, W, BS), bt[:, :NB])


def test_padding_and_unallocated_blocks_discard():
    bt = _table(1)
    bt[0, 5:] = 0  # blocks not allocated yet stay null
    slots = sr.ring_slot_mapping(np.array([10, 5 * BS + 1]), np.zeros(2, dtype=np.int64), bt, BS, W)
    assert slots[0] == int(bt[0, 0]) * BS + 10 and slots[1] == 0


def test_mixed_rows_one_step():
    """Packed step: a 2048-token prefill and two decode rows. Rows stay independent."""
    bt = _table(3, seed=3)
    pos = np.r_[np.arange(2048), [3000], [10]]
    req = np.r_[np.zeros(2048, dtype=np.int64), [1], [2]]
    slots = sr.ring_slot_mapping(pos, req, bt, BS, W)
    assert (slots[: 2048 - R] == 0).all() and (slots[2048 - R : 2048] != 0).all()
    assert slots[2048] == int(bt[1, (3000 % R) // BS]) * BS + (3000 % R) % BS
    assert slots[2049] == int(bt[2, 0]) * BS + 10


@pytest.mark.parametrize("L", [1, 100, 1022, 1023, 1024, 1025, 2048, 4096])
def test_decode_attention_set_every_step(L):
    """Prefill L tokens, then decode to L + 2100 (wraps the ring >= 2 times).

    At every decode step the attended set equals [p - 1023, p]. The read happens
    before the write of the step (attention, then the in-kernel update / the
    deferred scatter), so the write of p into slot p % R (which holds p - R,
    still inside the window of p) runs after the read.
    """
    bt = _table(1, seed=L)
    cache = Cache(int(bt.max()) + 1)
    pos = np.arange(L)
    cache.write(sr.ring_slot_mapping(pos, np.zeros(L, dtype=np.int64), bt, BS, W), pos)
    for p in range(L, L + 2100):
        got = _decode_attended(cache, bt[0], p)  # READ
        assert got == _expected(p), (L, p, sorted(_expected(p) ^ got)[:5])
        cache.write(sr.ring_slot_mapping(np.array([p]), np.zeros(1, dtype=np.int64), bt, BS, W),
                    np.array([p]))  # WRITE after the read


def test_write_before_read_is_wrong():
    """Negative control: write the active token first, then read. Once the ring is
    full the step loses p - 1023 (overwritten), so the harness must see it."""
    L = 2048
    bt = _table(1, seed=9)
    cache = Cache(int(bt.max()) + 1)
    pos = np.arange(L)
    cache.write(sr.ring_slot_mapping(pos, np.zeros(L, dtype=np.int64), bt, BS, W), pos)
    p = L
    cache.write(sr.ring_slot_mapping(np.array([p]), np.zeros(1, dtype=np.int64), bt, BS, W), np.array([p]))
    with pytest.raises(AssertionError):
        assert _decode_attended(cache, bt[0], p) == _expected(p)


def test_mod_1024_would_shadow():
    """Negative control: with modulus 1024 the stitched active slot hides a prior."""
    p = 2000  # 2000 % 1024 = 976 != 1023, so slot 1023 holds an in-window prior
    prior_in_last_slot = p - ((p - 1023) % 1024) - 1  # position stored at slot 1023
    assert p - W + 1 <= prior_in_last_slot < p


def test_linear_layout_prefill_breaks_ring_decode():
    """Gemma4 lesson: a linear prefill layout with ring decode is wrong for L > R."""
    L = 2048
    bt = _table(1, seed=11)
    cache = Cache(int(bt.max()) + 1)
    lin = bt[0, np.arange(L) // BS].astype(np.int64) * BS + np.arange(L) % BS
    cache.write(lin, np.arange(L))
    assert _decode_attended(cache, bt[0], L) != _expected(L)


def test_hold_blocks_cap():
    assert sr.ring_hold_blocks_for_tokens(2050, W, BS) == 33
    assert sr.ring_hold_blocks_for_tokens(4098, W, BS) == 33
    assert sr.ring_hold_blocks_for_tokens(1, W, BS) == 1
    assert sr.ring_hold_blocks_for_tokens(1056, W, BS) == 33
