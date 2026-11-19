# SPDX-License-Identifier: Apache-2.0
"""Unit tests for write_prefill_kv, the shared prefill KV-cache write.

The block path writes one cache block for each ``block_size`` tokens of a full
prefill. These tests compare it byte for byte against the per-token path that
the model code used before: ``index_put_`` for each (token, head), and for a
packed FP8 K cache ``scatter_packed_k`` or the whole-cache reswizzle. They also
check that the function selects the per-token path when the block path does not
apply. Pure CPU tests, no Neuron device needed.

Run:
    pytest test/vllm_neuron/model/test_kv_cache_prefill_write.py -v
"""

from __future__ import annotations

import pytest
import torch

import vllm_neuron.model.kv_cache as kvc
from vllm_neuron.functional.attention.attention_decode import (
    _swizzle_packed_k,
    _unswizzle_packed_k,
    scatter_packed_k,
)
from vllm_neuron.model.kv_cache import (
    paged_slot_indices,
    prefill_kv_block_write_ok,
    write_prefill_kv,
)

pytestmark = pytest.mark.fast

FP8 = torch.float8_e4m3fn
FP8_MAX = 240.0
NUM_BLOCKS = 24
D_HEAD = 8
SENTINEL = 0x11


def _reference_write(k_cache, v_cache, k_flat, v_flat, slots, block_size, packed_k):
    """The per-token write of the model code before write_prefill_kv."""
    nkh = k_flat.shape[0] // slots.shape[0]
    blk, pos = paged_slot_indices(slots, block_size, k_cache.shape[0])
    heads = torch.arange(nkh).repeat_interleave(slots.shape[0])
    index = (blk.repeat(nkh), heads, pos.repeat(nkh))
    v_cache.index_put_(index, v_flat.to(v_cache.dtype))
    if packed_k == "pairs":
        scatter_packed_k(k_cache, k_flat.to(k_cache.dtype), *index)
    elif packed_k == "reswizzle":
        k_unpacked = _unswizzle_packed_k(k_cache)
        k_unpacked.index_put_(index, k_flat.to(k_cache.dtype))
        k_cache.copy_(_swizzle_packed_k(k_unpacked))
    else:
        k_cache.index_put_(index, k_flat.to(k_cache.dtype))


def _caches(nkh, block_size, kind):
    """Make K/V caches filled with a sentinel byte, so a stray write shows."""
    dtype = torch.bfloat16 if kind == "bf16" else FP8
    k = torch.zeros(NUM_BLOCKS, nkh, block_size, D_HEAD, dtype=dtype)
    v = torch.zeros(NUM_BLOCKS, nkh, block_size, D_HEAD, dtype=dtype)
    k.view(torch.uint8).fill_(SENTINEL)
    v.view(torch.uint8).fill_(SENTINEL)
    if kind == "fp8_packed":
        k = k.view(NUM_BLOCKS, nkh, block_size // 2, D_HEAD, 2)
    return k, v


def _kv(nkh, tokens, kind, seed=0):
    """Make head-major K/V rows ``[nkh * T, d]`` as the models pass them."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(nkh * tokens, D_HEAD, generator=g) * 3
    v = torch.randn(nkh * tokens, D_HEAD, generator=g) * 3
    if kind == "bf16":
        return k.to(torch.bfloat16), v.to(torch.bfloat16)
    # Quantize as the models do: scale, clamp, cast.
    k_scale, v_scale = 0.5, 2.0
    return (
        (k * k_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8),
        (v * v_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8),
    )


def _slots(table, real, padded, block_size, pad_value):
    """Slots of one prefill: real tokens in ``table`` order, then pad slots."""
    p = torch.arange(real)
    real_slots = table[p // block_size] * block_size + p % block_size
    pad = torch.full((padded - real,), pad_value, dtype=torch.long)
    return torch.cat([real_slots, pad])


def _bytes(t):
    return t.contiguous().view(torch.uint8)


@pytest.fixture(autouse=True)
def _mode_all(monkeypatch):
    """Allow the block path for every cache, so each dtype is tested."""
    monkeypatch.setattr(kvc, "_PREFILL_KV_BLOCK_WRITE", "all")


def _block_path_spy(monkeypatch):
    """Record the decision of each write_prefill_kv call."""
    calls = []
    orig = kvc.prefill_kv_block_write_ok

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(kvc, "prefill_kv_block_write_ok", spy)
    return calls


# Out-of-order, non-contiguous block tables. Block 0 is the null block.
_TABLE = torch.tensor([3, 17, 5, 9, 2, 11, 7, 19, 13, 4, 15, 6, 8, 21, 1, 10])


@pytest.mark.parametrize(
    "kind,packed_k",
    [
        ("bf16", None),
        ("fp8", None),
        ("fp8_packed", "pairs"),
        ("fp8_packed", "reswizzle"),
    ],
    ids=["bf16", "fp8", "fp8_packed_pairs", "fp8_packed_reswizzle"],
)
@pytest.mark.parametrize("nkh", [1, 2, 8], ids=["kv1", "kv2", "kv8"])
@pytest.mark.parametrize("block_size", [16, 32], ids=["bs16", "bs32"])
@pytest.mark.parametrize(
    "real,padded",
    [
        (128, 128),  # whole blocks, no padding
        (100, 128),  # partial last real chunk
        (40, 128),  # partial last real chunk + all-pad chunks
        (1, 64),  # one real token
    ],
    ids=["full", "partial", "partial_allpad", "one_token"],
)
@pytest.mark.parametrize("pad_value", [0, -1], ids=["pad_null", "pad_minus1"])
def test_block_path_matches_per_token_path(
    monkeypatch, kind, packed_k, nkh, block_size, real, padded, pad_value
):
    calls = _block_path_spy(monkeypatch)
    nreq = (padded + block_size - 1) // block_size
    table = _TABLE[:nreq]
    slots = _slots(table, real, padded, block_size, pad_value)
    k_flat, v_flat = _kv(nkh, padded, kind)
    fp8_packed = kind == "fp8_packed"

    k_ref, v_ref = _caches(nkh, block_size, kind)
    _reference_write(
        k_ref,
        v_ref,
        k_flat,
        v_flat,
        slots,
        block_size,
        packed_k,
    )

    k_new, v_new = _caches(nkh, block_size, kind)
    write_prefill_kv(
        k_new,
        v_new,
        k_flat.view(nkh, padded, D_HEAD),
        v_flat.view(nkh, padded, D_HEAD),
        slots,
        block_size,
        full_prefill=True,
        fp8_packed=fp8_packed,
        packed_k_fallback=packed_k or "pairs",
    )
    assert calls == [True]

    k_ref_u = _unswizzle_packed_k(k_ref) if fp8_packed else k_ref
    k_new_u = _unswizzle_packed_k(k_new) if fp8_packed else k_new
    kr, kn, vr, vn = (_bytes(t) for t in (k_ref_u, k_new_u, v_ref, v_new))

    # Every real token's K and V bytes are identical to the per-token write.
    nreal_blk = (real + block_size - 1) // block_size
    for i, b in enumerate(table.tolist()):
        n = max(0, min((i + 1) * block_size, real) - i * block_size)
        assert torch.equal(kn[b, :, :n], kr[b, :, :n]), (i, b, n)
        assert torch.equal(vn[b, :, :n], vr[b, :, :n]), (i, b, n)
        if i >= nreal_blk:
            # A chunk of pads goes to the null block, so this block is untouched.
            assert bool((kn[b] == SENTINEL).all()) and bool((vn[b] == SENTINEL).all())
    # Blocks outside the table, other than the null block, are untouched.
    other = [b for b in range(1, NUM_BLOCKS) if b not in table.tolist()]
    assert torch.equal(kn[other], kr[other]) and bool((kn[other] == SENTINEL).all())
    assert torch.equal(vn[other], vr[other]) and bool((vn[other] == SENTINEL).all())
    if real == padded:
        # No padding: the whole caches are byte-identical, null block included.
        assert torch.equal(kn, kr) and torch.equal(vn, vr)


@pytest.mark.parametrize("kind", ["bf16", "fp8", "fp8_packed"])
def test_block_path_accepts_flat_and_3d_inputs(kind):
    """[nkh * T, d] and [nkh, T, d] inputs give the same cache."""
    nkh, bs, T = 2, 16, 64
    table = _TABLE[: T // bs]
    slots = _slots(table, T, T, bs, 0)
    k_flat, v_flat = _kv(nkh, T, kind)
    fp8_packed = kind == "fp8_packed"
    out = []
    for shape in ((nkh * T, D_HEAD), (nkh, T, D_HEAD)):
        k, v = _caches(nkh, bs, kind)
        write_prefill_kv(
            k,
            v,
            k_flat.view(shape),
            v_flat.view(shape),
            slots,
            bs,
            full_prefill=True,
            fp8_packed=fp8_packed,
        )
        out.append((_bytes(k), _bytes(v)))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1])


@pytest.mark.parametrize(
    "case",
    ["not_whole_blocks", "dcp", "not_full_prefill", "block_dim_mismatch", "disabled"],
)
@pytest.mark.parametrize("kind", ["bf16", "fp8", "fp8_packed"])
def test_fallback_to_per_token_path(monkeypatch, case, kind):
    """The function takes the per-token path, and the bytes match it exactly."""
    calls = _block_path_spy(monkeypatch)
    nkh, bs = 2, 16
    real, padded = (150, 150) if case == "not_whole_blocks" else (100, 128)
    cache_bs = 32 if case == "block_dim_mismatch" else bs
    if case == "disabled":
        monkeypatch.setattr(kvc, "_PREFILL_KV_BLOCK_WRITE", "off")
    table = _TABLE[: (padded + bs - 1) // bs]
    slots = _slots(table, real, padded, bs, 0)
    k_flat, v_flat = _kv(nkh, padded, kind)
    fp8_packed = kind == "fp8_packed"

    k_ref, v_ref = _caches(nkh, cache_bs, kind)
    _reference_write(
        k_ref, v_ref, k_flat, v_flat, slots, bs, "pairs" if fp8_packed else None
    )
    k_new, v_new = _caches(nkh, cache_bs, kind)
    write_prefill_kv(
        k_new,
        v_new,
        k_flat,
        v_flat,
        slots,
        bs,
        full_prefill=case != "not_full_prefill",
        dcp_size=2 if case == "dcp" else 1,
        fp8_packed=fp8_packed,
    )
    assert calls == [False]
    assert torch.equal(_bytes(k_new), _bytes(k_ref))
    assert torch.equal(_bytes(v_new), _bytes(v_ref))


def test_fallback_reswizzle_handles_odd_token_count():
    """The reswizzle fallback writes an odd, unpadded prefill (no pairs)."""
    nkh, bs, T = 2, 16, 37
    table = _TABLE[:3]
    slots = _slots(table, T, T, bs, 0)
    k_flat, v_flat = _kv(nkh, T, "fp8_packed")
    k_ref, v_ref = _caches(nkh, bs, "fp8_packed")
    _reference_write(k_ref, v_ref, k_flat, v_flat, slots, bs, "reswizzle")
    k_new, v_new = _caches(nkh, bs, "fp8_packed")
    write_prefill_kv(
        k_new,
        v_new,
        k_flat,
        v_flat,
        slots,
        bs,
        full_prefill=True,
        fp8_packed=True,
        packed_k_fallback="reswizzle",
    )
    assert torch.equal(_bytes(k_new), _bytes(k_ref))
    assert torch.equal(_bytes(v_new), _bytes(v_ref))


def test_block_path_rejects_unaligned_slots():
    """In eager mode, slots that do not start a block raise."""
    nkh, bs, T = 1, 16, 32
    k, v = _caches(nkh, bs, "bf16")
    k_flat, v_flat = _kv(nkh, T, "bf16")
    # Start at in-block position 3: the prefill does not start a block.
    slots = torch.arange(T) + 5 * bs + 3
    with pytest.raises(AssertionError, match="not block-aligned"):
        write_prefill_kv(k, v, k_flat, v_flat, slots, bs, full_prefill=True)
    # Two blocks inside one chunk.
    slots = torch.cat([torch.arange(8) + 2 * bs, torch.arange(8) + 7 * bs])
    slots = torch.cat([slots, torch.arange(16) + 9 * bs])
    with pytest.raises(AssertionError, match="not block-aligned"):
        write_prefill_kv(k, v, k_flat, v_flat, slots, bs, full_prefill=True)


def test_unknown_packed_k_fallback_raises():
    nkh, bs, T = 1, 16, 30
    k, v = _caches(nkh, bs, "fp8_packed")
    k_flat, v_flat = _kv(nkh, T, "fp8_packed")
    slots = _slots(_TABLE[:2], T, T, bs, 0)
    with pytest.raises(ValueError, match="packed_k_fallback"):
        write_prefill_kv(
            k,
            v,
            k_flat,
            v_flat,
            slots,
            bs,
            full_prefill=True,
            fp8_packed=True,
            packed_k_fallback="bogus",
        )


def test_block_write_ok_shape_rules():
    bs, nkh = 16, 2
    k = torch.zeros(4, nkh, bs, D_HEAD)
    v = torch.zeros(4, nkh, bs, D_HEAD)
    kp = torch.zeros(4, nkh, bs // 2, D_HEAD, 2)
    ok = prefill_kv_block_write_ok
    assert ok(k, v, 64, bs, nkh, full_prefill=True)
    assert ok(kp, v, 64, bs, nkh, full_prefill=True, fp8_packed=True)
    assert not ok(kp, v, 64, bs, nkh, full_prefill=True)  # packed K, flag off
    assert not ok(k, v, 64, bs, nkh, full_prefill=True, fp8_packed=True)
    assert not ok(k, v, 64, bs, 1, full_prefill=True)  # head count mismatch
    assert not ok(k, v, 64, 32, nkh, full_prefill=True)  # block dim mismatch
    assert not ok(k, v, 72, bs, nkh, full_prefill=True)
    assert not ok(k, v, 64, bs, nkh, full_prefill=False)
    assert not ok(k, v, 64, bs, nkh, full_prefill=True, dcp_size=2)


@pytest.mark.parametrize(
    "value,mode",
    [
        (None, "packed"),
        ("", "packed"),
        ("packed", "packed"),
        ("all", "all"),
        ("1", "all"),
        ("off", "off"),
        ("0", "off"),
        (" ALL ", "all"),
    ],
)
def test_mode_env(monkeypatch, value, mode):
    import vllm_neuron.envs as envs

    if value is None:
        monkeypatch.delenv("VLLM_NEURON_PREFILL_KV_BLOCK_WRITE", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_PREFILL_KV_BLOCK_WRITE", value)
    assert envs.VLLM_NEURON_PREFILL_KV_BLOCK_WRITE == mode


def test_mode_env_rejects_unknown(monkeypatch):
    import vllm_neuron.envs as envs

    monkeypatch.setenv("VLLM_NEURON_PREFILL_KV_BLOCK_WRITE", "bogus")
    with pytest.raises(ValueError, match="packed"):
        envs.VLLM_NEURON_PREFILL_KV_BLOCK_WRITE


@pytest.mark.parametrize("kind", ["bf16", "fp8", "fp8_packed"])
def test_default_mode_selects_packed_only(monkeypatch, kind):
    """The default mode writes whole blocks for a packed FP8 K cache only."""
    monkeypatch.setattr(kvc, "_PREFILL_KV_BLOCK_WRITE", "packed")
    calls = _block_path_spy(monkeypatch)
    nkh, bs, T = 2, 16, 64
    slots = _slots(_TABLE[: T // bs], 50, T, bs, 0)
    k_flat, v_flat = _kv(nkh, T, kind)
    fp8_packed = kind == "fp8_packed"
    k_ref, v_ref = _caches(nkh, bs, kind)
    _reference_write(
        k_ref, v_ref, k_flat, v_flat, slots, bs, "pairs" if fp8_packed else None
    )
    k_new, v_new = _caches(nkh, bs, kind)
    write_prefill_kv(
        k_new,
        v_new,
        k_flat,
        v_flat,
        slots,
        bs,
        full_prefill=True,
        fp8_packed=fp8_packed,
    )
    assert calls == [fp8_packed]
    kr = _unswizzle_packed_k(k_ref) if fp8_packed else k_ref
    kn = _unswizzle_packed_k(k_new) if fp8_packed else k_new
    # Real tokens: identical bytes on either path.
    blk, pos = paged_slot_indices(slots[:50], bs, NUM_BLOCKS)
    assert torch.equal(_bytes(kn)[blk, :, pos], _bytes(kr)[blk, :, pos])
    assert torch.equal(_bytes(v_new)[blk, :, pos], _bytes(v_ref)[blk, :, pos])


def test_block_path_traces_without_value_checks():
    """torch.compile traces the block path with no graph break."""
    nkh, bs, T = 2, 16, 64
    table = _TABLE[: T // bs]
    slots = _slots(table, 40, T, bs, 0)
    k_flat, v_flat = _kv(nkh, T, "fp8_packed")

    def fn(k_cache, v_cache, k_in, v_in, s):
        write_prefill_kv(
            k_cache, v_cache, k_in, v_in, s, bs, full_prefill=True, fp8_packed=True
        )
        return k_cache, v_cache

    k_ref, v_ref = _caches(nkh, bs, "fp8_packed")
    fn(k_ref, v_ref, k_flat, v_flat, slots)
    k_new, v_new = _caches(nkh, bs, "fp8_packed")
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    compiled(k_new, v_new, k_flat, v_flat, slots)
    assert torch.equal(_bytes(k_new), _bytes(k_ref))
    assert torch.equal(_bytes(v_new), _bytes(v_ref))
