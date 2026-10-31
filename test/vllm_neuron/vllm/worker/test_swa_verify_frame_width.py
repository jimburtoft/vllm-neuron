# SPDX-License-Identifier: Apache-2.0
"""The trimmed SWA decode frame has room for the active tokens of a verify step.

A multi-token decode step keeps its ``q`` active tokens in the last ``q`` slots of
the trimmed frame. ``_compute_swa_decode_tensors`` moves the block-aligned frame
start up so that the newest prior token stays below those slots, so the frame
must hold ``sliding_window + q + block_size - 2`` tokens for the window of the
first query to stay inside it. ``_compute_swa_num_blocks`` sizes for that.
"""

import functools
import types

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.fast]


def _cls():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    return NeuronModelRunner


def _old_count(window, block_size, bs, max_model_len, align_tokens=256, lnc=2):
    """The pre-change width, for the single-token regression check."""
    min_blocks = window // block_size + 1
    per = max(1, align_tokens // block_size)
    n = -(-min_blocks // per) * per
    if bs % lnc != 0:
        n = max(n, (2 * lnc * 128) // block_size)
    return min(n, -(-max_model_len // block_size))


def _fake(q, max_model_len=1 << 20):
    return types.SimpleNamespace(
        _dcp_size=1, max_model_len=max_model_len, _decode_token_threshold=lambda: q
    )


@pytest.mark.parametrize("window", [64, 100, 128, 1000, 2048, 4096])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("bs", [1, 2, 3, 16])
def test_single_token_width_is_unchanged(window, block_size, bs, monkeypatch):
    """Single-token decode keeps the old width, except where the old count was short.

    The old count ``window // block_size + 1`` covers ``window + block_size - 1``
    tokens only when the window is a block multiple. A window such as 1000 at block
    32 needed one more block, which the new count adds.
    """
    monkeypatch.delenv("VLLM_NEURON_SWA_ALIGN_TOKENS", raising=False)
    got = _cls()._compute_swa_num_blocks(_fake(1), window, block_size, bs)
    old = _old_count(window, block_size, bs, 1 << 20)
    if old * block_size >= window + block_size - 1:
        assert got == old
    else:
        assert got > old and got * block_size >= window + block_size - 1


@pytest.mark.parametrize("window", [64, 100, 128, 1000, 2048, 4096])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
@pytest.mark.parametrize("q", [1, 2, 4, 8, 16, 33])
@pytest.mark.parametrize("align", ["", "4096"])
def test_width_has_room_for_the_active_tokens(window, block_size, q, align, monkeypatch):
    if align:
        monkeypatch.setenv("VLLM_NEURON_SWA_ALIGN_TOKENS", align)
    else:
        monkeypatch.delenv("VLLM_NEURON_SWA_ALIGN_TOKENS", raising=False)
    got = _cls()._compute_swa_num_blocks(_fake(q), window, block_size, 2)
    assert got * block_size >= window + q + block_size - 2
    assert (got * block_size) % 256 == 0


def test_muse_glimmer_geometry_is_unchanged(monkeypatch):
    """Window 2048, block 32, DFlash q = 16: still 72 blocks (2304) / 128 with align 4096."""
    monkeypatch.delenv("VLLM_NEURON_SWA_ALIGN_TOKENS", raising=False)
    R = _cls()
    for q in (1, 16):
        assert R._compute_swa_num_blocks(_fake(q), 2048, 32, 16) == 72
    monkeypatch.setenv("VLLM_NEURON_SWA_ALIGN_TOKENS", "4096")
    for q in (1, 16):
        assert R._compute_swa_num_blocks(_fake(q, 8192), 2048, 32, 16) == 128


def test_max_model_len_clip_still_applies(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_SWA_ALIGN_TOKENS", raising=False)
    assert _cls()._compute_swa_num_blocks(_fake(16, 512), 512, 32, 2) == 16


def _decode_fake(ctx, q, window, block_size, table_width):
    R = _cls()
    fake = _fake(q)
    fake.max_model_len = table_width * block_size
    fake._swa_ring = False
    fake.device = torch.device("cpu")
    fake._swa_groups = {}
    fake._raw_block_table = R._raw_block_table
    fake._gather_swa_columns = R._gather_swa_columns
    fake._compute_swa_num_blocks = functools.partial(R._compute_swa_num_blocks, fake)
    fake.input_batch = types.SimpleNamespace(
        num_computed_tokens_cpu=np.asarray(ctx, dtype=np.int32)
    )
    table = (np.arange(len(ctx) * table_width, dtype=np.int32) + 1).reshape(
        len(ctx), table_width
    )
    blk = types.SimpleNamespace(
        block_table=types.SimpleNamespace(np=table, cpu=torch.from_numpy(table))
    )
    return R, fake, blk


@pytest.mark.parametrize(
    "window,block_size",
    [(64, 16), (100, 16), (127, 32), (2048, 32), (128, 128)],
)
@pytest.mark.parametrize("q", [1, 4, 16])
def test_frame_keeps_prior_tokens_and_first_query_window(window, block_size, q, monkeypatch):
    """Every context: the newest prior token is below the active slots, and the
    window of the first query starts inside the frame (no clip at the table end)."""
    monkeypatch.delenv("VLLM_NEURON_SWA_ALIGN_TOKENS", raising=False)
    table_width = 4 * (window // block_size + 8) + 64
    limit = table_width * block_size - q
    ctx = list(range(0, limit, 3))
    R, fake, blk = _decode_fake(ctx, q, window, block_size, table_width)
    n = fake._compute_swa_num_blocks(window, block_size, len(ctx))
    if n >= table_width:
        pytest.skip("no trim")
    _, offset = R._compute_swa_decode_tensors(
        fake, blk, len(ctx), window, block_size, 0, query_len=q
    )
    s_prior = n * block_size
    for c, start in zip(ctx, offset.tolist()):
        assert (c - 1) - start < s_prior - q
        assert start <= max(c - window + 1, 0)
