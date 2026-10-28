# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the async spec-decode position correction.

Run with:
    pytest test/vllm_neuron/vllm/spec_decode/test_spec_decode_correction.py -v
"""

import torch

from vllm_neuron.functional.spec_decode_correction import (
    correct_spec_decode_positions_and_slot_mapping,
)


def _meta(bs):
    table = torch.arange(bs * 4, dtype=torch.int32).view(bs, 4)
    return {"layer.0": {"block_table_tensor": table, "block_size": 16}}


def test_real_rows_are_corrected():
    pos = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    prev = torch.tensor([[100, 200, -1, -1]], dtype=torch.int32)
    out, meta = correct_spec_decode_positions_and_slot_mapping(
        pos, _meta(1), prev, torch.tensor([3], dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64), vocab_size=1000,
    )
    assert out.tolist() == [8, 9, 10, 11]
    assert meta["layer.0"]["slot_mapping"].tolist() == [8, 9, 10, 11]


def test_padded_row_stays_in_range():
    # Row 1 is padding: it copies a small position but keeps a stale previous-step
    # row with all drafts rejected, so the raw offset is larger than the position.
    pos = torch.tensor([20, 21, 1, 2], dtype=torch.int32)
    prev = torch.tensor([[5, 6, 7], [5, -1, -1]], dtype=torch.int32)
    out, meta = correct_spec_decode_positions_and_slot_mapping(
        pos, _meta(2), prev, torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([0, 0, 1, 1], dtype=torch.int64), vocab_size=1000,
    )
    assert out.tolist() == [20, 21, 0, 0]
    slots = meta["layer.0"]["slot_mapping"]
    assert int(slots.min()) >= 0
