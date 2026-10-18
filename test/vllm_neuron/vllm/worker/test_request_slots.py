# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RequestSlotTable (stable request slots for model-owned state).

Run with:
    pytest test/vllm_neuron/vllm/worker/test_request_slots.py -v
"""

import pytest

from vllm_neuron.vllm.worker.request_slots import RequestSlotTable

pytestmark = [pytest.mark.fast]


def test_assign_is_stable_and_lowest_free():
    t = RequestSlotTable(4)
    assert [t.assign("a"), t.assign("b"), t.assign("c")] == [0, 1, 2]
    assert t.assign("b") == 1
    t.release("a")
    assert t.assign("d") == 0
    assert t.num_free == 0


def test_lookup_pads_with_scratch():
    t = RequestSlotTable(4)
    t.assign("a")
    t.assign("b")
    assert t.lookup(["b", "a"], padded_len=4) == [1, 0, 3, 3]


def test_lookup_rejects_unknown_request():
    t = RequestSlotTable(3)
    with pytest.raises(RuntimeError, match="no request slot"):
        t.lookup(["x"], padded_len=1)


def test_full_table_takes_slot_of_request_not_running():
    t = RequestSlotTable(3)
    t.assign("a")
    t.assign("b")
    # "a" is not running (preempted), so "c" can take its slot.
    assert t.assign("c", running={"b"}) == 0
    assert "a" not in t


def test_full_table_with_all_running_raises():
    t = RequestSlotTable(2)
    t.assign("a")
    with pytest.raises(RuntimeError, match="No request slot is free"):
        t.assign("b", running={"a"})


def test_needs_scratch_slot():
    with pytest.raises(ValueError):
        RequestSlotTable(1)
