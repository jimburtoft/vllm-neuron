# SPDX-License-Identifier: Apache-2.0
"""Stable per-request slots for model-owned state (e.g. GDN recurrent state).

A model that keeps per-request state in its own device buffers sets
``uses_request_slots = True`` and ``num_request_slots``. The runner assigns
each live request one slot index that does not change when the persistent
batch is condensed. The last slot is reserved as a scratch slot for padded
batch rows and warmup; it is never handed to a request.
"""

from __future__ import annotations

from collections.abc import Iterable


class StateSlotTable:
    def __init__(self, num_slots: int) -> None:
        if num_slots < 2:
            raise ValueError(f"need >= 2 slots (one is scratch), got {num_slots}")
        self.num_slots = num_slots
        self.scratch_slot = num_slots - 1
        self._free = list(range(num_slots - 1))
        self._slot = {}

    def get(self, req_id: str) -> int:
        s = self._slot.get(req_id)
        if s is None:
            if not self._free:
                raise RuntimeError(
                    f"no free state slot for {req_id}: {self.num_slots - 1} slots in use; "
                    "num_request_slots must exceed max_num_seqs"
                )
            s = min(self._free)  # deterministic across ranks
            self._free.remove(s)
            self._slot[req_id] = s
        return s

    def release(self, req_ids: Iterable[str]) -> None:
        for r in req_ids or ():
            s = self._slot.pop(r, None)
            if s is not None:
                self._free.append(s)

    def lookup(self, req_ids: Iterable[str], padded_len: int) -> list[int]:
        out = [self.get(r) for r in req_ids]
        if len(out) > padded_len:
            raise RuntimeError(f"{len(out)} requests > padded batch {padded_len}")
        return out + [self.scratch_slot] * (padded_len - len(out))
