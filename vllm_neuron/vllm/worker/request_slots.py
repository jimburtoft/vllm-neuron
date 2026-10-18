# SPDX-License-Identifier: Apache-2.0
"""Stable request slots for model-owned per-request state.

Some models keep per-request state in model-owned device buffers, outside the
paged KV cache. Whisper is an example: it keeps the cross-attention K/V of each
request. The model runner gives each live request one slot index into such a
buffer. The slot does not change when ``condense()`` moves the request to a
different batch row.

The table holds ``num_slots`` slots. The last slot is the scratch slot. Padded
batch rows read the scratch slot, and the table never gives it to a request.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable


class RequestSlotTable:
    """Map each live request ID to a stable slot index.

    Args:
        num_slots: The number of slots in the model buffer, including the
            scratch slot. Must be 2 or more.

    Example:
        >>> table = RequestSlotTable(3)
        >>> table.assign("a"), table.assign("b"), table.assign("a")
        (0, 1, 0)
        >>> table.lookup(["b", "a"], padded_len=3)
        [1, 0, 2]
        >>> table.release("a")
        >>> table.assign("c")
        0
    """

    def __init__(self, num_slots: int) -> None:
        if num_slots < 2:
            raise ValueError(
                f"RequestSlotTable needs 2 or more slots, got {num_slots}. "
                "One slot is the scratch slot."
            )
        self.num_slots = num_slots
        self.scratch_slot = num_slots - 1
        self._free: list[int] = list(range(num_slots - 1))
        self._slot_of: dict[str, int] = {}

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._slot_of

    @property
    def num_free(self) -> int:
        """Return the number of slots that a new request can get now."""
        return len(self._free)

    def assign(self, req_id: str, running: Collection[str] = ()) -> int:
        """Return the slot of ``req_id``. Give a free slot to a new request.

        When no slot is free, take the slot of a request that is not in
        ``running``. Such a request is preempted. It runs its encoder again when
        it resumes, so it gets a slot again at that time.

        Args:
            req_id: The request ID.
            running: The IDs of the requests in the current batch. The table
                does not take a slot from these requests.

        Returns:
            The slot index.

        Raises:
            RuntimeError: If no slot is free and all holders are running.
        """
        slot = self._slot_of.get(req_id)
        if slot is not None:
            return slot
        if not self._free:
            for other in list(self._slot_of):
                if other not in running:
                    self.release(other)
                    break
        if not self._free:
            raise RuntimeError(
                f"No request slot is free for request {req_id}: all "
                f"{self.num_slots - 1} slots belong to running requests. "
                "Make the model buffer larger than max_num_seqs."
            )
        # Use the lowest free slot, so that every rank makes the same choice.
        slot = min(self._free)
        self._free.remove(slot)
        self._slot_of[req_id] = slot
        return slot

    def release(self, req_id: str) -> None:
        """Return the slot of ``req_id`` to the free list. Ignore an unknown ID."""
        slot = self._slot_of.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)

    def release_all(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self.release(req_id)

    def lookup(self, req_ids: Iterable[str], padded_len: int) -> list[int]:
        """Return the slot of each request, padded with the scratch slot.

        Raises:
            RuntimeError: If a request has no slot. Its model state was not
                written, so the model would read stale data.
        """
        slots = []
        for req_id in req_ids:
            slot = self._slot_of.get(req_id)
            if slot is None:
                raise RuntimeError(
                    f"Request {req_id} has no request slot. Its encoder did not "
                    "run in this engine. Make sure that the encoder runs for each "
                    "request before the first forward."
                )
            slots.append(slot)
        if len(slots) > padded_len:
            raise RuntimeError(
                f"The batch has {len(slots)} requests but the padded length is "
                f"{padded_len}."
            )
        return slots + [self.scratch_slot] * (padded_len - len(slots))
