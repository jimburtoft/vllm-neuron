# SPDX-License-Identifier: Apache-2.0
"""SWA ring buffer (VLLM_NEURON_SWA_RING=1) for sliding-window KV groups.

Each sliding-window (SWA) KV group of a request keeps a fixed ring of
``ring_blocks = cdiv(window, block_size)`` blocks. The ring is the first
``ring_blocks`` blocks that vLLM allocated to the request for that group, in
block-table order. For Gemma3 (window 1024, block 32) the ring has 32 blocks and
``ring_slots = 1024`` slots.

Ring modulus
------------
The decode kernel (nkilib ``attention_tkg``) always places the K/V of the
active token at the last table slot ``s_prior - 1``. The decode mask also marks
that slot as the active slot. The ring therefore reserves the last slot. A
position ``p`` maps to the ring slot ``p % R`` with ``R = ring_slots - 1``
(1023 for Gemma3). The ``R`` ring slots then hold exactly the prior window
``[p - R, p - 1]`` at decode position ``p``. Gemma3 attends ``k >= p - 1023``,
so ``R = window - 1`` is sufficient when ``ring_slots == window``.

The write of position ``p`` overwrites the slot of position ``p - R``. The
decode step reads the cache before it writes it (the in-kernel update runs after
the attention, and the external scatter runs after the layer loop). Position
``p - R`` is inside the window of ``p``, so the step reads it first and then
replaces it.

Decode mask
-----------
Prior slot ``s`` is valid if ``s < min(p, R)``. Before the ring is full the
valid slots are ``[0, p)``, which is the linear layout. After that every ring
slot is valid. Pass ``pos_ids = min(p, R)`` and ``start_pos = 0``.

Prefill
-------
Prefill writes only the last ``R`` positions of the prompt to the ring, at
``p % R``. All earlier positions go to the null block. Single-shot prefill
attends the raw K/V, so it never reads this layout. Segmented prefill and
prefix caching read the linear layout, so the ring refuses them.

Block management
----------------
vLLM's ``SlidingWindowManager`` allocates blocks for the whole sequence and
frees out-of-window head blocks (``remove_skipped_blocks``). The ring blocks are
those head blocks, so the scheduler must keep them. ``RingSlidingWindowManager``
below replaces the SWA manager when the ring is on:

* ``get_num_skipped_tokens`` returns 0, so vLLM never frees a ring block, and
  ``remove_skipped_blocks`` frees nothing.
* ``get_num_blocks_to_allocate`` / ``allocate_new_blocks`` clamp the request to
  ``hold_blocks = ring_blocks + 1`` blocks (33 for Gemma3). The ring uses only
  the first ``ring_blocks`` blocks. The extra block is the "+1" of the planned
  admission charge (6 x 33 + full); it is held but not written.

A request then holds at most ``hold_blocks`` blocks for each SWA group, in both
the prefill phase and the decode phase. Admission charges exactly that.

This module imports nothing from vLLM at module level, so the arithmetic can be
checked on any CPU.
"""

from __future__ import annotations

import numpy as np


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def ring_blocks(sliding_window: int, block_size: int) -> int:
    """Blocks in the ring of one SWA group.

    Example:
        >>> ring_blocks(1024, 32)
        32
    """
    return cdiv(sliding_window, block_size)


def ring_slots(sliding_window: int, block_size: int) -> int:
    """Slots in the ring table (the decode ``s_prior``).

    Example:
        >>> ring_slots(1024, 32)
        1024
    """
    return ring_blocks(sliding_window, block_size) * block_size


def ring_modulus(sliding_window: int, block_size: int) -> int:
    """Ring modulus R. The last table slot is kept for the active token.

    Example:
        >>> ring_modulus(1024, 32)
        1023
    """
    return ring_slots(sliding_window, block_size) - 1


def hold_blocks(sliding_window: int, block_size: int) -> int:
    """Blocks one request holds for one SWA group when the ring is on.

    Example:
        >>> hold_blocks(1024, 32)
        33
    """
    return ring_blocks(sliding_window, block_size) + 1


def check_ring_geometry(sliding_window: int, block_size: int) -> None:
    """Raise if the ring cannot hold the full window.

    The ring holds R = ring_slots - 1 prior tokens. The model attends the
    window - 1 prior tokens, so R must be at least window - 1.
    """
    R = ring_modulus(sliding_window, block_size)
    if R < sliding_window - 1:
        raise ValueError(
            f"SWA ring modulus {R} is smaller than window - 1 ({sliding_window - 1}). "
            "Use a block size that divides the sliding window."
        )


def ring_slot_mapping(
    positions: np.ndarray,
    req_indices: np.ndarray,
    block_table: np.ndarray,
    block_size: int,
    sliding_window: int,
    null_slot: int = 0,
) -> np.ndarray:
    """Map the SWA positions of one step to ring slots in the paged cache.

    In one step each request keeps only its last R positions; earlier positions
    of the same step go to ``null_slot``. A single-shot prefill of L tokens then
    writes ``[L - R, L)`` and a decode step writes its one token. Distinct kept
    positions of one request never share a ring slot.

    Args:
        positions: Absolute token positions, one per scheduled token.
        req_indices: Row in ``block_table`` for each token.
        block_table: Raw host block table of the SWA group [rows, width].
        block_size: Tokens per block.
        sliding_window: Window of the group.
        null_slot: Discard slot (vLLM null block 0).

    Returns:
        int64 slot per token.

    Example:
        >>> bt = np.arange(1, 41).reshape(1, 40)
        >>> ring_slot_mapping(np.array([1023]), np.array([0]), bt, 32, 1024).tolist()
        [32]
    """
    R = ring_modulus(sliding_window, block_size)
    nb = ring_blocks(sliding_window, block_size)
    positions = np.asarray(positions, dtype=np.int64)
    req_indices = np.asarray(req_indices, dtype=np.int64)
    if positions.size == 0:
        return positions.copy()
    assert block_table.shape[1] >= nb, (block_table.shape, nb)
    r = positions % R
    blocks = block_table[req_indices, r // block_size].astype(np.int64)
    slots = blocks * block_size + (r % block_size)
    # Step end per request: keep [end - R, end).
    end = np.zeros(int(req_indices.max()) + 1, dtype=np.int64)
    np.maximum.at(end, req_indices, positions + 1)
    drop = positions < end[req_indices] - R
    # A ring block that is still the null block (not allocated) also discards.
    drop |= blocks == 0
    return np.where(drop, null_slot, slots)


def ring_table(block_table: np.ndarray, sliding_window: int, block_size: int) -> np.ndarray:
    """Fixed decode table: the first ``ring_blocks`` columns of the block table.

    Example:
        >>> ring_table(np.arange(80).reshape(2, 40), 1024, 32).shape
        (2, 32)
    """
    return block_table[:, : ring_blocks(sliding_window, block_size)]


def ring_mask_pos(positions: np.ndarray, sliding_window: int, block_size: int) -> np.ndarray:
    """Decode-mask end position: prior slot s is valid iff s < min(p, R).

    Example:
        >>> ring_mask_pos(np.array([0, 5, 1023, 4000]), 1024, 32).tolist()
        [0, 5, 1023, 1023]
    """
    return np.minimum(np.asarray(positions), ring_modulus(sliding_window, block_size))


def ring_hold_blocks_for_tokens(num_tokens: int, sliding_window: int, block_size: int) -> int:
    """Blocks the ring manager lets a request hold for ``num_tokens`` tokens.

    Example:
        >>> ring_hold_blocks_for_tokens(2050, 1024, 32)
        33
        >>> ring_hold_blocks_for_tokens(100, 1024, 32)
        4
    """
    return min(cdiv(num_tokens, block_size), hold_blocks(sliding_window, block_size))


def make_ring_manager_class(base_cls):
    """Return a SlidingWindowManager subclass that keeps a fixed ring per request.

    ``base_cls`` is vLLM's ``SlidingWindowManager``. The import stays in the
    caller so this module has no vLLM import.
    """

    class RingSlidingWindowManager(base_cls):
        """SWA manager for VLLM_NEURON_SWA_RING=1.

        Holds at most ``ring_blocks + 1`` blocks per request and never frees a
        block before the request ends. The model runner maps every position to
        a ring slot inside the first ``ring_blocks`` blocks.
        """

        def __init__(self, kv_cache_spec, **kwargs) -> None:
            super().__init__(kv_cache_spec, **kwargs)
            check_ring_geometry(kv_cache_spec.sliding_window, self.block_size)
            self._ring_hold = hold_blocks(kv_cache_spec.sliding_window, self.block_size)
            cap = self._max_admission_blocks_per_request
            self._max_admission_blocks_per_request = (
                self._ring_hold if cap is None else min(cap, self._ring_hold)
            )

        def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
            # The ring reuses the head blocks. Nothing is out of the window.
            return 0

        def remove_skipped_blocks(
            self, request_id, processed_computed_tokens, num_prompt_tokens=None
        ) -> None:
            return None

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ) -> int:
            num_tokens = min(num_tokens, self._ring_hold * self.block_size)
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_local_computed_tokens,
                min(num_tokens_main_model, num_tokens),
                apply_admission_cap=apply_admission_cap,
            )

        def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
            num_tokens = min(num_tokens, self._ring_hold * self.block_size)
            return super().allocate_new_blocks(
                request_id, num_tokens, min(num_tokens_main_model, num_tokens)
            )

        def cache_blocks(self, request, num_tokens, retention_interval=None) -> None:
            # Ring blocks are rewritten in place, so they are never a valid prefix hit.
            return None

    return RingSlidingWindowManager


def install_ring_managers(kv_cache_manager) -> int:
    """Swap every plain SlidingWindowManager of ``kv_cache_manager`` for the ring one.

    Call this right after the KV cache manager is built, before any request.

    Returns:
        Number of managers that were replaced.
    """
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    ring_cls = make_ring_manager_class(SlidingWindowManager)
    n = 0
    for m in kv_cache_manager.coordinator.single_type_managers:
        if type(m) is SlidingWindowManager:
            assert not m.req_to_blocks, "install the SWA ring before the first request"
            m.__class__ = ring_cls
            check_ring_geometry(m.sliding_window, m.block_size)
            m._ring_hold = hold_blocks(m.sliding_window, m.block_size)
            cap = m._max_admission_blocks_per_request
            m._max_admission_blocks_per_request = (
                m._ring_hold if cap is None else min(cap, m._ring_hold)
            )
            n += 1
    return n
