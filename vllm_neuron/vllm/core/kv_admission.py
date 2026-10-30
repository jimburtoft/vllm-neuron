# SPDX-License-Identifier: Apache-2.0
"""Real-usage KV admission bound for hybrid sliding-window / full-attention models.

vLLM's per-request capacity (``get_max_concurrency_for_kv_cache_config``) charges
every sliding-window (SWA) group
``cdiv(min(window - 1 + max_concurrent_batches * max_num_batched_tokens,
max_model_len), block_size) + 1`` blocks. Under async scheduling (K = 2) with a
large ``max_num_batched_tokens`` this is the full ``max_model_len``, so a SWA group
is charged like a full-attention group.

What a request really holds, given vLLM 0.29's block manager
(``KVCacheManager.allocate_slots`` frees out-of-window SWA blocks on the
processed-token basis, ``max(0, computed - in_flight)``, before every allocation):

* Every group holds at most ``cdiv(computed + new + lookahead, block_size)`` blocks
  (blocks are a prefix of the sequence, minus the freed ones).
* Before its first post-prefill allocation (prefill phase, including the time a
  finished prefill sits hidden while the Neuron scheduler runs other prefills),
  computed + new <= P + K * d, where P is the prompt length and d the tokens per
  decode step. This is the *prefill-phase* bound.
* Afterwards a SWA group holds at most the window, the trailing retained tokens and
  the unprocessed + newly scheduled tokens (<= K * d), +1 block because the window
  need not start on a block boundary; a full-attention group grows to
  ``cdiv(min(P + M + K * d, max_model_len), block_size)``. This is the
  *decode-phase* bound.

The SWA groups shrink before the full-attention group grows, so the per-request
peak is ``max(prefill-phase total, decode-phase total)``, not the sum of per-group
maxima. If the scheduler admits only while the sum of these peaks over all live
requests fits the usable pool, no ``allocate_slots`` call can fail, so no request
is preempted.

With VLLM_NEURON_SWA_EARLY_FREE=1 the scheduler frees a request's out-of-window
SWA blocks as soon as its prefill output is processed (instead of at its first
decode allocation, which the Neuron scheduler can postpone for many steps while it
hides decodes behind other prefills), and from then on charges that request only
the decode-phase bound (``phase_charge``). With N live requests of which k_p are
still in the prefill phase, admission is then
``(N - k_p) * decode + k_p * prefill <= usable``.

This module imports nothing from vLLM so the arithmetic can be checked on any CPU.
"""

from dataclasses import dataclass


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class GroupDesc:
    """One KV cache group, reduced to what the bound needs.

    kind: "swa" (plain sliding window) or "full" (holds every block until the
        request finishes).
    upstream_blocks: vLLM's per-request block charge for the group
        (``cdiv(spec.max_memory_usage_bytes(cfg), spec.page_size_bytes)``); the real
        bound is clamped to it, so it can never exceed the upstream charge.
    ring_blocks: VLLM_NEURON_SWA_RING only (0 = off). A "swa" group then holds at
        most this many blocks in every phase (``swa_ring.hold_blocks``) and frees
        none before the request ends.
    """

    kind: str
    block_size: int
    upstream_blocks: int
    sliding_window: int = 0
    extra_retained_tokens: int = 0
    ring_blocks: int = 0


@dataclass(frozen=True)
class AdmissionParams:
    groups: tuple
    num_blocks: int
    usable_blocks: int
    max_model_len: int
    max_concurrent_batches: int
    decode_tokens: int
    lookahead_tokens: int

    @property
    def upstream_blocks_per_request(self) -> int:
        return sum(g.upstream_blocks for g in self.groups)

    @property
    def legacy_cap(self) -> int:
        w = self.upstream_blocks_per_request
        return self.num_blocks // w if w > 0 else 0


def make_params(
    groups,
    num_blocks: int,
    max_model_len: int,
    max_concurrent_batches: int,
    decode_tokens: int = 1,
    lookahead_tokens: int = 0,
    watermark_blocks: int = 0,
) -> AdmissionParams:
    # One block of the pool is vLLM's null block; the watermark is kept free.
    return AdmissionParams(
        groups=tuple(groups),
        num_blocks=num_blocks,
        usable_blocks=num_blocks - 1 - watermark_blocks,
        max_model_len=max_model_len,
        max_concurrent_batches=max(1, max_concurrent_batches),
        decode_tokens=max(1, decode_tokens),
        lookahead_tokens=max(0, lookahead_tokens),
    )


def _in_flight(p: AdmissionParams) -> int:
    return p.max_concurrent_batches * p.decode_tokens + p.lookahead_tokens


def prefill_phase_blocks(p: AdmissionParams, prefill_len: int) -> int:
    tokens = min(prefill_len + _in_flight(p), p.max_model_len)
    total = 0
    for g in p.groups:
        blocks = cdiv(tokens, g.block_size)
        if g.kind == "swa" and g.ring_blocks:
            blocks = min(blocks, g.ring_blocks)
        total += min(blocks, g.upstream_blocks)
    return total


def decode_phase_blocks(p: AdmissionParams, final_len: int) -> int:
    full_tokens = min(final_len + _in_flight(p), p.max_model_len)
    total = 0
    for g in p.groups:
        if g.kind == "swa" and g.ring_blocks:
            # Ring: the group holds its first ring_blocks blocks to the end.
            blocks = min(cdiv(full_tokens, g.block_size), g.ring_blocks)
        elif g.kind == "swa":
            swa_tokens = min(
                g.sliding_window - 1 + g.extra_retained_tokens + _in_flight(p),
                full_tokens,
            )
            blocks = cdiv(swa_tokens, g.block_size) + 1
        else:
            blocks = cdiv(full_tokens, g.block_size)
        total += min(blocks, g.upstream_blocks)
    return total


def request_peak_blocks(p: AdmissionParams, prefill_len: int, final_len: int) -> int:
    """Upper bound on the blocks one request holds at any time, all groups.

    prefill_len: tokens its (next) prefill covers -- the prompt, or every token
        so far for a preempted request that must re-prefill.
    final_len: prompt + max_tokens (clamped to max_model_len inside).
    """
    return max(prefill_phase_blocks(p, prefill_len), decode_phase_blocks(p, final_len))


def phase_charge(
    p: AdmissionParams, prefill_len: int, final_len: int, swa_freed: bool
) -> int:
    """Admission charge for one request, by phase (VLLM_NEURON_SWA_EARLY_FREE).

    swa_freed: the request's prefill output has been processed, no prefill
        tokens are still in flight, and its out-of-window SWA blocks were freed
        right away (``NeuronScheduler._swa_early_free``). From then on it never
        holds more than the decode-phase bound, so that is its charge. Otherwise
        the charge is the unchanged ``request_peak_blocks``.
    """
    if swa_freed:
        return decode_phase_blocks(p, final_len)
    return request_peak_blocks(p, prefill_len, final_len)


def uniform_cap(p: AdmissionParams, prompt_len: int, max_tokens: int) -> int:
    """Concurrent requests of one shape that fit the usable pool (log / tests)."""
    w = request_peak_blocks(p, prompt_len, prompt_len + max_tokens)
    return p.usable_blocks // w if w > 0 else 0
