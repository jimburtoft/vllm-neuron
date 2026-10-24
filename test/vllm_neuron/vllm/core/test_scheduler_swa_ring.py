# SPDX-License-Identifier: Apache-2.0
"""VLLM_NEURON_SWA_RING on vLLM's real Scheduler / KVCacheManager (async K=2).

Checks, for the Gemma3 shape (6 SWA groups of window 1024 + 1 full group, block 32):
* every SWA group of every request holds at most 33 blocks and never frees a block
  before the request ends (the ring = first 32 table entries never changes);
* no preemption at 212 x (2048 in, 2048 out) on the 70,778-block pool with real
  admission + early free + ring; the ring charge is 6 x 33 + full;
* the early-free hook frees nothing while the ring is on.
"""

from collections import deque
from unittest.mock import Mock

import pytest

from vllm_neuron.vllm.core import kv_admission as ka
from vllm_neuron.vllm.core import swa_ring as sr

from test.vllm_neuron.vllm.core.test_scheduler_real_swa_admission import (
    BLOCK,
    N_SWA,
    POOL,
    WINDOW,
    _kv_cache_config,
    _vllm_config,
)

ML = 4096
MNBT = 2176
HOLD = sr.hold_blocks(WINDOW, BLOCK)  # 33
NB = sr.ring_blocks(WINDOW, BLOCK)  # 32


def _groups(ring=True, ml=ML, k=2, mnbt=MNBT):
    swa_upstream = ka.cdiv(min(WINDOW - 1 + k * mnbt, ml), BLOCK) + 1
    return [
        ka.GroupDesc("swa", BLOCK, swa_upstream, WINDOW, ring_blocks=HOLD if ring else 0)
    ] * N_SWA + [ka.GroupDesc("full", BLOCK, ka.cdiv(ml, BLOCK))]


class TestRingArithmetic:
    def test_charges(self):
        p = ka.make_params(_groups(), POOL, ML, 2)
        assert ka.prefill_phase_blocks(p, 2048) == 6 * 33 + 65 == 263
        assert ka.decode_phase_blocks(p, 4096) == 6 * 33 + 128 == 326
        assert ka.request_peak_blocks(p, 2048, 4096) == 326
        # Admission with k_p <= 3 prefill-phase requests: (N - 3) x 326 + 3 x 263.
        assert (p.usable_blocks - 3 * 263) // 326 + 3 == 217
        # Short prompt: the ring never exceeds the sequence.
        assert ka.prefill_phase_blocks(p, 100) == 7 * ka.cdiv(102, 32)

    def test_ring_off_unchanged(self):
        p = ka.make_params(_groups(ring=False), POOL, ML, 2)
        assert ka.prefill_phase_blocks(p, 2048) == 455
        assert ka.decode_phase_blocks(p, 4096) == 332


def _run(monkeypatch, n, prompt, out_len, mns, num_blocks, ring=True, early=True, ml=ML,
         mnbt=MNBT, k=2):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    for var, on in (
        ("VLLM_NEURON_SWA_REAL_ADMISSION", True),
        ("VLLM_NEURON_SWA_EARLY_FREE", early),
        ("VLLM_NEURON_SWA_RING", ring),
    ):
        if on:
            monkeypatch.setenv(var, "1")
        else:
            monkeypatch.delenv(var, raising=False)
    sched = NeuronAsyncScheduler(
        vllm_config=_vllm_config(mns, ml, mnbt, k),
        kv_cache_config=_kv_cache_config(num_blocks, ml, BLOCK),
        structured_output_manager=Mock(should_advance=Mock(return_value=False)),
        block_size=BLOCK,
    )
    for i in range(n):
        sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
        sched.add_request(Request(f"r{i}", [7] * prompt, sp, None, arrival_time=float(i)))
    pool = sched.kv_cache_manager.block_pool
    mgrs = sched.kv_cache_manager.coordinator.single_type_managers
    first_ring: dict = {}
    s = dict(preempted=0, max_live=0, max_used=0, steps=0, finished=0, stalled=False,
             ring_changed=0, over_hold=0, nulled=0, over_charge=0)
    inflight: deque = deque()
    while sched.has_requests() or inflight:
        if sched.has_requests() and len(inflight) < k:
            out = sched.schedule()
            s["preempted"] += len(out.preempted_req_ids or ())
            s["max_live"] = max(s["max_live"], len(sched.running))
            s["max_used"] = max(s["max_used"], pool.num_gpu_blocks - pool.get_num_free_blocks())
            s["steps"] += 1
            for r in sched.running:
                rid = r.request_id
                held = 0
                for gi, m in enumerate(mgrs):
                    blocks = m.req_to_blocks.get(rid, [])
                    held += sum(1 for b in blocks if not b.is_null)
                    if gi >= N_SWA or not ring:
                        continue
                    if len(blocks) > HOLD:
                        s["over_hold"] += 1
                    if any(b.is_null for b in blocks):
                        s["nulled"] += 1
                    ids = tuple(b.block_id for b in blocks[:NB])
                    key = (rid, gi, r.num_preemptions)
                    if len(ids) == NB:
                        if key in first_ring and first_ring[key] != ids:
                            s["ring_changed"] += 1
                        first_ring.setdefault(key, ids)
                if held > sched._real_peak_blocks(r):
                    s["over_charge"] += 1
            samples = {rid: sched.requests[rid].num_computed_tokens
                       >= sched.requests[rid].num_prompt_tokens
                       for rid in out.num_scheduled_tokens}
            inflight.append((out, samples))
            if len(inflight) < k:
                continue
        out, samples = inflight.popleft()
        ids = list(out.num_scheduled_tokens)
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: j for j, r in enumerate(ids)},
                                sampled_token_ids=[[11] if samples[r] else [] for r in ids])
        for v in sched.update_from_output(out, mro).values():
            s["finished"] += sum(1 for o in v.outputs if o.finish_reason is not None)
        if s["steps"] >= 50 * (n + out_len):
            s["stalled"] = True
            break
    s["ring_on"] = sched._swa_ring
    s["early_freed_blocks"] = getattr(sched, "_swa_early_freed_blocks", 0)
    s["managers"] = [type(m).__name__ for m in mgrs]
    return s


def test_ring_small_no_preemption(monkeypatch):
    s = _run(monkeypatch, 24, 2048, 256, 24, 6_000)
    print("STATS_RING_SMALL", s)
    assert s["ring_on"] and s["managers"][:N_SWA] == ["RingSlidingWindowManager"] * N_SWA
    assert not s["stalled"] and s["finished"] == 24 and s["preempted"] == 0
    assert s["over_hold"] == 0 and s["nulled"] == 0 and s["ring_changed"] == 0
    assert s["early_freed_blocks"] == 0 and s["over_charge"] == 0


def test_ring_short_prompts(monkeypatch):
    # Prompt + output shorter than the window: the ring is never full.
    s = _run(monkeypatch, 8, 100, 300, 8, 2_000)
    print("STATS_RING_SHORT", s)
    assert not s["stalled"] and s["finished"] == 8 and s["preempted"] == 0
    assert s["over_hold"] == 0 and s["nulled"] == 0


def test_ring_long_prompt_4096(monkeypatch):
    s = _run(monkeypatch, 6, 4096, 300, 6, 3_000, ml=8192, mnbt=4224)
    print("STATS_RING_4096", s)
    assert not s["stalled"] and s["finished"] == 6 and s["preempted"] == 0
    assert s["over_hold"] == 0 and s["nulled"] == 0 and s["ring_changed"] == 0


def test_ring_off_managers_unchanged(monkeypatch):
    s = _run(monkeypatch, 2, 64, 4, 4, 2_000, ring=False)
    assert s["ring_on"] is False
    assert s["managers"][:N_SWA] == ["SlidingWindowManager"] * N_SWA


@pytest.mark.slow
def test_ring_212_x_2048_no_preemption(monkeypatch):
    s = _run(monkeypatch, 212, 2048, 2048, 212, POOL)
    print("STATS_RING_212", s)
    assert not s["stalled"] and s["finished"] == 212 and s["preempted"] == 0
    assert s["max_live"] == 212 and s["max_used"] <= POOL
    assert s["over_hold"] == 0 and s["nulled"] == 0 and s["ring_changed"] == 0
    assert s["over_charge"] == 0 and s["early_freed_blocks"] == 0


@pytest.mark.slow
def test_ring_217_x_2048_no_preemption(monkeypatch):
    s = _run(monkeypatch, 217, 2048, 2048, 217, POOL)
    print("STATS_RING_217", s)
    assert not s["stalled"] and s["finished"] == 217 and s["preempted"] == 0
    assert s["max_used"] <= POOL and s["over_charge"] == 0
