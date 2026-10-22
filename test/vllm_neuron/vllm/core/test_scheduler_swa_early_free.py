# SPDX-License-Identifier: Apache-2.0
"""SWA early free (VLLM_NEURON_SWA_EARLY_FREE) on top of real-peak KV admission.

The real NeuronAsyncScheduler runs on vLLM's real Scheduler / KVCacheManager with
two steps in flight (async K=2). Besides "no preemption", the harness checks the
async-safety invariant on every executed step: every sliding-window block that a
token of that step attends ([q - window + 1, q] for each scheduled position q) is
still the block the request held when the step was scheduled (not freed, not
null, not handed to another request).
"""

from collections import deque
from unittest.mock import Mock

import pytest

from vllm_neuron.vllm.core import kv_admission as ka

from test.vllm_neuron.vllm.core.test_scheduler_real_swa_admission import (
    BLOCK,
    N_SWA,
    POOL,
    WINDOW,
    _kv_cache_config,
    _vllm_config,
)

ML = 8192
MNBT = 4224
PROMPT = 4096
OUT = 4096


def _groups(ml=ML, k=2, mnbt=MNBT):
    swa_upstream = ka.cdiv(min(WINDOW - 1 + k * mnbt, ml), BLOCK) + 1
    return [ka.GroupDesc("swa", BLOCK, swa_upstream, WINDOW)] * N_SWA + [
        ka.GroupDesc("full", BLOCK, ka.cdiv(ml, BLOCK))
    ]


class TestArithmetic:
    def test_plan_charges(self):
        p = ka.make_params(_groups(), POOL, ML, 2)
        assert p.usable_blocks == 70_777
        assert ka.prefill_phase_blocks(p, PROMPT) == 903
        assert ka.decode_phase_blocks(p, PROMPT + OUT) == 460
        assert ka.request_peak_blocks(p, PROMPT, PROMPT + OUT) == 903
        assert ka.uniform_cap(p, PROMPT, OUT) == 78
        assert ka.phase_charge(p, PROMPT, PROMPT + OUT, swa_freed=False) == 903
        assert ka.phase_charge(p, PROMPT, PROMPT + OUT, swa_freed=True) == 460
        # 147 live with up to 3 in the prefill phase fits; the plan's table.
        for k_p, cap in ((1, 151), (2, 149), (3, 147), (4, 146)):
            assert (p.usable_blocks - k_p * 903) // 460 == cap

    def test_freed_per_request(self):
        # SlidingWindowManager.get_num_skipped_tokens(4096) = 4096 - 1024 + 1.
        assert (PROMPT - WINDOW + 1) // BLOCK * N_SWA == 576


def _swa_window_blocks(sched, rid, positions):
    """Per SWA group, the blocks a token at any of `positions` attends
    ([q - window + 1, q]) as (block_id, is_null, ref_cnt) tuples."""
    mgrs = sched.kv_cache_manager.coordinator.single_type_managers
    lo = max(0, positions[0] - WINDOW + 1) // BLOCK
    hi = positions[-1] // BLOCK
    out = []
    for m in mgrs[:N_SWA]:
        sl = m.req_to_blocks[rid][lo : hi + 1]
        if len(sl) != hi + 1 - lo:
            return None  # a needed block is missing
        out.append(tuple((b.block_id, b.is_null, b.ref_cnt) for b in sl))
    return tuple(out)


def _window_ok(view):
    return view is not None and all(
        not null and ref > 0 for g in view for (_, null, ref) in g
    )


def _run(
    monkeypatch,
    n,
    prompt,
    out_len,
    mns,
    num_blocks,
    early=True,
    real=True,
    ml=ML,
    mnbt=MNBT,
    k=2,
    check_windows=True,
    trace=None,
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    for var, on in (
        ("VLLM_NEURON_SWA_REAL_ADMISSION", real),
        ("VLLM_NEURON_SWA_EARLY_FREE", early),
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
        sched.add_request(
            Request(f"r{i}", [7] * prompt, sp, None, arrival_time=float(i))
        )
    pool = sched.kv_cache_manager.block_pool
    inflight: deque = deque()
    s = {
        "preempted": 0,
        "max_live": 0,
        "max_used": 0,
        "steps": 0,
        "finished": 0,
        "stalled": False,
        "window_violations": 0,
        "max_prefill_phase_live": 0,
        "over_charge": 0,
    }
    mgrs = sched.kv_cache_manager.coordinator.single_type_managers
    while sched.has_requests() or inflight:
        if sched.has_requests() and len(inflight) < k:
            out = sched.schedule()
            s["preempted"] += len(out.preempted_req_ids or ())
            s["max_live"] = max(s["max_live"], len(sched.running))
            s["max_used"] = max(
                s["max_used"], pool.num_gpu_blocks - pool.get_num_free_blocks()
            )
            s["max_prefill_phase_live"] = max(
                s["max_prefill_phase_live"],
                sum(
                    1
                    for r in sched.running
                    if getattr(sched, "_swa_freed", {}).get(r.request_id)
                    != r.num_preemptions
                ),
            )
            s["steps"] += 1
            if sched._real_admission is not None:
                # Invariant behind "no preemption": nobody holds more than its charge.
                for r in sched.running:
                    held = sum(
                        sum(
                            1
                            for b in m.req_to_blocks.get(r.request_id, ())
                            if not b.is_null
                        )
                        for m in mgrs
                    )
                    if held > sched._real_peak_blocks(r):
                        s["over_charge"] += 1
            if trace is not None:
                trace.append(tuple(sorted(out.num_scheduled_tokens.items())))
            samples, windows = {}, {}
            for rid, nt in out.num_scheduled_tokens.items():
                r = sched.requests[rid]
                samples[rid] = r.num_computed_tokens >= r.num_prompt_tokens
                if check_windows:
                    # _update_after_schedule already advanced num_computed_tokens.
                    pos = range(r.num_computed_tokens - nt, r.num_computed_tokens)
                    windows[rid] = (pos, _swa_window_blocks(sched, rid, pos))
            inflight.append((out, samples, windows))
            if len(inflight) < k:
                continue
        out, samples, windows = inflight.popleft()
        # "Execute" the step: everything it attends must still be held as scheduled.
        for rid, (pos, held) in windows.items():
            if (
                rid in sched.requests
                and not sched.requests[rid].is_finished()
                and not (
                    _window_ok(held) and _swa_window_blocks(sched, rid, pos) == held
                )
            ):
                s["window_violations"] += 1
        ids = list(out.num_scheduled_tokens)
        mro = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: j for j, r in enumerate(ids)},
            sampled_token_ids=[[11] if samples[rid] else [] for rid in ids],
        )
        for v in sched.update_from_output(out, mro).values():
            s["finished"] += sum(1 for o in v.outputs if o.finish_reason is not None)
        if s["steps"] >= 50 * (n + out_len):
            s["stalled"] = True
            break
    s["cap"] = sched._max_kv_concurrent
    s["early_free_on"] = getattr(sched, "_swa_early_free", False)
    s["early_freed_blocks"] = getattr(sched, "_swa_early_freed_blocks", 0)
    return s


def test_requires_real_admission(monkeypatch):
    s = _run(
        monkeypatch, 2, 64, 4, 4, 2_000, early=True, real=False, check_windows=False
    )
    assert s["early_free_on"] is False and s["early_freed_blocks"] == 0


def test_small_shape_no_preemption_and_window_safe(monkeypatch):
    # 24 x (4096 + 256) into 6,000 blocks: 24 x 341 > 5,999, so admission binds.
    s = _run(monkeypatch, 24, PROMPT, 256, 24, 6_000)
    print("STATS_SMALL", s)
    assert not s["stalled"] and s["finished"] == 24
    assert s["preempted"] == 0 and s["window_violations"] == 0
    assert s["over_charge"] == 0 and s["max_live"] < 24
    assert s["early_freed_blocks"] == 24 * 576


def test_chunked_prefill_window_safe(monkeypatch):
    # prompt > mnbt: the free fires only after the last chunk is processed. A
    # decode allocate_slots can run first (vLLM frees on the processed basis then,
    # e.g. at 2048), so the hook frees at most the remainder.
    s = _run(monkeypatch, 12, 3000, 200, 12, 4_000, mnbt=1024)
    print("STATS_CHUNKED", s)
    assert not s["stalled"] and s["finished"] == 12
    assert s["preempted"] == 0 and s["window_violations"] == 0
    assert s["over_charge"] == 0
    assert 0 < s["early_freed_blocks"] <= 12 * N_SWA * ((3000 - WINDOW + 1) // BLOCK)


def test_flag_off_schedule_unchanged(monkeypatch):
    """Flag off: the hook never runs and the schedule equals the f331fdfe one
    (real-peak admission only, prefill-phase charge 903 -> at most 78 live)."""
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    def boom(self, *a, **kw):
        raise AssertionError("early free ran with the flag off")

    monkeypatch.setattr(NeuronScheduler, "_free_swa_after_prefill", boom)
    tr = []
    s = _run(
        monkeypatch,
        90,
        PROMPT,
        64,
        90,
        POOL,
        early=False,
        check_windows=False,
        trace=tr,
    )
    print("STATS_OFF", s)
    assert s["early_free_on"] is False and s["early_freed_blocks"] == 0
    assert s["preempted"] == 0 and s["finished"] == 90
    assert s["max_live"] == 78


def test_negative_control_charge_without_free(monkeypatch):
    """Charge the decode phase but skip the actual free: the pool over-commits and
    the real KV manager must preempt or stall. Proves the free is what makes the
    lower charge safe and that the harness can see it."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    monkeypatch.setattr(
        KVCacheManager, "remove_skipped_blocks", lambda self, *a, **k: None
    )
    s = _run(monkeypatch, 24, PROMPT, 256, 24, 6_000, check_windows=False)
    print("STATS_NEG", s)
    assert s["preempted"] > 0 or s["stalled"]


def test_harness_detects_too_early_free(monkeypatch):
    """Mutant: free two blocks past the processed-token boundary. Steps still in
    flight then attend freed blocks and the window check must flag them."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    orig = KVCacheManager.remove_skipped_blocks

    def ahead(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        return orig(
            self, request_id, processed_computed_tokens + 2 * BLOCK, num_prompt_tokens
        )

    monkeypatch.setattr(KVCacheManager, "remove_skipped_blocks", ahead)
    s = _run(monkeypatch, 8, PROMPT, 256, 8, 6_000)
    print("STATS_MUTANT", s)
    assert s["window_violations"] > 0


@pytest.mark.slow
def test_147_x_8192_no_preemption(monkeypatch):
    """Target: 147 x (4096 in, 4096 out) in a 70,778-block pool, ml 8192, mnbt 4224,
    async K=2: zero preemptions, all in-flight windows intact."""
    s = _run(monkeypatch, 147, PROMPT, OUT, 147, POOL)
    print("STATS_147", s)
    assert not s["stalled"] and s["finished"] == 147
    assert s["preempted"] == 0 and s["window_violations"] == 0
    assert s["max_live"] >= 140
    assert s["max_used"] <= POOL and s["over_charge"] == 0
    assert s["early_freed_blocks"] == 147 * 576


@pytest.mark.slow
def test_147_x_8192_flag_off_caps_at_78(monkeypatch):
    s = _run(monkeypatch, 147, PROMPT, OUT, 147, POOL, early=False, check_windows=False)
    print("STATS_147_OFF", s)
    assert not s["stalled"] and s["finished"] == 147 and s["preempted"] == 0
    assert s["max_live"] == 78 and s["early_freed_blocks"] == 0
