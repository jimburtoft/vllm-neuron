# SPDX-License-Identifier: Apache-2.0
"""Real-peak KV admission (VLLM_NEURON_SWA_REAL_ADMISSION) for hybrid SWA models.

Two levels:
* the pure arithmetic in kv_admission.py (Gemma3-4B shape: 6 sliding-window
  groups of window 1024 + 1 full-attention group, block 32);
* an end-to-end run of the real NeuronAsyncScheduler on vLLM's real Scheduler /
  KVCacheManager, with two steps in flight (async K=2), asserting that no request
  is ever preempted and that the block pool never runs out.
"""

from collections import deque
from unittest.mock import Mock

import pytest

from vllm_neuron.vllm.core import kv_admission as ka

from test.vllm_neuron.vllm.core.conftest import (
    _make_mock_observability_config,
    _make_mock_vllm_config,
)

BLOCK = 32
WINDOW = 1024
N_SWA = 6
POOL = 70_778
ML = 4096
MNBT = 2176


def _gemma3_groups(ml=ML, k=2, mnbt=MNBT):
    swa_upstream = ka.cdiv(min(WINDOW - 1 + k * mnbt, ml), BLOCK) + 1
    full_upstream = ka.cdiv(ml, BLOCK)
    return [ka.GroupDesc("swa", BLOCK, swa_upstream, WINDOW)] * N_SWA + [
        ka.GroupDesc("full", BLOCK, full_upstream)
    ]


class TestArithmetic:
    def test_upstream_count_cap_is_78(self):
        p = ka.make_params(_gemma3_groups(), POOL, ML, 2)
        # 6 x (cdiv(4096, 32) + 1) + 128 = 902 -> 70,778 // 902 = 78 (matches the log).
        assert p.upstream_blocks_per_request == 902
        assert p.legacy_cap == 78

    def test_real_peak_2048_prompt(self):
        p = ka.make_params(_gemma3_groups(), POOL, ML, 2)
        # prefill end: 7 x cdiv(2048 + 2, 32) = 455; decode: 6 x 34 + 128 = 332.
        assert ka.prefill_phase_blocks(p, 2048) == 455
        assert ka.decode_phase_blocks(p, 4096) == 332
        assert ka.request_peak_blocks(p, 2048, 4096) == 455
        assert ka.uniform_cap(p, 2048, 2048) == 155

    def test_sync_scheduler_same_peak_class(self):
        p = ka.make_params(_gemma3_groups(k=1), POOL, ML, 1)
        assert ka.uniform_cap(p, 2048, 2048) == 155

    def test_never_above_upstream_charge(self):
        for prompt in (1, 31, 32, 33, 1000, 2048, 4000, 4095):
            for k in (1, 2):
                p = ka.make_params(_gemma3_groups(k=k), POOL, ML, k)
                assert ka.request_peak_blocks(p, prompt, ML) <= p.upstream_blocks_per_request

    def test_short_context_swa_equals_full(self):
        # window >= max_model_len: SWA holds everything, the bound equals full attention.
        groups = [ka.GroupDesc("swa", 16, ka.cdiv(128, 16), 1024), ka.GroupDesc("full", 16, 8)]
        p = ka.make_params(groups, 100, 128, 2)
        assert ka.request_peak_blocks(p, 100, 128) == 16


# ---------------------------------------------------------------------------
# End-to-end: real vLLM Scheduler + KVCacheManager under NeuronAsyncScheduler.
# ---------------------------------------------------------------------------


def _kv_cache_config(num_blocks, ml, block):
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    common = dict(block_size=block, num_kv_heads=1, head_size=256, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec([f"swa{i}"], SlidingWindowSpec(sliding_window=WINDOW, **common))
        for i in range(N_SWA)
    ] + [KVCacheGroupSpec(["full0"], FullAttentionSpec(**common))]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def _vllm_config(max_num_seqs, ml, mnbt, k):
    cfg = _make_mock_vllm_config()
    sc = cfg.scheduler_config
    sc.max_num_seqs = max_num_seqs
    sc.max_num_batched_tokens = mnbt
    sc.max_num_scheduled_tokens = mnbt
    sc.max_model_len = ml
    sc.async_scheduling = k > 1
    sc.scheduler_reserve_full_isl = True
    cfg.model_config.max_model_len = ml
    cfg.cache_config.block_size = BLOCK
    cfg.additional_config = {}
    cfg.parallel_config = Mock(
        data_parallel_rank=0,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    cfg.observability_config = _make_mock_observability_config()
    # Real VllmConfig properties the scheduler / KV manager read.
    cfg.max_concurrent_batches = k
    cfg.max_in_flight_tokens = k * mnbt
    cfg.num_speculative_tokens = 0
    cfg.num_lookahead_tokens = 0
    cfg.use_v2_model_runner = False
    return cfg


def _run(n_requests, prompt_len, out_len, max_num_seqs, num_blocks, ml=ML, mnbt=MNBT, k=2,
         real=True, monkeypatch=None):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    if real:
        monkeypatch.setenv("VLLM_NEURON_SWA_REAL_ADMISSION", "1")
    else:
        monkeypatch.delenv("VLLM_NEURON_SWA_REAL_ADMISSION", raising=False)
    cfg = _vllm_config(max_num_seqs, ml, mnbt, k)
    sched = NeuronAsyncScheduler(
        vllm_config=cfg,
        kv_cache_config=_kv_cache_config(num_blocks, ml, BLOCK),
        structured_output_manager=Mock(should_advance=Mock(return_value=False)),
        block_size=BLOCK,
    )
    for i in range(n_requests):
        sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
        sched.add_request(
            Request(f"r{i}", [7] * prompt_len, sp, None, arrival_time=float(i))
        )
    pool = sched.kv_cache_manager.block_pool
    inflight: deque = deque()
    stats = {"preempted": 0, "max_live": 0, "max_used": 0, "steps": 0, "finished": 0,
             "stalled": False}
    while sched.has_requests() or inflight:
        if sched.has_requests() and len(inflight) < k:
            out = sched.schedule()
            stats["preempted"] += len(out.preempted_req_ids or ())
            stats["max_live"] = max(stats["max_live"], len(sched.running))
            stats["max_used"] = max(stats["max_used"], pool.num_gpu_blocks - pool.get_num_free_blocks())
            stats["steps"] += 1
            # Which requests sample a token in this step (decode or last prefill
            # chunk), decided now, before a later in-flight step advances them.
            samples = {
                rid: sched.requests[rid].num_computed_tokens
                >= sched.requests[rid].num_prompt_tokens
                for rid in out.num_scheduled_tokens
            }
            inflight.append((out, samples))
            if len(inflight) < k:
                continue
        out, samples = inflight.popleft()
        ids = list(out.num_scheduled_tokens)
        sampled = [[11] if samples[rid] else [] for rid in ids]
        mro = ModelRunnerOutput(
            req_ids=ids, req_id_to_index={r: j for j, r in enumerate(ids)},
            sampled_token_ids=sampled,
        )
        eco = sched.update_from_output(out, mro)
        for v in eco.values():
            stats["finished"] += sum(1 for o in v.outputs if o.finish_reason is not None)
        if stats["steps"] >= 50 * (n_requests + out_len):
            stats["stalled"] = True
            break
    stats["cap"] = sched._max_kv_concurrent
    return stats


@pytest.mark.parametrize(
    "n,prompt,out,mns,pool,ml,mnbt",
    [
        # Scaled-down shape (fast): window 1024, prompt 2048, ml 4096, small pool.
        (24, 2048, 256, 24, 11_000, ML, MNBT),
        # Mixed lengths around a chunked prefill (prompt > mnbt).
        (16, 3000, 200, 16, 8_000, ML, MNBT),
    ],
)
def test_real_admission_no_preemption_small(monkeypatch, n, prompt, out, mns, pool, ml, mnbt):
    s = _run(n, prompt, out, mns, pool, ml, mnbt, monkeypatch=monkeypatch)
    assert not s["stalled"]
    assert s["preempted"] == 0
    assert s["finished"] == n
    assert s["max_used"] <= pool


def test_default_off_keeps_count_cap(monkeypatch):
    s = _run(4, 2048, 8, 155, POOL, monkeypatch=monkeypatch, real=False)
    assert s["cap"] == 78


@pytest.mark.slow
def test_real_admission_155_x_2048_no_preemption(monkeypatch):
    """The target config: 155 x (2048 in, 2048 out) in a 70,778-block pool, K=2."""
    s = _run(155, 2048, 2048, 155, POOL, monkeypatch=monkeypatch)
    print("STATS_155", s)
    assert not s["stalled"]
    assert s["preempted"] == 0
    assert s["finished"] == 155
    assert s["max_live"] >= 140
    assert s["max_used"] <= POOL


def test_harness_detects_preemption_when_bound_too_small(monkeypatch):
    """Negative control: charging only the decode-phase bound over-admits, and the
    real KV manager must then preempt. Proves the harness can see preemptions."""
    monkeypatch.setattr(
        ka, "request_peak_blocks", lambda p, pre, fin: ka.decode_phase_blocks(p, fin)
    )
    s = _run(24, 2048, 256, 24, 7_000, monkeypatch=monkeypatch)
    print("STATS_NEG", s)
    # Over-admission shows up either as preemptions or as a stall: vLLM's own
    # full-sequence gate refuses the over-admitted prefill while the Neuron
    # scheduler hides the decodes for it, so no step makes progress.
    assert s["preempted"] > 0 or s["stalled"]
