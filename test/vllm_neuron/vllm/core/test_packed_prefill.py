# SPDX-License-Identifier: Apache-2.0
"""Packed prefill (VLLM_NEURON_MAX_PREFILLS_PER_BATCH): scheduler admission.

The real NeuronAsyncScheduler runs on vLLM's real Scheduler / KVCacheManager with
two steps in flight. A fake model runner returns one token per request that
finished its prompt. Checks: every prefill step holds only whole prompts, at most
N of them, their total fits the packed limit, prefill and decode never mix, and
every request finishes. With the knob unset the scheduler keeps one prefill per
step.
"""

from collections import deque
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from vllm_neuron.vllm.core import packed_prefill as pp

BLOCK = 32


@dataclass
class _SchedCfg:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 4096
    max_num_scheduled_tokens: int = 4096
    max_model_len: int = 4096
    policy: str = "fcfs"
    enable_chunked_prefill: bool = True
    long_prefill_token_threshold: int = 0
    scheduler_reserve_full_isl: bool = True
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    max_num_encoder_input_tokens: int = 0
    disable_chunked_mm_input: bool = False
    disable_hybrid_kv_cache_manager: bool = False
    async_scheduling: bool = True
    encoder_cache_size: int = 0
    stream_interval: int = 0
    watermark: float = 0.0


class _Cfg:
    def __getattr__(self, name):
        return None


def _vllm_config(mns, ml, mnbt, buckets):
    import torch

    cfg = _Cfg()
    sc = _SchedCfg(max_num_seqs=mns, max_num_batched_tokens=mnbt,
                   max_num_scheduled_tokens=mnbt, max_model_len=ml)
    mc = MagicMock()
    for k, v in dict(max_model_len=ml, is_encoder_decoder=False, is_multimodal_model=False,
                     enable_return_routed_experts=False, uses_mrope=False, uses_xdrope=False,
                     architectures=["MistralForCausalLM"], runner_type="generate",
                     enable_prompt_embeds=False, model="m").items():
        setattr(mc, k, v)
    cc = MagicMock()
    for k, v in dict(block_size=BLOCK, num_gpu_blocks=4000, enable_prefix_caching=False,
                     cache_dtype=torch.bfloat16, mamba_cache_mode="none").items():
        setattr(cc, k, v)
    obs = MagicMock()
    for k, v in dict(kv_cache_metrics=False, kv_cache_metrics_sample=0.01, enable_mfu_metrics=False,
                     per_request_spec_decode_metrics="none").items():
        setattr(obs, k, v)
    vals = dict(
        scheduler_config=sc, model_config=mc, cache_config=cc, observability_config=obs,
        additional_config={"neuron_config": {"num_batched_tokens_buckets": buckets}},
        lora_config=None, kv_events_config=None, speculative_config=None, kv_transfer_config=None,
        ec_transfer_config=None, num_speculative_tokens=0, num_lookahead_tokens=0,
        ec_manager_config=Mock(get_encoder_cache_manager_obj=Mock(return_value=None)),
        parallel_config=Mock(data_parallel_rank=0, data_parallel_index=0, data_parallel_size=1,
                             pipeline_parallel_size=1, decode_context_parallel_size=1,
                             prefill_context_parallel_size=1),
        max_concurrent_batches=2, max_in_flight_tokens=2 * mnbt, use_v2_model_runner=False,
    )
    for k, v in vals.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _kv_cache_config(num_blocks):
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    common = dict(block_size=BLOCK, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec(["swa0"], SlidingWindowSpec(sliding_window=64, **common)),
        KVCacheGroupSpec(["full0"], FullAttentionSpec(**common)),
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def _run(monkeypatch, lens, n_max, out_len=6, mns=16, buckets=(128, 256, 512, 1024, 2048, 4096),
         limit=None, swa=False):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler

    monkeypatch.setenv(pp.ENV, str(n_max))
    if limit is not None:
        monkeypatch.setenv(pp.BUDGET_ENV, str(limit))
    else:
        monkeypatch.delenv(pp.BUDGET_ENV, raising=False)
    for k in ("VLLM_NEURON_SWA_REAL_ADMISSION", "VLLM_NEURON_SWA_EARLY_FREE"):
        if swa:
            monkeypatch.setenv(k, "1")
        else:
            monkeypatch.delenv(k, raising=False)
    ml = 4096
    cfg = _vllm_config(mns, ml, ml, list(buckets))
    sched = NeuronAsyncScheduler(
        vllm_config=cfg, kv_cache_config=_kv_cache_config(4000),
        structured_output_manager=Mock(should_advance=Mock(return_value=False)), block_size=BLOCK,
    )
    for i, L in enumerate(lens):
        sp = SamplingParams(max_tokens=out_len, ignore_eos=True, temperature=0.0)
        sched.add_request(Request(f"r{i}", [7] * L, sp, None, arrival_time=float(i)))
    prefill_steps = []
    inflight: deque = deque()
    finished = steps = 0
    while sched.has_requests() or inflight:
        if sched.has_requests() and len(inflight) < 2:
            out = sched.schedule()
            steps += 1
            new = {r.req_id: r.num_computed_tokens for r in out.scheduled_new_reqs}
            pre = []
            dec = 0
            for rid, n in out.num_scheduled_tokens.items():
                req = sched.requests[rid]
                c0 = new.get(rid, req.num_computed_tokens - n)
                if c0 < req.num_prompt_tokens:
                    pre.append((rid, n, c0, req.num_prompt_tokens))
                else:
                    dec += 1
            assert not (pre and dec), "prefill and decode mixed"
            if pre:
                prefill_steps.append((pre, dict(out.num_scheduled_tokens_padded)))
            samples = {rid: sched.requests[rid].num_computed_tokens >= sched.requests[rid].num_prompt_tokens
                       for rid in out.num_scheduled_tokens}
            inflight.append((out, samples))
            if len(inflight) < 2:
                continue
        out, samples = inflight.popleft()
        ids = list(out.num_scheduled_tokens)
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: j for j, r in enumerate(ids)},
                                sampled_token_ids=[[11] if samples[r] else [] for r in ids])
        for v in sched.update_from_output(out, mro).values():
            finished += sum(1 for o in v.outputs if o.finish_reason is not None)
        assert steps < 50 * (len(lens) + out_len), "stalled"
    return sched, prefill_steps, finished


def test_pack_padding_and_bounds():
    assert pp.pack_padding([13, 40, 70], [128, 256, 512]) == [13, 40, 75]
    lo, hi = pp.segment_bounds([13, 40], 128)
    assert lo[:13].tolist() == [0] * 13 and hi[12] == 13
    assert lo[13] == 13 and hi[52] == 53 and lo[53] == 53 and hi[127] == 128
    assert pp.packed_bucket_limit([128, 256, 512, 1024, 2048, 4096], 8192) == 2048
    assert pp.effective_rows(8, [128]) == 8


def test_default_one_prefill_per_step(monkeypatch):
    sched, steps, finished = _run(monkeypatch, [100] * 8, n_max=1)
    assert sched.max_prefills_per_batch == 1
    assert all(len(p) == 1 for p, _ in steps)
    assert finished == 8


@pytest.mark.parametrize("n_max", [2, 4, 8])
@pytest.mark.parametrize("swa", [False, True])
def test_packs_whole_prompts(monkeypatch, n_max, swa):
    lens = [100, 13, 40, 300, 5, 1500, 64, 65, 129, 2047, 30, 200, 3000, 17, 128, 900]
    sched, steps, finished = _run(monkeypatch, lens, n_max=n_max, swa=swa)
    assert sched.max_prefills_per_batch == n_max
    assert finished == len(lens)
    limit = sched._packed_token_budget
    assert limit == 2048
    for pre, padded in steps:
        assert len(pre) <= n_max
        total = sum(n for _, n, _, _ in pre)
        if len(pre) > 1:
            assert total <= limit
            for rid, n, c0, P in pre:
                assert c0 == 0 and n == P, "packed prompt not whole"
        for rid, n, c0, P in pre:
            want = n if 1 < n and pp.bucket_for(n, [128, 256, 512, 1024, 2048, 4096]) <= limit else None
            if want is not None:
                assert padded[rid] == n
    assert 2 <= max(len(p) for p, _ in steps) <= n_max
    # Prompts above the packed limit (2047 -> bucket 4096, 3000) run alone.
    big = {f"r{i}" for i, L in enumerate(lens) if L > limit}
    for pre, _ in steps:
        if any(rid in big for rid, *_ in pre):
            assert len(pre) == 1


def test_llama_family_opts_in(monkeypatch):
    cfg = _vllm_config(16, 2048, 2048, [128, 256, 512, 1024, 2048])
    monkeypatch.setenv(pp.ENV, "8")
    for arch in ("LlamaForCausalLM", "MistralForCausalLM"):
        cfg.model_config.architectures = [arch]
        cfg.model_config.hf_config.quantization_config = None
        assert pp.resolve_max_prefills(cfg) == (8, None)
        cfg.model_config.hf_config.quantization_config = {"quant_method": "modelopt"}
        n, reason = pp.resolve_max_prefills(cfg)
        assert n == 1 and "bf16" in reason


def test_limit_env(monkeypatch):
    sched, steps, finished = _run(monkeypatch, [100] * 12, n_max=8, limit=256)
    assert sched._packed_token_budget == 256
    assert all(sum(n for _, n, _, _ in p) <= 256 for p, _ in steps if len(p) > 1)
    assert max(len(p) for p, _ in steps) == 2
    assert finished == 12


def test_unsupported_model_stays_single(monkeypatch):
    from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler  # noqa: F401

    cfg = _vllm_config(16, 4096, 4096, [128, 256])
    cfg.model_config.architectures = ["Qwen3ForCausalLM"]
    monkeypatch.setenv(pp.ENV, "4")
    n, reason = pp.resolve_max_prefills(cfg)
    assert n == 1 and "supports_packed_prefill" in reason
    cfg.model_config.architectures = ["MistralForCausalLM"]
    cfg.scheduler_config.max_num_batched_tokens = 1024
    n, reason = pp.resolve_max_prefills(cfg)
    assert n == 1 and "segmented" in reason
