# SPDX-License-Identifier: Apache-2.0
"""Async prefill reorder for Qwen3-ASR (default on; ``VLLM_NEURON_ASYNC_PREFILL_REORDER=0`` = off).

The first decode step after a prefill used to read the prefill's sampled token back to the
host, because the prefill future has one row and the decode bucket has many. With the opt-in
the decode graph takes one more input, ``async_reorder_future_ext``, appended after the decode
future, and the permutation points the new request's slot at it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_neuron.model.qwen3_asr.model_bf16 import Qwen3ASRForConditionalGeneration
from vllm_neuron.vllm.worker import neuron_model_runner as nmr
from vllm_neuron.vllm.worker.neuron_model_runner import (
    NeuronModelRunner,
    _async_prefill_ext_noop,
    _supports_async_prefill_reorder,
)

ENV = "VLLM_NEURON_ASYNC_PREFILL_REORDER"


class _Runner:
    _use_async_reorder = NeuronModelRunner._use_async_reorder
    _build_async_reorder_kwargs = NeuronModelRunner._build_async_reorder_kwargs
    _noop_async_reorder_kwargs = staticmethod(NeuronModelRunner._noop_async_reorder_kwargs)
    _is_decode_row = NeuronModelRunner._is_decode_row
    _async_prefill_reorder_steps = 0

    def __init__(self, req_ids, computed=None, prompt=8):
        self.model = SimpleNamespace(
            supports_async_reorder=True,
            supports_async_reorder_multimodal=True,
            supports_async_reorder_prefill=True,
        )
        self.supports_mm_inputs = True
        self.use_async_scheduling = True
        self.is_spec_decode = False
        self._model_supports_async_reorder = None
        self.device = torch.device("cpu")
        self.async_execution_buffer = {}
        req_ids = list(req_ids)
        self.requests = {r: SimpleNamespace(num_prompt_tokens=prompt) for r in req_ids}
        if computed is None:
            computed = [prompt] * len(req_ids)
        self.input_batch = SimpleNamespace(
            req_ids=req_ids,
            num_reqs=len(req_ids),
            num_computed_tokens_cpu=np.array(computed or [prompt], dtype=np.int32),
        )


def _assemble(kwargs, host):
    """Mirror the model's forward: append the ext slot, then the shared reorder block."""
    future = kwargs["async_reorder_future"]
    if "async_reorder_future_ext" in kwargs:
        future = torch.cat([future, kwargs["async_reorder_future_ext"]])
    return torch.where(
        kwargs["async_reorder_mask"],
        host,
        torch.index_select(future, 0, kwargs["async_reorder_perm"]),
    )


def test_model_declares_prefill_capability():
    assert Qwen3ASRForConditionalGeneration.supports_async_reorder_prefill is True


@pytest.mark.parametrize(
    ("env", "flag", "expected"),
    [("1", True, True), ("0", True, False), (None, True, True), (None, False, False)],
    ids=["on", "env-0", "env-unset-default-on", "model-no-flag"],
)
def test_gate(monkeypatch, env, flag, expected):
    if env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    model = SimpleNamespace(supports_async_reorder_prefill=flag)
    assert _supports_async_prefill_reorder(model) is expected
    assert _supports_async_prefill_reorder(SimpleNamespace(_orig_mod=model)) is expected


def test_noop_ext_only_when_allowed():
    off = SimpleNamespace(_prefill_async_reorder_allowed=False)
    on = SimpleNamespace(_prefill_async_reorder_allowed=True)
    assert _async_prefill_ext_noop(off, torch.int32, torch.device("cpu")) == {}
    ext = _async_prefill_ext_noop(on, torch.int32, torch.device("cpu"))
    assert list(ext) == ["async_reorder_future_ext"]
    assert ext["async_reorder_future_ext"].shape == (nmr._ASYNC_PREFILL_REORDER_SLOTS,)
    assert ext["async_reorder_future_ext"].dtype == torch.int32


def _runner(monkeypatch, env, req_ids, computed=None):
    monkeypatch.setenv(ENV, env)
    monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", "1")
    runner = _Runner(req_ids, computed)
    assert runner._use_async_reorder()
    return runner


def test_pre_to_dec_reads_the_prefill_future(monkeypatch):
    """Prefill step ran only d (held-back decodes a, b). Next decode [a, b, d] + 1 pad slot.

    a and b were not in the prefill step (removed, re-added from all_token_ids) -> host ids.
    d takes the prefill's sampled id from the ext slot; the pad slot reads the host.
    """
    runner = _runner(monkeypatch, "1", ["a", "b", "d"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([555], dtype=torch.int32),
        "prev_req_ids_ordered": ["d"],
    }
    host = torch.tensor([11, 12, -1, 0], dtype=torch.int32)
    kwargs, usable = runner._build_async_reorder_kwargs(host)
    assert usable
    assert kwargs["async_reorder_mask"].tolist() == [True, True, False, True]
    assert kwargs["async_reorder_perm"].tolist()[2] == 4  # num_slots + 0
    assert _assemble(kwargs, host).tolist() == [11, 12, 555, 0]
    assert runner._async_prefill_reorder_steps == 1


def test_uint32_prefill_future_is_reinterpreted(monkeypatch):
    runner = _runner(monkeypatch, "1", ["d"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([7], dtype=torch.int32).view(torch.uint32),
        "prev_req_ids_ordered": ["d"],
    }
    host = torch.tensor([-1, 0], dtype=torch.int32)
    kwargs, usable = runner._build_async_reorder_kwargs(host)
    assert usable and kwargs["async_reorder_future_ext"].dtype == torch.int32
    assert _assemble(kwargs, host).tolist() == [7, 0]


def test_env_off_keeps_the_host_path(monkeypatch):
    """Without the opt-in the prefill-shaped future declines and no ext input is added."""
    runner = _runner(monkeypatch, "0", ["a", "d"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([555], dtype=torch.int32),
        "prev_req_ids_ordered": ["d"],
    }
    kwargs, usable = runner._build_async_reorder_kwargs(torch.tensor([1, 2], dtype=torch.int32))
    assert not usable
    assert "async_reorder_future_ext" not in kwargs
    assert kwargs["async_reorder_mask"].all()


def test_decode_to_decode_carries_a_noop_ext(monkeypatch):
    """A steady decode step keeps its own future and sends the zero ext slot (same NEFF inputs)."""
    runner = _runner(monkeypatch, "1", ["a", "b"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([101, 102], dtype=torch.int32),
        "prev_req_ids_ordered": ["a", "b"],
    }
    host = torch.tensor([-1, -1], dtype=torch.int32)
    kwargs, usable = runner._build_async_reorder_kwargs(host)
    assert usable
    assert kwargs["async_reorder_future_ext"].tolist() == [0]
    assert kwargs["async_reorder_perm"].tolist() == [0, 1]
    assert _assemble(kwargs, host).tolist() == [101, 102]
    assert runner._async_prefill_reorder_steps == 0


def test_noop_path_also_carries_ext(monkeypatch):
    """First step (no future): the no-op reorder still has the ext input."""
    runner = _runner(monkeypatch, "1", ["a"])
    kwargs, usable = runner._build_async_reorder_kwargs(torch.tensor([3, 0], dtype=torch.int32))
    assert not usable
    assert set(kwargs) == {
        "async_reorder_future",
        "async_reorder_perm",
        "async_reorder_mask",
        "async_reorder_future_ext",
    }


def test_partial_prefill_row_still_takes_host(monkeypatch):
    """The prefill future's request is still prefilling (chunked): its slot keeps the host id."""
    runner = _runner(monkeypatch, "1", ["d"], computed=[5])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([555], dtype=torch.int32),
        "prev_req_ids_ordered": ["d"],
    }
    host = torch.tensor([42, 0], dtype=torch.int32)
    kwargs, usable = runner._build_async_reorder_kwargs(host)
    assert usable
    assert _assemble(kwargs, host).tolist() == [42, 0]


def test_multi_row_prefill_future_declines(monkeypatch):
    """A prefill future wider than the ext slot is not used (host path)."""
    runner = _runner(monkeypatch, "1", ["a", "b", "c"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([1, 2], dtype=torch.int32),
        "prev_req_ids_ordered": ["a", "b"],
    }
    kwargs, usable = runner._build_async_reorder_kwargs(torch.tensor([0, 0, 0, 0], dtype=torch.int32))
    assert not usable


def test_forward_appends_the_ext_slot():
    seen = {}

    def language_model(input_ids, positions, **kw):
        seen["ids"] = input_ids.clone()
        seen["kw"] = set(kw)
        return torch.zeros(input_ids.shape[0], 4)

    stub = SimpleNamespace(
        world_size=1,
        language_model=language_model,
        _fused_lmhead=False,
        lm_head=lambda h: h,
        _gather_logits=False,
        on_device_sampling_config=None,
    )
    meta = {"layers.0.self_attn": {"max_query_len": 1, "decode_token_threshold": 1}}
    host = torch.tensor([11, -1, 0], dtype=torch.int32)
    Qwen3ASRForConditionalGeneration.forward(
        stub,
        host,
        torch.zeros(3, dtype=torch.int32),
        attn_metadata=meta,
        sampling_positions=torch.arange(3),
        async_reorder_future=torch.zeros(3, dtype=torch.int32),
        async_reorder_perm=torch.tensor([0, 3, 0]),
        async_reorder_mask=torch.tensor([True, False, True]),
        async_reorder_future_ext=torch.tensor([555], dtype=torch.int32),
    )
    assert seen["ids"].tolist() == [11, 555, 0]
    assert not any(k.startswith("async_reorder") for k in seen["kw"])
