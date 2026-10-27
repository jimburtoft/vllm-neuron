# SPDX-License-Identifier: Apache-2.0
"""In-graph async decode reorder for Qwen3-ASR (multimodal opt-in).

The runner keeps multimodal models on the host swap path. Qwen3-ASR opts out of that exclusion
with ``supports_async_reorder_multimodal`` (default on; ``VLLM_NEURON_MM_ASYNC_REORDER=0`` turns it off). These tests
check the gate, that the Qwen3-ASR forward holds the same reorder block as the GPT-OSS models,
and that the block assembles the right decode input_ids.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_neuron.model
from vllm_neuron.model.qwen3_asr.model_bf16 import Qwen3ASRForConditionalGeneration
from vllm_neuron.vllm.worker.neuron_model_runner import (
    NeuronModelRunner,
    _supports_async_reorder,
    _supports_async_reorder_multimodal,
)

MODEL_ROOT = Path(vllm_neuron.model.__file__).parent
BLOCK_START = "        # Assemble the decode input_ids on the device."


def _reorder_block(path: Path) -> str:
    source = path.read_text()
    start = source.index(BLOCK_START)
    end = source.index(
        "torch.index_select(async_reorder_future, 0, async_reorder_perm),", start
    )
    end = source.index("            )\n", end) + len("            )\n")
    return source[start:end]


class _Runner:
    _use_async_reorder = NeuronModelRunner._use_async_reorder
    _build_async_reorder_kwargs = NeuronModelRunner._build_async_reorder_kwargs
    _noop_async_reorder_kwargs = staticmethod(NeuronModelRunner._noop_async_reorder_kwargs)
    _is_decode_row = NeuronModelRunner._is_decode_row

    def __init__(self, model, mm=True, async_sched=True, spec=False, req_ids=()):
        self.model = model
        self.supports_mm_inputs = mm
        self.use_async_scheduling = async_sched
        self.is_spec_decode = spec
        self._model_supports_async_reorder = None
        self.device = torch.device("cpu")
        self.async_execution_buffer = {}
        req_ids = list(req_ids)
        self.requests = {r: SimpleNamespace(num_prompt_tokens=8) for r in req_ids}
        self.input_batch = SimpleNamespace(
            req_ids=req_ids,
            num_reqs=len(req_ids),
            num_computed_tokens_cpu=np.full(max(len(req_ids), 1), 8, dtype=np.int32),
        )


def test_qwen3_asr_declares_both_capabilities():
    assert Qwen3ASRForConditionalGeneration.supports_async_reorder is True
    assert Qwen3ASRForConditionalGeneration.supports_async_reorder_multimodal is True


def test_reorder_block_matches_gpt_oss():
    qwen = _reorder_block(MODEL_ROOT / "qwen3_asr" / "model_bf16.py")
    gpt = _reorder_block(MODEL_ROOT / "gpt_oss" / "model_mxfp4.py")
    assert qwen == gpt


def test_only_qwen3_asr_opts_into_multimodal():
    hits = sorted(
        str(p.relative_to(MODEL_ROOT))
        for p in MODEL_ROOT.rglob("*.py")
        if "supports_async_reorder_multimodal = True" in p.read_text()
    )
    assert hits == ["qwen3_asr/model_bf16.py"]


@pytest.mark.parametrize(
    ("env", "mm_flag", "mm", "expected"),
    [
        ("1", True, True, True),
        ("0", True, True, False),
        (None, True, True, True),
        ("1", False, True, False),
        ("1", True, False, True),
        ("0", True, False, True),
    ],
    ids=["on", "env-0", "env-unset", "model-no-mm-flag", "text-only-on", "text-only-env-0"],
)
def test_gate(monkeypatch, env, mm_flag, mm, expected):
    if env is None:
        monkeypatch.delenv("VLLM_NEURON_MM_ASYNC_REORDER", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", env)
    model = SimpleNamespace(supports_async_reorder=True)
    if mm_flag:
        model.supports_async_reorder_multimodal = True
    assert _Runner(model, mm=mm)._use_async_reorder() is expected


@pytest.mark.parametrize("spec,async_sched", [(True, True), (False, False)], ids=["spec", "sync"])
def test_gate_other_exclusions_hold(monkeypatch, spec, async_sched):
    monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", "1")
    model = SimpleNamespace(supports_async_reorder=True, supports_async_reorder_multimodal=True)
    assert _Runner(model, spec=spec, async_sched=async_sched)._use_async_reorder() is False


def test_probe_unwraps_compiled_module(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", "1")
    inner = SimpleNamespace(supports_async_reorder=True, supports_async_reorder_multimodal=True)
    wrapper = SimpleNamespace(_orig_mod=inner)
    assert _supports_async_reorder(wrapper) and _supports_async_reorder_multimodal(wrapper)


def test_finish_and_admit_composition(monkeypatch):
    """Previous decode [a, b, c]; c finished, d is new; condense moved d into slot 1.

    a keeps its own sampled token, b moved slots and still gets its own, d (new) and the pad
    slot read the host ids. This is the batch-composition change that used to force a
    blocking device-to-host read.
    """
    monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", "1")
    model = SimpleNamespace(supports_async_reorder=True, supports_async_reorder_multimodal=True)
    runner = _Runner(model, req_ids=["a", "d", "b"])
    future = torch.tensor([101, 102, 103, 0], dtype=torch.int32)
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": future,
        "prev_req_ids_ordered": ["a", "b", "c"],
    }
    host = torch.tensor([-1, 7, -1, 0], dtype=torch.int32)
    assert runner._use_async_reorder()
    kwargs, usable = runner._build_async_reorder_kwargs(host)
    assert usable
    out = torch.where(
        kwargs["async_reorder_mask"],
        host,
        torch.index_select(kwargs["async_reorder_future"], 0, kwargs["async_reorder_perm"]),
    )
    assert out.tolist() == [101, 7, 102, 0]


def test_prefill_shaped_future_declines(monkeypatch):
    """The prefill step's future has the prefill row count, not the decode bucket: no reorder."""
    monkeypatch.setenv("VLLM_NEURON_MM_ASYNC_REORDER", "1")
    model = SimpleNamespace(supports_async_reorder=True, supports_async_reorder_multimodal=True)
    runner = _Runner(model, req_ids=["a", "b"])
    runner.async_execution_buffer = {
        "futures_sampled_token_ids": torch.tensor([5], dtype=torch.int32),
        "prev_req_ids_ordered": ["a"],
    }
    kwargs, usable = runner._build_async_reorder_kwargs(torch.tensor([1, 2], dtype=torch.int32))
    assert not usable
    assert kwargs["async_reorder_mask"].all()


def test_forward_consumes_the_reorder_arguments():
    """Run the real forward on a stub self: the language model must see the assembled ids."""
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
    host = torch.tensor([-1, 7, -1, 0], dtype=torch.int32)
    out = Qwen3ASRForConditionalGeneration.forward(
        stub,
        host,
        torch.zeros(4, dtype=torch.int32),
        attn_metadata=meta,
        sampling_positions=torch.arange(4),
        async_reorder_future=torch.tensor([101, 102, 103, 0], dtype=torch.int32),
        async_reorder_perm=torch.tensor([0, 0, 1, 3]),
        async_reorder_mask=torch.tensor([False, True, False, True]),
    )
    assert seen["ids"].tolist() == [101, 7, 102, 0]
    assert not any(k.startswith("async_reorder") for k in seen["kw"])
    assert out.shape == (4, 4)
    # Without the arguments the host ids pass through (prefill / no-reorder steps).
    Qwen3ASRForConditionalGeneration.forward(
        stub, host, torch.zeros(4, dtype=torch.int32), attn_metadata=meta,
        sampling_positions=torch.arange(4),
    )
    assert seen["ids"].tolist() == host.tolist()
