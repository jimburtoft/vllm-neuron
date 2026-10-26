# SPDX-License-Identifier: Apache-2.0
"""Fused lm_head + greedy argmax (VLLM_NEURON_QWEN3_ASR_LMHEAD=fused) vs the stock lm_head -> Sampler path.

CPU (torch fallback) checks:
  * ``lmhead_argmax`` == ``torch.argmax(x @ W)`` incl. exact ties (first index) and ``combine_programs`` (LNC split).
  * ``Qwen3ASRForConditionalGeneration.forward`` tail on a stand-in model: fused returns the same token ids as the
    stock lm_head + all-greedy Sampler, at prefill (B=1) and decode (B=48) shapes; ``logit_mask`` falls back to stock.
  * the backend env switch parses and rejects unknown values; fused only engages at TP1 all-greedy without logits gather.

The NKI kernel itself is checked in the simulator and on trn2 by the Task 029 harness, not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.functional.lmhead_argmax import combine_programs, lmhead_argmax
from vllm_neuron.model.qwen3_asr import model_bf16
from vllm_neuron.model.qwen3_asr.model_bf16 import (
    Qwen3ASRForConditionalGeneration,
    get_lmhead_backend,
)

pytestmark = [pytest.mark.fast]


def _xw(B, H, V, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(H, V, generator=g) * 0.02).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("B", [1, 8, 48, 64])
def test_lmhead_argmax_matches_torch(B):
    x, w = _xw(B, 256, 5000, seed=B)
    ref = torch.argmax(torch.matmul(x, w), dim=-1)
    got = lmhead_argmax(x, w)
    assert got.dtype == torch.int32 and got.shape == (B,)
    assert torch.equal(got.long(), ref)


def test_lmhead_argmax_ties_first_index():
    x, w = _xw(4, 128, 3000)
    for b in range(4):
        col = x[b].float() / x[b].float().norm() * 3.0
        for c in (100 + b, 1700 + b, 2999):
            w[:, c] = col.to(torch.bfloat16)
    got = lmhead_argmax(x, w)
    lg = torch.matmul(x, w)
    assert torch.equal(got.long(), torch.argmax(lg, -1))
    assert got.tolist() == [100, 101, 102, 103]


def test_combine_programs():
    val = torch.tensor([[[1.0], [3.0], [5.0]], [[2.0], [3.0], [4.0]]])
    idx = torch.tensor([[[5], [7], [9]], [[1], [0], [2]]], dtype=torch.int32)
    assert combine_programs(val, idx, 10).tolist() == [11, 7, 9]
    assert combine_programs(val[:1], idx[:1], 10).tolist() == [5, 7, 9]


def test_backend_env(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_LMHEAD", raising=False)
    assert get_lmhead_backend() == "stock"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_LMHEAD", " Fused ")
    assert get_lmhead_backend() == "fused"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_LMHEAD", "fast")
    with pytest.raises(ValueError):
        get_lmhead_backend()


class _StandIn(torch.nn.Module):
    """Just enough of Qwen3ASRForConditionalGeneration for its forward tail."""

    def __init__(self, hidden, w, fused):
        super().__init__()
        self.world_size = 1
        self._gather_logits = False
        self._fused_lmhead = fused
        self.on_device_sampling_config = SimpleNamespace(all_greedy=True)
        self.language_model = lambda *a, **k: hidden
        self.lm_head = lambda x: torch.matmul(x, w)
        self.sampler = lambda logits, *a, **k: torch.argmax(logits, -1).to(torch.int32)


def _forward(stand_in, T, B, is_prefill, logit_mask=None):
    thr = 1
    md = {"layers.0.self_attn": {"max_query_len": T if is_prefill else 1, "decode_token_threshold": thr}}
    pos = torch.arange(T)
    sp = torch.tensor([T - 1]) if is_prefill else torch.arange(B)
    return Qwen3ASRForConditionalGeneration.forward(
        stand_in, torch.zeros(T, dtype=torch.long), pos, attn_metadata=md, sampling_positions=sp,
        logit_mask=logit_mask, rank=torch.tensor(0)
    )


@pytest.mark.parametrize("B,T,is_prefill", [(48, 48, False), (1, 64, True)])
def test_forward_fused_equals_stock(B, T, is_prefill):
    hidden, w = _xw(T, 256, 4096, seed=T)
    stock = _StandIn(hidden, w, fused=False)
    fused = _StandIn(hidden, w, fused=True)
    # fused reads only self.lm_head.weight; a non-callable stand-in proves the module is not called
    fused.lm_head = SimpleNamespace(weight=w)
    t_s, g_s = _forward(stock, T, B, is_prefill)
    t_f, g_f = _forward(fused, T, B, is_prefill)
    assert g_s is None and g_f is None
    assert t_f.dtype == torch.int32 and t_f.shape == t_s.shape
    assert torch.equal(t_f, t_s)


def test_forward_fused_logit_mask_falls_back():
    hidden, w = _xw(8, 256, 4096, seed=3)
    m = _StandIn(hidden, w, fused=True)
    mask = torch.ones(8, 4096, dtype=torch.bool)
    called = {}
    m.sampler = lambda logits, *a, logit_mask=None, **k: called.setdefault("t", torch.argmax(
        logits.masked_fill(~logit_mask, float("-inf")), -1).to(torch.int32))
    t, _ = _forward(m, 8, 8, False, logit_mask=mask)
    assert "t" in called and torch.equal(t, called["t"])


def test_fused_gate_conditions():
    src = open(model_bf16.__file__).read()
    gate = src[src.index("self._fused_lmhead = ("):src.index("logger.info(\n                \"qwen3_asr lm_head backend")]
    for cond in ('get_lmhead_backend() == "fused"', "self.world_size == 1", "all_greedy",
                 "not self._gather_logits", 'weight_layout == "in_out"', "bias is None"):
        assert cond in gate
