# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS FP8: prefill from the FP8 experts or from a BF16 copy (neuron_config gpt_oss_fp8_prefill)."""

import sys
import types

import pytest

from vllm_neuron.model.gpt_oss import fp8_prefill
from vllm_neuron.model.neuron_config import NeuronConfig

pytestmark = pytest.mark.fast


def _fake_bwmm(monkeypatch, direct, gu_major):
    """Install a fake nkilib bwmm_shard_on_I module with the given markers."""
    mod = types.ModuleType("nkilib.core.moe.moe_cte.bwmm_shard_on_I")
    if direct is not None:
        mod.FP8_DIRECT_MATMUL = direct
    if gu_major is not None:
        mod.SUPPORTS_GATE_UP_MAJOR = gu_major
    pkg = types.ModuleType("nkilib.core.moe.moe_cte")
    pkg.bwmm_shard_on_I = mod
    for name, m in [
        ("nkilib", types.ModuleType("nkilib")),
        ("nkilib.core", types.ModuleType("nkilib.core")),
        ("nkilib.core.moe", types.ModuleType("nkilib.core.moe")),
        ("nkilib.core.moe.moe_cte", pkg),
        ("nkilib.core.moe.moe_cte.bwmm_shard_on_I", mod),
    ]:
        monkeypatch.setitem(sys.modules, name, m)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("GPTOSS_HYBRID_PREFILL", raising=False)


@pytest.mark.parametrize(
    "direct, gu_major, expected_copy",
    [(True, True, False), (True, None, True), (None, True, True), (None, None, True)],
)
def test_auto_follows_nkilib(monkeypatch, direct, gu_major, expected_copy):
    _fake_bwmm(monkeypatch, direct, gu_major)
    assert fp8_prefill.keep_bf16_prefill_copy({}) is expected_copy
    assert (
        fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "auto"})
        is expected_copy
    )


def test_explicit_values(monkeypatch):
    _fake_bwmm(monkeypatch, True, True)
    assert (
        fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "bf16_copy"}) is True
    )
    _fake_bwmm(monkeypatch, None, None)
    assert fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "fp8"}) is False


def test_legacy_env_used_only_without_config_key(monkeypatch):
    _fake_bwmm(monkeypatch, True, True)
    monkeypatch.setenv("GPTOSS_HYBRID_PREFILL", "1")
    assert fp8_prefill.keep_bf16_prefill_copy({}) is True
    assert fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "fp8"}) is False
    monkeypatch.setenv("GPTOSS_HYBRID_PREFILL", "0")
    _fake_bwmm(monkeypatch, None, None)
    assert fp8_prefill.keep_bf16_prefill_copy({}) is False


def test_invalid_value():
    with pytest.raises(ValueError, match="gpt_oss_fp8_prefill"):
        fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "yes"})


def test_neuron_config_object(monkeypatch):
    _fake_bwmm(monkeypatch, None, None)
    nc = NeuronConfig.from_dict({"quantization": "fp8", "gpt_oss_fp8_prefill": "fp8"})
    assert nc.gpt_oss_fp8_prefill == "fp8"
    assert fp8_prefill.keep_bf16_prefill_copy(nc) is False
    assert NeuronConfig.from_dict({}).gpt_oss_fp8_prefill is None


@pytest.mark.parametrize(
    "inter_per_rank, expected_copy",
    [(None, False), (1440, False), (720, False), (719, True), (360, True), (45, True)],
)
def test_auto_keeps_copy_for_small_per_rank_intermediate(
    monkeypatch, inter_per_rank, expected_copy
):
    # GPT-OSS: intermediate 2880 -> 720 per rank at TP=4 (FP8 prefill compiles), 360 at TP=8 (it does not).
    _fake_bwmm(monkeypatch, True, True)
    assert fp8_prefill.keep_bf16_prefill_copy({}, inter_per_rank) is expected_copy


def test_explicit_fp8_ignores_per_rank_intermediate(monkeypatch):
    _fake_bwmm(monkeypatch, True, True)
    assert (
        fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "fp8"}, 360) is False
    )
    assert (
        fp8_prefill.keep_bf16_prefill_copy({"gpt_oss_fp8_prefill": "bf16_copy"}, 1440)
        is True
    )
