# SPDX-License-Identifier: Apache-2.0
"""Encoder backend switches VLLM_NEURON_QWEN3_ASR_ENC_MLP (torch|nki) and VLLM_NEURON_QWEN3_ASR_ENC_QKV (nki|torch).

CPU (torch fallback) checks:
  * the env switches parse, default to torch / nki, and reject unknown values.
  * one encoder layer: ENC_MLP=nki (nkilib MLP call with fused residual + LayerNorm + biases) matches the stock
    layer; ENC_QKV=torch (F.layer_norm + matmul + bias) matches NF.qkv_proj.
  * a 2-layer encoder forward gives the same result for every (mlp, qkv) combination.

On CPU NF.mlp / NF.qkv_proj run their torch fallbacks, so these tests check the wiring (argument order, residual
handling, bias placement), not the NKI kernels; those are checked on trn2 (Task 032 harness).
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import plan_audio_blocks

pytestmark = [pytest.mark.fast]

S = 128
COMBOS = [("torch", "nki"), ("nki", "nki"), ("torch", "torch"), ("nki", "torch")]


def test_backend_env(monkeypatch):
    from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import (
        get_encoder_mlp_backend,
        get_encoder_qkv_backend,
    )

    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_ENC_MLP", raising=False)
    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_ENC_QKV", raising=False)
    assert get_encoder_mlp_backend() == "torch"
    assert get_encoder_qkv_backend() == "nki"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_MLP", "NKI")
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_QKV", " Torch ")
    assert get_encoder_mlp_backend() == "nki"
    assert get_encoder_qkv_backend() == "torch"
    for name, fn in (("VLLM_NEURON_QWEN3_ASR_ENC_MLP", get_encoder_mlp_backend),
                     ("VLLM_NEURON_QWEN3_ASR_ENC_QKV", get_encoder_qkv_backend)):
        monkeypatch.setenv(name, "nope")
        with pytest.raises(ValueError):
            fn()
        monkeypatch.delenv(name)


def _init(m):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n and n.endswith("weight"):
                p.copy_((1.0 + 0.1 * torch.randn_like(p, dtype=torch.float32)).to(p.dtype))
            else:
                p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).to(p.dtype))


def _run_in_executor(load):
    from vllm_neuron.utils.executor import MPExecutor

    ex = MPExecutor(world_size=1, model_load=load, device="cpu")
    try:
        ex.dispatch()
        return ex.collect(timeout=900)[0]
    finally:
        ex.shutdown()


@pytest.mark.parametrize("mlp,qkv", COMBOS[1:])
def test_layer_matches_stock(monkeypatch, mlp, qkv):
    """One encoder layer, same weights, stock (torch, nki) vs the given combination."""

    def load():
        from vllm_neuron.functional.attention.encoder_window_attention import build_segment_factors
        from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import Qwen3ASRAudioEncoderLayer
        from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig

        cfg = Qwen3ASRAudioConfig()
        cfg.torch_dtype = torch.bfloat16
        layers = {}
        for arm in (("torch", "nki"), (mlp, qkv)):
            monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_MLP", arm[0])
            monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_QKV", arm[1])
            torch.manual_seed(0)
            m = Qwen3ASRAudioEncoderLayer(cfg)
            torch.manual_seed(1)
            _init(m)
            assert m.mlp_backend == arm[0] and m.attn.qkv_backend == arm[1]
            layers[arm] = m
        plan = plan_audio_blocks([3000], block_size=S, window_tokens=cfg.window_tokens, n_window=cfg.n_window,
                                 n_window_infer=cfg.n_window_infer, num_blocks=4)
        lo, hi = plan.block_bounds()
        seg = build_segment_factors(lo, hi, dtype=torch.bfloat16)
        torch.manual_seed(2)
        x = torch.randn(4, S, cfg.d_model).to(torch.bfloat16)

        def run():
            ref = layers[("torch", "nki")](x, lo, hi, seg)
            got = layers[(mlp, qkv)](x, lo, hi, seg)
            return ref, got, (hi > lo)[..., 0]
        return run

    ref, got, valid = _run_in_executor(load)
    r, g = ref[valid].float(), got[valid].float()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.isfinite(g).all()
    assert (g - r).norm() / r.norm() < 1e-2
    assert torch.nn.functional.cosine_similarity(g, r, dim=-1).min() > 0.999


def test_encoder_forward_all_combos(monkeypatch):
    """Two-layer encoder, 12 s clip in a 4-block bucket: every (mlp, qkv) combination vs stock."""

    def make(mlp, qkv):
        def load():
            from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import Qwen3ASREncoder
            from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig

            monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_MLP", mlp)
            monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_QKV", qkv)
            torch.manual_seed(0)
            cfg = Qwen3ASRAudioConfig(encoder_layers=2)
            cfg.torch_dtype = torch.bfloat16
            m = Qwen3ASREncoder(cfg)
            torch.manual_seed(1)
            _init(m)
            assert all(l.mlp_backend == mlp and l.attn.qkv_backend == qkv for l in m.layers)
            plan = plan_audio_blocks([1200], block_size=S, window_tokens=cfg.window_tokens, n_window=cfg.n_window,
                                     n_window_infer=cfg.n_window_infer, num_blocks=4)
            lo, hi = plan.block_bounds()
            torch.manual_seed(2)
            feats = torch.randn(1, cfg.num_mel_bins, plan.feature_length).to(torch.bfloat16)

            def run():
                y = m(feats, lo, hi, plan.chunk_slots())
                return y.reshape(-1, y.shape[-1])[plan.unpack_indices()].float()
            return run
        return load

    outs = {c: _run_in_executor(make(*c)) for c in COMBOS}
    r = outs[COMBOS[0]]
    for c in COMBOS[1:]:
        g = outs[c]
        assert torch.isfinite(g).all(), c
        assert torch.nn.functional.cosine_similarity(g, r, dim=-1).min() > 0.999, c
        assert (g - r).norm() / r.norm() < 2e-2, c
