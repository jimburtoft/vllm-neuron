# SPDX-License-Identifier: Apache-2.0
"""Fused audio conv frontend (VLLM_NEURON_QWEN3_ASR_ENC_CONV=nki) vs the stock torch frontend.

CPU (torch fallback) checks:
  * the [co, ci, 3, 3] <-> [ci, 9, co] weight pack is an exact round trip, and the checkpoint loader
    transform packs the same way.
  * audio_conv_frontend (torch) == conv2d1(addmm) / conv2d2 / conv2d3 + fp32 GELU, layer by layer.
  * Qwen3ASREncoder with ENC_CONV=nki gives the same encoder output as ENC_CONV=torch (same weights).
  * the backend env switch parses and rejects unknown values; the kernel gate rejects CPU tensors.

The NKI kernel itself is checked in the simulator and on trn2 by the Task 030 harness, not here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron.functional.audio_conv_frontend import (
    _torch_impl,
    audio_conv_frontend,
    can_use_audio_conv_frontend_kernel,
    pack_conv3x3_weight,
    unpack_conv3x3_weight,
)
from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import plan_audio_blocks

pytestmark = [pytest.mark.fast]

C, MEL, FR = 480, 128, 100


def _gelu(x):
    xf = x.float()
    return (0.5 * xf * (1.0 + torch.erf(xf * 0.7071067811865476))).to(x.dtype)


def _weights(seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    w1 = (torch.randn(C, 1, 3, 3, generator=g) / 3).to(dtype)
    w2 = (torch.randn(C, C, 3, 3, generator=g) * (1.5 / (9 * C) ** 0.5)).to(dtype)
    w3 = (torch.randn(C, C, 3, 3, generator=g) * (1.5 / (9 * C) ** 0.5)).to(dtype)
    b = [(torch.randn(C, generator=g) * 0.05).to(dtype) for _ in range(3)]
    return w1, w2, w3, b


def _conv1_addmm_ref(mel, w1k, b1):
    """Copy of Qwen3ASREncoder._conv1_addmm on [n, H, W] (independent of the module under test)."""
    n, H, W = mel.shape
    oh, ow = (H + 1) // 2, (W + 1) // 2
    p = F.pad(mel.unsqueeze(1), (1, 1, 1, 1))
    patches = torch.stack([p[:, 0, kr:kr + 2 * oh:2, kc:kc + 2 * ow:2] for kr in range(3) for kc in range(3)],
                          dim=-1)
    return torch.addmm(b1, patches.reshape(-1, 9), w1k).view(n, oh, ow, w1k.shape[1]).permute(0, 3, 1, 2)


def test_pack_roundtrip_and_loader():
    from vllm_neuron.model.qwen3_asr.weight_loader import audio_conv3x3_packed_weight_loader

    w = torch.randn(48, 32, 3, 3)
    p = pack_conv3x3_weight(w)
    assert p.shape == (32, 9, 48)
    assert torch.equal(unpack_conv3x3_weight(p), w)
    # p[ci, 3 kr + kc, co] == w[co, ci, kr, kc]
    assert p[5, 3 * 2 + 1, 7] == w[7, 5, 2, 1]

    class _Slice:
        def __init__(self, t):
            self.t = t

        def __getitem__(self, k):
            return self.t[k]

    got = audio_conv3x3_packed_weight_loader().load([_Slice(w)], 0)
    assert torch.equal(got, p)


@pytest.mark.parametrize("n", [1, 3])
def test_torch_impl_matches_stock_layers(n):
    w1, w2, w3, b = _weights()
    mel = (torch.rand(n, MEL, FR) * 2.5 - 1).to(torch.bfloat16)
    x = mel.unsqueeze(1)
    w1k = w1[:, 0].reshape(C, 9).t().contiguous()
    # conv1 reference = F.conv2d up to bf16 rounding (the module uses the stock addmm form)
    c1 = F.conv2d(x.float(), w1.float(), b[0].float(), stride=2, padding=1)
    got1 = _conv1_addmm_ref(mel, w1k, b[0]).float()
    assert (got1 - c1).abs().max() < 0.05
    want = _gelu(_conv1_addmm_ref(mel, w1k, b[0]))
    want = _gelu(F.conv2d(want, w2, b[1], stride=2, padding=1))
    want = _gelu(F.conv2d(want, w3, b[2], stride=2, padding=1))
    got = audio_conv_frontend(mel, w1k, b[0], pack_conv3x3_weight(w2), b[1], pack_conv3x3_weight(w3), b[2])
    assert got.shape == (n, C, 16, 13)
    assert torch.equal(got, want)
    assert torch.equal(_torch_impl(mel, w1k, b[0], pack_conv3x3_weight(w2), b[1],
                                   pack_conv3x3_weight(w3), b[2]), want)


def test_kernel_gate_rejects_cpu():
    w1 = torch.zeros(9, C, dtype=torch.bfloat16)
    assert not can_use_audio_conv_frontend_kernel(torch.zeros(9, MEL, FR, dtype=torch.bfloat16), w1)


def test_backend_env(monkeypatch):
    from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import get_encoder_conv_backend

    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_ENC_CONV", raising=False)
    assert get_encoder_conv_backend() == "torch"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_CONV", "NKI")
    assert get_encoder_conv_backend() == "nki"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_CONV", "nope")
    with pytest.raises(ValueError):
        get_encoder_conv_backend()


@pytest.mark.parametrize("feat,nb", [(1200, 2), (850, 1)])
def test_encoder_forward_nki_conv_matches_torch(feat, nb):
    """Two-layer encoder, identical random weights, ENC_CONV=nki (torch fallback on CPU) vs torch."""
    from vllm_neuron.utils.executor import MPExecutor

    def load():
        import os

        from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import Qwen3ASREncoder
        from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig

        def make(backend):
            os.environ["VLLM_NEURON_QWEN3_ASR_ENC_CONV"] = backend
            torch.manual_seed(0)
            cfg = Qwen3ASRAudioConfig(encoder_layers=2)
            cfg.torch_dtype = torch.bfloat16
            m = Qwen3ASREncoder(cfg)
            assert m.conv_backend == backend
            return m, cfg

        ref, cfg = make("torch")
        with torch.no_grad():
            for n_, p in ref.named_parameters():
                if "norm" in n_ and n_.endswith("weight"):
                    p.fill_(1.0)
                elif n_ in ("conv2d2.weight", "conv2d3.weight"):
                    p.copy_((torch.randn_like(p, dtype=torch.float32) * (1.5 / (9 * C) ** 0.5)).to(p.dtype))
                else:
                    p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).to(p.dtype))
        fused, _ = make("nki")
        os.environ.pop("VLLM_NEURON_QWEN3_ASR_ENC_CONV", None)
        sd = dict(ref.state_dict())
        for k in ("conv2d2.weight", "conv2d3.weight"):
            sd[k] = pack_conv3x3_weight(sd[k])
        fused.load_state_dict(sd, strict=True)

        plan = plan_audio_blocks([feat], block_size=128, window_tokens=cfg.window_tokens, n_window=cfg.n_window,
                                 n_window_infer=cfg.n_window_infer, num_blocks=nb)
        lo, hi = plan.block_bounds()
        feats = torch.randn(1, cfg.num_mel_bins, plan.feature_length).to(torch.bfloat16)
        idx = plan.unpack_indices()

        def run():
            with torch.no_grad():
                r = ref(feats, lo, hi, plan.chunk_slots())
                g = fused(feats, lo, hi, plan.chunk_slots())
            return (r.reshape(-1, r.shape[-1])[idx].float(), g.reshape(-1, g.shape[-1])[idx].float())
        return run

    ex = MPExecutor(world_size=1, model_load=load, device="cpu")
    try:
        ex.dispatch()
        r, g = ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()
    assert torch.isfinite(g).all()
    assert torch.equal(g, r)
