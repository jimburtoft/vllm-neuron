# SPDX-License-Identifier: Apache-2.0
"""Fused encoder window attention (VLLM_NEURON_QWEN3_ASR_ENC_ATTN=fused) vs the attention_cte path.

CPU (torch fallback) checks:
  * build_segment_factors gives exactly the bound mask (0 / -4096) for every valid query row, over
    plan_audio_blocks layouts (full windows, short tails, multi-segment blocks, padding blocks).
  * encoder_window_attention (torch) matches NF.flash_attention (torch fallback) on valid rows.
  * Qwen3ASRAudioAttention forward with seg_factors matches the default path (random weights).
  * the backend env switch parses and rejects unknown values.

The NKI kernel itself is checked in the simulator and on trn2 by the Task 028 harness, not here.
"""

from __future__ import annotations

import pytest
import torch

import vllm_neuron.functional as NF
from vllm_neuron.functional.attention.encoder_window_attention import (
    MASK_BIG,
    build_segment_factors,
    encoder_window_attention,
)
from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import plan_audio_blocks

pytestmark = [pytest.mark.fast]

NH, D, S = 16, 64, 128

LAYOUTS = [
    ([3000], True, 4),
    ([2500], True, 4),
    ([1200], True, 2),
    ([1200], True, 4),
    ([850], True, 1),
    ([100], True, 1),
    ([300, 800, 450, 120], False, 4),
    ([130, 140, 150, 160, 170, 180, 190, 200], False, 4),
]


def _plan(feats, one_item, nb):
    p = plan_audio_blocks(feats, block_size=S, window_tokens=104, n_window=50, n_window_infer=800,
                          num_blocks=nb, one_item_per_block=one_item)
    return p.block_bounds()


@pytest.mark.parametrize("feats,one_item,nb", LAYOUTS)
def test_segment_factors_match_bounds(feats, one_item, nb):
    lo, hi = _plan(feats, one_item, nb)
    sq, sk = build_segment_factors(lo, hi, dtype=torch.bfloat16)
    assert sq.shape == (nb, 16, S) and sk.shape == (nb, 16, S)
    bias = torch.einsum("brq,brk->bqk", sq.float(), sk.float())
    kp = torch.arange(S).view(1, 1, S)
    allowed = (kp >= lo) & (kp < hi)
    valid = (hi > lo)[..., 0]
    want = torch.where(allowed, 0.0, -MASK_BIG)
    assert torch.equal(bias[valid], want[valid])
    # padding rows: fully masked (uniform softmax, finite)
    assert torch.all(bias[~valid] == -MASK_BIG)


def _cte_reference(qkv, lo, hi, scale):
    nb = qkv.shape[0]
    hd = NH * D
    q, k, v = (qkv[..., i * hd:(i + 1) * hd].reshape(nb, S, NH, D).permute(0, 2, 1, 3).reshape(nb * NH, S, D)
               for i in range(3))
    out = NF.flash_attention(q, k, v, scale=scale, causal_mask=False, tp_q=True, tp_k=True, tp_out=True,
                             bound_min=lo.repeat_interleave(NH, dim=0), bound_max=hi.repeat_interleave(NH, dim=0))
    return out.view(nb, NH, D, S)


@pytest.mark.parametrize("feats,one_item,nb", LAYOUTS)
def test_window_attention_matches_flash_attention(feats, one_item, nb):
    torch.manual_seed(0)
    lo, hi = _plan(feats, one_item, nb)
    qkv = torch.randn(nb, S, 3 * NH * D).to(torch.bfloat16)
    sq, sk = build_segment_factors(lo, hi)
    got = encoder_window_attention(qkv, sq, sk, D ** -0.5, NH).float()
    ref = _cte_reference(qkv, lo, hi, D ** -0.5).float()
    assert got.shape == (nb, NH, D, S)
    valid = (hi > lo)[..., 0]                                   # [nb, S] query columns
    m = valid[:, None, None, :].expand_as(ref)
    err = (got - ref)[m].abs()
    assert torch.isfinite(got).all()
    assert err.max() < 3e-2, err.max()
    assert err.norm() / ref[m].norm() < 1e-2


def test_attention_module_fused_matches_cte():
    from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import Qwen3ASRAudioAttention
    from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig
    from vllm_neuron.utils.executor import MPExecutor

    def load():
        torch.manual_seed(0)
        cfg = Qwen3ASRAudioConfig()
        cfg.torch_dtype = torch.bfloat16
        m = Qwen3ASRAudioAttention(cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.03).to(p.dtype))
            m.norm.weight.fill_(1.0)
        lo, hi = _plan([3000], True, 4)
        x = torch.randn(4, S, cfg.d_model).to(torch.bfloat16)

        def run():
            ref = m(x, lo, hi)
            got = m(x, lo, hi, build_segment_factors(lo, hi))
            return ref, got, (hi > lo)[..., 0]
        return run

    ex = MPExecutor(world_size=1, model_load=load, device="cpu")
    try:
        ex.dispatch()
        ref, got, valid = ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()
    r, g = ref[valid].float(), got[valid].float()
    assert torch.isfinite(g).all()
    assert (g - r).norm() / r.norm() < 1e-2
    assert torch.nn.functional.cosine_similarity(g, r, dim=-1).min() > 0.999


def test_backend_env(monkeypatch):
    from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import get_encoder_attention_backend

    monkeypatch.delenv("VLLM_NEURON_QWEN3_ASR_ENC_ATTN", raising=False)
    assert get_encoder_attention_backend() == "cte"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_ATTN", "Fused")
    assert get_encoder_attention_backend() == "fused"
    monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_ATTN", "nope")
    with pytest.raises(ValueError):
        get_encoder_attention_backend()


def test_encoder_forward_fused_matches_cte(monkeypatch):
    """Two-layer encoder, random weights, 12 s clip in a 4-block bucket (2 padding blocks)."""
    from vllm_neuron.model.qwen3_asr.audio_encoder_bf16 import Qwen3ASREncoder
    from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig
    from vllm_neuron.utils.executor import MPExecutor

    def make(backend):
        def load():
            monkeypatch.setenv("VLLM_NEURON_QWEN3_ASR_ENC_ATTN", backend)
            torch.manual_seed(0)
            cfg = Qwen3ASRAudioConfig(encoder_layers=2)
            cfg.torch_dtype = torch.bfloat16
            m = Qwen3ASREncoder(cfg)
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if "norm" in n and n.endswith("weight"):
                        p.fill_(1.0)
                    else:
                        p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).to(p.dtype))
            assert m.attention_backend == backend
            plan = plan_audio_blocks([1200], block_size=S, window_tokens=cfg.window_tokens, n_window=cfg.n_window,
                                     n_window_infer=cfg.n_window_infer, num_blocks=4)
            lo, hi = plan.block_bounds()
            feats = torch.randn(1, cfg.num_mel_bins, plan.feature_length).to(torch.bfloat16)

            def run():
                y = m(feats, lo, hi, plan.chunk_slots())
                return y.reshape(-1, y.shape[-1])[plan.unpack_indices()].float()
            return run
        return load

    outs = {}
    for backend in ("cte", "fused"):
        ex = MPExecutor(world_size=1, model_load=make(backend), device="cpu")
        try:
            ex.dispatch()
            outs[backend] = ex.collect(timeout=600)[0]
        finally:
            ex.shutdown()
    r, g = outs["cte"], outs["fused"]
    assert torch.isfinite(g).all()
    assert torch.nn.functional.cosine_similarity(g, r, dim=-1).min() > 0.999
    assert (g - r).norm() / r.norm() < 2e-2
