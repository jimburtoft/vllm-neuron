# SPDX-License-Identifier: Apache-2.0
"""Selective (decode) routed experts == dense routed experts == fp32 reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hy4_stubs import Group, moe, set_group


class Cfg:
    torch_dtype = torch.bfloat16
    hidden_size = 256
    n_routed_experts = 32
    num_experts_per_tok = 8
    routed_scaling_factor = 2.827
    norm_topk_prob = True
    swiglu_limit = 10.0
    moe_intermediate_size = 64
    n_shared_experts = 0


def _quant_rows(w: torch.Tensor):
    """Per-output-row fp8 within the Trn2 range, as the converter writes it."""
    s = w.abs().amax(-1) / 240.0
    return (w / s.unsqueeze(-1)).clamp(-240, 240).to(moe.FP8), s


def build(seed: int):
    set_group(Group())
    torch.manual_seed(seed)
    c = Cfg()
    m = moe.HYV4MoE(c)
    e, h, i = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    gu_q, gu_s = _quant_rows(torch.randn(e, 2 * i, h) * 0.05)
    dn_q, dn_s = _quant_rows(torch.randn(e, h, i) * 0.05)
    # The loaders' layouts at rank 0 of 1: gate_up [E, H, 2, I], down_proj_t [E, I, H].
    g, u = gu_q[:, :i, :].view(torch.uint8), gu_q[:, i:, :].view(torch.uint8)
    m.experts_gate_up_weight.data = (
        torch.stack((g, u), 1).permute(0, 3, 1, 2).contiguous().view(moe.FP8)
    )
    m.experts_gate_up_scale.data = (
        torch.stack((gu_s[:, :i], gu_s[:, i:]), 1).reshape(1, -1).float()
    )
    m.experts_down_weight.data = dn_q.transpose(1, 2).contiguous()
    m.experts_down_scale.data = dn_s.unsqueeze(1).float()
    m.router_weight.data = (torch.randn(e, h) * 0.1).to(torch.bfloat16)
    m.e_score_correction_bias.data = torch.randn(e) * 0.01
    ref_w = (gu_q.float() * gu_s.unsqueeze(-1), dn_q.float() * dn_s.unsqueeze(-1))
    return m, c, ref_w


def reference(m, c, ref_w, x):
    gu, dn = ref_w
    i = c.moe_intermediate_size
    idx, w = moe.route_topk(
        x, m.router_weight, m.e_score_correction_bias, c.num_experts_per_tok,
        c.routed_scaling_factor, c.norm_topk_prob,
    )
    out = torch.zeros(x.shape[0], c.hidden_size)
    for t in range(x.shape[0]):
        for j in range(c.num_experts_per_tok):
            ex = idx[t, j]
            y = gu[ex] @ x[t].float()
            g, u = y[:i].clamp(max=10), y[i:].clamp(-10, 10)
            out[t] += w[t, j] * (dn[ex] @ (F.silu(g) * u))
    return out


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_selected_matches_dense_and_reference(tokens):
    m, c, ref_w = build(tokens)
    x = torch.randn(tokens, c.hidden_size).to(torch.bfloat16)
    with torch.no_grad():
        sel, den = m._forward_selected(x), m._forward_dense(x)
    ref = reference(m, c, ref_w, x)
    r_sel, r_den = rel(sel, ref), rel(den, ref)
    assert r_sel < 0.01 and r_den < 0.01
    assert r_sel <= r_den * 1.05 + 1e-4


def test_route_topk_matches_gather():
    """The one-hot weight selection equals ``torch.gather`` of the scores."""
    torch.manual_seed(0)
    x = torch.randn(5, 64).to(torch.bfloat16)
    router = (torch.randn(32, 64) * 0.1).to(torch.bfloat16)
    bias = torch.randn(32) * 0.01
    idx, w = moe.route_topk(x, router, bias, 8, 1.0, False)
    scores = torch.sigmoid(x.float() @ router.float().t())
    assert torch.equal(w, torch.gather(scores, 1, idx.long()))


def test_dispatch():
    m, c, _ = build(0)
    with torch.no_grad():
        x = torch.randn(moe.SELECTIVE_MOE_MAX_TOKENS, c.hidden_size).to(torch.bfloat16)
        assert torch.equal(m(x), m._forward_selected(x))
        x = torch.randn(moe.SELECTIVE_MOE_MAX_TOKENS + 1, c.hidden_size).to(torch.bfloat16)
        assert torch.equal(m(x), m._forward_dense(x))
