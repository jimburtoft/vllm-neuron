# SPDX-License-Identifier: Apache-2.0
"""MXFP8 experts, torch backend (``HY4_MX_BACKEND=torch``), on CPU.

* the loaders (real transforms, per TP rank) give the expected bytes/exponents;
* the in-graph dequantisation is bit-exact vs fp32 ``fp8 * 2**(e - 127)`` cast to bf16;
* ``_forward_selected_mx`` and ``_forward_dense_mx`` (summed over ranks) match an
  fp32 reference built from the whole dequantised checkpoint;
* dispatch, the default backend, and the load-time exponent checks.
"""

from __future__ import annotations

import os

import pytest
import torch

from hy4_stubs import Group, moe, set_group

LIMIT = 10.0
FP8 = torch.float8_e4m3fn


class Cfg:
    torch_dtype = torch.bfloat16
    hidden_size = 256
    n_routed_experts = 8
    num_experts_per_tok = 4
    routed_scaling_factor = 2.827
    norm_topk_prob = True
    swiglu_limit = LIMIT
    moe_intermediate_size = 64
    n_shared_experts = 0
    neuron_quantization = {"format": "mxfp8_experts"}


class Slice:
    """safetensors PySafeSlice stand-in: ``get_shape()`` and basic slicing."""

    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        return self.t[idx].clone()


def checkpoint(E, H, I, seed, lo=110, hi=130):
    """Random OCP e4m3fn weights (full range) + e8m0 exponents in [lo, hi]."""
    gen = torch.Generator().manual_seed(seed)

    def fp8(shape):
        return (torch.randn(shape, generator=gen) * 96.0).clamp(-448, 448).to(FP8)

    def e8m0(shape):
        return torch.randint(lo, hi + 1, shape, generator=gen, dtype=torch.uint8)

    # Exponents are centred so the expert outputs stay O(1) (SwiGLU clamp active
    # on some rows but not all): weight ~ 96 * 2**(e - 127), e ~ 120 -> ~0.75.
    return (
        fp8((E, 2 * I, H)),
        e8m0((E, 2 * I, H // 32)),
        fp8((E, H, I)),
        e8m0((E, H, I // 32)),
    )


def dequant32(w8, s):
    """fp32 ground truth: last dim blocked by 32."""
    return w8.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


NAMES = (
    "experts_gate_up_weight",
    "experts_gate_up_scale",
    "experts_down_weight",
    "experts_down_scale",
)


@pytest.fixture(autouse=True)
def torch_backend(monkeypatch):
    monkeypatch.setenv(moe.MX_BACKEND_ENV, moe.MX_BACKEND_TORCH)
    monkeypatch.delenv(moe.MX_FORCE_ENV, raising=False)  # no platform override needed


def _router(c, seed):
    gen = torch.Generator().manual_seed(seed)
    router = (torch.randn(c.n_routed_experts, c.hidden_size, generator=gen) * 0.05).to(
        torch.bfloat16
    )
    return router, torch.randn(c.n_routed_experts, generator=gen) * 0.01


def build(world, rank, ckpt, router, bias, c=None):
    c = c or Cfg()
    set_group(type("G", (Group,), {"world_size": world, "rank_in_group": rank})())
    prev = os.environ.get(moe.MX_BACKEND_ENV)
    os.environ[moe.MX_BACKEND_ENV] = moe.MX_BACKEND_TORCH
    try:
        m = moe.HYV4MoE(c)
    finally:
        if prev is None:
            os.environ.pop(moe.MX_BACKEND_ENV, None)
        else:
            os.environ[moe.MX_BACKEND_ENV] = prev
    assert m.mx_experts and m.mx_backend == moe.MX_BACKEND_TORCH
    for name, src in zip(NAMES, ckpt):
        p = getattr(m, name)
        t = p.weight_loader.load([Slice(src)], rank)
        assert t.dtype == p.dtype and tuple(t.shape) == tuple(p.shape), name
        p.data = t
    m.router_weight.data = router
    m.e_score_correction_bias.data = bias
    return m


@pytest.mark.parametrize("world", [1, 2])
def test_loader_layouts(world):
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    gu, gus, dn, dns = ckpt = checkpoint(E, H, I, seed=world)
    I_r = I // world
    for r in range(world):
        m = build(world, r, ckpt, *_router(c, 0))
        lo, hi = r * I_r, (r + 1) * I_r
        w = m.experts_gate_up_weight.view(torch.uint8)  # [E, H, 2, I_r]
        s = m.experts_gate_up_scale  # [E, H/32, 2, I_r]
        for g, off in ((0, 0), (1, I)):
            assert torch.equal(
                w[:, :, g, :], gu.view(torch.uint8)[:, off + lo : off + hi, :].transpose(1, 2)
            )
            assert torch.equal(s[:, :, g, :], moe.e8m0_to_bf16_host(gus[:, off + lo : off + hi, :].transpose(1, 2)))
        assert torch.equal(
            m.experts_down_weight.view(torch.uint8),
            dn.view(torch.uint8)[:, :, lo:hi].transpose(1, 2),
        )
        assert torch.equal(
            m.experts_down_scale, moe.e8m0_to_bf16_host(dns[:, :, lo // 32 : hi // 32].transpose(1, 2))
        )


def test_e8m0_to_bf16_exact():
    e = torch.arange(1, 255, dtype=torch.uint8)
    want = torch.exp2(e.double() - 127).to(torch.bfloat16)
    assert torch.equal(moe._e8m0_to_bf16(e), want)


@pytest.mark.parametrize("world", [1, 2])
def test_dequant_bit_exact(world):
    """In-graph bf16 dequant == fp32 dequant cast to bf16, every element."""
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    # The widest exponent range check_e8m0 accepts without a warning.
    ckpt = checkpoint(E, H, I, seed=7, lo=moe.MX_E8M0_EXACT_MIN, hi=moe.MX_E8M0_MAX_FINITE)
    gu_full, dn_full = dequant32(ckpt[0], ckpt[1]), dequant32(ckpt[2], ckpt[3])
    I_r = I // world
    for r in range(world):
        m = build(world, r, ckpt, *_router(c, 0))
        lo, hi = r * I_r, (r + 1) * I_r
        got_gu = moe.mx_dequant(m.experts_gate_up_weight, m.experts_gate_up_scale, dim=1)
        got_dn = moe.mx_dequant(m.experts_down_weight, m.experts_down_scale, dim=1)
        assert got_gu.dtype == got_dn.dtype == torch.bfloat16
        assert torch.isfinite(got_gu.float()).all() and torch.isfinite(got_dn.float()).all()
        for g, off in ((0, 0), (1, I)):
            want = gu_full[:, off + lo : off + hi, :].transpose(1, 2).to(torch.bfloat16)
            assert torch.equal(got_gu[:, :, g, :], want)
        want = dn_full[:, :, lo:hi].transpose(1, 2).to(torch.bfloat16)
        assert torch.equal(got_dn, want)
        # and the fp32 product was itself exact in bf16
        assert torch.equal(want.float(), dn_full[:, :, lo:hi].transpose(1, 2))


def reference(c, ckpt, router, bias, x):
    """fp32 per (token, expert) from the full dequantised checkpoint, clamped SwiGLU."""
    I = c.moe_intermediate_size
    wgu, wd = dequant32(ckpt[0], ckpt[1]), dequant32(ckpt[2], ckpt[3])
    idx, w = moe.route_topk(
        x, router, bias, c.num_experts_per_tok, c.routed_scaling_factor, c.norm_topk_prob
    )
    out = torch.zeros(x.shape[0], c.hidden_size)
    for t in range(x.shape[0]):
        for k in range(c.num_experts_per_tok):
            e = int(idx[t, k])
            y = wgu[e] @ x[t].float()
            g, u = y[:I].clamp(max=LIMIT), y[I:].clamp(-LIMIT, LIMIT)
            out[t] += float(w[t, k]) * (wd[e] @ (g * torch.sigmoid(g) * u))
    return out


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 5, 33])
@torch.no_grad()
def test_numerics_against_fp32(world, tokens):
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    ckpt = checkpoint(E, H, I, seed=100 + tokens)
    router, bias = _router(c, tokens)
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, H, generator=gen).to(torch.bfloat16)
    mods = [build(world, r, ckpt, router, bias) for r in range(world)]
    sel = sum(m._forward_selected_mx(x).float() for m in mods)
    den = sum(m._forward_dense_mx(x).float() for m in mods)
    ref = reference(c, ckpt, router, bias, x)
    r_sel, r_den = rel(sel, ref), rel(den, ref)
    print(f"world={world} tokens={tokens} rel_err selected={r_sel:.2e} dense={r_den:.2e}")
    assert r_sel < 1e-2 and r_den < 1e-2
    assert rel(sel, den) < 1e-2


@torch.no_grad()
def test_dispatch():
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    m = build(1, 0, checkpoint(E, H, I, seed=3), *_router(c, 3))
    n = moe.SELECTIVE_MOE_MAX_TOKENS
    x = torch.randn(n, H).to(torch.bfloat16)
    assert torch.equal(m(x), m._forward_selected_mx(x))
    x = torch.randn(n + 1, H).to(torch.bfloat16)
    assert torch.equal(m(x), m._forward_dense_mx(x))


def test_backend_env(monkeypatch):
    monkeypatch.delenv(moe.MX_BACKEND_ENV, raising=False)
    assert moe.mx_backend() == moe.MX_BACKEND_BF16  # the default
    monkeypatch.setenv(moe.MX_BACKEND_ENV, "nope")
    with pytest.raises(ValueError, match=moe.MX_BACKEND_ENV):
        moe.mx_backend()
    monkeypatch.setenv(moe.MX_BACKEND_ENV, moe.MX_BACKEND_KERNEL)
    set_group(Group())

    class K(Cfg):
        hidden_size = 512  # the kernel layouts need H % 512 == 0

    m = moe.HYV4MoE(K())
    assert m.mx_backend == moe.MX_BACKEND_KERNEL
    assert m.experts_gate_up_weight.dtype == torch.uint32


def test_rejects_split_blocks():
    class C(Cfg):
        moe_intermediate_size = 96  # I_r = 48 at world 2: splits a 32-wide block

    set_group(type("G", (Group,), {"world_size": 2, "rank_in_group": 0})())
    with pytest.raises(ValueError, match="multiples of 32"):
        moe.HYV4MoE(C())


def test_scale_checks(caplog):
    s = torch.full((2, 4, 8), 120, dtype=torch.uint8)
    with caplog.at_level("INFO", logger=moe.logger.name):
        moe.check_e8m0(s, "x", 0)
    assert "[120, 120]" in caplog.text
    s[0, 0, 0] = 255  # NaN
    with pytest.raises(ValueError, match="NaN"):
        moe.check_e8m0(s, "x", 0)
    s[0, 0, 0] = moe.MX_E8M0_MAX_FINITE + 1
    with pytest.raises(ValueError):
        moe.check_e8m0(s, "x", 0)
    # The largest e4m3fn value at the largest accepted exponent is finite in bf16.
    big = torch.tensor([448.0]).to(FP8).repeat(32)
    e = torch.tensor([moe.MX_E8M0_MAX_FINITE], dtype=torch.uint8)
    assert torch.isfinite(moe.mx_dequant(big, e, dim=0).float()).all()
    e = e + 1
    assert not torch.isfinite(moe.mx_dequant(big, e, dim=0).float()).all()


# ── HY4_MX_BACKEND=bf16: dequantised at load, bit-identical to the torch backend ──


def build_bf16(world, rank, ckpt, router, bias, monkeypatch, c=None):
    c = c or Cfg()
    monkeypatch.setenv(moe.MX_BACKEND_ENV, moe.MX_BACKEND_BF16)
    set_group(type("G", (Group,), {"world_size": world, "rank_in_group": rank})())
    m = moe.HYV4MoE(c)
    monkeypatch.delenv(moe.MX_BACKEND_ENV)
    assert m.mx_experts and m.mx_backend == moe.MX_BACKEND_BF16 and m.experts_bf16
    assert not hasattr(m, "experts_gate_up_scale") and not hasattr(m, "experts_down_scale")
    gu, gus, dn, dns = ckpt
    for name, srcs in (
        ("experts_gate_up_weight", (gu, gus)),
        ("experts_down_weight", (dn, dns)),
    ):
        p = getattr(m, name)
        t = p.weight_loader.load([Slice(s) for s in srcs], rank)
        assert t.dtype == p.dtype == torch.bfloat16 and tuple(t.shape) == tuple(p.shape), name
        p.data = t
    m.router_weight.data = router
    m.e_score_correction_bias.data = bias
    return m


@pytest.mark.parametrize("world", [1, 2])
def test_bf16_weights_equal_in_graph_dequant(world, monkeypatch):
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    ckpt = checkpoint(E, H, I, seed=11, lo=moe.MX_E8M0_EXACT_MIN, hi=moe.MX_E8M0_MAX_FINITE)
    for r in range(world):
        t = build(world, r, ckpt, *_router(c, 0))
        b = build_bf16(world, r, ckpt, *_router(c, 0), monkeypatch)
        want_gu = moe.mx_dequant(t.experts_gate_up_weight, t.experts_gate_up_scale, dim=1)
        want_dn = moe.mx_dequant(t.experts_down_weight, t.experts_down_scale, dim=1)
        assert torch.equal(b.experts_gate_up_weight.data, want_gu)
        assert torch.equal(b.experts_down_weight.data, want_dn)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("tokens", [1, 2, 5, 33])
@torch.no_grad()
def test_bf16_backend_bit_exact_vs_torch(world, tokens, monkeypatch):
    """Selective (decode) and dense (prefill) outputs equal the torch backend's, per rank."""
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    ckpt = checkpoint(E, H, I, seed=200 + tokens)
    router, bias = _router(c, tokens)
    gen = torch.Generator().manual_seed(tokens)
    x = torch.randn(tokens, H, generator=gen).to(torch.bfloat16)
    for r in range(world):
        t = build(world, r, ckpt, router, bias)
        b = build_bf16(world, r, ckpt, router, bias, monkeypatch)
        assert torch.equal(b._forward_selected(x), t._forward_selected_mx(x))
        assert torch.equal(b._forward_dense(x), t._forward_dense_mx(x))
        assert torch.equal(b(x), t(x))


@torch.no_grad()
def test_bf16_dispatch(monkeypatch):
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    m = build_bf16(1, 0, checkpoint(E, H, I, seed=3), *_router(c, 3), monkeypatch)
    n = moe.SELECTIVE_MOE_MAX_TOKENS
    x = torch.randn(n, H).to(torch.bfloat16)
    assert torch.equal(m(x), m._forward_selected(x))
    x = torch.randn(n + 1, H).to(torch.bfloat16)
    assert torch.equal(m(x), m._forward_dense(x))


@pytest.mark.parametrize(
    "backend", [moe.MX_BACKEND_TORCH, moe.MX_BACKEND_BF16, moe.MX_BACKEND_KERNEL]
)
def test_checkpoint_mappings_per_backend(backend, monkeypatch):
    from hy4_stubs import model

    class C:
        num_hidden_layers = 2
        indexer_types = ["none", "none"]
        tie_word_embeddings = True
        neuron_quantization = {"format": moe.FORMAT_MXFP8}

        @staticmethod
        def is_moe_layer(i):
            return i == 1

    monkeypatch.setenv(moe.MX_BACKEND_ENV, backend)
    m = model.checkpoint_mappings(C())
    mlp, hm = "model.layers.1.mlp", f"{model.HF_PREFIX}.layers.1.mlp.experts"
    if backend == moe.MX_BACKEND_BF16:
        assert m[f"{mlp}.experts_gate_up_weight"] == [f"{hm}.gate_up_proj", f"{hm}.gate_up_proj_scale"]
        assert m[f"{mlp}.experts_down_weight"] == [f"{hm}.down_proj", f"{hm}.down_proj_scale"]
        assert f"{mlp}.experts_gate_up_scale" not in m and f"{mlp}.experts_down_scale" not in m
    else:
        assert m[f"{mlp}.experts_gate_up_weight"] == f"{hm}.gate_up_proj"
        assert m[f"{mlp}.experts_down_scale"] == f"{hm}.down_proj_scale"
