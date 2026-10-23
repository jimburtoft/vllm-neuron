# SPDX-License-Identifier: Apache-2.0
"""Trn3 native MXFP8 experts (``mxfp8_experts``) on CPU.

* ``mx_experts`` tiling: decoding the kernel layouts back (indexing of nkilib's
  MX torch reference) gives the checkpoint bytes, bit for bit;
* the model's loaders, per TP rank, == ``convert_rank_experts(shard_experts_tp())``;
* the forward's dispatch to ``moe_tkg`` (selective / padded all-expert dynamic);
* with nkilib importable: the MoE output against an fp32 reference;
* the platform guard.
"""

from __future__ import annotations

import enum
import sys
import types

import pytest
import torch

from hy4_stubs import Group, moe, mx_experts as mx, set_group

LIMIT = 10.0


class Cfg:
    torch_dtype = torch.bfloat16
    hidden_size = 1024
    n_routed_experts = 16
    num_experts_per_tok = 8
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


def rand_fp8(shape, gen):
    return (torch.randn(shape, generator=gen) * 96.0).clamp(-448, 448).to(torch.float8_e4m3fn)


def rand_e8m0(shape, gen, lo, hi):
    return torch.randint(lo, hi + 1, shape, generator=gen, dtype=torch.uint8)


def checkpoint(E, H, I, seed):
    gen = torch.Generator().manual_seed(seed)
    return (
        rand_fp8((E, 2 * I, H), gen),
        rand_e8m0((E, 2 * I, H // 32), gen, 113, 118),
        rand_fp8((E, H, I), gen),
        rand_e8m0((E, H, I // 32), gen, 116, 121),
    )


def decode_layouts(t, E, H, I):
    """Invert the kernel layouts (see the ``mx_experts`` module docstring)."""
    I_tile = I if I <= 512 else 512
    c = torch.arange(I)
    tt, rem = c // 512, c % 512
    q, p = rem // (I_tile // 4), rem % (I_tile // 4)
    nat_i = 512 * tt + 4 * p + q  # natural I of kernel column c
    hperm = torch.arange(H).reshape(H // 4, 4).T.reshape(-1)  # kernel h' -> natural h

    gu_w = t["expert_gate_up_weights"]
    gu = torch.zeros(E, 2 * I, H, dtype=torch.uint8)
    gus = torch.zeros(E, 2 * I, H // 32, dtype=torch.uint8)
    for g in range(2):
        words = gu_w[:, :, g].permute(0, 2, 1, 3).reshape(E, H // 4, I)
        b = words.contiguous().view(torch.uint8).reshape(E, H // 4, I, 4)
        gu[:, g * I + nat_i] = b.permute(0, 2, 1, 3).reshape(E, I, H)
        s = t["expert_gate_up_weights_scale"][:, :, g].permute(0, 2, 1, 3).reshape(E, H // 32, I)
        gus[:, g * I + nat_i] = s.permute(0, 2, 1)
    words = t["expert_down_weights"].permute(0, 2, 1, 3).reshape(E, I // 4, H)
    b = words.contiguous().view(torch.uint8).reshape(E, I // 4, H, 4)
    dn = torch.empty(E, H, I, dtype=torch.uint8)
    dn[:, hperm] = b.permute(0, 2, 1, 3).reshape(E, H, I)
    s = t["expert_down_weights_scale"].permute(0, 2, 1, 3).reshape(E, I // 32, H).permute(0, 2, 1)
    dns = torch.empty_like(s)
    dns[:, hperm] = s
    return gu, gus, dn, dns


@pytest.mark.parametrize("E,H,I", [(8, 1024, 32), (8, 6144, 32), (4, 512, 64), (2, 512, 1024)])
def test_layouts_bit_exact(E, H, I):
    gu, gus, dn, dns = checkpoint(E, H, I, seed=E + I)
    t = mx.convert_rank_experts(gu, gus, dn, dns)
    for k, v in t.items():
        assert tuple(v.shape) == mx.expected_shapes(E, H, I)[k], k
    got = decode_layouts(t, E, H, I)
    assert torch.equal(got[0], gu.view(torch.uint8))
    assert torch.equal(got[1], gus)
    assert torch.equal(got[2], dn.view(torch.uint8))
    assert torch.equal(got[3], dns)
    x = torch.randn(3, H)
    assert torch.equal(mx.unshuffle_hidden(mx.shuffle_hidden(x)), x)


@pytest.fixture
def forced(monkeypatch):
    monkeypatch.setenv(moe.MX_FORCE_ENV, "1")


NAMES = {
    "experts_gate_up_weight": "expert_gate_up_weights",
    "experts_gate_up_scale": "expert_gate_up_weights_scale",
    "experts_down_weight": "expert_down_weights",
    "experts_down_scale": "expert_down_weights_scale",
}


def build(world, rank, ckpt, router, bias):
    set_group(type("G", (Group,), {"world_size": world, "rank_in_group": rank})())
    m = moe.HYV4MoE(Cfg())
    assert m.mx_experts
    want = mx.convert_rank_experts(*mx.shard_experts_tp(*ckpt, rank=rank, tp=world))
    for i, (name, key) in enumerate(NAMES.items()):
        p = getattr(m, name)
        t = p.weight_loader.load([Slice(ckpt[i])], rank)
        assert t.dtype == p.dtype and tuple(t.shape) == tuple(p.shape), name
        assert torch.equal(t, want[key]), name
        p.data = t
    m.router_weight.data = router
    m.e_score_correction_bias.data = bias
    return m


class _Act(enum.Enum):
    SiLU = 0


class _Scale(enum.Enum):
    POST_SCALE = 1


def _router(seed):
    c = Cfg()
    gen = torch.Generator().manual_seed(seed)
    router = (torch.randn(c.n_routed_experts, c.hidden_size, generator=gen) * 0.05).to(torch.bfloat16)
    return router, torch.randn(c.n_routed_experts, generator=gen) * 0.01


@pytest.mark.parametrize("world", [1, 2])
def test_loaders_per_rank(forced, world):
    c = Cfg()
    ckpt = checkpoint(c.n_routed_experts, c.hidden_size, c.moe_intermediate_size, seed=world)
    for r in range(world):
        build(world, r, ckpt, *_router(0))


@pytest.mark.parametrize("tokens", [1, 16, 17, 40, 64])
@torch.no_grad()
def test_dispatch(forced, monkeypatch, tokens):
    calls = []

    def fake_moe_tkg(hidden_input, **kw):
        assert kw["expert_index"].dtype == torch.int32
        assert kw["expert_affinities"].dtype == torch.float32
        calls.append((kw["is_all_expert"], kw.get("block_size"), hidden_input.shape[0]))
        return torch.zeros(hidden_input.shape, dtype=torch.float32)

    monkeypatch.setattr(moe, "_moe_tkg_api", lambda: (fake_moe_tkg, _Act, _Scale))
    c = Cfg()
    ckpt = checkpoint(c.n_routed_experts, c.hidden_size, c.moe_intermediate_size, seed=0)
    m = build(1, 0, ckpt, *_router(0))
    y = m(torch.randn(tokens, c.hidden_size).to(torch.bfloat16))
    assert y.shape == (tokens, c.hidden_size) and y.dtype == torch.bfloat16
    (all_expert, block, rows), = calls
    if tokens <= moe.MX_SELECTIVE_MAX_TOKENS:
        assert not all_expert and rows == tokens
    else:
        assert all_expert and rows % moe.MX_ALL_EXPERT_T_MULTIPLE == 0 and rows >= tokens
        assert block == moe.mx_block_size(rows) and rows % block == 0


def test_block_size():
    assert moe.mx_block_size(64) == 32
    assert moe.mx_block_size(96) == 32
    assert moe.mx_block_size(2048) == 128


def _dequant(w8, s):
    return w8.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("world,tokens", [(1, 1), (2, 3), (2, 40)])
@torch.no_grad()
def test_numerics_against_nkilib_reference(forced, monkeypatch, world, tokens):
    np = pytest.importorskip("numpy")
    pytest.importorskip("nkilib")
    dt = pytest.importorskip("neuron_dtypes")
    from nkilib.core.moe.moe_tkg.moe_tkg_torch import moe_tkg_torch_ref
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    def ref_moe_tkg(hidden_input, expert_gate_up_weights, expert_down_weights, **kw):
        for k in ("is_all_expert_dynamic", "block_size", "rank_id"):
            kw.pop(k, None)
        out = moe_tkg_torch_ref(
            hidden_input=hidden_input.float(),
            expert_gate_up_weights=expert_gate_up_weights.numpy().view(dt.float8_e4m3fn_x4),
            expert_down_weights=expert_down_weights.numpy().view(dt.float8_e4m3fn_x4),
            output_dtype=np.float32,
            **kw,
        )["out"]
        return torch.from_numpy(np.asarray(out, dtype=np.float32))

    monkeypatch.setattr(
        moe, "_moe_tkg_api", lambda: (ref_moe_tkg, ActFnType, ExpertAffinityScaleMode)
    )
    c = Cfg()
    E, H, I = c.n_routed_experts, c.hidden_size, c.moe_intermediate_size
    ckpt = checkpoint(E, H, I, seed=tokens)
    router, bias = _router(tokens)
    x = torch.randn(tokens, H, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    out = sum(build(world, r, ckpt, router, bias)(x).double() for r in range(world))

    idx, w = moe.route_topk(x, router, bias, c.num_experts_per_tok,
                            c.routed_scaling_factor, c.norm_topk_prob)
    wgu, wd = _dequant(ckpt[0], ckpt[1]).double(), _dequant(ckpt[2], ckpt[3]).double()
    ref = torch.zeros(tokens, H, dtype=torch.float64)
    for t in range(tokens):
        for k in range(c.num_experts_per_tok):
            e = int(idx[t, k])
            g = (wgu[e, :I] @ x[t].double()).clamp(max=LIMIT)
            u = (wgu[e, I:] @ x[t].double()).clamp(-LIMIT, LIMIT)
            ref[t] += float(w[t, k]) * (wd[e] @ (g * torch.sigmoid(g) * u))
    err = float((out - ref).norm() / ref.norm())
    assert 0.01 < err < 0.10  # MX activation quantisation inside the kernel


def test_platform_guard(monkeypatch):
    monkeypatch.delenv(moe.MX_FORCE_ENV, raising=False)
    fake = types.ModuleType("libtorch_neuronx_lite.compile.platform")
    for name in ("libtorch_neuronx_lite", "libtorch_neuronx_lite.compile"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite.compile.platform", fake)
    fake.get_platform_target = lambda: "trn2"
    with pytest.raises(RuntimeError, match="Trn3"):
        moe.require_mx_platform()
    fake.get_platform_target = lambda: "trn3"
    moe.require_mx_platform()
    monkeypatch.delitem(sys.modules, "libtorch_neuronx_lite.compile.platform")
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite.compile", None)
    with pytest.raises(RuntimeError):
        moe.require_mx_platform()
    monkeypatch.setenv(moe.MX_FORCE_ENV, "1")
    moe.require_mx_platform()
