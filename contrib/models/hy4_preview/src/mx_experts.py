# SPDX-License-Identifier: Apache-2.0
"""OCP MXFP8 routed-expert weights -> nkilib ``moe_tkg`` MX layouts (Trn3 only).

Pure torch, no vLLM imports. Used by ``moe.py`` when the checkpoint was written
with ``convert_mxfp8_to_neuron_fp8.py --keep-mx-experts``.

Checkpoint (ModelOpt MXFP8, per layer, copied through unchanged by the converter):
    gate_up_proj        [E, 2I, H]     float8_e4m3fn (OCP). Rows [0, I) gate, [I, 2I) up.
    gate_up_proj_scale  [E, 2I, H/32]  uint8 e8m0; one exponent per 32 consecutive H.
    down_proj           [E, H, I]      float8_e4m3fn.
    down_proj_scale     [E, H, I/32]   uint8 e8m0; one exponent per 32 consecutive I.
    value = fp8 * 2**(e8m0 - 127)

Kernel layouts produced (I = per-rank intermediate I_r, n_H = H/512, n_I = ceil(I/512),
I_p = I/4 if I <= 512 else 128; uint32 words hold 4 fp8 bytes, byte q at bits 8q, i.e.
``float8_e4m3fn_x4``):
    gate_up weight  uint32 [E, 128, 2, n_H, I]
        word [e, p, g, n, c] = bytes W_g[e, i(c), 4r .. 4r+3], r = 128 n + p (natural H)
    gate_up scale   uint8  [E, 16, 2, n_H, I]
        [e, s, g, n, c] = S_g[e, i(c), (128 n + 8 s) / 8] = e8m0 of H block 16 n + s
    down weight     uint32 [E, I_p, n_I, H]
        word [e, p, t, h'] = bytes W_d[e, h(h'), 512 t + 4p .. 512 t + 4p + 3]
    down scale      uint8  [E, I_p/8, n_I, H]
        [e, s, t, h'] = S_d[e, h(h'), (512 t + 32 s) / 32]
  where the I column permutation (inherited from gpt-oss, so the gate/up result lands in
  the [I_p, n_I, T, 4] SBUF layout the down projection contracts over) is, per 512 tile,
    c = 512 t + qw * (I_tile/4) + 8 qb + qh   <->   i(c) = 512 t + 32 qb + 4 qh + qw
  and the H output permutation is the gpt-oss 4-way shuffle:
    h' = hb * (H/4) + nb   <->   h(h') = 4 nb + hb.

Activations: the kernel reads hidden[t, q*H/4 + r] as element q of word r (nkilib
moe_tkg_torch.py line 598, mlp_proj_mx_torch.py lines 90-92), so feed ``shuffle_hidden(x)``;
each 8-partition x 4 MX block then covers 32 consecutive natural H (the OCP grouping). The
output comes back in shuffled order; apply ``unshuffle_hidden(y)``.

At I_r = 32 (TP=64): gate_up [E,128,2,H/512,32], gu scale [E,16,2,H/512,32],
down [E,8,1,H], down scale [E,1,1,H].
"""

from __future__ import annotations

import math

import torch

PMAX = 128
Q_WIDTH = 4
Q_HEIGHT = 8
MX_BLOCK = Q_WIDTH * Q_HEIGHT  # 32


# ---------------------------------------------------------------------------------------
# Hidden-dim shuffle (adapted from vllm_neuron/model/gpt_oss/weight_loaders_mxfp4.py
# ``_shuffle_hidden_dim`` (602-634) and ``unshuffle_hidden_dim_last`` (590-599)).
# ---------------------------------------------------------------------------------------
def shuffle_hidden(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Natural H -> kernel H order: [H] -> [H/4, 4] -> [4, H/4] -> [H]."""
    if dim < 0:
        dim += x.ndim
    H = x.shape[dim]
    assert H % 4 == 0, f"H={H} not divisible by 4"
    pre, post = x.shape[:dim], x.shape[dim + 1 :]
    return x.reshape(*pre, H // 4, 4, *post).transpose(dim, dim + 1).reshape(*pre, H, *post)


def unshuffle_hidden(y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Kernel H order -> natural H: [H] -> [4, H/4] -> [H/4, 4] -> [H]."""
    if dim < 0:
        dim += y.ndim
    H = y.shape[dim]
    assert H % 4 == 0, f"H={H} not divisible by 4"
    pre, post = y.shape[:dim], y.shape[dim + 1 :]
    return y.reshape(*pre, 4, H // 4, *post).transpose(dim, dim + 1).reshape(*pre, H, *post)


# ---------------------------------------------------------------------------------------
# Tiling parameters (gpt-oss ``_get_h_tiling_shard_i`` / ``_get_i_tiling_shard_i``, 884-955)
# ---------------------------------------------------------------------------------------
def _h_tiling(H: int) -> tuple[int, int]:
    assert H % (PMAX * Q_WIDTH) == 0, f"H={H} must be a multiple of 512"
    return H // (PMAX * Q_WIDTH), 512 // MX_BLOCK  # (n_H, 16)


def _i_tiling(I: int) -> tuple[int, int]:
    # Same rule as the kernel check in nkilib mlp_parameters.py:767-769.
    if I > 512:
        assert I % 512 == 0, f"I={I} > 512 must be a multiple of 512"
        return I // 512, 512 // MX_BLOCK
    assert I % MX_BLOCK == 0, f"I={I} must be a multiple of 32"
    return 1, I // MX_BLOCK


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    assert t.dtype == torch.float8_e4m3fn, f"expected float8_e4m3fn or uint8, got {t.dtype}"
    return t.view(torch.uint8)


def _pack_x4_last(u8: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K] -> uint32 [..., K/4]; element 4j+q goes to byte q of word j."""
    u8 = u8.contiguous()
    assert u8.shape[-1] % 4 == 0
    return u8.view(torch.uint32)


# ---------------------------------------------------------------------------------------
# Tilers (adapted from weight_loaders_mxfp4.py: ``_tile_gate_up_blocks`` 660-726,
# ``_tile_gate_up_scale`` 729-781, ``_tile_down_blocks`` 821-852, ``_tile_down_scale``
# 855-881). Differences: the packed word is uint32 (4 fp8) instead of uint16 (4 fp4); the
# q_packed dim is therefore 1 and dropped.
# ---------------------------------------------------------------------------------------
def _tile_gate_up_blocks(w: torch.Tensor) -> torch.Tensor:
    """[E, H/4, 2I] words -> [E, 128, 2, n_H, I]."""
    E, H4, I2 = w.shape
    n_H, qb_H = _h_tiling(H4 * 4)
    n_I, qb_I = _i_tiling(I2 // 2)
    w = w.reshape(E, n_H, qb_H, Q_HEIGHT, 2, n_I, qb_I, Q_HEIGHT, Q_WIDTH)
    # E, qb_H, qh_H, g, n_H, n_I, qw_I, qb_I, qh_I
    w = w.permute(0, 2, 3, 4, 1, 5, 8, 6, 7)
    return w.reshape(E, qb_H * Q_HEIGHT, 2, n_H, -1)


def _tile_gate_up_scale(s: torch.Tensor) -> torch.Tensor:
    """[E, H/32, 2I] -> [E, 16, 2, n_H, I]."""
    E, H32, I2 = s.shape
    n_H, qb_H = _h_tiling(H32 * 32)
    n_I, qb_I = _i_tiling(I2 // 2)
    s = s.reshape(E, n_H, qb_H, 2, n_I, qb_I, Q_HEIGHT, Q_WIDTH)
    s = s.permute(0, 2, 3, 1, 4, 7, 5, 6)
    return s.reshape(E, qb_H, 2, n_H, -1)


def _tile_down_blocks(w: torch.Tensor) -> torch.Tensor:
    """[E, I/4, H] words -> [E, I_p, n_I, H] (H shuffled)."""
    E, I4, H = w.shape
    n_I, qb_I = _i_tiling(I4 * 4)
    w = w.reshape(E, n_I, qb_I, Q_HEIGHT, H // 4, 4)
    w = w.permute(0, 2, 3, 1, 5, 4)
    return w.reshape(E, qb_I * Q_HEIGHT, n_I, H)


def _tile_down_scale(s: torch.Tensor) -> torch.Tensor:
    """[E, I/32, H] -> [E, I_p/8, n_I, H] (H shuffled)."""
    E, I32, H = s.shape
    n_I, qb_I = _i_tiling(I32 * 32)
    s = s.reshape(E, n_I, qb_I, H // 4, 4)
    s = s.permute(0, 2, 1, 4, 3)
    return s.reshape(E, qb_I, n_I, H)


# ---------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------
def check_dims(H: int, I_r: int) -> None:
    """Raise if (H, per-rank I) cannot be tiled for the MX ``moe_tkg`` kernel."""
    if H % (PMAX * Q_WIDTH):
        raise ValueError(f"MX experts need hidden_size % 512 == 0, got {H}")
    if not (I_r % 512 == 0 or (I_r < 512 and I_r % MX_BLOCK == 0)):
        raise ValueError(
            f"MX experts need the per-rank intermediate size to be a multiple of 512, "
            f"or < 512 and a multiple of 32; got {I_r}"
        )


def shard_experts_tp(gate_up, gate_up_scale, down, down_scale, rank: int, tp: int):
    """Slice one TP rank's I columns out of full-checkpoint expert tensors (contiguous halves).

    Returns (gate_up [E, 2I_r, H], gu_scale [E, 2I_r, H/32], down [E, H, I_r], d_scale [E, H, I_r/32]).
    """
    I = gate_up.shape[1] // 2
    assert I % tp == 0
    I_r = I // tp
    assert I_r % MX_BLOCK == 0, f"I_r={I_r} must be a multiple of 32 so down scales shard cleanly"
    lo, hi = rank * I_r, (rank + 1) * I_r
    gu = torch.cat([gate_up[:, lo:hi], gate_up[:, I + lo : I + hi]], dim=1)
    gus = torch.cat([gate_up_scale[:, lo:hi], gate_up_scale[:, I + lo : I + hi]], dim=1)
    dn = down[:, :, lo:hi]
    dns = down_scale[:, :, lo // MX_BLOCK : hi // MX_BLOCK]
    return gu, gus, dn, dns


def tile_gate_up_weight(gate_up: torch.Tensor) -> torch.Tensor:
    """fp8/uint8 [E, 2I, H] (gate rows then up rows) -> uint32 [E, 128, 2, H/512, I]."""
    u8 = _as_u8(gate_up)
    E, I2, H = u8.shape
    words = _pack_x4_last(u8)  # [E, 2I, H/4], word r = H 4r..4r+3
    words = words.transpose(1, 2)  # [E, H/4, 2I]
    return _tile_gate_up_blocks(words).contiguous()


def tile_gate_up_scale(gate_up_scale: torch.Tensor) -> torch.Tensor:
    """uint8 [E, 2I, H/32] -> uint8 [E, 16, 2, H/512, I]."""
    assert gate_up_scale.dtype == torch.uint8
    return _tile_gate_up_scale(gate_up_scale.transpose(1, 2)).contiguous()


def tile_down_weight(down: torch.Tensor) -> torch.Tensor:
    """fp8/uint8 [E, H, I] -> uint32 [E, I_p, ceil(I/512), H] (I_p = I/4 if I <= 512 else 128)."""
    u8 = _as_u8(down)
    E, H, I = u8.shape
    words = _pack_x4_last(u8)  # [E, H, I/4], word j = I 4j..4j+3
    words = words.transpose(1, 2)  # [E, I/4, H]
    return _tile_down_blocks(words).contiguous()


def tile_down_scale(down_scale: torch.Tensor) -> torch.Tensor:
    """uint8 [E, H, I/32] -> uint8 [E, I_p/8, ceil(I/512), H]."""
    assert down_scale.dtype == torch.uint8
    return _tile_down_scale(down_scale.transpose(1, 2)).contiguous()


def convert_rank_experts(gate_up, gate_up_scale, down, down_scale):
    """One rank's checkpoint slice -> dict of the four moe_tkg MX tensors."""
    return {
        "expert_gate_up_weights": tile_gate_up_weight(gate_up),
        "expert_gate_up_weights_scale": tile_gate_up_scale(gate_up_scale),
        "expert_down_weights": tile_down_weight(down),
        "expert_down_weights_scale": tile_down_scale(down_scale),
    }


def expected_shapes(E: int, H: int, I: int) -> dict:
    n_H = H // 512
    n_I = math.ceil(I / 512)
    I_p = I // 4 if I <= 512 else 128
    return {
        "expert_gate_up_weights": (E, 128, 2, n_H, I),
        "expert_gate_up_weights_scale": (E, 16, 2, n_H, I),
        "expert_down_weights": (E, I_p, n_I, H),
        "expert_down_weights_scale": (E, I_p // 8, n_I, H),
    }
