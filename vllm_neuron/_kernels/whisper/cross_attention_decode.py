# SPDX-License-Identifier: Apache-2.0
"""Whisper cross-attention decode kernel (``WHISPER_CROSS_MODE=xkernel``).

One call does the cross-attention of one decoder layer for B batch rows. Each row
attends over the 1500 encoder positions of its own request slot. The cross-KV does
not change during decode, so the kernel reads the K and V of each row one time, in
large contiguous DMAs, with the slot offset from a runtime scalar.

Layouts (H = heads per rank, S = slots, C = 1536 positions, of which 1500 are valid):
  q_t     [128, B*H/2] bf16. Column c holds the q of heads (2c, 2c+1) of the
          flattened (row, head) list: the even head on partitions 0-63, the odd
          head on 64-127. The caller applies the softmax scale to q.
  k_cache [S, H*64*C] bf16: K^T of each (slot, head), [64, C], d-major.
  v_cache [S, H*C*64] bf16: V of each (slot, head), [C, 64], position-major.
  slots   [1, B] int32: the slot of each row.
  qmask   [128, NP*U] bf16 (NP = U/2): 1 at (k, p, 2p) for k < 64 and at
          (k, p, 2p+1) for k >= 64, else 0. Built one time by the caller.
  dmask   [U, VB*64] fp32 (PV=1): 1 at (u, 64 (u % VB) + c), else 0.
  out     PV=1: [B*H, 64] bf16, row = row * H + head.
          PV=0: [B/G, 4, G*H/4 * 64] bf16. The context of unit u = 4 f + q of group g
          (unit u = row-in-group * H + head) is out[g, q, 64 f : 64 f + 64].

Math for each unit (row, head): s = q . K[:1500] (fp32), p = softmax(s) in fp32,
cast to bf16, ctx = p @ V with fp32 accumulation, cast to bf16.

Parameters:
  G   rows per group (G * H units on the score partitions, <= 128).
  KP  head pairs per K DMA (divides H / 2).
  VH  heads per V DMA (divides H).
  PV  1: blocked PV. Block b of VB units of one row: stationary = p^T with only the
      columns of block b (the other columns stay zero), moving = the V of the VB heads
      side by side (N = 64 VB). All blocks accumulate in one PSUM tile [U, 64 VB];
      unit u keeps column block u % VB (dmask), which is exact (the other blocks
      get p_u . V of a neighbour times 0). 0: one 1-column matmul per (unit, chunk).
  VB  units per PV block (PV=1; divides H and VH; 64 VB <= 512).
  HW  1: the K and V loads use hardware DGE (descriptors from the Sync / Scalar
      sequencers). 0: the compiler picks (software DGE on GpSimd for a runtime offset).
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

C = 1536
VALID = 1500
NCH = C // 128  # 12 position chunks of the PV contraction
QK_W = 512


@nki.jit
def whisper_xattn_decode(q_t, k_cache, v_cache, slots, qmask, dmask, G=4, KP=5, VH=10, PV=1, VB=5, HW=0):
    """See the module docstring."""
    B = slots.shape[1]
    H = 2 * q_t.shape[1] // B
    U = G * H
    NP = U // 2
    NG = B // G
    HP = H // 2
    F4 = U // 4
    NB = U // VB
    kernel_assert(U <= 128 and U % 4 == 0, "whisper_xattn_decode: G * H must be a multiple of 4 up to 128")
    kernel_assert(NG * G == B, "whisper_xattn_decode: G must divide the batch")
    kernel_assert(HP % KP == 0 and H % VH == 0, "whisper_xattn_decode: KP must divide H / 2 and VH must divide H")
    kernel_assert(PV == 0 or (VH % VB == 0 and VB * 64 <= 512), "whisper_xattn_decode: VB must divide VH, 64 VB <= 512")
    if PV == 1:
        out = nl.ndarray((B * H, 64), dtype=q_t.dtype, buffer=nl.shared_hbm)
        dm_sb = nl.ndarray((U, VB * 64), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dm_sb, src=dmask[0:U, 0:VB * 64])
        # One zero-padded p^T per PV block. Only the VB columns of the block are
        # written each group; the zeros stay.
        ptz = []
        for b in range(NB):
            t = nl.ndarray((128, NCH, U), dtype=q_t.dtype, buffer=nl.sbuf)
            nisa.memset(dst=t, value=0.0)
            ptz.append(t)
    else:
        out = nl.ndarray((NG, 4, F4 * 64), dtype=q_t.dtype, buffer=nl.shared_hbm)

    slot_sb = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_sb, src=slots[0:1, 0:B])

    # Masked q: qm[:, p, 2p] holds the even head of pair p on partitions 0-63 and
    # qm[:, p, 2p + 1] the odd head on 64-127; all else is zero. One matmul per pair
    # contracts both heads and writes the scores of each head to its own partition.
    mask_sb = nl.ndarray((128, NP, U), dtype=q_t.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mask_sb.reshape((128, NP * U)), src=qmask[0:128, 0:NP * U])

    for g in range(NG):
        qs = nl.ndarray((128, NP), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qs, src=q_t[0:128, g * NP:(g + 1) * NP])
        qm = nl.ndarray((128, NP, U), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qm, data1=qs.ap(pattern=[[NP, 128], [1, NP], [0, U]]),
                           data2=mask_sb, op=nl.multiply)

        sl = []
        for r in range(G):
            sl.append(nisa.scalar_load(slot_sb[0:1, g * G + r:g * G + r + 1]))

        # Scores [U, C], fp32, in three PSUM banks of 512 positions.
        sc = []
        for c in range(C // QK_W):
            sc.append(nl.ndarray((U, QK_W), dtype=nl.float32, buffer=nl.psum))
        for r in range(G):
            for kc in range(HP // KP):
                kb = nl.ndarray((128, KP, C), dtype=k_cache.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=kb,
                    src=k_cache.ap(pattern=[[C, 128], [128 * C, KP], [1, C]],
                                   offset=kc * KP * 128 * C, scalar_offset=sl[r], indirect_dim=0),
                    dge_mode=nisa.dge_mode.hwdge if HW == 1 else nisa.dge_mode.unknown,
                )
                for pk in range(KP):
                    p = r * HP + kc * KP + pk
                    for c in range(C // QK_W):
                        nisa.nc_matmul(dst=sc[c], stationary=qm[0:128, p, 0:U],
                                       moving=kb[0:128, pk, c * QK_W:(c + 1) * QK_W],
                                       accumulate=(p != 0))

        # Softmax over the 1500 valid positions, fp32. The pad positions get p = 0.
        widths = (QK_W, QK_W, VALID - 2 * QK_W)
        mx = nl.ndarray((U, 3), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(3):
            nisa.tensor_reduce(dst=mx[0:U, c:c + 1], op=nl.maximum, data=sc[c][0:U, 0:widths[c]], axis=1)
        m = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=mx, axis=1, negate=True)
        e = nl.ndarray((U, C), dtype=nl.float32, buffer=nl.sbuf)
        sums = nl.ndarray((U, 3), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(3):
            nisa.activation(dst=e[0:U, c * QK_W:c * QK_W + widths[c]], op=nl.exp,
                            data=sc[c][0:U, 0:widths[c]], bias=m, scale=1.0,
                            reduce_op=nl.add, reduce_res=sums[0:U, c:c + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
        tot = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=tot, op=nl.add, data=sums, axis=1)
        rs = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=tot)
        pb = nl.ndarray((U, 128, NCH), dtype=q_t.dtype, buffer=nl.sbuf)
        pbf = pb.reshape((U, C))
        nisa.memset(dst=pbf[0:U, VALID:C], value=0.0)
        nisa.tensor_scalar(dst=pbf[0:U, 0:VALID], data=e[0:U, 0:VALID], op0=nl.multiply, operand0=rs)

        # p^T: position 12 k + j goes to partition k of chunk j, the layout of a V DMA row.
        # One PSUM tile per chunk: 12 x U bf16 does not fit one 2 KiB PSUM bank for U > 85.
        pt = nl.ndarray((128, NCH, U), dtype=q_t.dtype, buffer=nl.sbuf)
        for j in range(NCH):
            pt_ps = nl.ndarray((128, U), dtype=q_t.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=pt_ps, data=pb[0:U, 0:128, j])
            if PV == 1:
                for b in range(NB):
                    nisa.tensor_copy(dst=ptz[b][0:128, j, b * VB:(b + 1) * VB],
                                     src=pt_ps[0:128, b * VB:(b + 1) * VB])
            else:
                nisa.tensor_copy(dst=pt[0:128, j, 0:U], src=pt_ps)
        if PV == 1:
            o_ps = nl.ndarray((U, VB * 64), dtype=nl.float32, buffer=nl.psum)
            first = 1
            for r in range(G):
                for vc in range(H // VH):
                    vb = nl.ndarray((128, VH, NCH * 64), dtype=v_cache.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=vb,
                        src=v_cache.ap(pattern=[[NCH * 64, 128], [C * 64, VH], [1, NCH * 64]],
                                       offset=vc * VH * C * 64, scalar_offset=sl[r], indirect_dim=0),
                    dge_mode=nisa.dge_mode.hwdge if HW == 1 else nisa.dge_mode.unknown,
                    )
                    for bb in range(VH // VB):
                        b = (r * H + vc * VH) // VB + bb
                        for j in range(NCH):
                            nisa.nc_matmul(dst=o_ps, stationary=ptz[b][0:128, j, 0:U],
                                           moving=vb[0:128, bb * VB:(bb + 1) * VB, 64 * j:64 * j + 64],
                                           accumulate=(first == 0))
                            first = 0
            o_m = nl.ndarray((U, VB, 64), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o_m.reshape((U, VB * 64)), data1=o_ps, data2=dm_sb, op=nl.multiply)
            o_sb = nl.ndarray((U, 64), dtype=q_t.dtype, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=o_sb, op=nl.add,
                               data=o_m.ap(pattern=[[VB * 64, U], [1, 64], [64, VB]]), axis=2)
            nisa.dma_copy(dst=out[g * U:(g + 1) * U, 0:64], src=o_sb)
            continue


        # PV: unit u = 4 f + qd accumulates on PSUM partition 32 qd, columns 64 f to
        # 64 f + 64. Column tiling runs the 4 partitions of one f at the same time.
        o_ps = nl.ndarray((128, F4 * 64), dtype=nl.float32, buffer=nl.psum)
        for r in range(G):
            for vc in range(H // VH):
                vb = nl.ndarray((128, VH, NCH * 64), dtype=v_cache.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=vb,
                    src=v_cache.ap(pattern=[[NCH * 64, 128], [C * 64, VH], [1, NCH * 64]],
                                   offset=vc * VH * C * 64, scalar_offset=sl[r], indirect_dim=0),
                    dge_mode=nisa.dge_mode.hwdge if HW == 1 else nisa.dge_mode.unknown,
                )
                for vk in range(VH):
                    u = r * H + vc * VH + vk
                    qd = u % 4
                    f = u // 4
                    for j in range(NCH):
                        nisa.nc_matmul(dst=o_ps[32 * qd:32 * qd + 1, 64 * f:64 * f + 64],
                                       stationary=pt[0:128, j, u:u + 1],
                                       moving=vb[0:128, vk, 64 * j:64 * j + 64],
                                       accumulate=(j != 0),
                                       tile_position=(0, 32 * qd), tile_size=(128, 32))
        o_sb = nl.ndarray((128, F4 * 64), dtype=q_t.dtype, buffer=nl.sbuf)
        for qd in range(4):
            nisa.tensor_copy(dst=o_sb[32 * qd:32 * qd + 1, 0:F4 * 64], src=o_ps[32 * qd:32 * qd + 1, 0:F4 * 64])
            nisa.dma_copy(dst=out[g, qd:qd + 1, 0:F4 * 64], src=o_sb[32 * qd:32 * qd + 1, 0:F4 * 64])
    return out


def xkernel_group(heads: int, rows: int) -> int:
    """Return the rows of one kernel group: the most rows with ``rows * heads <= 128``.

    The group must divide the batch and keep a multiple of 4 units. 20 heads give 4 rows
    (80 units) for a batch that is a multiple of 4.
    """
    g = max(1, 128 // heads)
    while g > 1 and (rows % g != 0 or (g * heads) % 4 != 0):
        g -= 1
    return g


def xkernel_qmask(heads: int, group: int, device=None, dtype=None):
    """Return the ``[128, NP * U]`` 0/1 head-pair mask of the kernel.

    Entry ``(k, p, u)`` is 1 when ``u == 2 p + (k >= 64)``. It is built from
    ``arange`` comparisons, so that it traces as device ops in a compiled graph.
    """
    import torch

    units = group * heads
    pairs = units // 2
    k = torch.arange(128, device=device).view(128, 1, 1)
    p = torch.arange(pairs, device=device).view(1, pairs, 1)
    u = torch.arange(units, device=device).view(1, 1, units)
    m = u == 2 * p + (k >= 64).to(p.dtype)
    return m.to(dtype or torch.float32).reshape(128, pairs * units)


def xkernel_dmask(heads: int, group: int, vb: int, device=None):
    """Return the ``[U, 64 vb]`` fp32 column-block mask of the blocked PV (PV=1)."""
    import torch

    units = group * heads
    col = torch.arange(64 * vb, device=device).view(1, -1) // 64
    unit = torch.arange(units, device=device).view(-1, 1) % vb
    return (col == unit).to(torch.float32)


def whisper_xattn_decode_torch(q, k_cache, v_cache, slots):
    """Return the context ``[B, H, 64]`` with the kernel math, in torch (CPU reference).

    Args:
        q: ``[B, H, 64]``, scaled.
        k_cache: ``[S, H, 64, C]``.
        v_cache: ``[S, H, C, 64]``.
        slots: ``[B]`` int.
    """
    import torch

    k = k_cache.index_select(0, slots.to(torch.int64))[..., :VALID]  # [B, H, 64, 1500]
    v = v_cache.index_select(0, slots.to(torch.int64))[:, :, :VALID]  # [B, H, 1500, 64]
    s = torch.einsum("bhd,bhdt->bht", q.float(), k.float())
    p = torch.softmax(s, dim=-1).to(v.dtype)
    return torch.einsum("bht,bhtd->bhd", p.float(), v.float()).to(q.dtype)
