# SPDX-License-Identifier: Apache-2.0
"""Whisper self-attention decode kernel over the slot self-KV (``WHISPER_SELF_ATTN=kernel``).

One call does the self-attention of one decoder layer for B batch rows. Each row attends
over the first ``CTX`` positions of its own request slot. The caller writes the new K/V
of each row before the call, and gives an additive mask for the positions after each
row's position.

Layouts (H = heads per rank, even; S = slots; CS = positions per slot; E = H * 64).
An odd head count (TP4: 5) is padded to the next even count by the caller (zero q, and
the buffers carry a zero pad head).
  q_t     [128, B*H/2] bf16. Column c holds the q of units (2c, 2c+1) of the flattened
          (row, head) list: the even unit on partitions 0-63, the odd unit on 64-127.
          The caller applies the softmax scale to q.
  k_cache [S, CS*E] bf16: K of each (slot, position), all heads side by side
          (the ``slot_k`` buffer ``[S, CS, E]`` as a 2D view).
  v_cache [S, CS*E] bf16: V, the same layout.
  slots   [1, B] int32: the slot of each row.
  qmask   [128, NP*U] bf16 (NP = U/2): the head-pair mask of the cross-attention kernel.
  dmask   [U, VB*64] fp32: 1 at (u, 64 (u % VB) + c), else 0.
  maskb   [B*H, CTX] fp32: 0 for a valid position, -1e30 for a masked one, in the
          column order of the kernel (see ``self_attn_col_pos``).
  out     [B*H, 64] bf16, row = row * H + head.

Position layout: M = positions per partition (``self_attn_m``), P = CTX / M partitions.
Partition k of the K or V tile of a row holds the positions k*M .. k*M + M - 1, which is
one contiguous run of M * E * 2 bytes in HBM. Chunk j is the positions k*M + j
(k = 0..P-1); score column j*P + k is position k*M + j.

Math for each unit (row, head): s = q . K (fp32), p = softmax(s + maskb) in fp32, cast
to bf16, ctx = p @ V with fp32 accumulation, cast to bf16.

LNC: the launch grid is the LNC degree. With 2 programs, program i runs the groups
g = i, i + 2, ... (each group is independent and writes its own output rows).
"""
import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert


@nki.jit
def whisper_self_attn_decode(q_t, k_cache, v_cache, slots, qmask, dmask, maskb, G=4, VB=5, M=1):
    """See the module docstring."""
    B = slots.shape[1]
    H = 2 * q_t.shape[1] // B
    E = H * 64
    U = G * H
    NP = U // 2
    NG = B // G
    HP = H // 2
    NB = U // VB
    CTX = maskb.shape[1]
    CS = k_cache.shape[1] // E
    P = CTX // M
    kernel_assert(H % 2 == 0, "whisper_self_attn_decode: H must be even (pad odd head counts)")
    kernel_assert(U <= 128, "whisper_self_attn_decode: G * H up to 128")
    kernel_assert(NG * G == B, "whisper_self_attn_decode: G must divide the batch")
    kernel_assert(P * M == CTX and P <= 128, "whisper_self_attn_decode: CTX must be M * P with P <= 128")
    kernel_assert(CTX <= CS and CTX <= 512, "whisper_self_attn_decode: CTX up to the slot length and 512")
    kernel_assert(H % VB == 0 and VB * 64 <= 512, "whisper_self_attn_decode: VB must divide H, 64 VB <= 512")

    out = nl.ndarray((B * H, 64), dtype=q_t.dtype, buffer=nl.shared_hbm)
    dm_sb = nl.ndarray((U, VB * 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dm_sb, src=dmask[0:U, 0:VB * 64])
    # One zero-padded p^T per PV block. Only the VB columns of the block are written
    # each group; the zeros stay.
    ptz = []
    for b in range(NB):
        t = nl.ndarray((128, M, U), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        ptz.append(t)

    slot_sb = nl.ndarray((1, B), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=slot_sb, src=slots[0:1, 0:B])
    mask_sb = nl.ndarray((128, NP, U), dtype=q_t.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mask_sb.reshape((128, NP * U)), src=qmask[0:128, 0:NP * U])

    n_prg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    prg = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    for g in range(prg, NG, n_prg):
        qs = nl.ndarray((128, NP), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qs, src=q_t[0:128, g * NP:(g + 1) * NP])
        qm = nl.ndarray((128, NP, U), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qm, data1=qs.ap(pattern=[[NP, 128], [1, NP], [0, U]]),
                           data2=mask_sb, op=nl.multiply)
        mb = nl.ndarray((U, CTX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mb, src=maskb[g * U:(g + 1) * U, 0:CTX])

        sl = []
        for r in range(G):
            sl.append(nisa.scalar_load(slot_sb[0:1, g * G + r:g * G + r + 1]))

        # Scores [U, CTX] fp32 in one PSUM bank. For each row and head pair: one PE
        # transpose of each chunk of K into [128 (pair, d), P], then one matmul with the
        # masked q of the pair over all CTX columns.
        sc = nl.ndarray((U, CTX), dtype=nl.float32, buffer=nl.psum)
        vbs = []
        for r in range(G):
            kb = nl.ndarray((128, M * E), dtype=k_cache.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=kb[0:P, 0:M * E],
                src=k_cache.ap(pattern=[[M * E, P], [1, M * E]],
                               offset=0, scalar_offset=sl[r], indirect_dim=0),
            )
            # Start the V load of the row now: it overlaps the scores and the softmax.
            vb = nl.ndarray((128, M * E), dtype=v_cache.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=vb[0:P, 0:M * E],
                src=v_cache.ap(pattern=[[M * E, P], [1, M * E]],
                               offset=0, scalar_offset=sl[r], indirect_dim=0),
            )
            vbs.append(vb)
            for pk in range(HP):
                kt_ps = nl.ndarray((128, CTX), dtype=k_cache.dtype, buffer=nl.psum)
                for j in range(M):
                    nisa.nc_transpose(dst=kt_ps[0:128, j * P:(j + 1) * P],
                                      data=kb[0:P, j * E + pk * 128:j * E + pk * 128 + 128])
                kt = nl.ndarray((128, CTX), dtype=k_cache.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=kt, src=kt_ps)
                p = r * HP + pk
                nisa.nc_matmul(dst=sc, stationary=qm[0:128, p, 0:U], moving=kt,
                               accumulate=(p != 0))

        # Softmax over the CTX columns, fp32. A masked position gets p = 0.
        s = nl.ndarray((U, CTX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=s, data1=sc, data2=mb, op=nl.add)
        m = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=s, axis=1, negate=True)
        e = nl.ndarray((U, CTX), dtype=nl.float32, buffer=nl.sbuf)
        tot = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=s, bias=m, scale=1.0,
                        reduce_op=nl.add, reduce_res=tot, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rs = nl.ndarray((U, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=tot)
        pb = nl.ndarray((U, CTX), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pb, data=e, op0=nl.multiply, operand0=rs)

        # p^T of chunk j: [P, U], partition k is position k*M + j, the layout of the V tile.
        for j in range(M):
            pt_ps = nl.ndarray((128, U), dtype=q_t.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=pt_ps[0:P, 0:U], data=pb[0:U, j * P:(j + 1) * P])
            for b in range(NB):
                nisa.tensor_copy(dst=ptz[b][0:P, j, b * VB:(b + 1) * VB],
                                 src=pt_ps[0:P, b * VB:(b + 1) * VB])

        # Blocked PV (as in the cross-attention kernel): block b of VB units of one row,
        # stationary = p^T with only the columns of block b, moving = the V of the VB heads
        # side by side. Unit u keeps column block u % VB (dmask).
        o_ps = nl.ndarray((U, VB * 64), dtype=nl.float32, buffer=nl.psum)
        first = 1
        for r in range(G):
            for bb in range(H // VB):
                b = r * (H // VB) + bb
                for j in range(M):
                    nisa.nc_matmul(dst=o_ps, stationary=ptz[b][0:P, j, 0:U],
                                   moving=vbs[r][0:P, j * E + bb * VB * 64:j * E + (bb + 1) * VB * 64],
                                   accumulate=(first == 0))
                    first = 0
        o_m = nl.ndarray((U, VB, 64), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o_m.reshape((U, VB * 64)), data1=o_ps, data2=dm_sb, op=nl.multiply)
        o_sb = nl.ndarray((U, 64), dtype=q_t.dtype, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=o_sb, op=nl.add,
                           data=o_m.ap(pattern=[[VB * 64, U], [1, 64], [64, VB]]), axis=2)
        nisa.dma_copy(dst=out[g * U:(g + 1) * U, 0:64], src=o_sb)
    return out


def self_attn_m(ctx: int) -> int:
    """Return M, the positions per partition: the smallest M >= ctx / 128 that divides ``ctx``."""
    m = -(-ctx // 128)
    while ctx % m:
        m += 1
    return m


def self_attn_col_pos(ctx: int, device=None):
    """Return the position of each score column, ``[ctx]`` int64 (column j*P + k is k*M + j)."""
    import torch

    m = self_attn_m(ctx)
    p = ctx // m
    col = torch.arange(ctx, device=device)
    return (col % p) * m + col // p


def self_attn_vb(heads: int, vb: int = 5) -> int:
    """Return the units per PV block: the largest divisor of ``heads`` up to ``vb`` (64 vb <= 512)."""
    return max(d for d in range(1, min(vb, 8) + 1) if heads % d == 0)


def whisper_self_attn_decode_torch(q, k_slots, v_slots, slots, positions, ctx):
    """Return the context ``[B, H, 64]`` with the kernel math, in torch (CPU reference).

    Args:
        q: ``[B, H, 64]``, scaled.
        k_slots: ``[S, CS, H, 64]``.
        v_slots: ``[S, CS, H, 64]``.
        slots: ``[B]`` int.
        positions: ``[B]`` int, the position of each row (positions above it are masked).
        ctx: the positions to read.
    """
    import torch

    k = k_slots.index_select(0, slots.to(torch.int64))[:, :ctx]
    v = v_slots.index_select(0, slots.to(torch.int64))[:, :ctx]
    s = torch.einsum("bhd,bthd->bht", q.float(), k.float())
    valid = torch.arange(ctx).view(1, 1, ctx) <= positions.view(-1, 1, 1).to(torch.int64)
    s = s.masked_fill(~valid, float("-inf"))
    p = torch.softmax(s, dim=-1).to(v.dtype)
    return torch.einsum("bht,bthd->bhd", p.float(), v.float()).to(q.dtype)
