# SPDX-License-Identifier: Apache-2.0
"""NKI kernel: block-local (windowed) bidirectional attention for 128-token blocks, head_dim 64.

Replaces the per-(block, head) ``attention_cte`` calls of a block-packed audio encoder. One call
does all blocks and all heads of one layer, reading the fused QKV projection output directly.

Inputs (bf16):
  qkv    [nb, 128, 3 * NH * 64]  fused projection output, token-major (q | k | v, head-major inside each)
  seg_q  [nb, 16, 128]           query-side segment factors (see ``build_segment_factors``)
  seg_k  [nb, 16, 128]           key-side segment factors
Output:
  out    [nb, NH, 64, 128]       attention output, feature-major (BNdS), the layout o_proj reads natively.

Masking: the additive mask ``bias[q, k] = 0 if q and k are in the same segment else -4096`` is the
rank-16 product ``seg_q[:, q]^T @ seg_k[:, k]``. The factors are appended as contraction rows 64..79
of Q^T / K^T, so the QK^T matmul (K = 80) returns masked scores with no masking instruction.
Factorization (no cancellation, so same-segment entries get an exact 0): key row r < 15 is the
one-hot of "key k is in segment r", key row 15 marks padding keys; query row r is -4096 unless the
query is in segment r, query row 15 is -4096. A padding query (no segment) gets -4096 everywhere, so
its softmax is uniform and finite; its output is never read by a real token.

Per work unit (one block x G = min(8, NH) heads):
  DMA   q, k, v column slabs [128, G*64] and the two [16, 128] factor tiles
  PE    16 transposes   [128 tok, 64] -> [64, 128 tok]   (Q^T, K^T, into one bf16 PSUM bank each)
  PE    G   QK^T        stationary Q^T_h [80, 128], moving K^T_h [80, 128] -> S [128 q, G, 128 k] fp32
  DVE   row max of S (negated), from PSUM, one instruction for G heads
  ACT   G   exp(scale * S - scale * max) from PSUM with fused row sum -> P bf16
  DVE   1/sum, P *= 1/sum (one instruction for G heads)
  PE    G   transposes  P_h -> P_h^T [128 k, 128 q]  (one bf16 PSUM bank)
  PE    G   PV          stationary V_h [128 k, 64], moving P_h^T -> O_h^T [64, 128 q]; two heads share
                        one 128-partition PSUM tile (partitions 0-63 / 64-127), so one eviction covers both
  DMA   O^T [G*64, 128] contiguous into out[b, g*G:(g+1)*G]
Every PSUM tile is written by single, complete (accumulate=False) matmuls into disjoint regions and
evicted before reuse; no PSUM accumulation, no ring/modulo buffers, no DMA skipping.

LNC: a 1-D grid of n programs splits the work units (blocks if nb % n == 0, else head groups).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128          # tokens per block (partition dim)
D = 64           # head dim
SEG_ROWS = 16    # segment factor rows (<= 15 segments per block + 1 constant row)
KA = D + SEG_ROWS
MASK_BIG = 4096.0
MAX_GROUP = 8    # heads per work unit: S [128, 8, 128] fp32 = 2 PSUM banks, P^T bf16 = 1 bank


def _unit(qkv2, sq2, sk2, out2, b, g, NH, G, scale, dt):
    """One block x one group of G heads. b may be symbolic (LNC grid), g is static or symbolic."""
    HD = NH * D
    row0 = b * P
    qcol = g * (G * D)

    # ---- loads -----------------------------------------------------------------------------
    x_q = nl.ndarray((P, G * D), dtype=dt, buffer=nl.sbuf)
    x_k = nl.ndarray((P, G * D), dtype=dt, buffer=nl.sbuf)
    x_v = nl.ndarray((P, G * D), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_q, src=qkv2[nl.ds(row0, P), nl.ds(qcol, G * D)])
    nisa.dma_copy(dst=x_k, src=qkv2[nl.ds(row0, P), nl.ds(HD + qcol, G * D)])
    nisa.dma_copy(dst=x_v, src=qkv2[nl.ds(row0, P), nl.ds(2 * HD + qcol, G * D)])

    qT = nl.ndarray((KA, G, P), dtype=dt, buffer=nl.sbuf)
    kT = nl.ndarray((KA, G, P), dtype=dt, buffer=nl.sbuf)
    # segment factor rows -> partitions 64..79, replicated over the G head slots by a stride-0 read
    nisa.dma_copy(dst=qT[D:KA, :, :],
                  src=sq2.ap(pattern=[[P, SEG_ROWS], [0, G], [1, P]], offset=b * SEG_ROWS * P))
    nisa.dma_copy(dst=kT[D:KA, :, :],
                  src=sk2.ap(pattern=[[P, SEG_ROWS], [0, G], [1, P]], offset=b * SEG_ROWS * P))

    # ---- Q^T, K^T (PE transposes, 64-wide so every head lands on partitions 0..63) ----------
    tq = nl.ndarray((D, G, P), dtype=dt, buffer=nl.psum)
    tk = nl.ndarray((D, G, P), dtype=dt, buffer=nl.psum)
    for h in range(G):
        nisa.nc_transpose(dst=tq[:, h, :], data=x_q[:, h * D:(h + 1) * D], engine=nisa.engine.tensor)
    for h in range(G):
        nisa.nc_transpose(dst=tk[:, h, :], data=x_k[:, h * D:(h + 1) * D], engine=nisa.engine.tensor)
    nisa.tensor_copy(dst=qT[0:D, :, :], src=tq, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=kT[0:D, :, :], src=tk, engine=nisa.engine.scalar)

    # ---- S = Q K^T + mask (K = 80) -------------------------------------------------------------
    s_ps = nl.ndarray((P, G, P), dtype=nl.float32, buffer=nl.psum)
    for h in range(G):
        nisa.nc_matmul(dst=s_ps[:, h, :], stationary=qT[:, h, :], moving=kT[:, h, :], accumulate=False)

    # ---- softmax ------------------------------------------------------------------------------
    nmax = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=nmax, op=nl.maximum, data=s_ps, axis=2, negate=True)
    nbias = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nbias, data=nmax, op0=nl.multiply, operand0=scale, engine=nisa.engine.gpsimd)
    p_sb = nl.ndarray((P, G, P), dtype=dt, buffer=nl.sbuf)
    rsum = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    for h in range(G):
        nisa.activation_reduce(dst=p_sb[:, h, :], op=nl.exp, data=s_ps[:, h, :], reduce_op=nl.add,
                               reduce_res=rsum[:, h:h + 1], bias=nbias[:, h:h + 1], scale=scale)
    rrec = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rrec, data=rsum)
    pn = nl.ndarray((P, G, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pn, data1=p_sb, data2=rrec.ap(pattern=[[G, P], [1, G], [0, P]], offset=0),
                       op=nl.multiply)

    # ---- P^T ----------------------------------------------------------------------------------
    pt_ps = nl.ndarray((P, G, P), dtype=dt, buffer=nl.psum)
    for h in range(G):
        nisa.nc_transpose(dst=pt_ps[:, h, :], data=pn[:, h, :], engine=nisa.engine.tensor)
    pt = nl.ndarray((P, G, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pt, src=pt_ps, engine=nisa.engine.vector)

    # ---- O^T = V^T P^T, two heads per 128-partition PSUM tile --------------------------------
    NPAIR = G // 2
    ot_ps = nl.ndarray((P, NPAIR, P), dtype=nl.float32, buffer=nl.psum)
    for j in range(NPAIR):
        for e in range(2):
            h = 2 * j + e
            nisa.nc_matmul(dst=ot_ps[e * D:(e + 1) * D, j, :], stationary=x_v[:, h * D:(h + 1) * D],
                           moving=pt[:, h, :], accumulate=False)
    o_sb = nl.ndarray((P, NPAIR, P), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_sb, src=ot_ps, engine=nisa.engine.scalar)

    # out2 is [nb * NH * 64, 128]; heads g*G .. g*G+G-1 of block b are G*64 contiguous rows.
    # SBUF o_sb[p, j, s] = row (2j + p // 64) * 64 + p % 64 = j * 128 + p of that slab.
    orow = (b * NH + g * G) * D
    nisa.dma_copy(dst=out2.ap(pattern=[[P, P], [P * P, NPAIR], [1, P]], offset=orow * P), src=o_sb)


@nki.jit
def encoder_window_attention_kernel(qkv, seg_q, seg_k, scale=0.125, num_heads=16):
    nb, S, F = qkv.shape
    NH = num_heads
    kernel_assert(S == P, "block size must be 128")
    kernel_assert(F == 3 * NH * D, "qkv last dim must be 3 * num_heads * 64")
    kernel_assert(NH % 2 == 0, "num_heads must be even")
    G = MAX_GROUP if NH >= MAX_GROUP else NH
    kernel_assert(NH % G == 0, "num_heads must be a multiple of the head group")
    n_grp = NH // G
    dt = qkv.dtype

    out = nl.ndarray((nb, NH, D, S), dtype=dt, buffer=nl.shared_hbm)
    qkv2 = qkv.reshape((nb * S, F))
    sq2 = seg_q.reshape((nb * SEG_ROWS, S))
    sk2 = seg_k.reshape((nb * SEG_ROWS, S))
    out2 = out.reshape((nb * NH * D, S))

    n_prg = 1
    if nl.program_ndim() > 0:
        n_prg = nl.num_programs(0)
    if n_prg == 1:
        for b in range(nb):
            for g in range(n_grp):
                _unit(qkv2, sq2, sk2, out2, b, g, NH, G, scale, dt)
    else:
        pid = nl.program_id(0)
        if nb % n_prg == 0:
            bpp = nb // n_prg
            for bi in range(bpp):
                for g in range(n_grp):
                    _unit(qkv2, sq2, sk2, out2, pid * bpp + bi, g, NH, G, scale, dt)
        else:
            kernel_assert(n_grp % n_prg == 0, "work units must split evenly across the LNC grid")
            gpp = n_grp // n_prg
            for b in range(nb):
                for gi in range(gpp):
                    _unit(qkv2, sq2, sk2, out2, b, pid * gpp + gi, NH, G, scale, dt)
    return out
