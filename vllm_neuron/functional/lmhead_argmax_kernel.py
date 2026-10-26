# SPDX-License-Identifier: Apache-2.0
"""NKI kernel: fused lm_head GEMM + greedy argmax (no logits written to HBM).

    idx[b] = argmax_v  bf16( x[b, :] @ w[:, v] )       (first index on ties, like torch.argmax)

Inputs:
  x  [B, H]  bf16 hidden states at the sampling positions (B <= 128, H % 128 == 0)
  w  [H, V]  bf16 lm_head weight in the ``in_out`` layout of ColumnParallelLinear (row h holds V
             contiguous vocab entries)
Outputs (per LNC program p, which owns vocab columns [p * Vp, (p + 1) * Vp), Vp = V / n_programs):
  val [n_prg, B, 1] fp32   max logit (bf16-rounded value, as the stock bf16 logits)
  idx [n_prg, B, 1] int32  argmax index LOCAL to the program's vocab range
The caller adds ``p * Vp`` and picks the program with the larger value (lower p on ties); at
n_prg == 1 ``idx[0, :, 0]`` is the answer.

Structure (weight-bound: the whole 2 x H x V-byte weight is streamed once, everything else hides
under it):
  - x is loaded once ([B, H], B partitions) and PE-transposed into x^T [128, H / 128, B]: the
    stationary operand of every matmul (K = 128 rows of h, M = B columns).
  - w is streamed in vocab tiles of VT columns; each tile is [128, H / 128, VT] in SBUF (k-tile
    index on the free axis), loaded with large DMAs (KG k-tiles each; 2 * VT contiguous bytes per
    partition row) that the compiler can issue ahead of the matmuls.
  - per vocab tile: H / 128 accumulating matmuls per 512-column PSUM bank into a FRESH PSUM tile
    [B, VT] fp32 (accumulate=False on k = 0), evicted once by ScalarE to a bf16 chunk buffer (this
    rounding is what the stock path's bf16 logits do).
  - every CT tiles (one chunk of up to 16384 columns, the max8 / nc_find_index8 limit), VectorE
    runs one max8 + one nc_find_index8 over the chunk (first occurrence inside the chunk) and records
    (chunk max, global index) in a per-chunk SBUF table. The table is reduced once at the end:
    gmax = max over chunks; idx = min over chunks whose max == gmax of their index. So the result is
    the FIRST index of the maximum over the whole vocab (torch.argmax semantics on bf16 logits),
    with no serial running-state chain across tiles.
No PSUM tile is shared across loop iterations, no ring/modulo buffers, no DMA skipping, no indirect
DMA. Indices are carried in fp32 (exact below 2^24).
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
BANK = 512            # fp32 columns per PSUM bank (moving free size per matmul)
FIND_MAX = 16384      # max elements per partition for max8 / nc_find_index8
BIG = 16777216.0      # 2^24: larger than any vocab index, exact in fp32


def _program(x, w, val_out, idx_out, B, H, V, Vp, col0_sym, prg, vt, ct, kg, round_bf16, dge):
    """Process vocab columns [col0_sym, col0_sym + Vp) of w. col0_sym may be symbolic (LNC grid)."""
    KT = H // P
    dt = x.dtype
    ldt = dt if round_bf16 else nl.float32

    # ---- x^T: [128 (h within k-tile), KT, B] ------------------------------------------------
    x_sb = nl.ndarray((B, H), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x[0:B, 0:H])
    xT = nl.ndarray((P, KT, B), dtype=dt, buffer=nl.sbuf)
    # PSUM slot per transpose padded to a multiple of 2 bf16 (4 bytes): an odd-width bf16 matmul output slot fails
    # the BIR matmul-output verifier (seen at B = 1).
    BS = B + (B % 2)
    per_bank = 2048 // (2 * BS)           # bf16 [128, BS] transposes that fit one PSUM bank
    if per_bank > KT:
        per_bank = KT
    k0 = 0
    while k0 < KT:
        nk = per_bank
        if k0 + nk > KT:
            nk = KT - k0
        tp = nl.ndarray((P, nk, BS), dtype=dt, buffer=nl.psum)
        for kk in range(nk):
            nisa.nc_transpose(dst=tp[:, kk, 0:B], data=x_sb[:, (k0 + kk) * P:(k0 + kk + 1) * P],
                              engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=xT[:, k0:k0 + nk, :], src=tp[:, :, 0:B], engine=nisa.engine.vector)
        k0 = k0 + nk

    n_tiles = (Vp + vt - 1) // vt
    n_chunks = (n_tiles + ct - 1) // ct
    cw_full = ct * vt
    cmax = nl.ndarray((B, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
    cidx = nl.ndarray((B, n_chunks), dtype=nl.float32, buffer=nl.sbuf)

    for c in range(n_chunks):
        c_lo = c * cw_full
        c_w = cw_full
        if c_lo + c_w > Vp:
            c_w = Vp - c_lo
        lg = nl.ndarray((B, cw_full), dtype=ldt, buffer=nl.sbuf)
        t_in_c = (c_w + vt - 1) // vt
        for tt in range(t_in_c):
            v_lo = c_lo + tt * vt          # program-local column of this tile
            wt = vt
            if v_lo + wt > Vp:
                wt = Vp - v_lo
            # ---- weight tile [128, KT, wt]: KG k-tiles per DMA ----
            w_sb = nl.ndarray((P, KT, vt), dtype=dt, buffer=nl.sbuf)
            g0 = 0
            while g0 < KT:
                ng = kg
                if g0 + ng > KT:
                    ng = KT - g0
                w_src = w.ap(pattern=[[V, P], [P * V, ng], [1, wt]], offset=g0 * P * V + col0_sym + v_lo)
                if dge == 1:
                    nisa.dma_copy(dst=w_sb[:, g0:g0 + ng, 0:wt], src=w_src, dge_mode=nisa.dge_mode.hwdge)
                elif dge == 3:
                    # two HWDGE rings (SP and ACT sequencers), alternating per DMA
                    eng = nisa.engine.sync
                    if ((v_lo // vt) * KT + g0 // kg) % 2 == 1:
                        eng = nisa.engine.scalar
                    nisa.dma_copy(dst=w_sb[:, g0:g0 + ng, 0:wt], src=w_src, dge_mode=nisa.dge_mode.hwdge,
                                  engine=eng)
                elif dge == 2:
                    nisa.dma_copy(dst=w_sb[:, g0:g0 + ng, 0:wt], src=w_src, dge_mode=nisa.dge_mode.none)
                else:
                    nisa.dma_copy(dst=w_sb[:, g0:g0 + ng, 0:wt], src=w_src)
                g0 = g0 + ng
            # ---- logits tile [B, wt] fp32: one fresh PSUM tensor per 512-column bank ----
            # k outer / bank inner, so each stationary x^T k-tile is loaded once per vocab tile.
            # Every bank is its own PSUM tensor with its own accumulation group (k = 0 overwrites).
            n_sub = (wt + BANK - 1) // BANK
            banks = []
            for j in range(n_sub):
                banks.append(nl.ndarray((B, BANK), dtype=nl.float32, buffer=nl.psum))
            for k in range(KT):
                for j in range(n_sub):
                    s_lo = j * BANK
                    s_w = BANK
                    if s_lo + s_w > wt:
                        s_w = wt - s_lo
                    nisa.nc_matmul(dst=banks[j][:, 0:s_w], stationary=xT[:, k, :],
                                   moving=w_sb[:, k, s_lo:s_lo + s_w], accumulate=(k != 0))
            for j in range(n_sub):
                s_lo = j * BANK
                s_w = BANK
                if s_lo + s_w > wt:
                    s_w = wt - s_lo
                o_lo = tt * vt + s_lo
                nisa.activation(dst=lg[:, o_lo:o_lo + s_w], op=nl.copy, data=banks[j][:, 0:s_w])
        # ---- chunk max + first index ----
        m8 = nl.ndarray((B, 8), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=m8, src=lg[:, 0:c_w])
        i8 = nl.ndarray((B, 8), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=i8, data=lg[:, 0:c_w], vals=m8)
        nisa.tensor_copy(dst=cmax[:, c:c + 1], src=m8[:, 0:1], engine=nisa.engine.gpsimd)
        nisa.tensor_scalar(dst=cidx[:, c:c + 1], data=i8[:, 0:1], op0=nl.add, operand0=float(c_lo),
                           engine=nisa.engine.gpsimd)

    # ---- combine chunks: first index of the global max ----
    gmax = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=gmax, op=nl.maximum, data=cmax, axis=1)
    pen = nl.ndarray((B, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pen, data=cmax, op0=nl.not_equal, operand0=gmax, op1=nl.multiply,
                       operand1=BIG)
    cand = nl.ndarray((B, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=cand, data1=pen, data2=cidx, op=nl.add)
    fidx = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=fidx, op=nl.minimum, data=cand, axis=1)
    iidx = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=iidx, src=fidx, engine=nisa.engine.vector)
    nisa.dma_copy(dst=val_out.ap(pattern=[[1, B], [1, 1]], offset=prg * B), src=gmax)
    nisa.dma_copy(dst=idx_out.ap(pattern=[[1, B], [1, 1]], offset=prg * B), src=iidx)


@nki.jit
def lmhead_argmax_kernel(x, w, vt=2048, ct=8, kg=8, round_bf16=True, dge=3):
    B, H = x.shape
    H2, V = w.shape
    kernel_assert(H == H2, "x and w contraction dims differ")
    kernel_assert(H % P == 0, "hidden size must be a multiple of 128")
    kernel_assert(B <= P, "at most 128 rows")
    kernel_assert(vt % BANK == 0, "vocab tile must be a multiple of 512")
    kernel_assert(ct * vt <= FIND_MAX, "chunk must be <= 16384 columns")

    n_prg = 1
    if nl.program_ndim() > 0:
        n_prg = nl.num_programs(0)
    kernel_assert(V % n_prg == 0, "vocab must split evenly across the LNC grid")
    Vp = V // n_prg
    kernel_assert(Vp >= 8, "at least 8 vocab columns per program")

    val_out = nl.ndarray((n_prg, B, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    idx_out = nl.ndarray((n_prg, B, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if n_prg == 1:
        _program(x, w, val_out, idx_out, B, H, V, Vp, 0, 0, vt, ct, kg, round_bf16, dge)
    else:
        pid = nl.program_id(0)
        _program(x, w, val_out, idx_out, B, H, V, Vp, pid * Vp, pid, vt, ct, kg, round_bf16, dge)
    return val_out, idx_out
