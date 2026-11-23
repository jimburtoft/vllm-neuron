"""megakernel building blocks shared by the MoE / GDN blocks and the M2 layer-pair kernel.

Residual stream: rows layout hR [B, H] bf16 in SBUF, full copy on both pcores (rebuilt with sendrecv after
every residual add). Column layouts for x-stationary matmuls are built with PE transposes:
  std   xT [128, KT*B]: element (p, kt*B + b) = x[b, kt*128 + p]   (GDN / GQA in_proj, tt012 layout)
  pmaj  xP [128, KT*B]: element (p, kt*B + b) = x[b, p*KT + kt]    (MoE: weight rows p*16..p*16+15 are
        contiguous in HBM, so a [H, 256] expert slice is 8 KB contiguous per partition, not 16 x 512 B)
LNC=2: pcore c owns H half c ([c*1024, (c+1)*1024)) for the all-reduce and the residual add.
All collectives are issued unconditionally by both pcores (no program_id guards). ranks=None is the
single-rank / simulator path: the all-reduce is an identity (nki.md "Megakernel correctness rules").
"""
import os

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

H = 2048
AR_MODE = os.environ.get("MK_AR", "sb")  # sb | hbm (A/B)
KT = H // 128
F32, BF = nl.float32, nl.bfloat16
EPS = 1e-6


def sb(shape, dt=F32, name=None):
    if name is None:
        return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf, name=name)


def rmsnorm_rows(hR, w_hbm, B):
    """Port RMSNorm (model.py:101-105): fp32 x * rsqrt(mean(x^2) + eps) * w, rounded to bf16.
    hR [B, H] bf16 SBUF, w_hbm [H] fp32 (the +1 is folded at load). Returns xn [B, H] bf16.
    MK_NORM=fast: same math in 3 instructions (ScalarE square + fused row reduce, rsqrt, one DVE
    (h * rs) * w with a single bf16 rounding) instead of 6."""
    w = sb((B, H))
    nisa.dma_copy(dst=w, src=w_hbm.ap(pattern=[[0, B], [1, H]], offset=0))
    if os.environ.get("MK_NORM", "") == "fast":
        rs = row_rstd(hR, B)
        xn = sb((B, H), BF)
        nisa.scalar_tensor_tensor(dst=xn, data=hR, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=w)
        return xn
    hf = sb((B, H))
    nisa.tensor_copy(dst=hf, src=hR)
    sq = sb((B, H))
    nisa.tensor_tensor(dst=sq, data1=hf, data2=hf, op=nl.multiply)
    ss = sb((B, 1))
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1)
    rs = sb((B, 1))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, scale=1.0 / H, bias=EPS)
    xf = sb((B, H))
    nisa.tensor_scalar(dst=xf, data=hf, op0=nl.multiply, operand0=rs)
    xn = sb((B, H), BF)
    nisa.tensor_tensor(dst=xn, data1=xf, data2=w, op=nl.multiply)
    return xn


def mk_fold():
    """Trace time. MK_FOLD=1: the RMSNorm weights are folded into the next projections at load time
    (in_proj / qkv / router / shared gate_up / shared gate); the kernel feeds the un-normed residual to those
    matmuls and scales their outputs by the per-row rsqrt (row_rstd). MK_FOLD=2: weights unchanged; the matmuls
    take xw = bf16(h * w) (one vector op) and the outputs are scaled by the rsqrt (the reduce -> rsqrt chain is
    off the critical path, bf16 rounding as the unfused norm up to the row scale). 0: unfused RMSNorm."""
    return int(os.environ.get("MK_FOLD", "0"))


def mk_fold_t4():
    """MK_FOLD for the T=4 verify kernel (MK_FOLD_T4, default = MK_FOLD)."""
    return int(os.environ.get("MK_FOLD_T4", os.environ.get("MK_FOLD", "0")))


def mul_w_rows(hR, w_hbm, B):
    """xw [B, H] bf16 = hR * w (w [H] fp32 in HBM, broadcast over rows)."""
    w = sb((B, H))
    nisa.dma_copy(dst=w, src=w_hbm.ap(pattern=[[0, B], [1, H]], offset=0))
    xw = sb((B, H), BF)
    nisa.tensor_tensor(dst=xw, data1=hR, data2=w, op=nl.multiply)
    return xw


def row_rstd(hR, B):
    """rs [B, 1] fp32 = rsqrt(mean(h^2) + eps) of the bf16 rows hR [B, H] (ScalarE square + fused reduce)."""
    sq = sb((B, H))
    ss = sb((B, 1))
    nisa.activation(dst=sq, op=nl.square, data=hR, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rs = sb((B, 1))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, scale=1.0 / H, bias=EPS)
    return rs


def evac(dst, src, rsc):
    """PSUM -> SBUF copy; with rsc [B, 1] (fold mode) multiplied by the per-row rsqrt."""
    if rsc is None:
        nisa.tensor_copy(dst=dst, src=src)
    else:
        nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=rsc)


def w_cols(w_hbm, B, pmaj=False):
    """MK_FOLD=3: norm weight w [H] fp32 in the column layout of rows_to_std (element (p, kt*B+b) =
    w[kt*128+p]) or rows_to_pmaj (w[p*KT+kt]), [128, KT*B] fp32 (DMA [128, KT] + free-dim broadcast copy)."""
    wr = sb((128, KT))
    if pmaj:
        nisa.dma_copy(dst=wr, src=w_hbm.ap(pattern=[[KT, 128], [1, KT]], offset=0))
    else:
        nisa.dma_copy(dst=wr, src=w_hbm.ap(pattern=[[1, 128], [128, KT]], offset=0))
    wc = sb((128, KT * B))
    nisa.tensor_copy(dst=wc, src=wr.ap(pattern=[[KT, 128], [1, KT], [0, B]], offset=0))
    return wc


def rs_cols(rs, ident, B):
    """MK_FOLD=4: per-row rsqrt rs [B, 1] fp32 -> [128, KT*B] fp32 with element (p, kt*B + b) = rs[b]
    (diag(rs) [B, B] times a ones stationary on the PE, then a free-dim broadcast over kt)."""
    idf = sb((B, B))
    nisa.tensor_copy(dst=idf, src=ident[0:B, 0:B])
    dg = sb((B, B))
    nisa.tensor_scalar(dst=dg, data=idf, op0=nl.multiply, operand0=rs)
    ones = sb((B, 128))
    nisa.memset(ones, 1.0)
    pb = nl.ndarray((128, B), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(pb, ones, dg)
    rb = sb((128, B))
    nisa.tensor_copy(dst=rb, src=pb)
    rk = sb((128, KT * B))
    nisa.tensor_copy(dst=rk, src=rb.ap(pattern=[[B, 128], [0, KT], [1, B]], offset=0))
    return rk


def _scale_cols(col, tp, wcol, rk, n):
    """col (bf16) = (tp * rk) * wcol, fp32 math, one rounding (= unfused RMSNorm op order)."""
    if rk is None:
        nisa.tensor_tensor(dst=col, data1=tp, data2=wcol, op=nl.multiply)
        return
    t = sb((128, n))
    nisa.tensor_tensor(dst=t, data1=tp, data2=rk, op=nl.multiply)
    nisa.tensor_tensor(dst=col, data1=t, data2=wcol, op=nl.multiply)


def rows_to_std(rows, ntile, ident, B, col0=0, wcol=None, rk=None):
    """rows [B, >= col0 + ntile*128] bf16 -> [128, ntile*B] bf16, element (p, t*B+b) = rows[b, col0 + t*128 + p]."""
    tp = nl.ndarray((128, ntile * B), dtype=F32, buffer=nl.psum)
    for t in range(ntile):
        nisa.nc_matmul(tp[:, t * B:(t + 1) * B], rows[0:B, col0 + t * 128:col0 + (t + 1) * 128], ident[0:B, 0:B])
    col = sb((128, ntile * B), BF)
    if wcol is None:
        nisa.tensor_copy(dst=col, src=tp)
    else:  # bf16(x [* rs] * w) in the column layout (the PE transpose of bf16 rows is exact)
        _scale_cols(col, tp, wcol, rk, ntile * B)
    return col


def rows_to_pmaj(rows, ident, B, dt=BF, wcol=None, rk=None):
    """rows [B, H] bf16 -> [128, KT*B], element (p, kt*B+b) = rows[b, p*KT + kt]."""
    tp = nl.ndarray((128, KT * B), dtype=F32, buffer=nl.psum)
    for kt in range(KT):
        nisa.nc_matmul(tp[:, kt * B:(kt + 1) * B], rows.ap(pattern=[[H, B], [KT, 128]], offset=kt), ident[0:B, 0:B])
    col = sb((128, KT * B), dt)
    if wcol is None:
        nisa.tensor_copy(dst=col, src=tp)
    else:
        _scale_cols(col, tp, wcol, rk, KT * B)
    return col


def all_reduce_sb(t, ranks, name):
    """In-kernel TP all-reduce of a 2-D SBUF tile (SBUF->SBUF). ranks=None -> identity."""
    if ranks is None:
        return t
    red = nl.ndarray(t.shape, dtype=t.dtype, buffer=nl.sbuf, name=name)
    ncc.all_reduce(dsts=[red], srcs=[t], op=nl.add, replica_group=ncc.ReplicaGroup([list(ranks)]))
    return red


def reduce_residual(y_half, hR, B, NC, pid, ranks, tag, after_ar=None):
    """y_half [B, H/NC] bf16 = this pcore's TP partial of its H half. All-reduce over TP, add the residual
    AFTER the reduce (never before: nki.md:1377), rebuild the full residual rows on both pcores."""
    hw = H // NC
    if ranks is not None and AR_MODE == "hbm":
        # M1 'hbm' shape (10.7 us vs 12.4 us SBUF): halves -> one shared [B, H], core_barrier, full-H AR issued
        # by both pcores, full sum read back by each pcore -> no sendrecv rebuild.
        src = nl.ndarray((B, H), dtype=BF, buffer=nl.shared_hbm, name=f"arin_{tag}")
        dst = nl.ndarray((B, H), dtype=BF, buffer=nl.shared_hbm, name=f"arout_{tag}")
        nisa.dma_copy(dst=src[:, nl.ds(pid * hw, hw)], src=y_half)
        if NC > 1:
            nisa.core_barrier(data=src, cores=(0, 1))
        ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=ncc.ReplicaGroup([list(ranks)]))
        full = sb((B, H), BF)
        nisa.dma_copy(dst=full, src=dst)
        hn = sb((B, H), BF)
        nisa.tensor_tensor(dst=hn, data1=full, data2=hR, op=nl.add)
        return hn
    red = all_reduce_sb(y_half, ranks, f"ar_{tag}")
    if after_ar is not None:  # M4 prefetch hook: called right after the AR is issued, gets its output tile
        red = after_ar(red)
    hn = sb((B, H), BF)
    nisa.tensor_tensor(dst=hn[:, nl.ds(pid * hw, hw)], data1=red, data2=hR[:, nl.ds(pid * hw, hw)], op=nl.add)
    if NC > 1:
        other = 1 - pid
        nisa.sendrecv(src=hn[:, nl.ds(pid * hw, hw)], dst=hn[:, nl.ds(other * hw, hw)], send_to_rank=other,
                      recv_from_rank=other, pipe_id=0)
    return hn


def fold_halves(y_full, B, NC, pid):
    """y_full [B, H] bf16 partial on each pcore -> this pcore's H half summed over both pcores."""
    hw = H // NC
    if NC == 1:
        return y_full
    other = 1 - pid
    recv = sb((B, hw), BF)
    nisa.sendrecv(src=y_full[:, nl.ds(other * hw, hw)], dst=recv, send_to_rank=other, recv_from_rank=other,
                  pipe_id=0)
    yh = sb((B, hw), BF)
    nisa.tensor_tensor(dst=yh, data1=y_full[:, nl.ds(pid * hw, hw)], data2=recv, op=nl.add)
    return yh


def store_rows(out, hR, NC, pid):
    """Each pcore writes its own half of the (identical) full residual rows: disjoint addresses."""
    hw = H // NC
    nisa.dma_copy(dst=out[:, nl.ds(pid * hw, hw)], src=hR[:, nl.ds(pid * hw, hw)])
