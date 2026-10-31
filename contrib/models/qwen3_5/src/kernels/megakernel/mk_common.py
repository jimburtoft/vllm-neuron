# SPDX-License-Identifier: Apache-2.0
"""Building blocks shared by the decode megakernel's GDN, attention, MoE and lm_head blocks.

Residual stream: rows layout hR [B, H] bf16 in SBUF, a full copy on both logical-core halves (pcores), rebuilt
with sendrecv after every residual add. Column layouts for x-stationary matmuls are built with PE transposes:
  std   xT [128, KT*B]: element (p, kt*B + b) = x[b, kt*128 + p]   (GDN / attention in_proj)
  pmaj  xP [128, KT*B]: element (p, kt*B + b) = x[b, p*KT + kt]    (MoE: weight rows p*16..p*16+15 are
        contiguous in HBM, so a [H, 256] expert slice is 8 KB contiguous per partition, not 16 x 512 B)
LNC=2: pcore c owns H half c ([c*1024, (c+1)*1024)) for the all-reduce and the residual add.
All collectives are issued unconditionally by both pcores (no program_id guards). ranks=None is the
single-rank path: the all-reduce is an identity.

RMSNorm placement: the per-row rsqrt(mean(h^2) + eps) is computed on the rows, and the scale x * rs * w is
applied in the PE-transposed column layout (same op order and rounding as the unfused norm), so the transposes
do not wait on the reduce -> rsqrt chain.
"""
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

H = 2048
KT = H // 128
F32, BF = nl.float32, nl.bfloat16
EPS = 1e-6


def sb(shape, dt=F32, name=None):
    if name is None:
        return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf, name=name)


def rmsnorm_rows(hR, w_hbm, B):
    """RMSNorm on rows: fp32 x * rsqrt(mean(x^2) + eps) * w, rounded to bf16 once.
    hR [B, H] bf16 SBUF, w_hbm [H] fp32 (the zero-centred weight's + 1 already applied). Returns xn [B, H] bf16."""
    w = sb((B, H))
    nisa.dma_copy(dst=w, src=w_hbm.ap(pattern=[[0, B], [1, H]], offset=0))
    rs = row_rstd(hR, B)
    xn = sb((B, H), BF)
    nisa.scalar_tensor_tensor(dst=xn, data=hR, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=w)
    return xn


def row_rstd(hR, B):
    """rs [B, 1] fp32 = rsqrt(mean(h^2) + eps) of the bf16 rows hR [B, H] (ScalarE square + fused reduce)."""
    sq = sb((B, H))
    ss = sb((B, 1))
    nisa.activation(dst=sq, op=nl.square, data=hR, reduce_op=nl.add, reduce_res=ss,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    rs = sb((B, 1))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, scale=1.0 / H, bias=EPS)
    return rs


def w_cols(w_hbm, B, pmaj=False):
    """Norm weight w [H] fp32 in the column layout of rows_to_std (element (p, kt*B+b) = w[kt*128+p]) or
    rows_to_pmaj (w[p*KT+kt]), [128, KT*B] fp32 (DMA [128, KT] + free-dim broadcast copy)."""
    wr = sb((128, KT))
    if pmaj:
        nisa.dma_copy(dst=wr, src=w_hbm.ap(pattern=[[KT, 128], [1, KT]], offset=0))
    else:
        nisa.dma_copy(dst=wr, src=w_hbm.ap(pattern=[[1, 128], [128, KT]], offset=0))
    wc = sb((128, KT * B))
    nisa.tensor_copy(dst=wc, src=wr.ap(pattern=[[KT, 128], [1, KT], [0, B]], offset=0))
    return wc


def rs_cols(rs, ident, B):
    """Per-row rsqrt rs [B, 1] fp32 -> [128, KT*B] fp32 with element (p, kt*B + b) = rs[b]
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


def reduce_residual(y_half, hR, B, NC, pid, ranks, tag):
    """y_half [B, H/NC] bf16 = this pcore's TP partial of its H half. All-reduce over TP, add the residual
    after the reduce, rebuild the full residual rows on both pcores."""
    hw = H // NC
    red = all_reduce_sb(y_half, ranks, f"ar_{tag}")
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
