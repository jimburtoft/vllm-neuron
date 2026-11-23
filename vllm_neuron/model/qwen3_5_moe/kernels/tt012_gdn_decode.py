"""Qwen3.5-35B-A3B GDN decode mixer for one TP=4 rank, S=1 (serving variant).

in_proj (x-stationary, weights streamed) -> PE transposes to channel layout -> conv1d(4)+silu ->
l2norm q,k -> gates -> gated delta rule vectorized over j=(v_head, b) -> gated RMSNorm -> out_proj.
Differences from the standalone kernel: one layer, no residual (the caller all-reduces the
out_proj partial and adds the residual), recurrent state read/written in the cache layout
[B, Hv, Dk, Dv] fp32, conv state carries K-1 = 3 columns.

LNC=2: pcore c owns v-heads [4c, 4c+4) (k-heads [2c, 2c+2)); out_proj is N-split (columns
[c*1024, (c+1)*1024)), o exchanged via HBM + core_barrier.
Inputs (per rank):
  h_col  [128, KT*B] bf16, element (p, kt*B + b) = x[b, kt*128 + p]
  Win    [NC*H*NIN_C] bf16, per pcore [H, NIN_C] columns [q | k | v | z | b | a] of its heads
  Wout   [VAL, H] bf16 (the model's out_proj_weight, read in place)
  cw     [NC*128*4*CT] fp32, per pcore [p, w, t] (channel t*128 + p of the pcore's [q|k|v] order)
  cst_in [NC*128*3*CT*B] fp32, per pcore [p, w, t, b]
  rs_in  [B, Hv, Dk, Dv] fp32
  alog, dtb [Hv] fp32; nw [Dv] fp32; ident [128,128] bf16; mk [J, J*128] bf16 (block-diag ones)
Outputs: y [B, H] bf16 (pre-all-reduce partial), cst_out (as cst_in), rs_out (as rs_in).
"""
import math

import nki
import nki.isa as nisa
import nki.language as nl

H, VAL, D = 2048, 1024, 128
HK, HV = 4, 8  # per TP rank
KCH = 4
KT, OKT = H // 128, VAL // 128
F32, BF = nl.float32, nl.bfloat16


def dims(nc, b):
    hk_c, hv_c = HK // nc, HV // nc
    ct = 2 * hk_c + hv_c
    nin_c = 128 * (ct + hv_c) + 2 * hv_c
    return hk_c, hv_c, ct, nin_c, hv_c * b, H // nc


def _ntiles(n, t=512):
    return [(s, min(t, n - s)) for s in range(0, n, t)]


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _colsum(dst_sb, src_bf, n, ones):
    ps = nl.ndarray((128, n), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ps, ones, src_bf)
    nisa.tensor_copy(dst=dst_sb, src=ps)


@nki.jit
def tt012_gdn_decode(h_col, Win, Wout, cw, cst_in, rs_in, alog, dtb, nw, ident_in, mk_in):
    NC = nl.num_programs(0)
    pid = nl.program_id(0) if NC > 1 else 0
    B = h_col.shape[1] // KT
    HK_C, HV_C, CT, NIN_C, J, NOUT_C = dims(NC, B)
    WIN_SZ = H * NIN_C
    CSZ = 128 * 3 * CT * B

    y_out = nl.ndarray((B, H), dtype=BF, buffer=nl.shared_hbm)
    cst_out = nl.ndarray(cst_in.shape, dtype=F32, buffer=nl.shared_hbm)
    rs_out = nl.ndarray(rs_in.shape, dtype=F32, buffer=nl.shared_hbm)
    oxch = nl.ndarray((128, OKT * B), dtype=BF, buffer=nl.shared_hbm)
    gsc = nl.ndarray((NC * B * 2 * HV_C,), dtype=BF, buffer=nl.shared_hbm)

    ident = _sb((128, 128), BF)
    nisa.dma_copy(dst=ident, src=ident_in)
    ones = _sb((128, 128), BF)
    nisa.memset(ones, 1.0)
    mk = _sb((J, J * 128), BF)
    nisa.dma_copy(dst=mk, src=mk_in)
    xT = _sb((128, KT * B), BF)
    nisa.dma_copy(dst=xT, src=h_col)

    # ---------------- in_proj ----------------
    nts = _ntiles(NIN_C)
    ps = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nts]
    for g in range(KT // KCH):
        w = _sb((128, KCH, NIN_C), BF)
        nisa.dma_copy(dst=w, src=Win.ap(pattern=[[NIN_C, 128], [128 * NIN_C, KCH], [1, NIN_C]],
                                         offset=pid * WIN_SZ + g * KCH * 128 * NIN_C))
        for kk in range(KCH):
            for ti, (n0, nn) in enumerate(nts):
                nisa.nc_matmul(ps[ti][0:B, 0:nn], xT[:, (g * KCH + kk) * B:(g * KCH + kk + 1) * B],
                               w[:, kk, n0:n0 + nn])
    a_rows = _sb((B, NIN_C), BF)
    for ti, (n0, nn) in enumerate(nts):
        nisa.tensor_copy(dst=a_rows[0:B, n0:n0 + nn], src=ps[ti][0:B, 0:nn])

    # ---------------- state / param loads ----------------
    hs = _sb((128, 4 * CT * B))
    nisa.dma_copy(dst=hs[:, 0:3 * CT * B], src=cst_in.ap(pattern=[[3 * CT * B, 128], [1, 3 * CT * B]], offset=pid * CSZ))
    wraw = _sb((128, 4 * CT))
    nisa.dma_copy(dst=wraw, src=cw.ap(pattern=[[4 * CT, 128], [1, 4 * CT]], offset=pid * 128 * 4 * CT))
    wexp = _sb((128, 4 * CT * B))
    nisa.tensor_copy(dst=wexp, src=wraw.ap(pattern=[[4 * CT, 128], [1, 4 * CT], [0, B]], offset=0))
    # recurrent state: cache layout [B, HV, K, V] -> S[k, (hv_local*B + b)*128 + v]
    S = _sb((128, J * 128))
    for i in range(HV_C):
        nisa.dma_copy(dst=S.ap(pattern=[[J * 128, 128], [128, B], [1, 128]], offset=i * B * 128),
                      src=rs_in.ap(pattern=[[D, 128], [HV * D * D, B], [1, D]], offset=(pid * HV_C + i) * D * D))
    al = _sb((128, J))
    dt = _sb((128, J))
    adr = _sb((128, 2 * HV_C))
    nisa.dma_copy(dst=adr[:, 0:HV_C], src=alog.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.dma_copy(dst=adr[:, HV_C:2 * HV_C], src=dtb.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.tensor_copy(dst=al, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=0))
    nisa.tensor_copy(dst=dt, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=HV_C))
    nwc = _sb((128, 1))
    nisa.dma_copy(dst=nwc, src=nw.ap(pattern=[[1, 128], [1, 1]], offset=0))
    # gates: raw b,a rows -> HBM scratch -> partition-broadcast [128, (c, b)]
    nisa.dma_copy(dst=gsc.ap(pattern=[[2 * HV_C, B], [1, 2 * HV_C]], offset=pid * B * 2 * HV_C),
                  src=a_rows[0:B, NIN_C - 2 * HV_C:NIN_C])
    gbb = _sb((128, 2 * J), BF)
    nisa.dma_copy(dst=gbb, src=gsc.ap(pattern=[[0, 128], [1, 2 * J]], offset=pid * B * 2 * HV_C))
    gb = _sb((128, 2 * J))
    nisa.tensor_copy(dst=gb, src=gbb.ap(pattern=[[2 * J, 128], [1, 2 * HV_C], [2 * HV_C, B]], offset=0))
    beta = _sb((128, J))
    nisa.activation(dst=beta, data=gb[:, 0:J], op=nl.sigmoid)
    t1 = _sb((128, J))
    nisa.tensor_tensor(dst=t1, data1=gb[:, J:2 * J], data2=dt, op=nl.add)
    nisa.activation(dst=t1, data=t1, op=nl.softplus)
    ea = _sb((128, J))
    nisa.activation(dst=ea, data=al, op=nl.exp)
    gg = _sb((128, J))
    nisa.scalar_tensor_tensor(dst=gg, data=t1, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=ea)
    eg = _sb((128, J))
    nisa.activation(dst=eg, data=gg, op=nl.exp)

    # ---------------- channel layout + conv1d + silu ----------------
    xz = _sb((128, (CT + HV_C) * B))
    for t in range(CT + HV_C):
        tp = nl.ndarray((128, B), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(tp, a_rows[0:B, t * 128:(t + 1) * 128], ident[0:B, 0:B])
        nisa.tensor_copy(dst=xz[:, t * B:(t + 1) * B], src=tp)
    nisa.tensor_copy(dst=hs[:, 3 * CT * B:4 * CT * B], src=xz[:, 0:CT * B])
    nisa.dma_copy(dst=cst_out.ap(pattern=[[3 * CT * B, 128], [1, 3 * CT * B]], offset=pid * CSZ),
                  src=hs[:, CT * B:4 * CT * B])
    prod = _sb((128, 4 * CT * B))
    nisa.tensor_tensor(dst=prod, data1=hs, data2=wexp, op=nl.multiply)
    n = CT * B
    acc = _sb((128, n))
    nisa.tensor_tensor(dst=acc, data1=prod[:, 0:n], data2=prod[:, n:2 * n], op=nl.add)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=prod[:, 2 * n:3 * n], op=nl.add)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=prod[:, 3 * n:4 * n], op=nl.add)
    xc = _sb((128, n))
    nisa.activation(dst=xc, data=acc, op=nl.silu)

    # ---------------- l2norm q, k ----------------
    nqk = 2 * HK_C * B
    sq = _sb((128, nqk), BF)
    nisa.activation(dst=sq, data=xc[:, 0:nqk], op=nl.square)
    ss = _sb((128, nqk))
    _colsum(ss, sq, nqk, ones)
    rs = _sb((128, nqk))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, bias=1e-6)
    qk = _sb((128, nqk))
    nisa.tensor_tensor(dst=qk, data1=xc[:, 0:nqk], data2=rs, op=nl.multiply)
    nisa.tensor_scalar(dst=qk[:, 0:HK_C * B], data=qk[:, 0:HK_C * B], op0=nl.multiply, operand0=1.0 / math.sqrt(D))
    Qe = _sb((128, J))
    Ke = _sb((128, J))
    for hv in range(HV_C):
        hk = hv // (HV_C // HK_C)
        nisa.tensor_copy(dst=Qe[:, hv * B:(hv + 1) * B], src=qk[:, hk * B:(hk + 1) * B])
        nisa.tensor_copy(dst=Ke[:, hv * B:(hv + 1) * B], src=qk[:, (HK_C + hk) * B:(HK_C + hk + 1) * B])
    QK2 = _sb((128, J, 2), BF)
    nisa.tensor_copy(dst=QK2[:, :, 0], src=Ke)
    nisa.tensor_copy(dst=QK2[:, :, 1], src=Qe)
    Ke_bf = _sb((128, J), BF)
    nisa.tensor_copy(dst=Ke_bf, src=Ke)
    pqk = _sb((128, J), BF)
    nisa.tensor_tensor(dst=pqk, data1=Qe, data2=Ke, op=nl.multiply)
    qkd = _sb((128, J))
    _colsum(qkd, pqk, J, ones)

    # ---------------- recurrence ----------------
    S_bf = _sb((128, J * 128), BF)
    nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
    psA = nl.ndarray((128, J, 2), dtype=F32, buffer=nl.psum)
    for j in range(J):
        nisa.nc_matmul(psA[:, j, 0:2], S_bf[:, j * 128:(j + 1) * 128], QK2[:, j, 0:2])
    kq = _sb((128, J, 2))
    nisa.tensor_copy(dst=kq, src=psA)
    kv = _sb((128, J))
    nisa.tensor_tensor(dst=kv, data1=kq[:, :, 0], data2=eg, op=nl.multiply)
    vcol = xc[:, 2 * HK_C * B:CT * B]
    delta = _sb((128, J))
    nisa.tensor_tensor(dst=delta, data1=vcol, data2=kv, op=nl.subtract)
    nisa.tensor_tensor(dst=delta, data1=delta, data2=beta, op=nl.multiply)
    o = _sb((128, J))
    nisa.tensor_tensor(dst=o, data1=kq[:, :, 1], data2=eg, op=nl.multiply)
    t2 = _sb((128, J))
    nisa.tensor_tensor(dst=t2, data1=qkd, data2=delta, op=nl.multiply)
    nisa.tensor_tensor(dst=o, data1=o, data2=t2, op=nl.add)

    # ---------------- state update S = S*eg + k (x) delta ----------------
    d_bf = _sb((128, J), BF)
    nisa.tensor_copy(dst=d_bf, src=delta)
    pT = nl.ndarray((J, 128), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(pT, d_bf, ident)
    Drows = _sb((J, 128), BF)
    nisa.tensor_copy(dst=Drows, src=pT)
    pK = nl.ndarray((J, 128), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(pK, Ke_bf, ident)
    Krows = _sb((J, 128), BF)
    nisa.tensor_copy(dst=Krows, src=pK)
    Dblk = _sb((J, J * 128), BF)
    nisa.tensor_tensor(dst=Dblk, data1=Drows.ap(pattern=[[128, J], [0, J], [1, 128]], offset=0), data2=mk, op=nl.multiply)
    for c in range(J // 4):
        po = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(po, Krows, Dblk[0:J, c * 512:(c + 1) * 512])
        Sc = S[:, c * 512:(c + 1) * 512]
        nisa.tensor_tensor(dst=Sc, data1=Sc, data2=eg.ap(pattern=[[J, 128], [1, 4], [0, 128]], offset=c * 4), op=nl.multiply)
        nisa.tensor_tensor(dst=Sc, data1=Sc, data2=po, op=nl.add)
    for i in range(HV_C):
        nisa.dma_copy(dst=rs_out.ap(pattern=[[D, 128], [HV * D * D, B], [1, D]], offset=(pid * HV_C + i) * D * D),
                      src=S.ap(pattern=[[J * 128, 128], [128, B], [1, 128]], offset=i * B * 128))

    # ---------------- gated RMSNorm ----------------
    o2 = _sb((128, J), BF)
    nisa.activation(dst=o2, data=o, op=nl.square)
    ms = _sb((128, J))
    _colsum(ms, o2, J, ones)
    rr = _sb((128, J))
    nisa.activation(dst=rr, data=ms, op=nl.rsqrt, scale=1.0 / D, bias=1e-6)
    sz = _sb((128, J))
    nisa.activation(dst=sz, data=xz[:, CT * B:(CT + HV_C) * B], op=nl.silu)
    y = _sb((128, J))
    nisa.scalar_tensor_tensor(dst=y, data=o, op0=nl.multiply, operand0=nwc, op1=nl.multiply, operand1=rr)
    oT = _sb((128, OKT * B), BF)
    if NC == 1:
        nisa.tensor_tensor(dst=oT, data1=y, data2=sz, op=nl.multiply)
    else:
        oTl = _sb((128, J), BF)
        nisa.tensor_tensor(dst=oTl, data1=y, data2=sz, op=nl.multiply)
        nisa.dma_copy(dst=oxch.ap(pattern=[[OKT * B, 128], [1, J]], offset=pid * J), src=oTl)
        nisa.core_barrier(data=oxch, cores=(0, 1))
        nisa.dma_copy(dst=oT, src=oxch)

    # ---------------- out_proj (N-split, Wout read in place) ----------------
    nto = _ntiles(NOUT_C)
    pso = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nto]
    for g in range(OKT // KCH):
        w = _sb((128, KCH, NOUT_C), BF)
        nisa.dma_copy(dst=w, src=Wout.ap(pattern=[[H, 128], [128 * H, KCH], [1, NOUT_C]],
                                          offset=g * KCH * 128 * H + pid * NOUT_C))
        for kk in range(KCH):
            kt = g * KCH + kk
            for ti, (n0, nn) in enumerate(nto):
                nisa.nc_matmul(pso[ti][0:B, 0:nn], oT[:, kt * B:(kt + 1) * B], w[:, kk, n0:n0 + nn])
    y_rows = _sb((B, NOUT_C), BF)
    for ti, (n0, nn) in enumerate(nto):
        nisa.tensor_copy(dst=y_rows[0:B, n0:n0 + nn], src=pso[ti][0:B, 0:nn])
    nisa.dma_copy(dst=y_out.ap(pattern=[[H, B], [1, NOUT_C]], offset=pid * NOUT_C), src=y_rows)
    return y_out, cst_out, rs_out
