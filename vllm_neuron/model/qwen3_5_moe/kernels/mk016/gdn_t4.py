"""GDN mixer for the MTP verify step (T tokens per request, B requests, R = B*T rows b-major: r = b*T + t).

Same math as gdn_block.gdn_mixer (port layouts, sio path only). in_proj / gates / gated RMSNorm / out_proj run on all
R rows at once; conv + recurrence run token by token per request. After token t the conv history and the recurrent
state are written to checkpoint row widx[t*B + b] (= slot*n_ckpt + (p0+t) % n_ckpt, position-indexed ring);
the initial state is read from ridx[b] (= slot*n_ckpt + (p0-1) % n_ckpt).
"""
import math

import nki.isa as nisa
import nki.language as nl

from .gdn_block import (BF, D, F32, H, HK, HV, KCH, KT, MK_WOUT_EARLY, OKT, _colsum, _ix, _ntiles, _sb, dims,
                       gdn_win_load, gdn_wout_load)
from .mk_common import evac


def slot_io_t4(ridx, widx, B, T, NC, pid):
    """ridx [B] int32 read rows, widx [T*B] int32 write rows (token-major). Returns cr (1,B), cw[t] (1,B) uint32
    tiles of row * NC + pid (rec state viewed [N*NC, HV_C*D*D], conv state viewed [N*NC, 128*3*CT])."""
    def mk(src, off):
        t = _sb((1, B), nl.int32)
        nisa.dma_copy(dst=t, src=src.ap(pattern=[[B, 1], [1, B]], offset=off))
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=NC, op1=nl.add, operand1=pid)
        u = nl.ndarray((1, B), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=t)
        return u

    return dict(cr=mk(ridx, 0), cw=[mk(widx, t * B) for t in range(T)])


def gdn_mixer_t4(xT, Win, Wout, cw, cs_io, rs_io, alog, dtb, nw, ident, ones, mk, gsc, B, T, NC, pid, sio, rsc=None, w_pre=()):
    """xT [128, KT*R] bf16 std column layout (column kt*R + r). cs_io / rs_io: port state pools (updated in place).
    mk [J, J*128] block mask for J = HV_C*B. gsc HBM scratch [NC*R*2*HV_C]. Returns [R, H/NC] bf16."""
    R = B * T
    HK_C, HV_C, CT, NIN_C, J, NOUT_C = dims(NC, B)
    JR = HV_C * R
    NR = rs_io.shape[0] * rs_io.shape[1]
    rsN = rs_io.reshape((NR * NC, HV_C * D * D))
    RSK = len(rs_io.shape) == 3
    rs_ap = ([[HV_C * D, 128], [D, HV_C], [1, D]] if RSK else [[D, 128], [D * D, HV_C], [1, D]])
    cs2 = cs_io.reshape((NR * NC, 128 * 3 * CT))
    W7 = (3 + T) * CT * B   # conv window columns: w*CT*B + tile*B + b, w = 0..2 history, 3+t token t

    # ---------------- in_proj (R rows) ----------------
    nts = _ntiles(NIN_C)
    ps = [nl.ndarray((R, 512), dtype=F32, buffer=nl.psum) for _ in nts]
    for g in range(KT // KCH):
        w = w_pre[g] if g < len(w_pre) else gdn_win_load(Win, g, NC, pid, B)
        for kk in range(KCH):
            for ti, (n0, nn) in enumerate(nts):
                nisa.nc_matmul(ps[ti][0:R, 0:nn], xT[:, (g * KCH + kk) * R:(g * KCH + kk + 1) * R],
                               w[:, kk, n0:n0 + nn])
    a_rows = _sb((R, NIN_C), BF)
    for ti, (n0, nn) in enumerate(nts):
        evac(a_rows[0:R, n0:n0 + nn], ps[ti][0:R, 0:nn], rsc)
    wouts = [gdn_wout_load(Wout, g, NC, pid) for g in range(OKT // KCH)] if MK_WOUT_EARLY else None

    # ---------------- state / param loads ----------------
    hs = _sb((128, W7))
    cstg = _sb((128, B * 3 * CT))
    for b in range(B):
        nisa.dma_copy(dst=cstg[:, b * 3 * CT:(b + 1) * 3 * CT],
                      src=cs2.ap(pattern=[[3 * CT, 128], [1, 3 * CT]], offset=0,
                                 scalar_offset=_ix(sio["cr"], b, B), indirect_dim=0))
    nisa.tensor_copy(dst=hs.ap(pattern=[[W7, 128], [1, B], [B, 3 * CT]], offset=0),
                     src=cstg.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0))
    wraw = _sb((128, 4 * CT))
    nisa.dma_copy(dst=wraw, src=cw.ap(pattern=[[4 * CT, 128], [1, 4 * CT]], offset=pid * 128 * 4 * CT))
    wexp = _sb((128, 4 * CT * B))
    nisa.tensor_copy(dst=wexp, src=wraw.ap(pattern=[[4 * CT, 128], [1, 4 * CT], [0, B]], offset=0))
    S = _sb((128, J * 128))
    for b in range(B):
        nisa.dma_copy(dst=S.ap(pattern=[[J * 128, 128], [B * 128, HV_C], [1, 128]], offset=b * 128),
                      src=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cr"], b, B), indirect_dim=0))
    al = _sb((128, JR))
    dt = _sb((128, JR))
    adr = _sb((128, 2 * HV_C))
    nisa.dma_copy(dst=adr[:, 0:HV_C], src=alog.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.dma_copy(dst=adr[:, HV_C:2 * HV_C], src=dtb.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.tensor_copy(dst=al, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, R]], offset=0))
    nisa.tensor_copy(dst=dt, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, R]], offset=HV_C))
    nwc = _sb((128, 1))
    nisa.dma_copy(dst=nwc, src=nw.ap(pattern=[[1, 128], [1, 1]], offset=0))
    # gates for all R rows: column c*R + r
    nisa.dma_copy(dst=gsc.ap(pattern=[[2 * HV_C, R], [1, 2 * HV_C]], offset=pid * R * 2 * HV_C),
                  src=a_rows[0:R, NIN_C - 2 * HV_C:NIN_C])
    gbb = _sb((128, 2 * JR), BF)
    nisa.dma_copy(dst=gbb, src=gsc.ap(pattern=[[0, 128], [1, 2 * JR]], offset=pid * R * 2 * HV_C))
    gb = _sb((128, 2 * JR))
    nisa.tensor_copy(dst=gb, src=gbb.ap(pattern=[[2 * JR, 128], [1, 2 * HV_C], [2 * HV_C, R]], offset=0))
    beta = _sb((128, JR))
    nisa.activation(dst=beta, data=gb[:, 0:JR], op=nl.sigmoid)
    t1 = _sb((128, JR))
    nisa.tensor_tensor(dst=t1, data1=gb[:, JR:2 * JR], data2=dt, op=nl.add)
    nisa.activation(dst=t1, data=t1, op=nl.softplus)
    ea = _sb((128, JR))
    nisa.activation(dst=ea, data=al, op=nl.exp)
    gg = _sb((128, JR))
    nisa.scalar_tensor_tensor(dst=gg, data=t1, op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=ea)
    eg = _sb((128, JR))
    nisa.activation(dst=eg, data=gg, op=nl.exp)

    # ---------------- channel layout: xz column tile*R + r ----------------
    NT = CT + HV_C
    xz = _sb((128, NT * R))
    for t in range(NT):
        tp = nl.ndarray((128, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(tp, a_rows[0:R, t * 128:(t + 1) * 128], ident[0:R, 0:R])
        nisa.tensor_copy(dst=xz[:, t * R:(t + 1) * R], src=tp)
    # new conv inputs: hs column (3+tok)*CT*B + tile*B + b  <-  xz column tile*R + b*T + tok
    nisa.tensor_copy(dst=hs.ap(pattern=[[W7, 128], [CT * B, T], [B, CT], [1, B]], offset=3 * CT * B),
                     src=xz.ap(pattern=[[NT * R, 128], [1, T], [R, CT], [T, B]], offset=0))

    oall = _sb((128, JR))
    Sp = S
    n = CT * B
    nqk = 2 * HK_C * B
    for tok in range(T):
        # ---- conv1d + silu on window w = tok .. tok+3 ----
        prod = _sb((128, 4 * n))
        nisa.tensor_tensor(dst=prod, data1=hs[:, tok * n:(tok + 4) * n], data2=wexp, op=nl.multiply)
        acc = _sb((128, n))
        nisa.tensor_tensor(dst=acc, data1=prod[:, 0:n], data2=prod[:, n:2 * n], op=nl.add)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=prod[:, 2 * n:3 * n], op=nl.add)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=prod[:, 3 * n:4 * n], op=nl.add)
        xc = _sb((128, n))
        nisa.activation(dst=xc, data=acc, op=nl.silu)
        # ---- conv checkpoint after token tok: window w = tok+1 .. tok+3 ----
        cstw = _sb((128, B * 3 * CT))
        nisa.tensor_copy(dst=cstw.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0),
                         src=hs.ap(pattern=[[W7, 128], [1, B], [B, 3 * CT]], offset=(tok + 1) * n))
        for b in range(B):
            nisa.dma_copy(dst=cs2.ap(pattern=[[3 * CT, 128], [1, 3 * CT]], offset=0,
                                     scalar_offset=_ix(sio["cw"][tok], b, B), indirect_dim=0),
                          src=cstw[:, b * 3 * CT:(b + 1) * 3 * CT])
        # ---- per-token gates [128, J] column c*B + b ----
        bt_ = _sb((128, J))
        nisa.tensor_copy(dst=bt_, src=beta.ap(pattern=[[JR, 128], [R, HV_C], [T, B]], offset=tok))
        et_ = _sb((128, J))
        nisa.tensor_copy(dst=et_, src=eg.ap(pattern=[[JR, 128], [R, HV_C], [T, B]], offset=tok))
        # ---- l2norm q, k ----
        sq = _sb((128, nqk), BF)
        nisa.activation(dst=sq, data=xc[:, 0:nqk], op=nl.square)
        ss = _sb((128, nqk))
        _colsum(ss, sq, nqk, ones)
        rsq = _sb((128, nqk))
        nisa.activation(dst=rsq, data=ss, op=nl.rsqrt, bias=1e-6)
        qk = _sb((128, nqk))
        nisa.tensor_tensor(dst=qk, data1=xc[:, 0:nqk], data2=rsq, op=nl.multiply)
        nisa.tensor_scalar(dst=qk[:, 0:HK_C * B], data=qk[:, 0:HK_C * B], op0=nl.multiply,
                           operand0=1.0 / math.sqrt(D))
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
        # ---- recurrence ----
        S_bf = _sb((128, J * 128), BF)
        nisa.tensor_copy(dst=S_bf, src=Sp, engine=nisa.scalar_engine)
        psA = nl.ndarray((128, J, 2), dtype=F32, buffer=nl.psum)
        for j in range(J):
            nisa.nc_matmul(psA[:, j, 0:2], S_bf[:, j * 128:(j + 1) * 128], QK2[:, j, 0:2])
        kq = _sb((128, J, 2))
        nisa.tensor_copy(dst=kq, src=psA)
        kv = _sb((128, J))
        nisa.tensor_tensor(dst=kv, data1=kq[:, :, 0], data2=et_, op=nl.multiply)
        vcol = xc[:, 2 * HK_C * B:CT * B]
        delta = _sb((128, J))
        nisa.tensor_tensor(dst=delta, data1=vcol, data2=kv, op=nl.subtract)
        nisa.tensor_tensor(dst=delta, data1=delta, data2=bt_, op=nl.multiply)
        o = _sb((128, J))
        nisa.tensor_tensor(dst=o, data1=kq[:, :, 1], data2=et_, op=nl.multiply)
        t2 = _sb((128, J))
        nisa.tensor_tensor(dst=t2, data1=qkd, data2=delta, op=nl.multiply)
        nisa.tensor_tensor(dst=o, data1=o, data2=t2, op=nl.add)
        nisa.tensor_copy(dst=oall.ap(pattern=[[JR, 128], [R, HV_C], [T, B]], offset=tok), src=o)
        # ---- state update S_new = S*eg + k (x) delta (new tile per token: the checkpoint DMA reads it) ----
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
        nisa.tensor_tensor(dst=Dblk, data1=Drows.ap(pattern=[[128, J], [0, J], [1, 128]], offset=0), data2=mk,
                           op=nl.multiply)
        Sn = _sb((128, J * 128))
        for c in range(J // 4):
            po = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(po, Krows, Dblk[0:J, c * 512:(c + 1) * 512])
            Sc = Sn[:, c * 512:(c + 1) * 512]
            nisa.tensor_tensor(dst=Sc, data1=Sp[:, c * 512:(c + 1) * 512],
                               data2=et_.ap(pattern=[[J, 128], [1, 4], [0, 128]], offset=c * 4), op=nl.multiply)
            nisa.tensor_tensor(dst=Sc, data1=Sc, data2=po, op=nl.add)
        for b in range(B):
            nisa.dma_copy(dst=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cw"][tok], b, B),
                                     indirect_dim=0),
                          src=Sn.ap(pattern=[[J * 128, 128], [B * 128, HV_C], [1, 128]], offset=b * 128))
        Sp = Sn

    # ---------------- gated RMSNorm (R rows, column c*R + r) ----------------
    o2 = _sb((128, JR), BF)
    nisa.activation(dst=o2, data=oall, op=nl.square)
    ms = _sb((128, JR))
    _colsum(ms, o2, JR, ones)
    rr = _sb((128, JR))
    nisa.activation(dst=rr, data=ms, op=nl.rsqrt, scale=1.0 / D, bias=1e-6)
    sz = _sb((128, JR))
    nisa.activation(dst=sz, data=xz[:, CT * R:NT * R], op=nl.silu)
    y = _sb((128, JR))
    nisa.scalar_tensor_tensor(dst=y, data=oall, op0=nl.multiply, operand0=nwc, op1=nl.multiply, operand1=rr)
    oT = _sb((128, OKT * R), BF)
    if NC == 1:
        nisa.tensor_tensor(dst=oT, data1=y, data2=sz, op=nl.multiply)
    else:
        oTl = _sb((128, JR), BF)
        nisa.tensor_tensor(dst=oTl, data1=y, data2=sz, op=nl.multiply)
        other = 1 - pid
        peer = _sb((128, JR), BF)
        nisa.sendrecv(src=oTl, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.tensor_copy(dst=oT[:, pid * JR:(pid + 1) * JR], src=oTl)
        nisa.tensor_copy(dst=oT[:, other * JR:(other + 1) * JR], src=peer)

    # ---------------- out_proj (N-split) ----------------
    nto = _ntiles(NOUT_C)
    pso = [nl.ndarray((R, 512), dtype=F32, buffer=nl.psum) for _ in nto]
    for g in range(OKT // KCH):
        w = wouts[g] if wouts is not None else gdn_wout_load(Wout, g, NC, pid)
        for kk in range(KCH):
            kt = g * KCH + kk
            for ti, (n0, nn) in enumerate(nto):
                nisa.nc_matmul(pso[ti][0:R, 0:nn], oT[:, kt * R:(kt + 1) * R], w[:, kk, n0:n0 + nn])
    y_rows = _sb((R, NOUT_C), BF)
    for ti, (n0, nn) in enumerate(nto):
        nisa.tensor_copy(dst=y_rows[0:R, n0:n0 + nn], src=pso[ti][0:R, 0:nn])
    return y_rows
