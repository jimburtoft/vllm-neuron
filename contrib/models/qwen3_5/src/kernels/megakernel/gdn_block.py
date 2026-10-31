# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet decode mixer as a megakernel block (no residual), one TP=4 rank, LNC=2, one token per request.

Takes the normed input in the std column layout xT [128, KT*B] bf16 (SBUF) and returns this pcore's H half of the
out_proj rank partial ([B, H/NC] bf16 SBUF); the caller all-reduces over TP and adds the residual.

LNC=2 split: pcore c owns key heads [2c, 2c+2) and value heads [4c, 4c+4) of the rank, i.e. conv channel tiles
(128 channels each) q {2c, 2c+1}, k {4+2c, 5+2c}, v {8+4c .. 11+4c} of the rank's [q | k | v] channels, and the
out_proj output columns [c*H/2, (c+1)*H/2) (N-split, after the two pcores exchange their gated-norm outputs).

State: read and updated in place in the model's paged state cache (``Qwen3_5GatedDeltaNet.bind_state``): two
float32 halves A / B of shape [pages, rows, 128]; one page per request holds the conv window [K-1, conv_dim]
followed by the recurrent state [Hv, Dk, Dv], back to back across A then B. The page id of request b is
pages[b] (dead batch rows point at the null page 0). Each pcore reads and writes only its own channel tiles and
heads, so the two pcores never touch the same bytes.
"""
import math

import nki.isa as nisa
import nki.language as nl

H, VAL, D = 2048, 1024, 128
HK, HV = 4, 8  # per TP rank
KCH = 4
KT, OKT = H // 128, VAL // 128
F32, BF = nl.float32, nl.bfloat16
K1 = 3          # conv kernel size - 1
CONV_ROWS = K1 * (2 * HK * D + HV * D) // 128   # conv window rows of 128 floats in a page (48)


def dims(nc, b):
    hk_c, hv_c = HK // nc, HV // nc
    ct = 2 * hk_c + hv_c
    nin_c = 128 * (ct + hv_c) + 2 * hv_c
    return hk_c, hv_c, ct, nin_c, hv_c * b, H // nc


def tile_runs(pid, nc):
    """This pcore's conv channel tiles as runs (first rank-local tile T0, first pcore-local tile t0, count)."""
    hk_c, hv_c = HK // nc, HV // nc
    return ((pid * hk_c, 0, hk_c), (HK + pid * hk_c, hk_c, hk_c), (2 * HK + pid * hv_c, 2 * hk_c, hv_c))


def _ntiles(n, t=512):
    return [(s, min(t, n - s)) for s in range(0, n, t)]


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _colsum(dst_sb, src_bf, n, ones):
    ps = nl.ndarray((128, n), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ps, ones, src_bf)
    nisa.tensor_copy(dst=dst_sb, src=ps)


def gdn_win_load(Win, g, NC, pid, B):
    """k-tile group g (KCH x 128 rows) of this pcore's in_proj slice -> SBUF (cross-layer prefetch unit)."""
    NIN_C = dims(NC, B)[3]
    w = _sb((128, KCH, NIN_C), BF)
    nisa.dma_copy(dst=w, src=Win.ap(pattern=[[NIN_C, 128], [128 * NIN_C, KCH], [1, NIN_C]],
                                     offset=pid * H * NIN_C + g * KCH * 128 * NIN_C))
    return w


def gdn_wout_load(Wout, g, NC, pid):
    """k-tile group g of this pcore's out_proj N-half [KCH*128, H/NC] -> SBUF."""
    NOUT_C = H // NC
    w = _sb((128, KCH, NOUT_C), BF)
    nisa.dma_copy(dst=w, src=Wout.ap(pattern=[[H, 128], [128 * H, KCH], [1, NOUT_C]],
                                      offset=g * KCH * 128 * H + pid * NOUT_C))
    return w


def page_tiles(pages, g, B):
    """pages [NGRP, 2, B] int32 HBM, state group g (row 0: page read, row 1: page written; a dead batch row reads
    the null page 0 and writes an out-of-range page, which the DMA drops) -> two (1, B) uint32 SBUF tiles."""
    out = []
    for r in range(2):
        t = _sb((1, B), nl.int32)
        nisa.dma_copy(dst=t, src=pages.ap(pattern=[[2 * B, 1], [1, B]], offset=(g * 2 + r) * B))
        u = nl.ndarray((1, B), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=t)
        out.append(u)
    return out


def _ix(t, b, B):
    return t.ap(pattern=[[B, 1], [1, 1]], offset=b)


def _rec_segments(rows, pid, nc):
    """Trace-time split of this pcore's recurrent heads over the two halves.
    Returns [(half, row0, head0, nheads, k0, nk)]: rows [row0, ...) of half (0 = A, 1 = B) hold heads
    [head0, head0 + nheads) (pcore-local index) at key rows [k0, k0 + nk)."""
    hv_c = HV // nc
    segs = []
    for i in range(hv_c):
        a0 = CONV_ROWS + (pid * hv_c + i) * D   # page row of key row 0 of this head
        if a0 + D <= rows:
            segs.append((0, a0, i, 1, 0, D))
        elif a0 >= rows:
            segs.append((1, a0 - rows, i, 1, 0, D))
        else:
            split = rows - a0
            segs.append((0, a0, i, 1, 0, split))
            segs.append((1, 0, i, 1, split, D - split))
    # merge consecutive whole heads in the same half into one multi-head segment
    out = []
    for s in segs:
        if out and s[4] == 0 and s[5] == D and out[-1][4] == 0 and out[-1][5] == D and out[-1][0] == s[0] \
                and out[-1][1] + out[-1][3] * D == s[1] and out[-1][2] + out[-1][3] == s[2]:
            h, r0, i0, n, _, _ = out[-1]
            out[-1] = (h, r0, i0, n + 1, 0, D)
        else:
            out.append(s)
    return out


def gdn_mixer(xT, Win, Wout, cw, sa, sbh, alog, dtb, nw, ident, ones, mk, gsc, pg, B, NC, pid, w_pre=()):
    """xT std column layout of the normed input. Win [NC, H, NIN_C] flat bf16 (per pcore [q_c|k_c|v_c|z_c|b_c|a_c]),
    Wout [1024, H] bf16, cw [NC, 128, K, CT] flat fp32, sa / sbh state halves [pages, rows, 128] fp32 (in place),
    alog / dtb [HV] fp32, nw [128] fp32, gsc HBM scratch [NC * B * 2 * HV_C] bf16, pg = page_tiles(...).
    Returns (y_rows [B, H/NC] bf16, (sa, sbh))."""
    HK_C, HV_C, CT, NIN_C, J, NOUT_C = dims(NC, B)
    rows = sa.shape[1]
    runs = tile_runs(pid, NC)
    pr_, pw_ = pg
    skip = {"oob_mode": nisa.oob_mode.skip}
    halves = (sa, sbh)

    # ---------------- state / param loads ----------------
    # conv window: page rows w*16 + T (all 16 rank tiles; only this pcore's are used) -> rows tile [16, B, K1*128]
    NT_R = CONV_ROWS // K1
    crow = _sb((NT_R, B * K1 * 128))
    for b in range(B):
        nisa.dma_copy(dst=crow.ap(pattern=[[B * K1 * 128, NT_R], [128, K1], [1, 128]], offset=b * K1 * 128),
                      src=sa.ap(pattern=[[128, NT_R], [NT_R * 128, K1], [1, 128]], offset=0,
                                scalar_offset=_ix(pr_, b, B), indirect_dim=0))
    crow_bf = _sb((NT_R, B * K1 * 128), BF)  # the window holds bf16 values: exact
    nisa.tensor_copy(dst=crow_bf, src=crow)
    hs = _sb((128, 4 * CT * B))   # column w*CT*B + t*B + b (w = 3 is the new token)
    for b in range(B):
        pcv = nl.ndarray((128, K1 * NT_R), dtype=F32, buffer=nl.psum)   # column w*16 + T
        for w in range(K1):
            c0 = (b * K1 + w) * 128
            nisa.nc_matmul(pcv[:, w * NT_R:(w + 1) * NT_R], crow_bf[0:NT_R, c0:c0 + 128], ident[0:NT_R, 0:NT_R])
        for T0, t0, n in runs:
            nisa.tensor_copy(dst=hs.ap(pattern=[[4 * CT * B, 128], [CT * B, K1], [B, n]], offset=t0 * B + b),
                             src=pcv.ap(pattern=[[K1 * NT_R, 128], [NT_R, K1], [1, n]], offset=T0))
    wraw = _sb((128, 4 * CT))
    nisa.dma_copy(dst=wraw, src=cw.ap(pattern=[[4 * CT, 128], [1, 4 * CT]], offset=pid * 128 * 4 * CT))
    wexp = _sb((128, 4 * CT * B))
    nisa.tensor_copy(dst=wexp, src=wraw.ap(pattern=[[4 * CT, 128], [1, 4 * CT], [0, B]], offset=0))
    # recurrent state [Hv, Dk, Dv] -> S[k, (hv_local*B + b)*128 + v]
    S = _sb((128, J * 128))
    segs = _rec_segments(rows, pid, NC)
    for b in range(B):
        for hf, r0, i0, nh, k0, nk in segs:
            nisa.dma_copy(dst=S[k0:k0 + nk, :].ap(pattern=[[J * 128, nk], [B * 128, nh], [1, 128]], offset=(i0 * B + b) * 128),
                          src=halves[hf].ap(pattern=[[128, nk], [D * 128, nh], [1, 128]], offset=r0 * 128,
                                            scalar_offset=_ix(pr_, b, B), indirect_dim=0))
    al = _sb((128, J))
    dt = _sb((128, J))
    adr = _sb((128, 2 * HV_C))
    nisa.dma_copy(dst=adr[:, 0:HV_C], src=alog.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.dma_copy(dst=adr[:, HV_C:2 * HV_C], src=dtb.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
    nisa.tensor_copy(dst=al, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=0))
    nisa.tensor_copy(dst=dt, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=HV_C))
    nwc = _sb((128, 1))
    nisa.dma_copy(dst=nwc, src=nw.ap(pattern=[[1, 128], [1, 1]], offset=0))

    # ---------------- in_proj ----------------
    nts = _ntiles(NIN_C)
    ps = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nts]
    for g in range(KT // KCH):
        w = w_pre[g] if g < len(w_pre) else gdn_win_load(Win, g, NC, pid, B)
        for kk in range(KCH):
            for ti, (n0, nn) in enumerate(nts):
                nisa.nc_matmul(ps[ti][0:B, 0:nn], xT[:, (g * KCH + kk) * B:(g * KCH + kk + 1) * B],
                               w[:, kk, n0:n0 + nn])
    a_rows = _sb((B, NIN_C), BF)
    for ti, (n0, nn) in enumerate(nts):
        nisa.tensor_copy(dst=a_rows[0:B, n0:n0 + nn], src=ps[ti][0:B, 0:nn])

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
    # conv window update, this pcore's tiles only: rows w' = 0, 1 <- old w = 1, 2; row w' = 2 <- this token
    a32 = _sb((B, CT * 128))
    nisa.tensor_copy(dst=a32, src=a_rows[0:B, 0:CT * 128])
    for b in range(B):
        for T0, t0, n in runs:
            nisa.dma_copy(dst=sa.ap(pattern=[[128, n], [NT_R * 128, K1 - 1], [1, 128]], offset=T0 * 128,
                                    scalar_offset=_ix(pw_, b, B), indirect_dim=0),
                          src=crow[T0:T0 + n, :].ap(pattern=[[B * K1 * 128, n], [128, K1 - 1], [1, 128]],
                                                    offset=b * K1 * 128 + 128), **skip)
            nisa.dma_copy(dst=sa.ap(pattern=[[n * 128, 1], [1, n * 128]], offset=((K1 - 1) * NT_R + T0) * 128,
                                    scalar_offset=_ix(pw_, b, B), indirect_dim=0),
                          src=a32[b:b + 1, :].ap(pattern=[[CT * 128, 1], [1, n * 128]], offset=t0 * 128), **skip)
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
    for b in range(B):
        for hf, r0, i0, nh, k0, nk in segs:
            nisa.dma_copy(dst=halves[hf].ap(pattern=[[128, nk], [D * 128, nh], [1, 128]], offset=r0 * 128,
                                            scalar_offset=_ix(pw_, b, B), indirect_dim=0),
                          src=S[k0:k0 + nk, :].ap(pattern=[[J * 128, nk], [B * 128, nh], [1, 128]], offset=(i0 * B + b) * 128),
                          **skip)

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
        other = 1 - pid
        peer = _sb((128, J), BF)
        nisa.sendrecv(src=oTl, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        nisa.tensor_copy(dst=oT[:, pid * J:(pid + 1) * J], src=oTl)
        nisa.tensor_copy(dst=oT[:, other * J:(other + 1) * J], src=peer)

    # ---------------- out_proj (N-split, Wout read in place) ----------------
    nto = _ntiles(NOUT_C)
    pso = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nto]
    for g in range(OKT // KCH):
        w = gdn_wout_load(Wout, g, NC, pid)
        for kk in range(KCH):
            kt = g * KCH + kk
            for ti, (n0, nn) in enumerate(nto):
                nisa.nc_matmul(pso[ti][0:B, 0:nn], oT[:, kt * B:(kt + 1) * B], w[:, kk, n0:n0 + nn])
    y_rows = _sb((B, NOUT_C), BF)
    for ti, (n0, nn) in enumerate(nto):
        nisa.tensor_copy(dst=y_rows[0:B, n0:n0 + nn], src=pso[ti][0:B, 0:nn])
    return y_rows
