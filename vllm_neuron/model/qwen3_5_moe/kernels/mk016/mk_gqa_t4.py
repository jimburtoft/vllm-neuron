"""GQA mixer for the MTP verify step: B requests x T query tokens (R = B*T rows, r = b*T + t).

Same math as mk_gqa.gqa_mixer. Per request: queries at positions p0..p0+T-1 attend to the cached keys at
positions < p0 (cache window, masked by p0) plus the T new keys from SBUF (causal within the T; constant mask
cm_in [16, T], row h*T + t, NEG where t' > t). The T new K/V rows are written to the cache at slot_mapping
(pcore 0, no read-after-write: the cache read masks positions >= p0).
"""
import os

import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, H, KT, evac, fold_halves, sb
from .mk_gqa import BS, D, KCH, NEG, NQH, NQKV, RD, SCALE, _oproj_nsplit, _rmsnorm_rope
from .mk_flash import FlashBT, attend_t4_flash, flash_mode


def _attend_t4(heads, kv, Kc, Vc, btu, b, NB, posf, io, cm, ident, odst, col, cstride, ldo, B, T):
    """heads = [(tile, row0, col)]: q|gate of head h for the T tokens = tile rows row0..row0+T-1, columns col..
    kv = (tile, row0, col) of [k_new | v_new] rows. Rows of the score tile: h*T + t.
    Writes the gated out^T: odst column (h*2 + half)*cstride + col + t."""
    nh = len(heads)
    M = nh * T
    CTX = NB * BS
    q2 = sb((M, D), BF)
    g2 = sb((M, D), BF)
    for h, (qt, r0, c) in enumerate(heads):
        nisa.dma_copy(dst=q2[h * T:(h + 1) * T, :], src=qt[r0:r0 + T, c:c + D])
        nisa.dma_copy(dst=g2[h * T:(h + 1) * T, :], src=qt[r0:r0 + T, c + D:c + 2 * D])
    kt_, kr, kc = kv
    kn = sb((T, D), BF)
    vn = sb((T, D), BF)
    nisa.dma_copy(dst=kn, src=kt_[kr:kr + T, kc:kc + D])
    nisa.dma_copy(dst=vn, src=kt_[kr:kr + T, kc + D:kc + 2 * D])
    pq = nl.ndarray((128, 2, M), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(pq[:, hf, :], q2[0:M, hf * 128:(hf + 1) * 128], ident[0:M, 0:M])
    qT = sb((128, 2, M), BF)
    nisa.tensor_copy(dst=qT, src=pq)
    pkn = nl.ndarray((128, 2, T), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(pkn[:, hf, :], kn[0:T, hf * 128:(hf + 1) * 128], ident[0:T, 0:T])
    knT = sb((128, 2, T), BF)
    nisa.tensor_copy(dst=knT, src=pkn)
    S = sb((M, CTX + T))
    msk = sb((M, CTX))
    nisa.tensor_scalar(dst=msk, data=io[0:M, :], op0=nl.greater_equal, operand0=posf[0:M, b:b + 1],
                       op1=nl.multiply, operand1=NEG)
    Vt = []
    for j in range(NB):
        iv = btu.ap(pattern=[[B * NB, 1], [1, 1]], offset=b * NB + j)
        kt = sb((128, 2, D), BF)
        nisa.dma_copy(dst=kt, src=Kc.ap(pattern=[[D, 128], [128 * D, 2], [1, D]], offset=0, scalar_offset=iv,
                                        indirect_dim=0))
        vt = sb((128, 2, D), BF)
        nisa.dma_copy(dst=vt, src=Vc.ap(pattern=[[D, 128], [128 * D, 2], [1, D]], offset=0, scalar_offset=iv,
                                        indirect_dim=0))
        Vt.append(vt)
        pk = nl.ndarray((128, 2, BS), dtype=F32, buffer=nl.psum)
        for hf in range(2):
            for sub in range(2):
                nisa.nc_matmul(pk[:, hf, sub * 128:(sub + 1) * 128], kt[:, sub, hf * 128:(hf + 1) * 128], ident)
        kT = sb((128, 2, BS), BF)
        nisa.tensor_copy(dst=kT, src=pk)
        pss = nl.ndarray((M, BS), dtype=F32, buffer=nl.psum)
        for hf in range(2):
            nisa.nc_matmul(pss, qT[:, hf, :], kT[:, hf, :])
        nisa.scalar_tensor_tensor(dst=S[:, j * BS:(j + 1) * BS], data=pss, op0=nl.multiply, operand0=SCALE,
                                  op1=nl.add, operand1=msk[:, j * BS:(j + 1) * BS])
    psn = nl.ndarray((M, T), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(psn, qT[:, hf, :], knT[:, hf, :])
    nisa.scalar_tensor_tensor(dst=S[:, CTX:CTX + T], data=psn, op0=nl.multiply, operand0=SCALE, op1=nl.add,
                              operand1=cm[0:M, :])
    mx = sb((M, 1))
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
    nmx = sb((M, 1))
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
    P = sb((M, CTX + T))
    nisa.activation(dst=P, data=S, op=nl.exp, bias=nmx)
    lsum = sb((M, 1))
    nisa.tensor_reduce(dst=lsum, op=nl.add, data=P, axis=1)
    rl = sb((M, 1))
    nisa.reciprocal(dst=rl, data=lsum)
    Pb = sb((M, CTX + T), BF)
    nisa.tensor_copy(dst=Pb, src=P)
    po = nl.ndarray((M, D), dtype=F32, buffer=nl.psum)
    for j in range(NB):
        pp = nl.ndarray((128, 2, M), dtype=F32, buffer=nl.psum)
        for sub in range(2):
            c0 = j * BS + sub * 128
            nisa.nc_matmul(pp[:, sub, :], Pb[0:M, c0:c0 + 128], ident[0:M, 0:M])
        pT = sb((128, 2, M), BF)
        nisa.tensor_copy(dst=pT, src=pp)
        for sub in range(2):
            nisa.nc_matmul(po, pT[:, sub, :], Vt[j][:, sub, :])
    ppn = nl.ndarray((T, M), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ppn, Pb[0:M, CTX:CTX + T], ident[0:M, 0:M])
    pnT = sb((T, M), BF)
    nisa.tensor_copy(dst=pnT, src=ppn)
    nisa.nc_matmul(po, pnT, vn)
    ob = sb((M, D), BF)
    nisa.tensor_scalar(dst=ob, data=po, op0=nl.multiply, operand0=rl)
    sg = sb((M, D), BF)
    nisa.activation(dst=sg, data=g2, op=nl.sigmoid)
    og = sb((M, D), BF)
    nisa.tensor_tensor(dst=og, data1=ob, data2=sg, op=nl.multiply)
    pt = nl.ndarray((128, 2, M), dtype=F32, buffer=nl.psum)   # [d, half, h*T + t]
    for hf in range(2):
        nisa.nc_matmul(pt[:, hf, :], og[0:M, hf * 128:(hf + 1) * 128], ident[0:M, 0:M])
    ptb = sb((128, 2, M), BF)
    nisa.tensor_copy(dst=ptb, src=pt)
    # k-tile kt = h*2 + half -> column kt*cstride + col + t
    dst = odst.ap(pattern=[[ldo, 128], [cstride, 2], [2 * cstride, nh], [1, T]], offset=col)
    nisa.tensor_copy(dst=dst, src=ptb)


def gqa_mixer_t4(xT, Wqkv, Wo, qnw, knw, cosb, sinb, Kc, Vc, bt_in, slot_in, pos0_in, cm_in, ident, B, T, NC, pid, rsc=None):
    """xT [128, KT*R] bf16 (column kt*R + r). cosb/sinb [R, 64] fp32, slot_in [R] int32, pos0_in [B] int32 (position
    of each request's first verify token), cm_in [16, T] fp32 causal mask. Returns [R, H/NC] bf16 (pcore half)."""
    assert NC == 2
    R = B * T
    NB = bt_in.shape[1]
    CTX = NB * BS
    NIN = 2 * 2 * D + 2 * D
    QW = 2 * 2 * D
    NBLK = Kc.shape[0]
    MX = NQH * T

    bti = sb((1, B * NB), nl.int32)
    nisa.dma_copy(dst=bti, src=bt_in.ap(pattern=[[B * NB, 1], [1, B * NB]], offset=0))
    nisa.tensor_scalar(dst=bti, data=bti, op0=nl.maximum, operand0=0)
    btu = nl.ndarray((1, B * NB), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=btu, src=bti)
    psi = sb((MX, B), nl.int32)
    nisa.dma_copy(dst=psi, src=pos0_in.ap(pattern=[[0, MX], [1, B]], offset=0))
    posf = sb((MX, B))
    nisa.tensor_copy(dst=posf, src=psi)
    fmode = flash_mode(NB)   # flash-decode attention
    csplit = bool(fmode) and B % 2 == 1 and os.environ.get("MK_FLASH_SPLIT", "0") == "1"
    if fmode:
        fb = FlashBT(bti, B, NB, fmode, MX, split=(2, pid) if csplit else (1, 0))
        io = None
    else:
        io = sb((MX, CTX))
        nisa.iota(dst=io, pattern=[[1, CTX]], offset=0, channel_multiplier=0)
    cm = sb((MX, T))
    nisa.dma_copy(dst=cm, src=cm_in[0:MX, :])
    qw = sb((R, D))
    nisa.dma_copy(dst=qw, src=qnw.ap(pattern=[[0, R], [1, D]], offset=0))
    kw = sb((R, D))
    nisa.dma_copy(dst=kw, src=knw.ap(pattern=[[0, R], [1, D]], offset=0))
    cs = sb((R, RD))
    nisa.dma_copy(dst=cs, src=cosb)
    sn = sb((R, RD))
    nisa.dma_copy(dst=sn, src=sinb)

    nts = [(s, 512) for s in range(0, NIN, 512)]
    ps = [nl.ndarray((R, 512), dtype=F32, buffer=nl.psum) for _ in nts]
    for g in range(KT // KCH):
        w = sb((128, KCH, NIN), BF)
        nisa.dma_copy(dst=w[:, :, 0:QW], src=Wqkv.ap(pattern=[[NQKV, 128], [128 * NQKV, KCH], [1, QW]],
                                                    offset=g * KCH * 128 * NQKV + pid * QW))
        nisa.dma_copy(dst=w[:, :, QW:NIN], src=Wqkv.ap(pattern=[[NQKV, 128], [128 * NQKV, KCH], [1, 2 * D]],
                                                      offset=g * KCH * 128 * NQKV + NQH * 2 * D))
        for kk in range(KCH):
            for ti, (n0, nn) in enumerate(nts):
                nisa.nc_matmul(ps[ti][0:R, 0:nn], xT[:, (g * KCH + kk) * R:(g * KCH + kk + 1) * R],
                               w[:, kk, n0:n0 + nn])
    a = sb((R, NIN), BF)
    for ti, (n0, nn) in enumerate(nts):
        evac(a[0:R, n0:n0 + nn], ps[ti][0:R, 0:nn], rsc)
    pr = sb((R, NIN), BF)
    for hl in range(2):
        _rmsnorm_rope(pr[:, hl * 2 * D:hl * 2 * D + D], a[:, hl * 2 * D:hl * 2 * D + D], qw, cs, sn, R)
        nisa.tensor_copy(dst=pr[:, hl * 2 * D + D:(hl + 1) * 2 * D], src=a[:, hl * 2 * D + D:(hl + 1) * 2 * D])
    _rmsnorm_rope(pr[:, QW:QW + D], a[:, QW:QW + D], kw, cs, sn, R)
    nisa.tensor_copy(dst=pr[:, QW + D:NIN], src=a[:, QW + D:NIN])

    if pid == 0:
        sli = sb((R, 1), nl.int32)
        nisa.dma_copy(dst=sli, src=slot_in.ap(pattern=[[1, R], [1, 1]], offset=0))
        slu = nl.ndarray((R, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slu, src=sli)
        for cache, c0 in ((Kc, QW), (Vc, QW + D)):
            nisa.dma_copy(dst=cache.reshape((NBLK * BS, D)).ap(pattern=[[1, R], [D, 1], [1, D]], offset=0,
                                                                vector_offset=slu, indirect_dim=0),
                          src=pr.ap(pattern=[[NIN, R], [D, 1], [1, D]], offset=c0), oob_mode=nisa.oob_mode.skip)

    if csplit:   # context split over the 2 pcores (all 4 heads each, LSE merge), o_proj N-split
        other = 1 - pid
        qpeer = sb((R, NIN), BF)
        nisa.sendrecv(src=pr, dst=qpeer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        q_of = [pr, qpeer] if pid == 0 else [qpeer, pr]
        oT = sb((128, 8 * R), BF)                 # column kt*R + b*T + t
        for b in range(B):
            heads = [(q_of[c], b * T, hl * 2 * D) for c in range(2) for hl in range(2)]
            attend_t4_flash(heads, (pr, b * T, QW), Kc, Vc, fb, b, posf, cm, ident, oT, b * T, R, 8 * R, T)
        return _oproj_nsplit(oT, Wo, R, NC, pid)
    if B % 2 == 0:
        other = 1 - pid
        qpeer = sb((R, NIN), BF)
        nisa.sendrecv(src=pr, dst=qpeer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        q_of = [pr, qpeer] if pid == 0 else [qpeer, pr]
        NH2 = B // 2
        own = sb((128, NH2 * 8 * T), BF)          # column (i*8 + kt)*T + t
        for i in range(NH2):
            b = 2 * i + pid
            heads = [(q_of[c], b * T, hl * 2 * D) for c in range(2) for hl in range(2)]
            if fmode:
                attend_t4_flash(heads, (pr, b * T, QW), Kc, Vc, fb, b, posf, cm, ident, own, i * 8 * T, T,
                                NH2 * 8 * T, T)
            else:
                _attend_t4(heads, (pr, b * T, QW), Kc, Vc, btu, b, NB, posf, io, cm, ident, own, i * 8 * T, T,
                           NH2 * 8 * T, B, T)
        peer = sb((128, NH2 * 8 * T), BF)
        nisa.sendrecv(src=own, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        oT = sb((128, 8 * R), BF)                 # column kt*R + b*T + t, b = 2i + c
        for c, t_ in ((pid, own), (other, peer)):
            nisa.tensor_copy(dst=oT.ap(pattern=[[8 * R, 128], [2 * T, NH2], [R, 8], [1, T]], offset=c * T),
                             src=t_.ap(pattern=[[NH2 * 8 * T, 128], [8 * T, NH2], [T, 8], [1, T]], offset=0))
        HW = H // NC
        wo = sb((128, 8, HW), BF)
        nisa.dma_copy(dst=wo, src=Wo.ap(pattern=[[H, 128], [128 * H, 8], [1, HW]], offset=pid * HW))
        pso = [nl.ndarray((R, 512), dtype=F32, buffer=nl.psum) for _ in range(HW // 512)]
        for kt in range(8):
            for ti in range(HW // 512):
                nisa.nc_matmul(pso[ti][0:R, 0:512], oT[:, kt * R:(kt + 1) * R], wo[:, kt, ti * 512:(ti + 1) * 512])
        yh = sb((R, HW), BF)
        for ti in range(HW // 512):
            nisa.tensor_copy(dst=yh[0:R, ti * 512:(ti + 1) * 512], src=pso[ti][0:R, 0:512])
        return yh
    oT = sb((128, 4 * R), BF)    # column (hl*2 + half)*R + r
    for b in range(B):
        heads = [(pr, b * T, hl * 2 * D) for hl in range(2)]
        if fmode:
            attend_t4_flash(heads, (pr, b * T, QW), Kc, Vc, fb, b, posf, cm, ident, oT, b * T, R, 4 * R, T)
        else:
            _attend_t4(heads, (pr, b * T, QW), Kc, Vc, btu, b, NB, posf, io, cm, ident, oT, b * T, R, 4 * R, B, T)
    wo = sb((128, 4, H), BF)
    nisa.dma_copy(dst=wo, src=Wo.ap(pattern=[[H, 128], [128 * H, 4], [1, H]], offset=pid * 4 * 128 * H))
    pso = [nl.ndarray((R, 512), dtype=F32, buffer=nl.psum) for _ in range(H // 512)]
    for hl in range(2):
        for hf in range(2):
            col = (hl * 2 + hf) * R
            for ti in range(H // 512):
                nisa.nc_matmul(pso[ti][0:R, 0:512], oT[:, col:col + R], wo[:, hl * 2 + hf, ti * 512:(ti + 1) * 512])
    y = sb((R, H), BF)
    for ti in range(H // 512):
        nisa.tensor_copy(dst=y[0:R, ti * 512:(ti + 1) * 512], src=pso[ti][0:R, 0:512])
    return fold_halves(y, R, NC, pid)
