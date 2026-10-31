# SPDX-License-Identifier: Apache-2.0
"""Gated GQA decode mixer as a megakernel block (full attention), one TP=4 rank, LNC=2, one token per request.

Math (``Qwen3_5Attention.forward_decode``):
  qkv = x @ Wqkv                        [B, 2560] bf16; per local head h: [q_h | gate_h] (2 x 256), then k, v (256 each)
  q, k = RMSNorm(q), RMSNorm(k)         zero-centred weights (1 + w, fp32), fp32 math, bf16 result
  q, k = partial RoPE on dims [0, 64)   (rotate_half over 64, cos/sin [B, 64] fp32 from the caller)
  scores over the cached keys t < pos plus the new token; fp32 softmax(q.K^T / 16)
  out = (P @ V) -> bf16, * sigmoid(gate) -> bf16, @ Wo [1024, H]
Per rank: 4 q heads, 1 kv head (replicated: ranks 0,1 hold kv head 0, ranks 2,3 kv head 1; each rank passes its
own cache). KV cache per layer [NBLK, 1, BS=256, 256] bf16, block_table [B, NB] int32 (-1 -> block 0, masked),
slot_mapping [B] int32 (flat row = block*256 + offset), pos [B] int32 (position of the new token: the cache holds
pos tokens of context). Fixed context bucket CTX = NB * 256.

LNC=2: pcore c computes the qkv columns of local q heads 2c, 2c+1 plus the shared k/v columns. The new token is folded
in as one extra score column from SBUF (no read-after-write on the cache); pcore 0 writes K/V into the cache in place
(vector_offset DMA). Odd B (B=1): pcore c attends its 2 heads for every request, o_proj K-split + fold_halves. Even B:
pcore c attends requests 2i+c over all 4 heads (KV read once per rank, the peer's q by SBUF sendrecv), outputs
exchanged by sendrecv, o_proj N-split. Contexts above 2048 tokens use the flash-decode core (mk_flash.py).
Both return this pcore's H half of the rank partial [B, H/NC] bf16.
"""
import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, EPS, F32, H, KT, fold_halves, sb
from .mk_flash import FlashBT, attend_one_flash, flash_on

D, RD, BS = 256, 64, 256      # head_dim, rotary dims, KV block size
NQH = 4                       # q heads per rank
NQKV = NQH * 2 * D + 2 * D    # 2560
SCALE = D ** -0.5
NEG = -1.0e30
KCH = 4


def _rmsnorm_rope(dst, src, w, cs, sn, B):
    """dst/src [B, 256] bf16 rows; w [B, 256] fp32 (broadcast); cs/sn [B, 64] fp32."""
    xf = sb((B, D))
    nisa.tensor_copy(dst=xf, src=src)
    sq = sb((B, D))
    nisa.tensor_tensor(dst=sq, data1=xf, data2=xf, op=nl.multiply)
    ss = sb((B, 1))
    nisa.tensor_reduce(dst=ss, op=nl.add, data=sq, axis=1)
    rs = sb((B, 1))
    nisa.activation(dst=rs, data=ss, op=nl.rsqrt, scale=1.0 / D, bias=EPS)
    nisa.tensor_scalar(dst=xf, data=xf, op0=nl.multiply, operand0=rs)
    nb = sb((B, D), BF)
    nisa.tensor_tensor(dst=nb, data1=xf, data2=w, op=nl.multiply)
    # partial RoPE on the bf16-rounded normed values
    xr = sb((B, RD))
    nisa.tensor_copy(dst=xr, src=nb[:, 0:RD])
    t1 = sb((B, RD))
    nisa.tensor_tensor(dst=t1, data1=xr, data2=cs, op=nl.multiply)
    t2 = sb((B, RD))
    h = RD // 2
    nisa.tensor_tensor(dst=t2[:, 0:h], data1=xr[:, h:RD], data2=sn[:, 0:h], op=nl.multiply)
    nisa.tensor_tensor(dst=t2[:, h:RD], data1=xr[:, 0:h], data2=sn[:, h:RD], op=nl.multiply)
    nisa.tensor_tensor(dst=t1[:, 0:h], data1=t1[:, 0:h], data2=t2[:, 0:h], op=nl.subtract)
    nisa.tensor_tensor(dst=t1[:, h:RD], data1=t1[:, h:RD], data2=t2[:, h:RD], op=nl.add)
    nisa.tensor_copy(dst=dst[:, 0:RD], src=t1)
    nisa.tensor_copy(dst=dst[:, RD:D], src=nb[:, RD:D])


def _attend_one(heads, kv_off, Kc, Vc, btu, b, NB, posf, io, ident, odst, col, cstride, ldo, B,
                sbuf_out=False):
    """One request, nh = len(heads) q heads sharing the rank's kv head. qs = SBUF rows tile; heads = (row, col) of
    each head's [q | gate], kv_off = (row, col) of [k_new | v_new] (SBUF->SBUF row DMAs). Writes out^T (gated) as k-tiles (h*2 + half): column (h*2+half)*B + col
    of odst ([128, ldo], SBUF or HBM)."""
    nh = len(heads)
    CTX = NB * BS
    q2 = sb((nh, D), BF)
    g2 = sb((nh, D), BF)
    kt_, kr, kc = kv_off
    kn = sb((nh, D), BF)
    vn = sb((nh, D), BF)
    for h, (qt, r, c) in enumerate(heads):
        nisa.dma_copy(dst=q2[h:h + 1, :], src=qt[r:r + 1, c:c + D])
        nisa.dma_copy(dst=g2[h:h + 1, :], src=qt[r:r + 1, c + D:c + 2 * D])
        nisa.dma_copy(dst=kn[h:h + 1, :], src=kt_[kr:kr + 1, kc:kc + D])
        nisa.dma_copy(dst=vn[h:h + 1, :], src=kt_[kr:kr + 1, kc + D:kc + 2 * D])
    pq = nl.ndarray((128, 2, nh), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(pq[:, hf, :], q2[0:nh, hf * 128:(hf + 1) * 128], ident[0:nh, 0:nh])
    qT = sb((128, 2, nh), BF)
    nisa.tensor_copy(dst=qT, src=pq)
    S = sb((nh, CTX + 1))
    msk = sb((nh, CTX))
    nisa.tensor_scalar(dst=msk, data=io[0:nh, :], op0=nl.greater_equal, operand0=posf[0:nh, b:b + 1],
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
        pk = nl.ndarray((128, 2, BS), dtype=F32, buffer=nl.psum)   # [d, half, t]
        for hf in range(2):
            for sub in range(2):
                nisa.nc_matmul(pk[:, hf, sub * 128:(sub + 1) * 128], kt[:, sub, hf * 128:(hf + 1) * 128], ident)
        kT = sb((128, 2, BS), BF)
        nisa.tensor_copy(dst=kT, src=pk)
        pss = nl.ndarray((nh, BS), dtype=F32, buffer=nl.psum)
        for hf in range(2):
            nisa.nc_matmul(pss, qT[:, hf, :], kT[:, hf, :])
        nisa.scalar_tensor_tensor(dst=S[:, j * BS:(j + 1) * BS], data=pss, op0=nl.multiply, operand0=SCALE,
                                  op1=nl.add, operand1=msk[:, j * BS:(j + 1) * BS])
    qk = sb((nh, D))
    nisa.tensor_tensor(dst=qk, data1=q2, data2=kn, op=nl.multiply)
    sn1 = sb((nh, 1))
    nisa.tensor_reduce(dst=sn1, op=nl.add, data=qk, axis=1)
    nisa.tensor_scalar(dst=S[:, CTX:CTX + 1], data=sn1, op0=nl.multiply, operand0=SCALE)
    mx = sb((nh, 1))
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
    nmx = sb((nh, 1))
    nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0)
    P = sb((nh, CTX + 1))
    nisa.activation(dst=P, data=S, op=nl.exp, bias=nmx)
    lsum = sb((nh, 1))
    nisa.tensor_reduce(dst=lsum, op=nl.add, data=P, axis=1)
    rl = sb((nh, 1))
    nisa.reciprocal(dst=rl, data=lsum)
    Pb = sb((nh, CTX), BF)
    nisa.tensor_copy(dst=Pb, src=P[:, 0:CTX])
    po = nl.ndarray((nh, D), dtype=F32, buffer=nl.psum)
    for j in range(NB):
        pp = nl.ndarray((128, 2, nh), dtype=F32, buffer=nl.psum)
        for sub in range(2):
            c0 = j * BS + sub * 128
            nisa.nc_matmul(pp[:, sub, :], Pb[0:nh, c0:c0 + 128], ident[0:nh, 0:nh])
        pT = sb((128, 2, nh), BF)
        nisa.tensor_copy(dst=pT, src=pp)
        for sub in range(2):
            nisa.nc_matmul(po, pT[:, sub, :], Vt[j][:, sub, :])
    o = sb((nh, D))
    nisa.tensor_scalar(dst=o, data=vn, op0=nl.multiply, operand0=P[:, CTX:CTX + 1])
    nisa.tensor_tensor(dst=o, data1=o, data2=po, op=nl.add)
    ob = sb((nh, D), BF)
    nisa.tensor_scalar(dst=ob, data=o, op0=nl.multiply, operand0=rl)
    sg = sb((nh, D), BF)
    nisa.activation(dst=sg, data=g2, op=nl.sigmoid)
    og = sb((nh, D), BF)
    nisa.tensor_tensor(dst=og, data1=ob, data2=sg, op=nl.multiply)
    pt = nl.ndarray((128, 2, nh), dtype=F32, buffer=nl.psum)   # [d, half, head]
    for hf in range(2):
        nisa.nc_matmul(pt[:, hf, :], og[0:nh, hf * 128:(hf + 1) * 128], ident[0:nh, 0:nh])
    ptb = sb((128, nh, 2), BF)                                     # [d, head, half] -> k-tile h*2 + half
    for hf in range(2):
        nisa.tensor_copy(dst=ptb[:, :, hf], src=pt[:, hf, :])
    dst = odst.ap(pattern=[[ldo, 128], [cstride, nh * 2]], offset=col)
    if sbuf_out:
        nisa.tensor_copy(dst=dst, src=ptb)
    else:
        nisa.dma_copy(dst=dst, src=ptb)


def _oproj_nsplit(oT, Wo, B, NC, pid):
    """o_proj N-split: oT [128, 8*B] (column kt*B + b, all 4 heads) -> this pcore's H half [B, H/NC] bf16."""
    HW = H // NC
    wo = sb((128, 8, HW), BF)
    nisa.dma_copy(dst=wo, src=Wo.ap(pattern=[[H, 128], [128 * H, 8], [1, HW]], offset=pid * HW))
    pso = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in range(HW // 512)]
    for kt in range(8):
        for ti in range(HW // 512):
            nisa.nc_matmul(pso[ti][0:B, 0:512], oT[:, kt * B:(kt + 1) * B], wo[:, kt, ti * 512:(ti + 1) * 512])
    yh = sb((B, HW), BF)
    for ti in range(HW // 512):
        nisa.tensor_copy(dst=yh[0:B, ti * 512:(ti + 1) * 512], src=pso[ti][0:B, 0:512])
    return yh


def gqa_mixer(xT, Wqkv, Wo, qnw, knw, cosb, sinb, Kc, Vc, bt_in, slot_in, pos_in, ident, B, NC, pid):
    """xT [128, KT*B] bf16 std column layout of the normed input. Wqkv [H, 2560] bf16, Wo [1024, H] bf16,
    qnw/knw [256] fp32 (1 + w), cosb/sinb [B, 64] fp32, Kc/Vc [NBLK, 1, 256, 256] bf16 (written in place),
    bt_in [B, NB] int32, slot_in [B] int32, pos_in [B] int32, ident [128,128] bf16 SBUF.
    Returns [B, H/NC] bf16 SBUF (pcore half of the rank partial)."""
    assert NC == 2, "gqa_mixer splits the 4 local q heads 2/2 over pcores (LNC=2 only)"
    NB = bt_in.shape[1]
    CTX = NB * BS
    NIN = 2 * 2 * D + 2 * D       # 1536 columns per pcore: q0 g0 q1 g1 k v
    QW = 2 * 2 * D                # 1024 q/gate columns per pcore
    NBLK = Kc.shape[0]

    # ---------------- small runtime inputs ----------------
    bti = sb((1, B * NB), nl.int32)
    nisa.dma_copy(dst=bti, src=bt_in.ap(pattern=[[B * NB, 1], [1, B * NB]], offset=0))
    nisa.tensor_scalar(dst=bti, data=bti, op0=nl.maximum, operand0=0)
    btu = nl.ndarray((1, B * NB), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=btu, src=bti)
    psi = sb((4, B), nl.int32)
    nisa.dma_copy(dst=psi, src=pos_in.ap(pattern=[[0, 4], [1, B]], offset=0))
    posf = sb((4, B))
    nisa.tensor_copy(dst=posf, src=psi)
    fmode = flash_on(NB)   # flash-decode attention (online softmax over KV chunks) above 2048 tokens
    if fmode:
        fb = FlashBT(bti, B, NB, 4)
        io = None
    else:
        io = sb((4, CTX))
        nisa.iota(dst=io, pattern=[[1, CTX]], offset=0, channel_multiplier=0)
    qw = sb((B, D))
    nisa.dma_copy(dst=qw, src=qnw.ap(pattern=[[0, B], [1, D]], offset=0))
    kw = sb((B, D))
    nisa.dma_copy(dst=kw, src=knw.ap(pattern=[[0, B], [1, D]], offset=0))
    cs = sb((B, RD))
    nisa.dma_copy(dst=cs, src=cosb)
    sn = sb((B, RD))
    nisa.dma_copy(dst=sn, src=sinb)

    # ---------------- qkv projection (pcore columns) ----------------
    nts = [(s, 512) for s in range(0, NIN, 512)]
    ps = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nts]
    for g in range(KT // KCH):
        w = sb((128, KCH, NIN), BF)
        nisa.dma_copy(dst=w[:, :, 0:QW], src=Wqkv.ap(pattern=[[NQKV, 128], [128 * NQKV, KCH], [1, QW]],
                                                    offset=g * KCH * 128 * NQKV + pid * QW))
        nisa.dma_copy(dst=w[:, :, QW:NIN], src=Wqkv.ap(pattern=[[NQKV, 128], [128 * NQKV, KCH], [1, 2 * D]],
                                                      offset=g * KCH * 128 * NQKV + NQH * 2 * D))
        for kk in range(KCH):
            for ti, (n0, nn) in enumerate(nts):
                nisa.nc_matmul(ps[ti][0:B, 0:nn], xT[:, (g * KCH + kk) * B:(g * KCH + kk + 1) * B],
                               w[:, kk, n0:n0 + nn])
    a = sb((B, NIN), BF)
    for ti, (n0, nn) in enumerate(nts):
        nisa.tensor_copy(dst=a[0:B, n0:n0 + nn], src=ps[ti][0:B, 0:nn])

    # ---------------- q/k norm + RoPE (rows), gate/v pass-through ----------------
    pr = sb((B, NIN), BF)
    for hl in range(2):
        _rmsnorm_rope(pr[:, hl * 2 * D:hl * 2 * D + D], a[:, hl * 2 * D:hl * 2 * D + D], qw, cs, sn, B)
        nisa.tensor_copy(dst=pr[:, hl * 2 * D + D:(hl + 1) * 2 * D], src=a[:, hl * 2 * D + D:(hl + 1) * 2 * D])
    _rmsnorm_rope(pr[:, QW:QW + D], a[:, QW:QW + D], kw, cs, sn, B)
    nisa.tensor_copy(dst=pr[:, QW + D:NIN], src=a[:, QW + D:NIN])

    # ---------------- KV cache write (pcore 0, in place, slot_mapping rows) ----------------
    if NC == 1 or pid == 0:
        sli = sb((B, 1), nl.int32)
        nisa.dma_copy(dst=sli, src=slot_in.ap(pattern=[[1, B], [1, 1]], offset=0))
        slu = nl.ndarray((B, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slu, src=sli)
        for cache, c0 in ((Kc, QW), (Vc, QW + D)):
            nisa.dma_copy(dst=cache.reshape((NBLK * BS, D)).ap(pattern=[[1, B], [D, 1], [1, D]], offset=0,
                                                                vector_offset=slu, indirect_dim=0),
                          src=pr.ap(pattern=[[NIN, B], [D, 1], [1, D]], offset=c0), oob_mode=nisa.oob_mode.skip)

    # ---------------- attention ----------------
    # B even: request split (pcore c runs requests b = 2i + c over all 4 local heads; o_proj N-split).
    # B odd (B=1): head split (pcore c runs its own 2 heads for every request; o_proj K-split + fold_halves).
    rsplit = NC == 2 and B % 2 == 0
    if rsplit:
        # pcore exchange by sendrecv (SBUF <-> SBUF).
        other = 1 - pid
        qpeer = sb((B, NIN), BF)
        nisa.sendrecv(src=pr, dst=qpeer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        q_of = [pr, qpeer] if pid == 0 else [qpeer, pr]      # tile holding pcore c's heads
        NH2 = B // 2
        own = sb((128, NH2 * 8), BF)                         # column i*8 + kt
        for i in range(NH2):
            b = 2 * i + pid
            heads = [(q_of[c], b, hl * 2 * D) for c in range(2) for hl in range(2)]
            if fmode:
                attend_one_flash(heads, (pr, b, QW), Kc, Vc, fb, b, posf, ident, own, i * 8, 1, NH2 * 8, sbuf_out=True)
            else:
                _attend_one(heads, (pr, b, QW), Kc, Vc, btu, b, NB, posf, io, ident, own, i * 8, 1, NH2 * 8, B,
                            sbuf_out=True)
        peer = sb((128, NH2 * 8), BF)
        nisa.sendrecv(src=own, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        oT = sb((128, 8 * B), BF)        # column kt*B + b, b = 2i + c
        for c, t in ((pid, own), (other, peer)):
            nisa.tensor_copy(dst=oT.ap(pattern=[[8 * B, 128], [2, NH2], [B, 8]], offset=c),
                             src=t.ap(pattern=[[NH2 * 8, 128], [8, NH2], [1, 8]], offset=0))
        return _oproj_nsplit(oT, Wo, B, NC, pid)
    oT = sb((128, 4 * B), BF)    # column (hl*2 + half)*B + b
    for b in range(B):
        heads = [(pr, b, hl * 2 * D) for hl in range(2)]
        if fmode:
            attend_one_flash(heads, (pr, b, QW), Kc, Vc, fb, b, posf, ident, oT, b, B, 4 * B, sbuf_out=True)
        else:
            _attend_one(heads, (pr, b, QW), Kc, Vc, btu, b, NB, posf, io, ident, oT, b, B, 4 * B, B, sbuf_out=True)

    # ---------------- o_proj, K-split over pcores (rows [pid*512, +512)) ----------------
    wo = sb((128, 4, H), BF)    # k-tile kt = hl*2 + half
    nisa.dma_copy(dst=wo, src=Wo.ap(pattern=[[H, 128], [128 * H, 4], [1, H]], offset=pid * 4 * 128 * H))
    pso = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in range(H // 512)]
    for hl in range(2):
        for hf in range(2):
            col = (hl * 2 + hf) * B
            for ti in range(H // 512):
                nisa.nc_matmul(pso[ti][0:B, 0:512], oT[:, col:col + B], wo[:, hl * 2 + hf, ti * 512:(ti + 1) * 512])
    y = sb((B, H), BF)
    for ti in range(H // 512):
        nisa.tensor_copy(dst=y[0:B, ti * 512:(ti + 1) * 512], src=pso[ti][0:B, 0:512])
    return fold_halves(y, B, NC, pid)
