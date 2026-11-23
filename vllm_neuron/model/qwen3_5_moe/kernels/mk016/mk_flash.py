"""flash-decode (online softmax) GQA attention core for the megakernel, T=1 decode and T=4 verify.

Replaces the bucket-wide score row + all-V-tiles-resident form of mk_gqa._attend_one / mk_gqa_t4._attend_t4
(per-partition SBUF ~22 B x CTX: does not fit past ~8K). Per request, the cached context (keys t < pos) is
walked in chunks of G KV blocks (CW = G*256 tokens). Per chunk: G indirect K/V block DMAs (block-table entry per
block), scores [M, CW] fp32 = q.K^T * 1/16 + mask, running max m / sum l / accumulator acc [M, 256] fp32 in SBUF:
  mnew = max(m, rowmax S); alpha = exp(m - mnew); P = exp(S - mnew); l = l*alpha + rowsum P
  acc = acc*alpha + bf16(P) @ V;  m = mnew
The new token(s) (k/v from SBUF, never read back from the cache) are folded in LAST; m is seeded with the new
token's score (always valid), so fully masked chunk tails give exp(-1e30) = 0 and not 1.

Loop forms (trace-time env MK_FLASH):
  1 = runtime-bounded nl.fori_loop over n = ceil(pos / CW) chunks (n from SBUF via scalar_load; short requests
      skip the bucket's dead chunks). Loop-carried state lives in SBUF only (no PSUM live across the loop:
      NCC_IBIR092); per-chunk block-table entries come from offset-0 compacted tiles (one per g) indexed
      by the loop register (static offset + register scalar_offset was mis-lowered on HW).
  2 = static unroll over all ceil(NB/G) chunks of the bucket (masking only), same body.
  auto (default) = MK_FLASH_AUTO (default 2 = static) when the bucket CTX = NB*256 > MK_FLASH_MIN (2048), else 0.
  0 = original kernel.
MK_FLASH_G = KV blocks per chunk (default 8, capped at NB).
MK_FLASH_SPLIT=1 (odd B, i.e. B=1): instead of the 2/2 head split, each pcore runs all 4 heads over half of the
chunks (chunk c on pcore c % 2: K/V read once per rank instead of twice), then the partial (m, l, acc) are
merged by LSE over one SBUF sendrecv; o_proj N-split.
"""
import os

import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, sb

D, BS = 256, 256
SCALE = D ** -0.5
NEG = -1.0e30


def flash_mode(NB):
    """'' (original kernel) | 'fori' | 'static' for a bucket of NB blocks."""
    m = os.environ.get("MK_FLASH", "auto")
    if m == "auto":
        # HW: static beats fori at full bucket (fori ~17-20 us/iter, no cross-chunk DMA overlap);
        # serving buckets the context, so auto = static. MK_FLASH_AUTO=1 selects fori.
        m = os.environ.get("MK_FLASH_AUTO", "2") if NB * BS > int(os.environ.get("MK_FLASH_MIN", "2048")) else "0"
    return {"0": "", "1": "fori", "2": "static"}[m]


class FlashBT:
    """Per-kernel block-table tiles for the flash loop. bti [1, B*NB] int32 SBUF (already clamped / offset).
    split=(NS, pid): this pcore walks chunks pid, pid+NS, ... (context split across NS pcores, LSE merge after)."""

    def __init__(self, bti, B, NB, mode, rows, split=(1, 0)):
        self.mode, self.B, self.NB = mode, B, NB
        self.NS, self.pid = split
        self.G = max(1, min(int(os.environ.get("MK_FLASH_G", "8")), NB))
        self.NCH = (NB + self.G - 1) // self.G                  # chunks in the bucket
        self.NCHL = (self.NCH + self.NS - 1) // self.NS         # chunk slots of this pcore
        self.CW = self.G * BS
        NP = self.NCHL * self.NS * self.G
        btp = nl.ndarray((1, B * NP), dtype=nl.int32, buffer=nl.sbuf)
        if NP != NB:
            nisa.memset(btp, 0)
        for b in range(B):
            nisa.tensor_copy(dst=btp[:, b * NP:b * NP + NB], src=bti[:, b * NB:(b + 1) * NB])
        self.NP = NP
        if mode == "static":
            self.btu = nl.ndarray((1, B * NP), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=self.btu, src=btp)
        else:
            self.btp = btp
        # chunk-local token index 0..CW-1 (fp32), shared by all requests
        self.io = sb((rows, self.CW))
        nisa.iota(dst=self.io, pattern=[[1, self.CW]], offset=0, channel_multiplier=0)
        # start token of this pcore's k-th chunk: (k*NS + pid) * CW
        self.thr = sb((1, self.NCHL))
        nisa.iota(dst=self.thr, pattern=[[self.NS * self.CW, self.NCHL]], offset=self.pid * self.CW,
                  channel_multiplier=0)
        # uniform trip count (HW: fori loops with different trip counts on the two pcores hang, NRT_TIMEOUT):
        # every loop runs n = #{k : k*NS*CW < max_b pos_b} (pid-0 thresholds); dead chunks are masked.
        # MK_FLASH_UNI: 2 (default) = max over the request pair (2i, 2i+1) that runs concurrently on the two pcores
        # (row split, even B), else this request; 1 = max over the whole batch; 0 = per-request (hangs on HW).
        self.uni = os.environ.get("MK_FLASH_UNI", "2")
        if self.uni != "0":
            self.thr0 = sb((1, self.NCHL))
            nisa.iota(dst=self.thr0, pattern=[[self.NS * self.CW, self.NCHL]], offset=0, channel_multiplier=0)
            self.nuni = None

    def request_tiles(self, b):
        """fori mode: G offset-0 tiles [1, NCHL] int32, tile g element k = bt[b, (k*NS + pid)*G + g]."""
        out = []
        for g in range(self.G):
            t = nl.ndarray((1, self.NCHL), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t, src=self.btp.ap(pattern=[[self.B * self.NP, 1], [self.NS * self.G, self.NCHL]],
                                                     offset=b * self.NP + self.pid * self.G + g))
            out.append(t)
        return out

    def trip_count(self, posf1, posall=None):
        """n = #{k : (k*NS + pid)*CW < pos} as an NkiScalar. posf1 = [1, 1] fp32 SBUF AP of pos.
        uni: n from max over posall [1, B] with pid-0 thresholds (same n for every loop on both pcores)."""
        if self.uni != "0" and posall is not None:
            pm = sb((1, 1))
            nisa.tensor_reduce(dst=pm, op=nl.maximum, data=posall, axis=1)
            gt = sb((1, self.NCHL))
            nisa.tensor_scalar(dst=gt, data=self.thr0, op0=nl.less, operand0=pm)
            nf = sb((1, 1))
            nisa.tensor_reduce(dst=nf, op=nl.add, data=gt, axis=1)
            ni = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ni, src=nf)
            return nisa.scalar_load(ni)
        gt = sb((1, self.NCHL))
        nisa.tensor_scalar(dst=gt, data=self.thr, op0=nl.less, operand0=posf1)
        nf = sb((1, 1))
        nisa.tensor_reduce(dst=nf, op=nl.add, data=gt, axis=1)
        ni = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ni, src=nf)
        return nisa.scalar_load(ni)


def _chunk(c, st):
    """One chunk of G KV blocks. c = python int (static) or loop register (fori)."""
    fb, b, M, qT, Kc, Vc, ident, m, l, acc, posr, btg = st
    G, CW = fb.G, fb.CW
    msk = sb((M, CW))
    nisa.tensor_scalar(dst=msk, data=fb.io[0:M, :], op0=nl.greater_equal, operand0=posr, op1=nl.multiply,
                       operand1=NEG)
    S = sb((M, CW))
    vts = []
    for g in range(G):
        if fb.mode == "static":
            iv = fb.btu.ap(pattern=[[fb.B * fb.NP, 1], [1, 1]], offset=b * fb.NP + c * G + g)
        else:
            iv = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=iv, src=btg[g].ap(pattern=[[fb.NCHL, 1], [1, 1]], offset=0, scalar_offset=c,
                                                 indirect_dim=1))
        kt = sb((128, 2, D), BF)
        nisa.dma_copy(dst=kt, src=Kc.ap(pattern=[[D, 128], [128 * D, 2], [1, D]], offset=0, scalar_offset=iv,
                                        indirect_dim=0))
        vt = sb((128, 2, D), BF)
        nisa.dma_copy(dst=vt, src=Vc.ap(pattern=[[D, 128], [128 * D, 2], [1, D]], offset=0, scalar_offset=iv,
                                        indirect_dim=0))
        vts.append(vt)
        pk = nl.ndarray((128, 2, BS), dtype=F32, buffer=nl.psum)   # [d, half, t]
        for hf in range(2):
            for sub in range(2):
                nisa.nc_matmul(pk[:, hf, sub * 128:(sub + 1) * 128], kt[:, sub, hf * 128:(hf + 1) * 128], ident)
        kT = sb((128, 2, BS), BF)
        nisa.tensor_copy(dst=kT, src=pk)
        pss = nl.ndarray((M, BS), dtype=F32, buffer=nl.psum)
        for hf in range(2):
            nisa.nc_matmul(pss, qT[:, hf, :], kT[:, hf, :])
        nisa.scalar_tensor_tensor(dst=S[:, g * BS:(g + 1) * BS], data=pss, op0=nl.multiply, operand0=SCALE,
                                  op1=nl.add, operand1=msk[:, g * BS:(g + 1) * BS])
    mloc = sb((M, 1))
    nisa.tensor_reduce(dst=mloc, op=nl.maximum, data=S, axis=1)
    mnew = sb((M, 1))
    nisa.tensor_tensor(dst=mnew, data1=m, data2=mloc, op=nl.maximum)
    alpha = sb((M, 1))
    nisa.tensor_tensor(dst=alpha, data1=m, data2=mnew, op=nl.subtract)
    nisa.activation(dst=alpha, data=alpha, op=nl.exp)
    nmx = sb((M, 1))
    nisa.tensor_scalar(dst=nmx, data=mnew, op0=nl.multiply, operand0=-1.0)
    P = sb((M, CW))
    nisa.activation(dst=P, data=S, op=nl.exp, bias=nmx)
    ls = sb((M, 1))
    nisa.tensor_reduce(dst=ls, op=nl.add, data=P, axis=1)
    nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=ls)
    Pb = sb((M, CW), BF)
    nisa.tensor_copy(dst=Pb, src=P)
    po = nl.ndarray((M, D), dtype=F32, buffer=nl.psum)
    for g in range(G):
        pp = nl.ndarray((128, 2, M), dtype=F32, buffer=nl.psum)
        for sub in range(2):
            c0 = g * BS + sub * 128
            nisa.nc_matmul(pp[:, sub, :], Pb[0:M, c0:c0 + 128], ident[0:M, 0:M])
        pT = sb((128, 2, M), BF)
        nisa.tensor_copy(dst=pT, src=pp)
        for sub in range(2):
            nisa.nc_matmul(po, pT[:, sub, :], vts[g][:, sub, :])
    nisa.scalar_tensor_tensor(dst=acc, data=acc, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=po)
    nisa.tensor_copy(dst=m, src=mnew)
    nisa.tensor_scalar(dst=posr, data=posr, op0=nl.add, operand0=float(-CW * fb.NS))


def flash_context(fb, b, M, qT, Kc, Vc, ident, m, posf, rows_pos):
    """Walk request b's cached context. m [M, 1] fp32 = seed max (the new token's score). posf [*, B] fp32
    (cache length per request, all rows equal). Returns (m, l, acc) fp32 SBUF state."""
    l = sb((M, 1))
    nisa.memset(l, 0.0)
    acc = sb((M, D))
    nisa.memset(acc, 0.0)
    posr = sb((M, 1))
    nisa.tensor_copy(dst=posr, src=posf[0:M, b:b + 1])
    if fb.pid:
        nisa.tensor_scalar(dst=posr, data=posr, op0=nl.add, operand0=float(-fb.pid * fb.CW))
    if fb.mode == "static":
        st = (fb, b, M, qT, Kc, Vc, ident, m, l, acc, posr, None)
        for c in range(fb.pid, fb.NCH, fb.NS):
            _chunk(c, st)
    else:
        btg = fb.request_tiles(b)
        if fb.uni == "1":
            grp = posf[0:1, 0:fb.B]
        elif fb.NS == 1 and fb.B % 2 == 0:
            grp = posf[0:1, (b // 2) * 2:(b // 2) * 2 + 2]
        elif fb.NS > 1:
            grp = posf[0:1, b:b + 1]   # context split: pid-0 count on both pcores (pid 1 may need one fewer)
        else:
            grp = None   # head split: both pcores walk the same request, same n already
        n = fb.trip_count(posf[0:1, b:b + 1], grp)
        st = (fb, b, M, qT, Kc, Vc, ident, m, l, acc, posr, btg)
        nl.fori_loop(0, n, lambda c: _chunk(c, st))
    if fb.NS > 1:
        _merge_peer(fb, M, m, l, acc)
    return m, l, acc


def _merge_peer(fb, M, m, l, acc):
    """LSE merge of the two pcores' partial (m, l, acc) via one SBUF<->SBUF sendrecv; both end identical."""
    W = D + 2
    mine = sb((M, W))
    nisa.tensor_copy(dst=mine[:, 0:1], src=m)
    nisa.tensor_copy(dst=mine[:, 1:2], src=l)
    nisa.tensor_copy(dst=mine[:, 2:W], src=acc)
    peer = sb((M, W))
    other = 1 - fb.pid
    nisa.sendrecv(src=mine, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
    mt = sb((M, 1))
    nisa.tensor_tensor(dst=mt, data1=m, data2=peer[:, 0:1], op=nl.maximum)
    a0 = sb((M, 1))
    nisa.tensor_tensor(dst=a0, data1=m, data2=mt, op=nl.subtract)
    nisa.activation(dst=a0, data=a0, op=nl.exp)
    a1 = sb((M, 1))
    nisa.tensor_tensor(dst=a1, data1=peer[:, 0:1], data2=mt, op=nl.subtract)
    nisa.activation(dst=a1, data=a1, op=nl.exp)
    t1 = sb((M, 1))
    nisa.tensor_tensor(dst=t1, data1=peer[:, 1:2], data2=a1, op=nl.multiply)
    nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=a0, op1=nl.add, operand1=t1)
    ta = sb((M, D))
    nisa.tensor_scalar(dst=ta, data=peer[:, 2:W], op0=nl.multiply, operand0=a1)
    nisa.scalar_tensor_tensor(dst=acc, data=acc, op0=nl.multiply, operand0=a0, op1=nl.add, operand1=ta)
    nisa.tensor_copy(dst=m, src=mt)


def fold_new(m, l, acc, snew, M, TN):
    """Fold the TN new-token scores snew [M, TN] fp32 (scaled + causal-masked) in last. Returns (pn fp32 [M, TN]
    probabilities w.r.t. the final max) and rescales l / acc in place (acc still lacks the new-token PV term)."""
    mx = sb((M, 1))
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=snew, axis=1)
    mf = sb((M, 1))
    nisa.tensor_tensor(dst=mf, data1=m, data2=mx, op=nl.maximum)
    alpha = sb((M, 1))
    nisa.tensor_tensor(dst=alpha, data1=m, data2=mf, op=nl.subtract)
    nisa.activation(dst=alpha, data=alpha, op=nl.exp)
    nmx = sb((M, 1))
    nisa.tensor_scalar(dst=nmx, data=mf, op0=nl.multiply, operand0=-1.0)
    pn = sb((M, TN))
    nisa.activation(dst=pn, data=snew, op=nl.exp, bias=nmx)
    ls = sb((M, 1))
    nisa.tensor_reduce(dst=ls, op=nl.add, data=pn, axis=1)
    nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=ls)
    nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply, operand0=alpha)
    return pn


def attend_one_flash(heads, kv_off, Kc, Vc, fb, b, posf, ident, odst, col, cstride, ldo, sbuf_out=False):
    """Flash version of mk_gqa._attend_one (same arguments / outputs, block table via FlashBT)."""
    nh = len(heads)
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
    qk = sb((nh, D))
    nisa.tensor_tensor(dst=qk, data1=q2, data2=kn, op=nl.multiply)
    sn1 = sb((nh, 1))
    nisa.tensor_reduce(dst=sn1, op=nl.add, data=qk, axis=1)
    snew = sb((nh, 1))
    nisa.tensor_scalar(dst=snew, data=sn1, op0=nl.multiply, operand0=SCALE)
    m = sb((nh, 1))
    nisa.tensor_copy(dst=m, src=snew)
    m, l, acc = flash_context(fb, b, nh, qT, Kc, Vc, ident, m, posf, nh)
    pn = fold_new(m, l, acc, snew, nh, 1)
    o = sb((nh, D))
    nisa.tensor_scalar(dst=o, data=vn, op0=nl.multiply, operand0=pn)
    nisa.tensor_tensor(dst=o, data1=o, data2=acc, op=nl.add)
    rl = sb((nh, 1))
    nisa.reciprocal(dst=rl, data=l)
    ob = sb((nh, D), BF)
    nisa.tensor_scalar(dst=ob, data=o, op0=nl.multiply, operand0=rl)
    sg = sb((nh, D), BF)
    nisa.activation(dst=sg, data=g2, op=nl.sigmoid)
    og = sb((nh, D), BF)
    nisa.tensor_tensor(dst=og, data1=ob, data2=sg, op=nl.multiply)
    pt = nl.ndarray((128, 2, nh), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(pt[:, hf, :], og[0:nh, hf * 128:(hf + 1) * 128], ident[0:nh, 0:nh])
    ptb = sb((128, nh, 2), BF)
    for hf in range(2):
        nisa.tensor_copy(dst=ptb[:, :, hf], src=pt[:, hf, :])
    dst = odst.ap(pattern=[[ldo, 128], [cstride, nh * 2]], offset=col)
    if sbuf_out:
        nisa.tensor_copy(dst=dst, src=ptb)
    else:
        nisa.dma_copy(dst=dst, src=ptb)


def attend_t4_flash(heads, kv, Kc, Vc, fb, b, posf, cm, ident, odst, col, cstride, ldo, T):
    """Flash version of mk_gqa_t4._attend_t4 (same arguments / outputs, block table via FlashBT)."""
    nh = len(heads)
    M = nh * T
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
    psn = nl.ndarray((M, T), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(psn, qT[:, hf, :], knT[:, hf, :])
    snew = sb((M, T))
    nisa.scalar_tensor_tensor(dst=snew, data=psn, op0=nl.multiply, operand0=SCALE, op1=nl.add, operand1=cm[0:M, :])
    m = sb((M, 1))
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=snew, axis=1)
    m, l, acc = flash_context(fb, b, M, qT, Kc, Vc, ident, m, posf, M)
    pn = fold_new(m, l, acc, snew, M, T)
    pnb = sb((M, T), BF)
    nisa.tensor_copy(dst=pnb, src=pn)
    ppn = nl.ndarray((T, M), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ppn, pnb[0:M, 0:T], ident[0:M, 0:M])
    pnT = sb((T, M), BF)
    nisa.tensor_copy(dst=pnT, src=ppn)
    po = nl.ndarray((M, D), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(po, pnT, vn)
    o = sb((M, D))
    nisa.tensor_tensor(dst=o, data1=acc, data2=po, op=nl.add)
    rl = sb((M, 1))
    nisa.reciprocal(dst=rl, data=l)
    ob = sb((M, D), BF)
    nisa.tensor_scalar(dst=ob, data=o, op0=nl.multiply, operand0=rl)
    sg = sb((M, D), BF)
    nisa.activation(dst=sg, data=g2, op=nl.sigmoid)
    og = sb((M, D), BF)
    nisa.tensor_tensor(dst=og, data1=ob, data2=sg, op=nl.multiply)
    pt = nl.ndarray((128, 2, M), dtype=F32, buffer=nl.psum)
    for hf in range(2):
        nisa.nc_matmul(pt[:, hf, :], og[0:M, hf * 128:(hf + 1) * 128], ident[0:M, 0:M])
    ptb = sb((128, 2, M), BF)
    nisa.tensor_copy(dst=ptb, src=pt)
    dst = odst.ap(pattern=[[ldo, 128], [cstride, 2], [2 * cstride, nh], [1, T]], offset=col)
    nisa.tensor_copy(dst=dst, src=ptb)
