# SPDX-License-Identifier: Apache-2.0
"""Flash-decode (online softmax) attention core for the decode megakernel, used for context buckets above
2048 tokens (FLASH_MIN).

Replaces the bucket-wide score row + all-V-tiles-resident form of mk_gqa._attend_one, whose per-partition SBUF
use (~22 B x CTX) does not fit past ~8K tokens. Per request, the cached context (keys t < pos) is walked in chunks
of G = 8 KV blocks (CW = G*256 tokens), statically unrolled over the bucket (dead chunks are masked). Per chunk:
G indirect K/V block DMAs (block-table entry per block), scores [M, CW] fp32 = q.K^T * 1/16 + mask, running max m /
sum l / accumulator acc [M, 256] fp32 in SBUF:
  mnew = max(m, rowmax S); alpha = exp(m - mnew); P = exp(S - mnew); l = l*alpha + rowsum P
  acc = acc*alpha + bf16(P) @ V;  m = mnew
The new token (k/v from SBUF, never read back from the cache) is folded in last; m is seeded with the new token's
score (always valid), so fully masked chunk tails give exp(-1e30) = 0 and not 1.
"""
import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, sb

D, BS = 256, 256
SCALE = D ** -0.5
NEG = -1.0e30
FLASH_MIN = 2048   # context bucket (tokens) above which the flash core is used
FLASH_G = 8        # KV blocks per chunk


def flash_on(NB):
    """Use the flash core for a bucket of NB KV blocks?"""
    return NB * BS > FLASH_MIN


class FlashBT:
    """Per-kernel block-table tiles for the flash loop. bti [1, B*NB] int32 SBUF (already clamped)."""

    def __init__(self, bti, B, NB, rows):
        self.B, self.NB = B, NB
        self.G = max(1, min(FLASH_G, NB))
        self.NCH = (NB + self.G - 1) // self.G                  # chunks in the bucket
        self.CW = self.G * BS
        NP = self.NCH * self.G
        btp = nl.ndarray((1, B * NP), dtype=nl.int32, buffer=nl.sbuf)
        if NP != NB:
            nisa.memset(btp, 0)
        for b in range(B):
            nisa.tensor_copy(dst=btp[:, b * NP:b * NP + NB], src=bti[:, b * NB:(b + 1) * NB])
        self.NP = NP
        self.btu = nl.ndarray((1, B * NP), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=self.btu, src=btp)
        # chunk-local token index 0..CW-1 (fp32), shared by all requests
        self.io = sb((rows, self.CW))
        nisa.iota(dst=self.io, pattern=[[1, self.CW]], offset=0, channel_multiplier=0)


def _chunk(c, st):
    """One chunk of G KV blocks (c = static chunk index)."""
    fb, b, M, qT, Kc, Vc, ident, m, l, acc, posr = st
    G, CW = fb.G, fb.CW
    msk = sb((M, CW))
    nisa.tensor_scalar(dst=msk, data=fb.io[0:M, :], op0=nl.greater_equal, operand0=posr, op1=nl.multiply,
                       operand1=NEG)
    S = sb((M, CW))
    vts = []
    for g in range(G):
        iv = fb.btu.ap(pattern=[[fb.B * fb.NP, 1], [1, 1]], offset=b * fb.NP + c * G + g)
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
    nisa.tensor_scalar(dst=posr, data=posr, op0=nl.add, operand0=float(-CW))


def flash_context(fb, b, M, qT, Kc, Vc, ident, m, posf):
    """Walk request b's cached context. m [M, 1] fp32 = seed max (the new token's score). posf [*, B] fp32
    (cache length per request, all rows equal). Returns (m, l, acc) fp32 SBUF state."""
    l = sb((M, 1))
    nisa.memset(l, 0.0)
    acc = sb((M, D))
    nisa.memset(acc, 0.0)
    posr = sb((M, 1))
    nisa.tensor_copy(dst=posr, src=posf[0:M, b:b + 1])
    st = (fb, b, M, qT, Kc, Vc, ident, m, l, acc, posr)
    for c in range(fb.NCH):
        _chunk(c, st)
    return m, l, acc


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
    m, l, acc = flash_context(fb, b, nh, qT, Kc, Vc, ident, m, posf)
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
