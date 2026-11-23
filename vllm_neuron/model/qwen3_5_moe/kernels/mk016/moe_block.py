"""MoE decode block for one TP=4 rank (port layout, weights read in place), LNC=2.

Per rank (model.py:1091-1109, ep_degree=1): router [H, E] fp32 (replicated), experts gate_up [E, H, 2I] /
down [E, I, H] with I = 512/TP = 128, shared expert gate_up [H, 2I_sh] / down [I_sh, H] / gate [H, 1].
Math = forward_decode_gather (model.py:1281-1314) + _shared (model.py:1169-1173):
  logits = x @ Wr (fp32) -> softmax -> top-8 -> renorm      (== exp(l - lmax) / sum over the top 8)
  y = sum_k w_k * (silu(x Wg_e) * (x Wu_e)) Wd_e  +  sigmoid(x Wsg) * (silu(x Wsg_g) * (x Wsu)) Wsd
Output: this pcore's H half of the rank partial ([B, H/NC] bf16); the caller all-reduces over TP.

LNC=2 split (no control flow depends on program_id, only offsets):
  router + shared gate_up: K-split (pcore c contracts rows p*16 + [8c, 8c+8)), partials swapped by sendrecv
    and added, so both pcores hold bit-identical logits and pick the same experts.
  routed experts: slot s = b*8 + k; pcore c runs slots 2i + c. Each slot DMAs its expert by runtime index
    (scalar_offset on dim 0 of the HBM stack, nkilib selective_expert_impl.py:250-262 pattern).
    Weighted sum = PSUM accumulation of down matmuls whose stationary is the [128, B] matrix with column b =
    w_s * h_s (zeros elsewhere), so every slot accumulates straight into the token's output row.
  shared down: N-split (pcore c computes its own H half).
"""
import os

import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, H, KT, evac, rows_to_pmaj, sb

TOPK = 8
# Optimization knobs; each is an A/B switch.
FUSE_SWIGLU = os.environ.get("MK_FUSE_SWIGLU", "1") == "1"
TAIL_ONE_MM = os.environ.get("MK_TAIL_ONE_MM", "0") == "1"
SHARED_FIRST = os.environ.get("MK_SHARED_FIRST", "1") == "1"


def moe_static_load(Wr, Wsgu, Wsg, NC, pid, after=None):
    """Router (fp32) + shared gate_up + shared gate weights of this pcore's K half -> SBUF (M4 prefetch unit)."""
    E, IS2, KC = Wr.shape[1], Wsgu.shape[1], KT // NC
    kw = {"dge_mode": nisa.dge_mode.hwdge} if os.environ.get("MK_DGE") == "hw" else {}
    wr = sb((128, KC, E))
    i_ = nisa.dma_copy(dst=wr, src=Wr.ap(pattern=[[KT * E, 128], [E, KC], [1, E]], offset=pid * KC * E), **kw)
    if after is not None:
        i_.depends_on(after, kind=nisa.dependency.runtime)
    wsgu = sb((128, KC, IS2), BF)
    i_ = nisa.dma_copy(dst=wsgu, src=Wsgu.ap(pattern=[[KT * IS2, 128], [IS2, KC], [1, IS2]], offset=pid * KC * IS2), **kw)
    if after is not None:
        i_.depends_on(after, kind=nisa.dependency.runtime)
    wsg = sb((128, KC, 1), BF)
    i_ = nisa.dma_copy(dst=wsg, src=Wsg.ap(pattern=[[KT, 128], [1, KC], [1, 1]], offset=pid * KC), **kw)
    if after is not None:
        i_.depends_on(after, kind=nisa.dependency.runtime)
    return wr, wsgu, wsg


def dedup_on(R):
    """MK_DEDUP=1 (trace time) enables the unique-expert path for R >= MK_DEDUP_MIN rows (default 8)."""
    return os.environ.get("MK_DEDUP", "0") == "1" and R >= int(os.environ.get("MK_DEDUP_MIN", "8"))


def dedup_bufs(Wgu, Wd):
    """NBUF rotating expert weight buffers, zeroed once per kernel. A unique slot past the runtime count
    skips its DMA (oob), so its buffer keeps the previous (finite) expert or zeros; its routing weight is 0."""
    KTI2, I, Hh = Wgu.shape[1] * Wgu.shape[2] // 128, Wd.shape[1], Wd.shape[2]
    nb = int(os.environ.get("MK_DEDUP_NBUF", "4"))
    bufs = []
    for _ in range(nb):
        wgu = sb((128, KTI2), BF)
        nisa.memset(wgu, 0.0)
        wd = sb((I, Hh), BF)
        nisa.memset(wd, 0.0)
        bufs.append((wgu, wd))
    return bufs  # bufs[0] doubles as the overflow-tail buffer (the tail runs after all static slots)


def routed_dedup(full, tv, nmax, er, xP, Wgu, Wd, ident, py, R, E, NC, pid, ebuf, thr=None):
    """routed experts with each unique expert loaded once for all R rows.
      dense [R, E] = renormed top-8 weight (exp(l - lmax) / sum, 0 off the top 8: mask l >= 8th largest)
      present [1, E] = column sums -> nonzero_with_count -> u [1, E+1]: unique ids ascending, padded with E (out of
        bounds: the slot's DMAs are skipped), count last.
      Wsel [R, CAP] = dense @ onehot(u)  (the [rows, unique] routing-weight mask)
      unique slot j (pcore c runs j = NC*i + c, CAP = min(8R, E) slots, statically unrolled):
        gu [R, 2I] = x @ Wgu[u_j]; h = silu(g) * u; h *= Wsel[:, j]; py += h @ Wd[u_j]   (dense over rows)."""
    I2 = Wgu.shape[2]
    I = I2 // 2
    CAPT = min(TOPK * R, E)
    # static unique-slot cap: ~p80-p90 of real GSM8K traffic; overflow -> runtime tail loop
    dflt = {8: 56, 16: 96, 32: 128, 64: 192}.get(R, 256)  # R=16 96, R=64 192 (HW A/B)
    CAP = min(CAPT, int(os.environ.get(f"MK_DEDUP_CAP{R}", os.environ.get("MK_DEDUP_CAP", str(dflt)))))
    CAP -= CAP % NC
    EC = min(128, E)
    NE = E // EC
    # dense routing weights
    # thr [R, 1]: null router -> max(8th largest, null logit), so null-dropped experts are absent from the
    # presence list (their weights are never DMA'd)
    dm = sb((R, E))
    nisa.tensor_scalar(dst=dm, data=full[:, 0:E], op0=nl.greater_equal,
                       operand0=tv[:, TOPK - 1:TOPK] if thr is None else thr)
    de = sb((R, E))
    nisa.activation(dst=de, data=full[:, 0:E], op=nl.exp, bias=nmax)
    dense = sb((R, E))
    nisa.scalar_tensor_tensor(dst=dense, data=de, op0=nl.multiply, operand0=er, op1=nl.multiply, operand1=dm)
    # presence + unique list
    onesr = sb((R, 1))
    nisa.memset(onesr, 1.0)
    ppr = nl.ndarray((1, E), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ppr, onesr, dm)
    pres = sb((1, E))
    nisa.tensor_copy(dst=pres, src=ppr)
    u = nl.ndarray((1, E + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=u, src=pres, index_offset=0, padding_val=E)
    # one-hot [E, CAP] and Wsel [R, CAP]
    uf = sb((1, CAPT))
    nisa.tensor_copy(dst=uf, src=u[:, 0:CAPT])
    ones1 = sb((1, 128))
    nisa.memset(ones1, 1.0)
    pub = nl.ndarray((128, CAPT), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(pub, ones1, uf)
    ub = sb((128, CAPT))
    nisa.tensor_copy(dst=ub, src=pub)
    identf = sb((R, R))
    nisa.tensor_copy(dst=identf, src=ident[0:R, 0:R])
    pws = nl.ndarray((R, CAPT), dtype=F32, buffer=nl.psum)
    for h in range(NE):
        io = sb((EC, 1))
        nisa.iota(io, [[0, 1]], offset=EC * h, channel_multiplier=1)
        oh = sb((EC, CAPT))
        nisa.tensor_scalar(dst=oh, data=ub[0:EC, :], op0=nl.equal, operand0=io)
        pdt = nl.ndarray((EC, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pdt, dense[:, h * EC:(h + 1) * EC], identf)
        dT = sb((EC, R))
        nisa.tensor_copy(dst=dT, src=pdt)
        nisa.nc_matmul(pws, dT, oh)
    wsel = sb((R, CAPT))
    nisa.tensor_copy(dst=wsel, src=pws)
    # unique slots
    bufs = ebuf
    for i in range(CAP // NC):
        j = NC * i + pid
        iv = u.ap(pattern=[[E + 1, 1], [1, 1]], offset=j)
        wgu, wd = bufs[i % len(bufs)]
        nisa.dma_copy(dst=wgu, src=Wgu.ap(pattern=[[KT * I2, 128], [1, KT * I2]], offset=0, scalar_offset=iv,
                                          indirect_dim=0), oob_mode=nisa.oob_mode.skip)
        nisa.dma_copy(dst=wd, src=Wd.ap(pattern=[[H, I], [1, H]], offset=0, scalar_offset=iv, indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip)
        pgu = nl.ndarray((R, I2), dtype=F32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(pgu, xP[:, kt * R:(kt + 1) * R], wgu[:, kt * I2:(kt + 1) * I2])
        hw = sb((R, I), BF)
        if FUSE_SWIGLU:  # silu straight from PSUM, one STT for *up*w (drops 2 DVE ops)
            sg = sb((R, I), BF)
            nisa.activation(dst=sg, data=pgu[:, 0:I], op=nl.silu)
            nisa.scalar_tensor_tensor(dst=hw, data=pgu[:, I:I2], op0=nl.multiply, operand0=wsel[:, j:j + 1],
                                      op1=nl.multiply, operand1=sg)
        else:
            gu = sb((R, I2), BF)
            nisa.tensor_copy(dst=gu, src=pgu)
            sg = sb((R, I), BF)
            nisa.activation(dst=sg, data=gu[:, 0:I], op=nl.silu)
            hr = sb((R, I), BF)
            nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, I:I2], op=nl.multiply)
            nisa.tensor_scalar(dst=hw, data=hr, op0=nl.multiply, operand0=wsel[:, j:j + 1])
        ph = nl.ndarray((I, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ph, hw, ident[0:R, 0:R])
        hs = sb((I, R), BF)
        nisa.tensor_copy(dst=hs, src=ph)
        for t in range(H // 512):
            nisa.nc_matmul(py[t], hs, wd[:, t * 512:(t + 1) * 512])
    if CAP >= CAPT:
        return None
    # overflow tail (unique count > CAP): runtime loop, one unique expert per pcore per iteration, partials
    # accumulated in SBUF fp32 (no loop-carried PSUM). Trip count n = #{k : count > CAP + NC*k} (exact, no division).
    NX = (CAPT - CAP + NC - 1) // NC
    cntf = sb((1, 1))
    nisa.tensor_copy(dst=cntf, src=u[:, E:E + 1])
    thr = sb((1, NX))
    nisa.iota(thr, [[NC, NX]], offset=CAP, channel_multiplier=0)
    gt = sb((1, NX))
    nisa.tensor_scalar(dst=gt, data=thr, op0=nl.less, operand0=cntf)
    nf = sb((1, 1))
    nisa.tensor_reduce(dst=nf, op=nl.add, data=gt, axis=1)
    ni = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ni, src=nf)
    yacc = sb((R, H))  # PSUM must not be live across the runtime loop (NCC_IBIR092): evacuate py first
    for t in range(H // 512):
        nisa.tensor_copy(dst=yacc[:, t * 512:(t + 1) * 512], src=py[t])
    n = nisa.scalar_load(ni)
    twgu, twd = ebuf[0]  # finite (zeros or an earlier expert): a skipped tail DMA must not leave garbage
    # compact this pcore's tail ids / weights (offset-0 tiles: on HW a static offset + SBUF scalar_offset was
    # mis-lowered, G=4 probe)
    ut = nl.ndarray((1, NX), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ut, src=u.ap(pattern=[[E + 1, 1], [NC, NX]], offset=CAP + pid))
    wt = sb((1, NX * R))  # tail weights on one partition: [k*R:(k+1)*R] = Wsel[:, CAP + pid + NC*k]
    if TAIL_ONE_MM:  # one strided transpose matmul instead of NX one-row matmuls
        wst = sb((R, NX))
        nisa.tensor_copy(dst=wst, src=wsel.ap(pattern=[[CAPT, R], [NC, NX]], offset=CAP + pid))
        pwt = nl.ndarray((NX, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pwt, wst, identf)
        wtc = sb((NX, R))
        nisa.tensor_copy(dst=wtc, src=pwt)
        for k in range(NX):  # partition k -> free [k*R:(k+1)*R] on partition 0 (SBUF->SBUF DMA)
            nisa.dma_copy(dst=wt[:, k * R:(k + 1) * R], src=wtc[k:k + 1, :])
    else:
        for k in range(NX):
            pwt = nl.ndarray((1, R), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(pwt, wsel[:, CAP + pid + NC * k:CAP + pid + NC * k + 1], identf)
            nisa.tensor_copy(dst=wt[:, k * R:(k + 1) * R], src=pwt)
    ctx = (ut, wt, xP, Wgu, Wd, ident, yacc, R, NX, NC, pid, I, I2, twgu, twd)
    nl.fori_loop(0, n, lambda i: _dedup_tail_body(i, ctx))
    return yacc


def _dedup_tail_body(i, ctx):
    ut, wt, xP, Wgu, Wd, ident, yacc, R, NX, NC, pid, I, I2, wgu, wd = ctx
    e1 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=e1, src=ut.ap(pattern=[[NX, 1], [1, 1]], offset=0, scalar_offset=i, indirect_dim=1))
    wr1 = sb((1, R))
    nisa.dma_copy(dst=wr1, src=wt.ap(pattern=[[NX * R, 1], [1, R]], offset=0, scalar_offset=i * R, indirect_dim=1))
    pwc = nl.ndarray((R, 1), dtype=F32, buffer=nl.psum)
    one1 = sb((1, 1))
    nisa.memset(one1, 1.0)
    nisa.nc_matmul(pwc, wr1, one1)
    wc = sb((R, 1))
    nisa.tensor_copy(dst=wc, src=pwc)
    nisa.dma_copy(dst=wgu, src=Wgu.ap(pattern=[[KT * I2, 128], [1, KT * I2]], offset=0, scalar_offset=e1,
                                      indirect_dim=0), oob_mode=nisa.oob_mode.skip)
    nisa.dma_copy(dst=wd, src=Wd.ap(pattern=[[H, I], [1, H]], offset=0, scalar_offset=e1, indirect_dim=0),
                  oob_mode=nisa.oob_mode.skip)
    pgu = nl.ndarray((R, I2), dtype=F32, buffer=nl.psum)
    for kt in range(KT):
        nisa.nc_matmul(pgu, xP[:, kt * R:(kt + 1) * R], wgu[:, kt * I2:(kt + 1) * I2])
    gu = sb((R, I2), BF)
    nisa.tensor_copy(dst=gu, src=pgu)
    sg = sb((R, I), BF)
    nisa.activation(dst=sg, data=gu[:, 0:I], op=nl.silu)
    hr = sb((R, I), BF)
    nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, I:I2], op=nl.multiply)
    hw = sb((R, I), BF)
    nisa.tensor_scalar(dst=hw, data=hr, op0=nl.multiply, operand0=wc)
    ph = nl.ndarray((I, R), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ph, hw, ident[0:R, 0:R])
    hs = sb((I, R), BF)
    nisa.tensor_copy(dst=hs, src=ph)
    for t in range(H // 512):
        pt = nl.ndarray((R, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pt, hs, wd[:, t * 512:(t + 1) * 512])
        nisa.tensor_tensor(dst=yacc[:, t * 512:(t + 1) * 512], data1=yacc[:, t * 512:(t + 1) * 512], data2=pt,
                           op=nl.add)


def moe_block(xn, Wr, Wgu, Wd, Wsgu, Wsd, Wsg, ident, ones_f, bmask, B, NC, pid, tag="", pre=None, fold=None, wcol=None, rk=None,
              ebuf=None):
    """xn [B, H] bf16 SBUF (post-attention RMSNorm output). ones_f [B, 128] fp32 ones; bmask [B, 8B] fp32
    block-diagonal ones (bmask[b, b*8 + k] = 1). Returns yh [B, H/NC] fp32 SBUF (pcore half, rank partial).
    fold=(rsc [B, 1] fp32, ln2_w HBM [H] fp32): xn is the UN-normed residual; router / shared gate_up /
    shared gate weights carry the folded norm weight and their outputs are scaled by rsc; the routed experts get
    the normed rows xn * rsc * w, computed after the router matmuls are issued."""
    ER = Wr.shape[1]     # router width: E, or E + 1 with the Isaac-0.5 null-expert row (last column)
    E = Wgu.shape[0]     # real experts
    NULL = ER > E
    I2 = Wgu.shape[2]
    I = I2 // 2
    IS2 = Wsgu.shape[1]
    IS = IS2 // 2
    KC = KT // NC        # k-tiles contracted by this pcore (K-split)
    HW = H // NC         # this pcore's H half
    NSL = TOPK * B // NC  # routed slots per pcore

    xP = rows_to_pmaj(xn, ident, B, BF, wcol=wcol, rk=rk)  # [128, KT*B] bf16, element (p, kt*B+b) = x[b, p*KT+kt]
    xPf = sb((128, KT * B))
    nisa.tensor_copy(dst=xPf, src=xP)
    rsc = fold[0] if fold is not None else None

    # ---- router (fp32) + shared gate + shared gate_up, K-split over pcores ----
    wr, wsgu, wsg = pre if pre is not None else moe_static_load(Wr, Wsgu, Wsg, NC, pid)
    NP = ER + IS2 + 1
    pr = nl.ndarray((B, ER), dtype=F32, buffer=nl.psum)
    psg = nl.ndarray((B, IS2), dtype=F32, buffer=nl.psum)
    pgt = nl.ndarray((B, 1), dtype=F32, buffer=nl.psum)
    for kk in range(KC):
        col = nl.ds((pid * KC + kk) * B, B)
        nisa.nc_matmul(pr, xPf[:, col], wr[:, kk, :])
        nisa.nc_matmul(psg, xP[:, col], wsgu[:, kk, :])
        nisa.nc_matmul(pgt, xP[:, col], wsg[:, kk, :])
    part = sb((B, NP))
    evac(part[:, 0:ER], pr, rsc)
    evac(part[:, ER:ER + IS2], psg, rsc)
    evac(part[:, ER + IS2:NP], pgt, rsc)
    if fold is not None:  # routed experts read the normed rows (norm weight NOT folded into the expert stacks)
        xnn = sb((B, H), BF)
        if fold[1] is not None:  # MK_FOLD=1: xn = h (un-normed)
            w2 = sb((B, H))
            nisa.dma_copy(dst=w2, src=fold[1].ap(pattern=[[0, B], [1, H]], offset=0))
            nisa.scalar_tensor_tensor(dst=xnn, data=xn, op0=nl.multiply, operand0=rsc, op1=nl.multiply, operand1=w2)
        else:  # MK_FOLD=2: xn = h * w already
            nisa.tensor_scalar(dst=xnn, data=xn, op0=nl.multiply, operand0=rsc)
        xP = rows_to_pmaj(xnn, ident, B, BF)
    if NC > 1:
        other = 1 - pid
        peer = sb((B, NP))
        nisa.sendrecv(src=part, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=1)
        full = sb((B, NP))
        nisa.tensor_tensor(dst=full, data1=part, data2=peer, op=nl.add)
    else:
        full = part

    # ---- top-8 + renormalized softmax weights (identical on both pcores) ----
    tv = sb((B, TOPK))
    nisa.max8(dst=tv, src=full[:, 0:E])
    ti = nl.ndarray((B, TOPK), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=ti, data=full[:, 0:E], vals=tv)
    nmax = sb((B, 1))
    nisa.tensor_scalar(dst=nmax, data=tv[:, 0:1], op0=nl.multiply, operand0=-1.0)
    ev = sb((B, TOPK))
    nisa.activation(dst=ev, data=tv, op=nl.exp, bias=nmax)
    dthr = None
    if NULL:  # Isaac-0.5: a top-8 pick survives iff its logit >= the null logit (ties -> real expert)
        km = sb((B, TOPK))
        nisa.tensor_scalar(dst=km, data=tv, op0=nl.greater_equal, operand0=full[:, E:E + 1])
        evm = sb((B, TOPK))
        nisa.tensor_tensor(dst=evm, data1=ev, data2=km, op=nl.multiply)
        ev = evm
        dthr = sb((B, 1))
        nisa.tensor_tensor(dst=dthr, data1=tv[:, TOPK - 1:TOPK], data2=full[:, E:E + 1], op=nl.maximum)
    es = sb((B, 1))
    nisa.tensor_reduce(dst=es, op=nl.add, data=ev, axis=1)
    if NULL:  # all-null rows: es = 0 -> weights 0 (shared expert only)
        es2 = sb((B, 1))
        nisa.tensor_scalar(dst=es2, data=es, op0=nl.maximum, operand0=1e-30)
        es = es2
    er = sb((B, 1))
    nisa.reciprocal(dst=er, data=es)
    aff = sb((B, TOPK))
    nisa.tensor_scalar(dst=aff, data=ev, op0=nl.multiply, operand0=er)
    # broadcast affinities to all partitions: wb[p, b*8+k] = aff[b, k]
    # (hit-list path only; skipped under dedup -- at B >= 64 the [128, 8B] broadcast exceeds one matmul)
    if not dedup_on(B):
        ablk = sb((B, TOPK * B))
        nisa.tensor_tensor(dst=ablk, data1=aff.ap(pattern=[[TOPK, B], [0, B], [1, TOPK]], offset=0), data2=bmask,
                           op=nl.multiply)
        pwb = nl.ndarray((128, TOPK * B), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pwb, ones_f, ablk)
        wb = sb((128, TOPK * B))
        nisa.tensor_copy(dst=wb, src=pwb)

    # ---- shared expert first (MK_SHARED_FIRST=1): its Wsd DMA and matmuls do not
    # depend on routing, so issue them before the routed loop to fill the PE gap while the first expert loads.
    if SHARED_FIRST:
        gus = sb((B, IS2), BF)
        nisa.tensor_copy(dst=gus, src=full[:, ER:ER + IS2])
        ssg = sb((B, IS), BF)
        nisa.activation(dst=ssg, data=gus[:, 0:IS], op=nl.silu)
        hsh = sb((B, IS), BF)
        nisa.tensor_tensor(dst=hsh, data1=ssg, data2=gus[:, IS:IS2], op=nl.multiply)
        phs = nl.ndarray((IS, B), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(phs, hsh, ident[0:B, 0:B])
        hcol = sb((IS, B), BF)
        nisa.tensor_copy(dst=hcol, src=phs)
        wsd = sb((IS, HW), BF)
        nisa.dma_copy(dst=wsd, src=Wsd.ap(pattern=[[H, IS], [1, HW]], offset=pid * HW))
        gate = sb((B, 1))
        nisa.activation(dst=gate, data=full[:, ER + IS2:NP], op=nl.sigmoid)
        ysh_all = sb((B, HW))
        for t in range(HW // 512):
            pys = nl.ndarray((B, 512), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(pys, hcol, wsd[:, t * 512:(t + 1) * 512])
            nisa.tensor_scalar(dst=ysh_all[:, t * 512:(t + 1) * 512], data=pys, op0=nl.multiply, operand0=gate)

    NT = H // 512
    py = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in range(NT)]
    yacc = None
    if dedup_on(B):
        yacc = routed_dedup(full, tv, nmax, er, xP, Wgu, Wd, ident, py, B, E, NC, pid,
                     ebuf if ebuf is not None else dedup_bufs(Wgu, Wd), thr=dthr)
    for i in range(0 if dedup_on(B) else NSL):
        s = NC * i + pid          # global slot; b = s // 8 is pid-independent because 8 % NC == 0
        b = (NC * i) // TOPK
        iv = ti.ap(pattern=[[TOPK, 1], [1, 1]], offset=s)
        wgu = sb((128, KT * I2), BF)
        nisa.dma_copy(dst=wgu, src=Wgu.ap(pattern=[[KT * I2, 128], [1, KT * I2]], offset=0, scalar_offset=iv,
                                          indirect_dim=0))
        wd = sb((I, H), BF)
        nisa.dma_copy(dst=wd, src=Wd.ap(pattern=[[H, I], [1, H]], offset=0, scalar_offset=iv, indirect_dim=0))
        pgu = nl.ndarray((1, I2), dtype=F32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(pgu, xP[:, kt * B + b:kt * B + b + 1], wgu[:, kt * I2:(kt + 1) * I2])
        gu = sb((1, I2), BF)
        nisa.tensor_copy(dst=gu, src=pgu)
        sg = sb((1, I), BF)
        nisa.activation(dst=sg, data=gu[:, 0:I], op=nl.silu)
        hr = sb((1, I), BF)
        nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, I:I2], op=nl.multiply)
        ph = nl.ndarray((I, 1), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ph, hr, ident[0:1, 0:1])
        hs = sb((I, B), BF)
        nisa.memset(hs, 0.0)
        nisa.tensor_scalar(dst=hs[:, b:b + 1], data=ph, op0=nl.multiply, operand0=wb[0:I, nl.ds(s, 1)])
        for t in range(NT):
            nisa.nc_matmul(py[t], hs, wd[:, t * 512:(t + 1) * 512])

    # ---- fold routed partials to this pcore's half ----
    yr = sb((B, H))
    for t in range(NT):
        nisa.tensor_copy(dst=yr[:, t * 512:(t + 1) * 512], src=py[t] if yacc is None else yacc[:, t * 512:(t + 1) * 512])
    if NC > 1:
        other = 1 - pid
        recv = sb((B, HW))
        nisa.sendrecv(src=yr[:, nl.ds(other * HW, HW)], dst=recv, send_to_rank=other, recv_from_rank=other,
                      pipe_id=2)
        yh = sb((B, HW))
        nisa.tensor_tensor(dst=yh, data1=yr[:, nl.ds(pid * HW, HW)], data2=recv, op=nl.add)
    else:
        yh = yr

    if SHARED_FIRST:
        nisa.tensor_tensor(dst=yh, data1=yh, data2=ysh_all, op=nl.add)
        return yh
    # ---- shared expert: h from the K-split gate_up, down N-split to this pcore's half ----
    gus = sb((B, IS2), BF)
    nisa.tensor_copy(dst=gus, src=full[:, ER:ER + IS2])
    ssg = sb((B, IS), BF)
    nisa.activation(dst=ssg, data=gus[:, 0:IS], op=nl.silu)
    hsh = sb((B, IS), BF)
    nisa.tensor_tensor(dst=hsh, data1=ssg, data2=gus[:, IS:IS2], op=nl.multiply)
    phs = nl.ndarray((IS, B), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(phs, hsh, ident[0:B, 0:B])
    hcol = sb((IS, B), BF)
    nisa.tensor_copy(dst=hcol, src=phs)
    wsd = sb((IS, HW), BF)
    nisa.dma_copy(dst=wsd, src=Wsd.ap(pattern=[[H, IS], [1, HW]], offset=pid * HW))
    gate = sb((B, 1))
    nisa.activation(dst=gate, data=full[:, ER + IS2:NP], op=nl.sigmoid)
    for t in range(HW // 512):
        pys = nl.ndarray((B, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pys, hcol, wsd[:, t * 512:(t + 1) * 512])
        ysh = sb((B, 512), BF)
        nisa.tensor_copy(dst=ysh, src=pys)
        nisa.scalar_tensor_tensor(dst=yh[:, t * 512:(t + 1) * 512], data=ysh, op0=nl.multiply, operand0=gate,
                                  op1=nl.add, operand1=yh[:, t * 512:(t + 1) * 512])
    return yh
