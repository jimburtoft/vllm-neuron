# SPDX-License-Identifier: Apache-2.0
"""Sparse MoE decode block of the megakernel for one TP=4 rank (weights read in place), LNC=2.

Per rank (``Qwen3_5SparseMoeBlock``, ep_degree=1): router [H, E] fp32 (a load-time copy of the replicated
router), experts gate_up [E, H, 2I] (a view of ``experts_gate_up_weight`` [E, H, 2, I]) / down [E, I, H] with
I = 512 / TP = 128, shared expert gate_up [H, 2I_sh] / down [I_sh, H] / gate [H, 1].
Math (``forward_decode`` + ``_shared_expert``):
  logits = x @ Wr (fp32) -> softmax -> top-8 -> renorm      (== exp(l - lmax) / sum over the top 8)
  y = sum_k w_k * (silu(x Wg_e) * (x Wu_e)) Wd_e  +  sigmoid(x Wsg) * (silu(x Wsg_g) * (x Wsu)) Wsd
Output: this pcore's H half of the rank partial ([B, H/NC] fp32); the caller all-reduces over TP.

LNC=2 split (no control flow depends on program_id, only offsets):
  router + shared gate_up: K-split (pcore c contracts rows p*16 + [8c, 8c+8)), partials swapped by sendrecv
    and added, so both pcores hold bit-identical logits and pick the same experts.
  routed experts, B < 8: slot s = b*8 + k; pcore c runs slots 2i + c, each DMAs its expert by runtime index.
    Weighted sum = PSUM accumulation of down matmuls whose stationary is the [128, B] matrix with column b =
    w_s * h_s (zeros elsewhere), so every slot accumulates straight into the token's output row.
  routed experts, B >= 8 (expert dedup): each unique expert of the step is loaded once and applied to all rows,
    with a static number of unique-expert slots and a runtime loop for any overflow (routed_dedup).
  shared down: N-split (pcore c computes its own H half).
"""
import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, H, KT, rows_to_pmaj, sb

TOPK = 8
DEDUP_MIN = 8   # rows (batch) from which routed experts are deduplicated
DEDUP_NBUF = 4  # rotating expert weight buffers of the dedup path
# static unique-expert slots per batch size (about the p80-p90 unique count of real traffic); overflow -> loop
DEDUP_CAP = {8: 56, 16: 80, 32: 128}


def _i2(Wgu):
    """2I of an expert gate_up stack [E, H, 2I] or [E, H, 2, I] (the model's own layout, read in place)."""
    return Wgu.shape[2] if len(Wgu.shape) == 3 else Wgu.shape[2] * Wgu.shape[3]


def moe_static_load(Wr, Wsgu, Wsg, NC, pid):
    """Router (fp32) + shared gate_up + shared gate weights of this pcore's K half -> SBUF."""
    E, IS2, KC = Wr.shape[1], Wsgu.shape[1], KT // NC
    wr = sb((128, KC, E))
    nisa.dma_copy(dst=wr, src=Wr.ap(pattern=[[KT * E, 128], [E, KC], [1, E]], offset=pid * KC * E))
    wsgu = sb((128, KC, IS2), BF)
    nisa.dma_copy(dst=wsgu, src=Wsgu.ap(pattern=[[KT * IS2, 128], [IS2, KC], [1, IS2]], offset=pid * KC * IS2))
    wsg = sb((128, KC, 1), BF)
    nisa.dma_copy(dst=wsg, src=Wsg.ap(pattern=[[KT, 128], [1, KC], [1, 1]], offset=pid * KC))
    return wr, wsgu, wsg


def dedup_on(R):
    """Unique-expert path for R >= DEDUP_MIN rows."""
    return R >= DEDUP_MIN


def dedup_bufs(Wgu, Wd):
    """DEDUP_NBUF rotating expert weight buffers, zeroed once per kernel. A unique slot past the runtime count
    skips its DMA (oob), so its buffer keeps the previous (finite) expert or zeros; its routing weight is 0."""
    KTI2, I, Hh = Wgu.shape[1] * _i2(Wgu) // 128, Wd.shape[1], Wd.shape[2]
    bufs = []
    for _ in range(DEDUP_NBUF):
        wgu = sb((128, KTI2), BF)
        nisa.memset(wgu, 0.0)
        wd = sb((I, Hh), BF)
        nisa.memset(wd, 0.0)
        bufs.append((wgu, wd))
    return bufs  # bufs[0] doubles as the overflow-tail buffer (the tail runs after all static slots)


def routed_dedup(full, tv, nmax, er, xP, Wgu, Wd, ident, py, R, E, NC, pid, ebuf):
    """Routed experts with each unique expert loaded once for all R rows.
      dense [R, E] = renormed top-8 weight (exp(l - lmax) / sum, 0 off the top 8: mask l >= 8th largest)
      present [1, E] = column sums -> nonzero_with_count -> u [1, E+1]: unique ids ascending, padded with E (out of
        bounds: the slot's DMAs are skipped), count last.
      Wsel [R, CAP] = dense @ onehot(u)  (the [rows, unique] routing-weight mask)
      unique slot j (pcore c runs j = NC*i + c, CAP = min(8R, E) slots, statically unrolled):
        gu [R, 2I] = x @ Wgu[u_j]; h = silu(g) * u; h *= Wsel[:, j]; py += h @ Wd[u_j]   (dense over rows)."""
    I2 = _i2(Wgu)
    I = I2 // 2
    CAPT = min(TOPK * R, E)
    CAP = min(CAPT, DEDUP_CAP.get(R, 256))
    CAP -= CAP % NC
    EC = min(128, E)
    NE = E // EC
    # dense routing weights
    dm = sb((R, E))
    nisa.tensor_scalar(dst=dm, data=full[:, 0:E], op0=nl.greater_equal, operand0=tv[:, TOPK - 1:TOPK])
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
        gu = sb((R, I2), BF)
        nisa.tensor_copy(dst=gu, src=pgu)
        sg = sb((R, I), BF)
        nisa.activation(dst=sg, data=gu[:, 0:I], op=nl.silu)
        hr = sb((R, I), BF)
        nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, I:I2], op=nl.multiply)
        hw = sb((R, I), BF)
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
    # compact this pcore's tail ids / weights into offset-0 tiles (indexing a static offset plus a runtime
    # scalar_offset into one SBUF tile did not lower correctly on hardware)
    ut = nl.ndarray((1, NX), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ut, src=u.ap(pattern=[[E + 1, 1], [NC, NX]], offset=CAP + pid))
    wt = sb((1, NX * R))  # tail weights on one partition: [k*R:(k+1)*R] = Wsel[:, CAP + pid + NC*k]
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


def moe_block(xn, Wr, Wgu, Wd, Wsgu, Wsd, Wsg, ident, ones_f, bmask, B, NC, pid, wcol, rk):
    """xn [B, H] bf16 SBUF: the UN-normed residual rows; the post-attention RMSNorm is applied in the column layout
    (wcol = norm weight columns, rk = per-row rsqrt columns). ones_f [B, 128] fp32 ones; bmask [B, 8B] fp32
    block-diagonal ones (bmask[b, b*8 + k] = 1). Returns yh [B, H/NC] fp32 SBUF (pcore half, rank partial)."""
    E = Wr.shape[1]
    I2 = _i2(Wgu)
    if len(Wgu.shape) == 4:   # the model's [E, H, 2, I] stack, viewed as [E, H, 2I] (same bytes)
        Wgu = Wgu.reshape((Wgu.shape[0], Wgu.shape[1], I2))
    I = I2 // 2
    IS2 = Wsgu.shape[1]
    IS = IS2 // 2
    KC = KT // NC        # k-tiles contracted by this pcore (K-split)
    HW = H // NC         # this pcore's H half
    NSL = TOPK * B // NC  # routed slots per pcore

    xP = rows_to_pmaj(xn, ident, B, BF, wcol=wcol, rk=rk)  # [128, KT*B] bf16, element (p, kt*B+b) = x[b, p*KT+kt]
    xPf = sb((128, KT * B))
    nisa.tensor_copy(dst=xPf, src=xP)

    # ---- router (fp32) + shared gate + shared gate_up, K-split over pcores ----
    wr, wsgu, wsg = moe_static_load(Wr, Wsgu, Wsg, NC, pid)
    NP = E + IS2 + 1
    pr = nl.ndarray((B, E), dtype=F32, buffer=nl.psum)
    psg = nl.ndarray((B, IS2), dtype=F32, buffer=nl.psum)
    pgt = nl.ndarray((B, 1), dtype=F32, buffer=nl.psum)
    for kk in range(KC):
        col = nl.ds((pid * KC + kk) * B, B)
        nisa.nc_matmul(pr, xPf[:, col], wr[:, kk, :])
        nisa.nc_matmul(psg, xP[:, col], wsgu[:, kk, :])
        nisa.nc_matmul(pgt, xP[:, col], wsg[:, kk, :])
    part = sb((B, NP))
    nisa.tensor_copy(dst=part[:, 0:E], src=pr)
    nisa.tensor_copy(dst=part[:, E:E + IS2], src=psg)
    nisa.tensor_copy(dst=part[:, E + IS2:NP], src=pgt)
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
    es = sb((B, 1))
    nisa.tensor_reduce(dst=es, op=nl.add, data=ev, axis=1)
    er = sb((B, 1))
    nisa.reciprocal(dst=er, data=es)
    aff = sb((B, TOPK))
    nisa.tensor_scalar(dst=aff, data=ev, op0=nl.multiply, operand0=er)
    # broadcast affinities to all partitions: wb[p, b*8+k] = aff[b, k]
    ablk = sb((B, TOPK * B))
    nisa.tensor_tensor(dst=ablk, data1=aff.ap(pattern=[[TOPK, B], [0, B], [1, TOPK]], offset=0), data2=bmask,
                       op=nl.multiply)
    pwb = nl.ndarray((128, TOPK * B), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(pwb, ones_f, ablk)
    wb = sb((128, TOPK * B))
    nisa.tensor_copy(dst=wb, src=pwb)

    NT = H // 512
    py = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in range(NT)]
    yacc = None
    if dedup_on(B):
        yacc = routed_dedup(full, tv, nmax, er, xP, Wgu, Wd, ident, py, B, E, NC, pid, dedup_bufs(Wgu, Wd))
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

    # ---- shared expert: h from the K-split gate_up, down N-split to this pcore's half ----
    gus = sb((B, IS2), BF)
    nisa.tensor_copy(dst=gus, src=full[:, E:E + IS2])
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
    nisa.activation(dst=gate, data=full[:, E + IS2:NP], op=nl.sigmoid)
    for t in range(HW // 512):
        pys = nl.ndarray((B, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pys, hcol, wsd[:, t * 512:(t + 1) * 512])
        ysh = sb((B, 512), BF)
        nisa.tensor_copy(dst=ysh, src=pys)
        nisa.scalar_tensor_tensor(dst=yh[:, t * 512:(t + 1) * 512], data=ysh, op0=nl.multiply, operand0=gate,
                                  op1=nl.add, operand1=yh[:, t * 512:(t + 1) * 512])
    return yh
