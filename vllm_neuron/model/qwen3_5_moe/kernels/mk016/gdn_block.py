"""GDN decode mixer as a megakernel block (no residual): the tt012 kernel body
(vllm-neuron/.../kernels/tt012_gdn_decode.py:79-257, verbatim math) refactored into a function that takes the
column-layout normed input xT from SBUF and returns this pcore's H half of the out_proj rank partial
([B, H/NC] bf16 SBUF). The caller all-reduces over TP and adds the residual. Layouts and LNC=2 split are as in
tt012 (pcore c owns v-heads [4c, 4c+4), out_proj N-split = H half c). State in/out: conv [NC*128*3*CT*B] fp32
kernel-native, rec [B, Hv, Dk, Dv] fp32 cache layout; the caller allocates cst_out / rs_out / oxch / gsc.
"""
import math
import os

import nki.isa as nisa
import nki.language as nl

from .mk_common import evac

H, VAL, D = 2048, 1024, 128
HK, HV = 4, 8  # per TP rank
KCH = 4
KT, OKT = H // 128, VAL // 128
F32, BF = nl.float32, nl.bfloat16
MK_RS1 = os.environ.get("MK_RS1") == "1"  # M5: 1 = one rec-state DMA per (request, head) (first version)
# MK_WOUT_EARLY=1 issues the out_proj weight DMAs right after the in_proj matmuls (streams under the
# conv / recurrence vector chain instead of after it)
MK_WOUT_EARLY = os.environ.get("MK_WOUT_EARLY", "0") == "1"
# MK_STEARLY=1 issues the GDN conv / rec state + small param DMAs before the in_proj matmuls
MK_STEARLY = os.environ.get("MK_STEARLY", "0") == "1"
MK_GDN_XCH = os.environ.get("MK_GDN_XCH", "sendrecv")  # sendrecv (default, M3) | hbm (core_barrier; wrong on HW after a GQA layer)



GDN_SPLIT = os.environ.get("MK_GDN_SPLIT", "0") == "1"

def dims(nc, b):
    hk_c, hv_c = HK // nc, HV // nc
    ct = 2 * hk_c + hv_c
    nin_c = 128 * (ct + hv_c) + 2 * hv_c
    return hk_c, hv_c, ct, nin_c, hv_c * b, H // nc


def load_mk(mk_in):
    """GDN block mask kron(eye(J), ones(1, 128)) in SBUF, capped at 128 rows (grouped recurrence uses the
    top-left [JL, JL*128] block, JL <= 128). mk_in may be [J, J*128] (any J) or already capped."""
    J0, W0 = mk_in.shape
    JM = min(J0, 128)
    mk = _sb((JM, JM * 128), BF)
    nisa.dma_copy(dst=mk, src=mk_in.ap(pattern=[[W0, JM], [1, JM * 128]], offset=0))
    return mk


def _ntiles(n, t=512):
    return [(s, min(t, n - s)) for s in range(0, n, t)]


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _colsum(dst_sb, src_bf, n, ones):
    ps = nl.ndarray((128, n), dtype=F32, buffer=nl.psum)
    nisa.nc_matmul(ps, ones, src_bf)
    nisa.tensor_copy(dst=dst_sb, src=ps)


def gdn_win_load(Win, g, NC, pid, B):
    """k-tile group g (KCH x 128 rows) of this pcore's in_proj slice -> SBUF (M4 cross-layer prefetch unit)."""
    NIN_C = dims(NC, B)[3]
    w = _sb((128, KCH, NIN_C), BF)
    kw = {"dge_mode": nisa.dge_mode.hwdge} if os.environ.get("MK_DGE") == "hw" else {}
    nisa.dma_copy(dst=w, src=Win.ap(pattern=[[NIN_C, 128], [128 * NIN_C, KCH], [1, NIN_C]],
                                     offset=pid * H * NIN_C + g * KCH * 128 * NIN_C), **kw)
    return w


def gdn_wout_load(Wout, g, NC, pid):
    """k-tile group g of this pcore's out_proj N-half [KCH*128, H/NC] -> SBUF."""
    NOUT_C = H // NC
    w = _sb((128, KCH, NOUT_C), BF)
    nisa.dma_copy(dst=w, src=Wout.ap(pattern=[[H, 128], [128 * H, KCH], [1, NOUT_C]],
                                      offset=g * KCH * 128 * H + pid * NOUT_C))
    return w


def slot_io_tiles(ridx, widx, B, NC, pid):
    """per-step row indices for slot-indexed state I/O (shared by all GDN layers).
    ridx/widx [B] int32 HBM = flat (slot * n_ckpt + ckpt) read / write rows. Returns dict of (1, B) uint32 tiles:
    rr[i]/rw[i] = row * HV + pid*HV_C + i (rec state viewed [N*HV, D*D]); cr/cw = row * NC + pid (conv state
    viewed [N*NC, 128*3*CT])."""
    HV_C = HV // NC

    def mk(src, mul, add):
        t = _sb((1, B), nl.int32)
        nisa.dma_copy(dst=t, src=src.ap(pattern=[[B, 1], [1, B]], offset=0))
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=mul, op1=nl.add, operand1=add)
        u = nl.ndarray((1, B), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=u, src=t)
        return u

    return dict(rr=[mk(ridx, HV, pid * HV_C + i) for i in range(HV_C)],
                rw=[mk(widx, HV, pid * HV_C + i) for i in range(HV_C)],
                cr=mk(ridx, NC, pid), cw=mk(widx, NC, pid))


def contig_io_tiles(act, B):
    """contiguous (slot = kernel row) state I/O. act [B] fp32 HBM: 1 = row decodes this step, 0 = row's
    state must stay unchanged (hole / padding). Returns dict(contig=True, act=[128, B], nact=[128, B]) fp32."""
    a = _sb((128, B))
    nisa.dma_copy(dst=a, src=act.ap(pattern=[[0, 128], [1, B]], offset=0))
    na = _sb((128, B))
    nisa.tensor_scalar(dst=na, data=a, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    return dict(contig=True, act=a, nact=na)


def _ix(t, b, B):
    return t.ap(pattern=[[B, 1], [1, 1]], offset=b)


def gdn_mixer(xT, Win, Wout, cw, cst_in, rs_in, alog, dtb, nw, ident, ones, mk, cst_out, rs_out, oxch, gsc,
              B, NC, pid, w_pre=(), sio=None, rsc=None):
    """sio=None: M2-M4 layouts (cst kernel-native [NC*128*3*CT*B], rs [B, Hv, Dk, Dv], separate outputs).
    sio=slot_io_tiles(...): M5 port layouts, updated in place (cst_in is cst_out, rs_in is rs_out):
    rs [slots, ckpt, Hv, Dk, Dv] fp32, cst [slots, ckpt, NC*128*3*CT] fp32 (per slot [NC, 128, 3(w), CT])."""
    HK_C, HV_C, CT, NIN_C, J, NOUT_C = dims(NC, B)
    # contiguous port layout -- kernel row b IS pool row b (slot b, n_ckpt 1): static large DMAs (the M4
    # patterns over the first B pool rows) + an active-row mask that keeps inactive rows' state bit-exact.
    CTG = sio is not None and sio.get("contig", False)
    if CTG:
        NR = rs_in.shape[0] * rs_in.shape[1]
        assert rs_in.shape[1] == 1 and cst_in.shape[1] == 1, "contig state path needs n_ckpt 1"
        cs2 = cst_in.reshape((NR * NC, 128 * 3 * CT))
        rs_c = rs_in.reshape((NR, HV, D, D))
        act, nact = sio["act"], sio["nact"]
        sio_c, sio = sio, None   # rec-state DMAs below take the M4 (sio None) form on rs_c
        rs_in = rs_out = rs_c
    if sio is not None:
        NR = rs_in.shape[0] * rs_in.shape[1]
        rs2 = rs_in.reshape((NR * HV, D * D))
        rsN = rs_in.reshape((NR * NC, HV_C * D * D))   # row = slot_row * NC + pid (same index tiles as conv)
        # rank-3 rs [slots, ckpt, NC*128(k)*HV_C*128(v)]: kernel-native layout (2 KB contiguous per partition row)
        RSK = len(rs_in.shape) == 3
        sdk = {"dge_mode": nisa.dge_mode.hwdge} if os.environ.get("MK_SDGE") == "hw" else {}
        rs_ap = ([[HV_C * D, 128], [D, HV_C], [1, D]] if RSK else [[D, 128], [D * D, HV_C], [1, D]])
        cs2 = cst_in.reshape((NR * NC, 128 * 3 * CT))
    WIN_SZ = H * NIN_C
    CSZ = 128 * 3 * CT * B
    # request groups for the recurrence (J = HV_C*B state columns must sit on <= 128 partitions in the
    # outer-product update, and the fp32 state is 512 B x J per partition). BG requests per group, state rows
    # streamed per group (MK_GDN_BG, default: one group up to J = 128).
    BG = int(os.environ.get("MK_GDN_BG", "0")) or (B if J <= 128 else max(1, 128 // HV_C))
    BG = min(BG, B)
    assert B % BG == 0, (B, BG)
    NGR = B // BG
    def _state_loads():
        # ---------------- state / param loads ----------------
        hs = _sb((128, 4 * CT * B))
        if CTG:  # pool rows 0..B-1, this pcore's [128, 3*CT] block of each: one DMA -> staging (same order as below)
            cstg = _sb((128, B * 3 * CT))
            nisa.dma_copy(dst=cstg.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0),
                          src=cs2.ap(pattern=[[3 * CT, 128], [NC * 128 * 3 * CT, B], [1, 3 * CT]],
                                     offset=pid * 128 * 3 * CT))
            nisa.tensor_copy(dst=hs.ap(pattern=[[4 * CT * B, 128], [1, B], [B, 3 * CT]], offset=0),
                             src=cstg.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0))
        elif sio is None:
            nisa.dma_copy(dst=hs[:, 0:3 * CT * B], src=cst_in.ap(pattern=[[3 * CT * B, 128], [1, 3 * CT * B]],
                                                                 offset=pid * CSZ))
        else:  # per request b: slot row [128(p), 3(w)*CT(t)] contiguous -> staging -> hs column w*CT*B + t*B + b
            cstg = _sb((128, B * 3 * CT))
            for b in range(B):
                nisa.dma_copy(dst=cstg[:, b * 3 * CT:(b + 1) * 3 * CT],
                              src=cs2.ap(pattern=[[3 * CT, 128], [1, 3 * CT]], offset=0,
                                         scalar_offset=_ix(sio["cr"], b, B), indirect_dim=0))
            nisa.tensor_copy(dst=hs.ap(pattern=[[4 * CT * B, 128], [1, B], [B, 3 * CT]], offset=0),
                             src=cstg.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0))
        wraw = _sb((128, 4 * CT))
        nisa.dma_copy(dst=wraw, src=cw.ap(pattern=[[4 * CT, 128], [1, 4 * CT]], offset=pid * 128 * 4 * CT))
        wexp = _sb((128, 4 * CT * B))
        nisa.tensor_copy(dst=wexp, src=wraw.ap(pattern=[[4 * CT, 128], [1, 4 * CT], [0, B]], offset=0))
        # recurrent state: cache layout [B, HV, K, V] -> S[k, (hv_local*B + b)*128 + v]
        S = _sb((128, J * 128)) if NGR == 1 else None
        for i in range(HV_C if NGR == 1 else 0):
            if sio is None:
                nisa.dma_copy(dst=S.ap(pattern=[[J * 128, 128], [128, B], [1, 128]], offset=i * B * 128),
                              src=rs_in.ap(pattern=[[D, 128], [HV * D * D, B], [1, D]], offset=(pid * HV_C + i) * D * D))
            elif MK_RS1:
                for b in range(B):
                    nisa.dma_copy(dst=S[:, (i * B + b) * 128:(i * B + b + 1) * 128],
                                  src=rs2.ap(pattern=[[D, 128], [1, D]], offset=0,
                                             scalar_offset=_ix(sio["rr"][i], b, B), indirect_dim=0))
        if sio is not None and not MK_RS1 and NGR == 1:  # one DMA per request: the pcore's HV_C heads are contiguous in the slot row
            for b in range(B):
                nisa.dma_copy(dst=S.ap(pattern=[[J * 128, 128], [B * 128, HV_C], [1, 128]], offset=b * 128),
                              src=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cr"], b, B), indirect_dim=0),
                              **sdk)
        al = _sb((128, J))
        dt = _sb((128, J))
        adr = _sb((128, 2 * HV_C))
        nisa.dma_copy(dst=adr[:, 0:HV_C], src=alog.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
        nisa.dma_copy(dst=adr[:, HV_C:2 * HV_C], src=dtb.ap(pattern=[[0, 128], [1, HV_C]], offset=pid * HV_C))
        nisa.tensor_copy(dst=al, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=0))
        nisa.tensor_copy(dst=dt, src=adr.ap(pattern=[[2 * HV_C, 128], [1, HV_C], [0, B]], offset=HV_C))
        nwc = _sb((128, 1))
        nisa.dma_copy(dst=nwc, src=nw.ap(pattern=[[1, 128], [1, 1]], offset=0))
        return hs, wexp, S, al, dt, nwc

    st_early = _state_loads() if MK_STEARLY else None  # state DMAs before in_proj
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
        evac(a_rows[0:B, n0:n0 + nn], ps[ti][0:B, 0:nn], rsc)
    wouts = [gdn_wout_load(Wout, g, NC, pid) for g in range(OKT // KCH)] if MK_WOUT_EARLY else None

    hs, wexp, S, al, dt, nwc = st_early if st_early is not None else _state_loads()
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
    if CTG:  # inactive rows: decay 1, update 0 -> S*1 + 0 = S (bit-exact); active rows unchanged (x*1 + 0)
        actJ = act.ap(pattern=[[B, 128], [0, HV_C], [1, B]], offset=0)
        nactJ = nact.ap(pattern=[[B, 128], [0, HV_C], [1, B]], offset=0)
        nisa.tensor_tensor(dst=eg, data1=eg, data2=actJ, op=nl.multiply)
        nisa.tensor_tensor(dst=eg, data1=eg, data2=nactJ, op=nl.add)
        nisa.tensor_tensor(dst=beta, data1=beta, data2=actJ, op=nl.multiply)

    # ---------------- channel layout + conv1d + silu ----------------
    xz = _sb((128, (CT + HV_C) * B))
    for t in range(CT + HV_C):
        tp = nl.ndarray((128, B), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(tp, a_rows[0:B, t * 128:(t + 1) * 128], ident[0:B, 0:B])
        nisa.tensor_copy(dst=xz[:, t * B:(t + 1) * B], src=tp)
    nisa.tensor_copy(dst=hs[:, 3 * CT * B:4 * CT * B], src=xz[:, 0:CT * B])
    if CTG:  # new conv state = new * act + old * (1 - act), columns (w, t, b); one DMA per pcore
        cnew = _sb((128, 3 * CT * B))
        cold = _sb((128, 3 * CT * B))
        nisa.tensor_tensor(dst=cnew, data1=hs[:, CT * B:4 * CT * B],
                           data2=act.ap(pattern=[[B, 128], [0, 3 * CT], [1, B]], offset=0), op=nl.multiply)
        nisa.tensor_tensor(dst=cold, data1=hs[:, 0:3 * CT * B],
                           data2=nact.ap(pattern=[[B, 128], [0, 3 * CT], [1, B]], offset=0), op=nl.multiply)
        nisa.tensor_tensor(dst=cnew, data1=cnew, data2=cold, op=nl.add)
        cstw = _sb((128, B * 3 * CT))
        nisa.tensor_copy(dst=cstw.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0),
                         src=cnew.ap(pattern=[[3 * CT * B, 128], [1, B], [B, 3 * CT]], offset=0))
        nisa.dma_copy(dst=cs2.ap(pattern=[[3 * CT, 128], [NC * 128 * 3 * CT, B], [1, 3 * CT]], offset=pid * 128 * 3 * CT),
                      src=cstw.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0))
    elif sio is None:
        nisa.dma_copy(dst=cst_out.ap(pattern=[[3 * CT * B, 128], [1, 3 * CT * B]], offset=pid * CSZ),
                      src=hs[:, CT * B:4 * CT * B])
    else:
        cstw = _sb((128, B * 3 * CT))
        nisa.tensor_copy(dst=cstw.ap(pattern=[[B * 3 * CT, 128], [3 * CT, B], [1, 3 * CT]], offset=0),
                         src=hs.ap(pattern=[[4 * CT * B, 128], [1, B], [B, 3 * CT]], offset=CT * B))
        for b in range(B):
            nisa.dma_copy(dst=cs2.ap(pattern=[[3 * CT, 128], [1, 3 * CT]], offset=0,
                                     scalar_offset=_ix(sio["cw"], b, B), indirect_dim=0),
                          src=cstw[:, b * 3 * CT:(b + 1) * 3 * CT])
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

    # ---------------- recurrence + state update (per request group) ----------------
    def _recur(S, QK2, eg, beta, vcol, qkd, Ke_bf, JL):
        S_bf = _sb((128, JL * 128), BF)
        nisa.tensor_copy(dst=S_bf, src=S, engine=nisa.scalar_engine)
        psA = nl.ndarray((128, JL, 2), dtype=F32, buffer=nl.psum)
        for j in range(JL):
            nisa.nc_matmul(psA[:, j, 0:2], S_bf[:, j * 128:(j + 1) * 128], QK2[:, j, 0:2])
        kq = _sb((128, JL, 2))
        nisa.tensor_copy(dst=kq, src=psA)
        kv = _sb((128, JL))
        nisa.tensor_tensor(dst=kv, data1=kq[:, :, 0], data2=eg, op=nl.multiply)
        delta = _sb((128, JL))
        nisa.tensor_tensor(dst=delta, data1=vcol, data2=kv, op=nl.subtract)
        nisa.tensor_tensor(dst=delta, data1=delta, data2=beta, op=nl.multiply)
        o = _sb((128, JL))
        nisa.tensor_tensor(dst=o, data1=kq[:, :, 1], data2=eg, op=nl.multiply)
        t2 = _sb((128, JL))
        nisa.tensor_tensor(dst=t2, data1=qkd, data2=delta, op=nl.multiply)
        nisa.tensor_tensor(dst=o, data1=o, data2=t2, op=nl.add)
        # state update S = S*eg + k (x) delta
        d_bf = _sb((128, JL), BF)
        nisa.tensor_copy(dst=d_bf, src=delta)
        pT = nl.ndarray((JL, 128), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pT, d_bf, ident)
        Drows = _sb((JL, 128), BF)
        nisa.tensor_copy(dst=Drows, src=pT)
        pK = nl.ndarray((JL, 128), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pK, Ke_bf, ident)
        Krows = _sb((JL, 128), BF)
        nisa.tensor_copy(dst=Krows, src=pK)
        Dblk = _sb((JL, JL * 128), BF)
        nisa.tensor_tensor(dst=Dblk, data1=Drows.ap(pattern=[[128, JL], [0, JL], [1, 128]], offset=0),
                           data2=mk[0:JL, 0:JL * 128], op=nl.multiply)
        for c in range(JL // 4):
            po = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(po, Krows, Dblk[0:JL, c * 512:(c + 1) * 512])
            Sc = S[:, c * 512:(c + 1) * 512]
            # MK_GDN_SPLIT=1 runs alternate chunks' state update on GpSimd, so two of the
            # four [128, 512] fp32 chunk updates proceed in parallel with the other two on Vector.
            eng = {"engine": nisa.gpsimd_engine} if (GDN_SPLIT and c % 2 == 1) else {}
            nisa.tensor_tensor(dst=Sc, data1=Sc, data2=eg.ap(pattern=[[JL, 128], [1, 4], [0, 128]], offset=c * 4),
                               op=nl.multiply, **eng)
            if eng:  # GpSimd cannot read PSUM: evacuate on ScalarE (lightly loaded) first
                pos_ = _sb((128, 512))
                nisa.activation(dst=pos_, data=po, op=nl.copy)
                nisa.tensor_tensor(dst=Sc, data1=Sc, data2=pos_, op=nl.add, **eng)
            else:
                nisa.tensor_tensor(dst=Sc, data1=Sc, data2=po, op=nl.add)
        return o

    vcol = xc[:, 2 * HK_C * B:CT * B]
    if NGR == 1:
        o = _recur(S, QK2, eg, beta, vcol, qkd, Ke_bf, J)
    else:
        JG = HV_C * BG
        o = _sb((128, J))
        for g in range(NGR):
            gpat = lambda t, w: t.ap(pattern=[[w, 128], [B, HV_C], [1, BG]], offset=g * BG)   # noqa: E731
            Sg = _sb((128, JG * 128))
            if sio is None:  # M4 layout [B, HV, D, D]: one DMA per head over the group's BG requests
                for i in range(HV_C):
                    nisa.dma_copy(dst=Sg.ap(pattern=[[JG * 128, 128], [128, BG], [1, 128]], offset=i * BG * 128),
                                  src=rs_in.ap(pattern=[[D, 128], [HV * D * D, BG], [1, D]],
                                               offset=(g * BG * HV + pid * HV_C + i) * D * D))
            for bl in range(BG if sio is not None else 0):
                b = g * BG + bl
                if True:
                    nisa.dma_copy(dst=Sg.ap(pattern=[[JG * 128, 128], [BG * 128, HV_C], [1, 128]], offset=bl * 128),
                                  src=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cr"], b, B), indirect_dim=0),
                                  **sdk)
            QKg = _sb((128, JG, 2), BF)
            nisa.tensor_copy(dst=QKg, src=QK2.ap(pattern=[[2 * J, 128], [2 * B, HV_C], [1, 2 * BG]], offset=g * BG * 2))
            loc = []
            for t_ in (eg, beta, qkd):
                u_ = _sb((128, JG))
                nisa.tensor_copy(dst=u_, src=gpat(t_, J))
                loc.append(u_)
            vg = _sb((128, JG))
            nisa.tensor_copy(dst=vg, src=xc.ap(pattern=[[CT * B, 128], [B, HV_C], [1, BG]], offset=2 * HK_C * B + g * BG))
            kg = _sb((128, JG), BF)
            nisa.tensor_copy(dst=kg, src=gpat(Ke_bf, J))
            og_ = _recur(Sg, QKg, loc[0], loc[1], vg, loc[2], kg, JG)
            nisa.tensor_copy(dst=gpat(o, J), src=og_)
            if sio is None:
                for i in range(HV_C):
                    nisa.dma_copy(dst=rs_out.ap(pattern=[[D, 128], [HV * D * D, BG], [1, D]],
                                                offset=(g * BG * HV + pid * HV_C + i) * D * D),
                                  src=Sg.ap(pattern=[[JG * 128, 128], [128, BG], [1, 128]], offset=i * BG * 128))
            for bl in range(BG if sio is not None else 0):
                b = g * BG + bl
                if True:
                    nisa.dma_copy(dst=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cw"], b, B), indirect_dim=0),
                                  src=Sg.ap(pattern=[[JG * 128, 128], [BG * 128, HV_C], [1, 128]], offset=bl * 128), **sdk)
    for i in range(HV_C if NGR == 1 else 0):
        if sio is None:
            nisa.dma_copy(dst=rs_out.ap(pattern=[[D, 128], [HV * D * D, B], [1, D]], offset=(pid * HV_C + i) * D * D),
                          src=S.ap(pattern=[[J * 128, 128], [128, B], [1, 128]], offset=i * B * 128))
        elif MK_RS1:
            for b in range(B):
                nisa.dma_copy(dst=rs2.ap(pattern=[[D, 128], [1, D]], offset=0,
                                         scalar_offset=_ix(sio["rw"][i], b, B), indirect_dim=0),
                              src=S[:, (i * B + b) * 128:(i * B + b + 1) * 128])
    if sio is not None and not MK_RS1 and NGR == 1:
        for b in range(B):
            nisa.dma_copy(dst=rsN.ap(pattern=rs_ap, offset=0, scalar_offset=_ix(sio["cw"], b, B), indirect_dim=0),
                          src=S.ap(pattern=[[J * 128, 128], [B * 128, HV_C], [1, 128]], offset=b * 128), **sdk)

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
        if MK_GDN_XCH == "sendrecv":
            # SBUF<->SBUF exchange (the shared-HBM + core_barrier exchange gave wrong data on HW for
            # the GQA block; same pattern here is suspected in the GGGAGGGA HW failure)
            other = 1 - pid
            peer = _sb((128, J), BF)
            nisa.sendrecv(src=oTl, dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
            nisa.tensor_copy(dst=oT[:, pid * J:(pid + 1) * J], src=oTl)
            nisa.tensor_copy(dst=oT[:, other * J:(other + 1) * J], src=peer)
        else:
            nisa.dma_copy(dst=oxch.ap(pattern=[[OKT * B, 128], [1, J]], offset=pid * J), src=oTl)
            nisa.core_barrier(data=oxch, cores=(0, 1))
            nisa.dma_copy(dst=oT, src=oxch)

    # ---------------- out_proj (N-split, Wout read in place) ----------------
    nto = _ntiles(NOUT_C)
    pso = [nl.ndarray((B, 512), dtype=F32, buffer=nl.psum) for _ in nto]
    for g in range(OKT // KCH):
        w = wouts[g] if wouts is not None else gdn_wout_load(Wout, g, NC, pid)
        for kk in range(KCH):
            kt = g * KCH + kk
            for ti, (n0, nn) in enumerate(nto):
                nisa.nc_matmul(pso[ti][0:B, 0:nn], oT[:, kt * B:(kt + 1) * B], w[:, kk, n0:n0 + nn])
    y_rows = _sb((B, NOUT_C), BF)
    for ti, (n0, nn) in enumerate(nto):
        nisa.tensor_copy(dst=y_rows[0:B, n0:n0 + nn], src=pso[ti][0:B, 0:nn])
    return y_rows
