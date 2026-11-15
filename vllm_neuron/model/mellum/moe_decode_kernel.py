"""Expert-major MoE decode kernel for Mellum on trn2 (one EP rank): each local expert's weights are loaded once per
step and applied to all R decode rows (tokens are the stationary operand, weights the moving one).

Adapted from working/qwen3_omni/kernels/moe_dedup_decode.py (itself from torchtpu `routed_dedup`), changed for the
Mellum EP4 shape (E_L 16, H 2304, I 896, R <= 128):

  - H tail: 2304 = 4 x 512 + 256 output columns per down matmul group; K tiles = H / 128 = 18 (any H % 128 == 0).
  - 2I = 1792 gate/up columns: the gate/up product is split into <= 512-column matmuls.
  - I split across the two physical cores (LNC=2, NC=2): core c computes I rows [c * I/2, (c+1) * I/2) of every
    expert (gate, up and the matching down rows), so each core streams exactly half of every expert's bytes and the
    load balance does not depend on routing. The two fp32 partials of y are summed with sendrecv by H halves.
  - Unique-expert dedupe: experts with no routed row on this rank are skipped (runtime-indexed DMA, oob skip);
    a skipped slot still costs its (cheap, data-independent) matmuls on zeroed buffers, but no HBM bytes.
  - FP8 weights (float8_e4m3, |w| <= 240) with one scale per (expert, gate/up output column) and one scale per
    (expert, down output column). Scales are applied after each matmul: gate/up scale on the [R, 2I] product,
    down scale folded into a [R, H] multiply per slot... -- to keep the bytes low, the down scale is per expert
    (scalar, folded into the routing weight); the gate/up scale is per output column.

Inputs (HBM):
  x     bf16 [R, H]          post-norm rows (R <= 128)
  aff   fp32 [R, E]          routing weights of this rank's E local experts (0 where not routed)
  Wgu   bf16|fp8 [E, H, 2I]  gate | up (gate columns 0..I-1, up I..2I-1)
  Wd    bf16|fp8 [E, I, H]
  gu_scale fp32 [E, 2I] (FP8 only), d_scale fp32 [E, 1] (FP8 only)
Output: y fp32 [R, H] (rank partial; caller all-reduces over the MoE group).
"""
import nki
import nki.isa as nisa
import nki.language as nl

F32, BF = nl.float32, nl.bfloat16


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _col_tiles(n, w=512):
    return [(o, min(w, n - o)) for o in range(0, n, w)]


@nki.jit
def mellum_moe_decode(x, aff, Wgu, Wd, gu_scale=None, d_scale=None, nbuf=4, dedupe=1):
    R, H = x.shape
    NC = nl.num_programs(0) if nl.program_ndim() > 0 else 1
    pid = nl.program_id(0) if nl.program_ndim() > 0 else 0
    E = aff.shape[1]
    I2 = Wgu.shape[2]
    I = I2 // 2
    KT = H // 128
    WDT = Wgu.dtype
    FP8 = gu_scale is not None
    IC = I // NC  # this core's I rows per expert
    I_TILES = [(o, min(128, IC - o)) for o in range(0, IC, 128)]  # down K tiles of this core's I slice
    HT = _col_tiles(H)  # down output column groups (<= 512, one PSUM bank each)
    GT = [(0, o, n) for (o, n) in _col_tiles(IC)] + [(1, o, n) for (o, n) in _col_tiles(IC)]  # (gate/up, col, n)
    i0 = pid * IC

    ident = _sb((128, 128), BF)
    io_p = _sb((128, 1))
    nisa.iota(io_p, [[0, 1]], offset=0, channel_multiplier=1)
    io_f = _sb((128, 128))
    nisa.iota(io_f, [[1, 128]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=ident, data=io_f, op0=nl.equal, operand0=io_p)

    # x -> K-major tiles xP[p, kt * R + r] = x[r, kt * 128 + p]
    xs = _sb((R, H), BF)
    nisa.dma_copy(dst=xs, src=x)
    xP = _sb((128, KT * R), BF)
    for kt in range(KT):
        pt = nl.ndarray((128, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pt, xs[:, kt * 128:(kt + 1) * 128], ident[0:R, 0:R])
        nisa.tensor_copy(dst=xP[:, kt * R:(kt + 1) * R], src=pt)

    dense = _sb((R, E))
    nisa.dma_copy(dst=dense, src=aff)
    if dedupe:
        dm = _sb((R, E))
        nisa.tensor_scalar(dst=dm, data=dense, op0=nl.greater, operand0=0.0)
        onesr = _sb((R, 1))
        nisa.memset(onesr, 1.0)
        ppr = nl.ndarray((1, E), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(ppr, onesr, dm)
        pres = _sb((1, E))
        nisa.tensor_copy(dst=pres, src=ppr)
        u = nl.ndarray((1, E + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nonzero_with_count(dst=u, src=pres, index_offset=0, padding_val=E)
        uf = _sb((1, E))
        nisa.tensor_copy(dst=uf, src=u[:, 0:E])
        ones1 = _sb((1, 128))
        nisa.memset(ones1, 1.0)
        pub = nl.ndarray((128, E), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pub, ones1, uf)
        ub = _sb((128, E))
        nisa.tensor_copy(dst=ub, src=pub)
        identf = _sb((R, R))
        nisa.tensor_copy(dst=identf, src=ident[0:R, 0:R])
        ioe = _sb((E, 1))
        nisa.iota(ioe, [[0, 1]], offset=0, channel_multiplier=1)
        oh = _sb((E, E))
        nisa.tensor_scalar(dst=oh, data=ub[0:E, :], op0=nl.equal, operand0=ioe)
        pdt = nl.ndarray((E, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pdt, dense, identf)
        dT = _sb((E, R))
        nisa.tensor_copy(dst=dT, src=pdt)
        pws = nl.ndarray((R, E), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pws, dT, oh)
        wsel = _sb((R, E))  # wsel[r, j] = routing weight of row r for unique slot j
        nisa.tensor_copy(dst=wsel, src=pws)

    # weight ring; zeroed once so that a skipped (padded-slot) DMA leaves finite data (its routing weight is 0)
    bufs = []
    for bi in range(nbuf):
        wg = _sb((128, KT * IC), WDT)   # gate slice:  wg[p, kt*IC + c] = Wgu[e, kt*128+p, i0 + c]
        wu = _sb((128, KT * IC), WDT)   # up slice:    wu[p, kt*IC + c] = Wgu[e, kt*128+p, I + i0 + c]
        wd = [_sb((n, H), WDT) for (_, n) in I_TILES]
        if dedupe:
            nisa.memset(wg, 0.0, engine=nisa.gpsimd_engine)
            nisa.memset(wu, 0.0, engine=nisa.gpsimd_engine)
            for t in wd:
                nisa.memset(t, 0.0, engine=nisa.gpsimd_engine)
        bufs.append((wg, wu, wd))

    py = [nl.ndarray((R, n), dtype=F32, buffer=nl.psum) for (_, n) in HT]
    for j in range(E):
        wg, wu, wd = bufs[j % nbuf]
        if dedupe:
            iv = u.ap(pattern=[[E + 1, 1], [1, 1]], offset=j)
            dk = dict(scalar_offset=iv, indirect_dim=0)
            ok = dict(oob_mode=nisa.oob_mode.skip)
            base = 0
        else:
            dk, ok = {}, {}
            base = j
        nisa.dma_copy(dst=wg, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * H * I2 + i0, **dk), **ok)
        nisa.dma_copy(dst=wu, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * H * I2 + I + i0, **dk), **ok)
        for ti, (o, n) in enumerate(I_TILES):
            nisa.dma_copy(dst=wd[ti], src=Wd.ap(pattern=[[H, n], [1, H]], offset=base * I * H + (i0 + o) * H, **dk), **ok)
        if FP8:
            gsc = _sb((R, 2 * IC))
            dsc = _sb((R, 1))
            if dedupe:
                nisa.memset(gsc, 0.0)
                nisa.memset(dsc, 0.0)
            nisa.dma_copy(dst=gsc[:, 0:IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + i0, **dk), **ok)
            nisa.dma_copy(dst=gsc[:, IC:2 * IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + I + i0, **dk), **ok)
            nisa.dma_copy(dst=dsc, src=d_scale.ap(pattern=[[0, R], [1, 1]], offset=base, **dk), **ok)
        # gate | up for this core's I slice: [R, 2 IC] in <= 512-column PSUM groups
        gu = _sb((R, 2 * IC), BF)
        for (which, co, cn) in GT:
            wsrc = wg if which == 0 else wu
            dco = which * IC + co  # column in gu = [gate(IC) | up(IC)]
            pgu = nl.ndarray((R, cn), dtype=F32, buffer=nl.psum)
            for kt in range(KT):
                nisa.nc_matmul(pgu, xP[:, kt * R:(kt + 1) * R], wsrc[:, kt * IC + co: kt * IC + co + cn])
            if FP8:
                nisa.tensor_tensor(dst=gu[:, dco:dco + cn], data1=pgu, data2=gsc[:, dco:dco + cn], op=nl.multiply)
            else:
                nisa.tensor_copy(dst=gu[:, dco:dco + cn], src=pgu)
        sg = _sb((R, IC), BF)
        nisa.activation(dst=sg, data=gu[:, 0:IC], op=nl.silu)
        hr = _sb((R, IC), BF)
        nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, IC:2 * IC], op=nl.multiply)
        wcol = (wsel if dedupe else dense)[:, j:j + 1]
        if FP8:
            wds = _sb((R, 1))
            nisa.tensor_tensor(dst=wds, data1=wcol, data2=dsc, op=nl.multiply)
            wcol = wds
        hw = _sb((R, IC), BF)
        nisa.tensor_scalar(dst=hw, data=hr, op0=nl.multiply, operand0=wcol)
        for ti, (o, n) in enumerate(I_TILES):
            ph = nl.ndarray((n, R), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(ph, hw[:, o:o + n], ident[0:R, 0:R])
            hs = _sb((n, R), BF)
            nisa.tensor_copy(dst=hs, src=ph)
            for t, (ho, hn) in enumerate(HT):
                nisa.nc_matmul(py[t], hs, wd[ti][:, ho:ho + hn])

    y = nl.ndarray((R, H), dtype=F32, buffer=nl.shared_hbm)
    ys = _sb((R, H))
    for t, (ho, hn) in enumerate(HT):
        nisa.tensor_copy(dst=ys[:, ho:ho + hn], src=py[t])
    if NC > 1:
        HW = H // NC
        other = 1 - pid
        peer = _sb((R, HW))
        nisa.sendrecv(src=ys[:, nl.ds(other * HW, HW)], dst=peer, send_to_rank=other, recv_from_rank=other, pipe_id=0)
        yh = _sb((R, HW))
        nisa.tensor_tensor(dst=yh, data1=ys[:, nl.ds(pid * HW, HW)], data2=peer, op=nl.add)
        nisa.dma_copy(dst=y[:, nl.ds(pid * HW, HW)], src=yh)
    else:
        nisa.dma_copy(dst=y, src=ys)
    return y
