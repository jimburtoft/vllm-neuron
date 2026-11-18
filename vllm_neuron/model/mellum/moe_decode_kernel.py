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
def mellum_moe_decode(x, aff, Wgu, Wd, gu_scale=None, d_scale=None, nbuf=4, dedupe=1, dr=0, pipe=0):
    R, H = x.shape
    NC = nl.num_programs(0) if nl.program_ndim() > 0 else 1
    pid = nl.program_id(0) if nl.program_ndim() > 0 else 0
    E = aff.shape[1]
    I2 = Wgu.shape[2]
    I = I2 // 2
    KT = H // 128
    # storage row widths may exceed the compute H (weights zero-padded to Hp for the prefill kernel):
    # only the first H rows of Wgu / columns of Wd are read.
    HSG = Wgu.shape[1]
    HSD = Wd.shape[2]
    WDT = Wgu.dtype
    FP8 = gu_scale is not None
    IC = I // NC  # this core's I rows per expert
    I_TILES = [(o, min(128, IC - o)) for o in range(0, IC, 128)]  # down K tiles of this core's I slice
    HT = _col_tiles(H)  # down output column groups (<= 512, one PSUM bank each)
    GT = [(0, o, n) for (o, n) in _col_tiles(IC)] + [(1, o, n) for (o, n) in _col_tiles(IC)]  # (gate/up, col, n)
    i0 = pid * IC
    # dr=1: FP8 double_row gate/up matmuls (2x contraction per instruction). x is quantized per token to e4m3
    # (scale = amax / 240) and laid out [128, KT/2, 2, Rp] (paired contraction rows kt2*256 + j*128 + p; Rp >= 16,
    # a multiple of 16, so the pair stride is legal). Needs FP8 weights and H % 256 == 0.
    DR = bool(dr) and FP8 and H % 256 == 0
    KT2 = KT // 2
    Rp = ((R + 15) // 16) * 16
    NI = (IC + 127) // 128
    NI2 = (NI + 1) // 2
    DRD = DR and int(dr) == 2  # dr=2: double_row down, per-slot scale, SBUF accumulation
    DR3 = DR and int(dr) == 3  # dr=3: two-pass: all h first, one per-row scale, double_row down in PSUM

    ident = _sb((128, 128), BF)
    io_p = _sb((128, 1))
    nisa.iota(io_p, [[0, 1]], offset=0, channel_multiplier=1)
    io_f = _sb((128, 128))
    nisa.iota(io_f, [[1, 128]], offset=0, channel_multiplier=0)
    nisa.tensor_scalar(dst=ident, data=io_f, op0=nl.equal, operand0=io_p)

    identf_r = _sb((R, R))
    nisa.tensor_copy(dst=identf_r, src=ident[0:R, 0:R])
    ones1r = _sb((1, 128))
    nisa.memset(ones1r, 1.0)
    # x -> K-major tiles xP[p, kt * R + r] = x[r, kt * 128 + p]
    xs = _sb((R, H), BF)
    nisa.dma_copy(dst=xs, src=x)
    xP = _sb((128, KT * R), BF)
    for kt in range(KT):
        pt = nl.ndarray((128, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pt, xs[:, kt * 128:(kt + 1) * 128], ident[0:R, 0:R])
        nisa.tensor_copy(dst=xP[:, kt * R:(kt + 1) * R], src=pt)
    if DR:
        # per-token e4m3 quantization of x: xsc[r] = amax_r / 240, x8 = x / xsc
        xa = _sb((R, H))
        nisa.activation(dst=xa, op=nl.abs, data=xs)
        am = _sb((R, 1))
        nisa.tensor_reduce(dst=am, op=nl.maximum, data=xa, axis=1, keepdims=True)
        nisa.tensor_scalar(dst=am, data=am, op0=nl.maximum, operand0=1e-6)
        xsc = _sb((R, 1))
        nisa.tensor_scalar(dst=xsc, data=am, op0=nl.multiply, operand0=1.0 / 240.0)
        xinv = _sb((R, 1))
        nisa.reciprocal(dst=xinv, data=xsc)
        xq = _sb((R, H), BF)
        nisa.tensor_scalar(dst=xq, data=xs, op0=nl.multiply, operand0=xinv)
        x8 = _sb((128, KT2, 2, Rp), nl.float8_e4m3)
        if Rp > R:
            nisa.memset(x8, 0.0)
        for kt in range(KT):
            pt = nl.ndarray((128, R), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(pt, xq[:, kt * 128:(kt + 1) * 128], ident[0:R, 0:R])
            nisa.tensor_copy(dst=x8[:, kt // 2, kt % 2, 0:R], src=pt)

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

    wd8 = []
    ysb = None
    if DRD:
        for bi in range(nbuf):
            t8 = _sb((128, NI2, 2, H), WDT)
            nisa.memset(t8, 0.0, engine=nisa.gpsimd_engine)
            wd8.append(t8)
        ysb = _sb((R, H))
        nisa.memset(ysb, 0.0)
    sbufs = []
    if FP8:
        for bi in range(nbuf):
            gsc0, dsc0 = _sb((R, 2 * IC)), _sb((R, 1))
            if dedupe:
                nisa.memset(gsc0, 0.0)
                nisa.memset(dsc0, 0.0)
            sbufs.append((gsc0, dsc0))
    py = [nl.ndarray((R, n), dtype=F32, buffer=nl.psum) for (_, n) in HT]
    def _stage_a(j):
        """weight / scale loads + gate|up matmuls + dequant for slot j -> (j, gu, dsc, wd buffers)"""
        wg, wu, wd = bufs[j % nbuf]
        if dedupe:
            iv = u.ap(pattern=[[E + 1, 1], [1, 1]], offset=j)
            dk = dict(scalar_offset=iv, indirect_dim=0)
            ok = dict(oob_mode=nisa.oob_mode.skip)
            base = 0
        else:
            dk, ok = {}, {}
            base = j
        nisa.dma_copy(dst=wg, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * HSG * I2 + i0, **dk), **ok)
        nisa.dma_copy(dst=wu, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * HSG * I2 + I + i0, **dk), **ok)
        if DRD:
            for ti, (o, n) in enumerate(I_TILES):
                nisa.dma_copy(dst=wd8[j % nbuf][0:n, ti // 2, ti % 2, :],
                              src=Wd.ap(pattern=[[HSD, n], [1, H]], offset=base * I * HSD + (i0 + o) * HSD, **dk), **ok)
        else:
            for ti, (o, n) in enumerate(I_TILES):
                nisa.dma_copy(dst=wd[ti], src=Wd.ap(pattern=[[HSD, n], [1, H]], offset=base * I * HSD + (i0 + o) * HSD, **dk), **ok)
        if FP8:
            # scale buffers zeroed once (ring), not per slot: a skipped slot's routing weight is 0 anyway, and the
            # buffers only need finite contents
            gsc, dsc = sbufs[j % nbuf]
            nisa.dma_copy(dst=gsc[:, 0:IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + i0, **dk), **ok)
            nisa.dma_copy(dst=gsc[:, IC:2 * IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + I + i0, **dk), **ok)
            nisa.dma_copy(dst=dsc, src=d_scale.ap(pattern=[[0, R], [1, 1]], offset=base, **dk), **ok)
        # gate | up for this core's I slice: [R, 2 IC] in <= 512-column PSUM groups
        gu = _sb((R, 2 * IC), BF)
        for (which, co, cn) in GT:
            wsrc = wg if which == 0 else wu
            dco = which * IC + co  # column in gu = [gate(IC) | up(IC)]
            if DR:
                w3 = wsrc.reshape((128, KT2, 2, IC))
                pgu = nl.ndarray((Rp, cn), dtype=F32, buffer=nl.psum)
                for kt2 in range(KT2):
                    nisa.nc_matmul(pgu, x8[:, kt2, :, :], w3[:, kt2, :, co:co + cn],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
                # dequant: per-column weight scale (PSUM x SBUF), then per-token x scale (per-partition scalar)
                gtmp = _sb((R, cn))
                nisa.tensor_tensor(dst=gtmp, data1=pgu[0:R, :], data2=gsc[:, dco:dco + cn], op=nl.multiply)
                nisa.tensor_scalar(dst=gu[:, dco:dco + cn], data=gtmp, op0=nl.multiply, operand0=xsc)
            else:
                pgu = nl.ndarray((R, cn), dtype=F32, buffer=nl.psum)
                for kt in range(KT):
                    nisa.nc_matmul(pgu, xP[:, kt * R:(kt + 1) * R], wsrc[:, kt * IC + co: kt * IC + co + cn])
                if FP8:
                    nisa.tensor_tensor(dst=gu[:, dco:dco + cn], data1=pgu, data2=gsc[:, dco:dco + cn], op=nl.multiply)
                else:
                    nisa.tensor_copy(dst=gu[:, dco:dco + cn], src=pgu)
        return (j, gu, (dsc if FP8 else None), wd)

    def _stage_b(st):
        """SiLU * up, routing weight, transpose, down matmuls for a slot from _stage_a"""
        j, gu, dsc, wd = st
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
        if DRD:
            # down in double_row: per-token e4m3 quantization of hw (scale hsc[r] = amax/240, folded back into the
            # y accumulation: y += hsc * (h8 @ Wd8)). The fp32 y accumulator must take the per-row scale before the
            # sum over experts, so each slot accumulates into its own PSUM tile and is added into an SBUF fp32 y.
            ha = _sb((R, IC))
            nisa.activation(dst=ha, op=nl.abs, data=hw)
            hm = _sb((R, 1))
            nisa.tensor_reduce(dst=hm, op=nl.maximum, data=ha, axis=1, keepdims=True)
            nisa.tensor_scalar(dst=hm, data=hm, op0=nl.maximum, operand0=1e-12)
            hsc = _sb((R, 1))
            nisa.tensor_scalar(dst=hsc, data=hm, op0=nl.multiply, operand0=1.0 / 240.0)
            hinv = _sb((R, 1))
            nisa.reciprocal(dst=hinv, data=hsc)
            hq = _sb((R, IC), BF)
            nisa.tensor_scalar(dst=hq, data=hw, op0=nl.multiply, operand0=hinv)
            h8 = _sb((128, NI2, 2, Rp), nl.float8_e4m3)
            if Rp > R or IC % 256:
                nisa.memset(h8, 0.0)
            for ti, (o, n) in enumerate(I_TILES):
                ph = nl.ndarray((n, R), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(ph, hq[:, o:o + n], ident[0:R, 0:R])
                nisa.tensor_copy(dst=h8[0:n, ti // 2, ti % 2, 0:R], src=ph)
            for t, (ho, hn) in enumerate(HT):
                pyd = nl.ndarray((Rp, hn), dtype=F32, buffer=nl.psum)
                for t2 in range(NI2):
                    nisa.nc_matmul(pyd, h8[:, t2, :, :], wd8[j % nbuf][:, t2, :, ho:ho + hn],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
                # ysb += hsc * pyd  (PSUM slice -> SBUF first: the fused SB+SB op needs equal base partitions)
                yt = _sb((R, hn))
                nisa.tensor_scalar(dst=yt, data=pyd[0:R, :], op0=nl.multiply, operand0=hsc)
                nisa.tensor_tensor(dst=ysb[:, ho:ho + hn], data1=ysb[:, ho:ho + hn], data2=yt, op=nl.add)
        else:
            for ti, (o, n) in enumerate(I_TILES):
                ph = nl.ndarray((n, R), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(ph, hw[:, o:o + n], ident[0:R, 0:R])
                hs = _sb((n, R), BF)
                nisa.tensor_copy(dst=hs, src=ph)
                for t, (ho, hn) in enumerate(HT):
                    nisa.nc_matmul(py[t], hs, wd[ti][:, ho:ho + hn])

    # software pipeline (pipe=1): the tensor engine runs in program order, so slot j's gate|up matmuls are
    # emitted before slot j-1's transposes / down matmuls, which wait on the vector engine's SiLU of slot j-1
    if DR3:
        # down weights for ALL slots stay resident (FP8: E * NI2*2*H bytes = 16 x 9.2 KB per partition = 147 KB? too big)
        # -> keep a ring, but issue the first nbuf slots' down loads now so pass 2 starts without waiting
        dbufs = []
        for bi in range(nbuf):
            t8 = _sb((128, NI2, 2, H), WDT)
            nisa.memset(t8, 0.0, engine=nisa.gpsimd_engine)
            dbufs.append(t8)

        def _load_down(j):
            if dedupe:
                ivd = u.ap(pattern=[[E + 1, 1], [1, 1]], offset=j)
                dkd = dict(scalar_offset=ivd, indirect_dim=0)
                okd = dict(oob_mode=nisa.oob_mode.skip)
                based = 0
            else:
                dkd, okd = {}, {}
                based = j
            db = dbufs[j % nbuf]
            for ti, (o, n) in enumerate(I_TILES):
                nisa.dma_copy(dst=db[0:n, ti // 2, ti % 2, :],
                              src=Wd.ap(pattern=[[HSD, n], [1, H]], offset=based * I * HSD + (i0 + o) * HSD, **dkd), **okd)
            return db

        # pass 1: gate|up (double_row) + SiLU * up * routing weight * down scale for every slot -> hall [R, E*IC]
        hall = _sb((R, E * IC), BF)
        # x^T layout for h: per-slot transposes happen in pass 1 right after h is formed (overlaps pass-1 matmuls)
        hT = _sb((128, E, NI, R), BF)
        for j in range(E):
            wg, wu, _ = bufs[j % nbuf]
            if dedupe:
                iv = u.ap(pattern=[[E + 1, 1], [1, 1]], offset=j)
                dk = dict(scalar_offset=iv, indirect_dim=0)
                ok = dict(oob_mode=nisa.oob_mode.skip)
                base = 0
            else:
                dk, ok = {}, {}
                base = j
            nisa.dma_copy(dst=wg, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * HSG * I2 + i0, **dk), **ok)
            nisa.dma_copy(dst=wu, src=Wgu.ap(pattern=[[I2, 128], [128 * I2, KT], [1, IC]], offset=base * HSG * I2 + I + i0, **dk), **ok)
            gsc, dsc = sbufs[j % nbuf]
            nisa.dma_copy(dst=gsc[:, 0:IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + i0, **dk), **ok)
            nisa.dma_copy(dst=gsc[:, IC:2 * IC], src=gu_scale.ap(pattern=[[0, R], [1, IC]], offset=base * I2 + I + i0, **dk), **ok)
            nisa.dma_copy(dst=dsc, src=d_scale.ap(pattern=[[0, R], [1, 1]], offset=base, **dk), **ok)
            gu = _sb((R, 2 * IC), BF)
            for (which, co, cn) in GT:
                wsrc = wg if which == 0 else wu
                dco = which * IC + co
                w3 = wsrc.reshape((128, KT2, 2, IC))
                pgu = nl.ndarray((Rp, cn), dtype=F32, buffer=nl.psum)
                for kt2 in range(KT2):
                    nisa.nc_matmul(pgu, x8[:, kt2, :, :], w3[:, kt2, :, co:co + cn],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
                gtmp = _sb((R, cn))
                nisa.tensor_tensor(dst=gtmp, data1=pgu[0:R, :], data2=gsc[:, dco:dco + cn], op=nl.multiply)
                nisa.tensor_scalar(dst=gu[:, dco:dco + cn], data=gtmp, op0=nl.multiply, operand0=xsc)
            sg = _sb((R, IC), BF)
            nisa.activation(dst=sg, data=gu[:, 0:IC], op=nl.silu)
            hr = _sb((R, IC), BF)
            nisa.tensor_tensor(dst=hr, data1=sg, data2=gu[:, IC:2 * IC], op=nl.multiply)
            wcol = (wsel if dedupe else dense)[:, j:j + 1]
            wds = _sb((R, 1))
            nisa.tensor_tensor(dst=wds, data1=wcol, data2=dsc, op=nl.multiply)
            nisa.tensor_scalar(dst=hall[:, j * IC:(j + 1) * IC], data=hr, op0=nl.multiply, operand0=wds)
            for ti, (o, n) in enumerate(I_TILES):
                ph = nl.ndarray((n, R), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(ph, hall[:, j * IC + o:j * IC + o + n], ident[0:R, 0:R])
                nisa.tensor_copy(dst=hT[0:n, j, ti, :], src=ph)
        # one e4m3 scale per row over all slots: hsc[r] = amax_r / 240
        ha = _sb((R, E * IC))
        nisa.activation(dst=ha, op=nl.abs, data=hall)
        hm = _sb((R, 1))
        nisa.tensor_reduce(dst=hm, op=nl.maximum, data=ha, axis=1, keepdims=True)
        nisa.tensor_scalar(dst=hm, data=hm, op0=nl.maximum, operand0=1e-20)
        hsc = _sb((R, 1))
        nisa.tensor_scalar(dst=hsc, data=hm, op0=nl.multiply, operand0=1.0 / 240.0)
        hinv = _sb((R, 1))
        nisa.reciprocal(dst=hinv, data=hsc)
        # hinv as a row [128, R] (same value down each partition): ones[1,128]^T x hinv^T[1,R] via a K=1 matmul
        hinvT = _sb((1, R))
        pti = nl.ndarray((1, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pti, hinv, identf_r)
        nisa.tensor_copy(dst=hinvT, src=pti)
        pbc = nl.ndarray((128, R), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(pbc, ones1r, hinvT)
        hinvB = _sb((128, R))
        nisa.tensor_copy(dst=hinvB, src=pbc)
        h8 = _sb((128, E, NI2, 2, Rp), nl.float8_e4m3)
        nisa.memset(h8, 0.0)
        for j in range(E):
            for ti, (o, n) in enumerate(I_TILES):
                nisa.tensor_tensor(dst=h8[0:n, j, ti // 2, ti % 2, 0:R], data1=hT[0:n, j, ti, :], data2=hinvB[0:n, :],
                                   op=nl.multiply)
        # pass 2: down in double_row, accumulated over slots in PSUM (scale applied once at the end)
        py3 = [nl.ndarray((Rp, n), dtype=F32, buffer=nl.psum) for (_, n) in HT]
        for j in range(E):
            db = _load_down(j)
            for t, (ho, hn) in enumerate(HT):
                for t2 in range(NI2):
                    nisa.nc_matmul(py3[t], h8[:, j, t2, :, :], db[:, t2, :, ho:ho + hn],
                                   perf_mode=nisa.matmul_perf_mode.double_row)
        ys3 = _sb((R, H))
        for t, (ho, hn) in enumerate(HT):
            nisa.tensor_scalar(dst=ys3[:, ho:ho + hn], data=py3[t][0:R, :], op0=nl.multiply, operand0=hsc)
    elif pipe:
        prev = None
        for j in range(E):
            cur = _stage_a(j)
            if prev is not None:
                _stage_b(prev)
            prev = cur
        _stage_b(prev)
    else:
        for j in range(E):
            _stage_b(_stage_a(j))

    y = nl.ndarray((R, H), dtype=F32, buffer=nl.shared_hbm)
    if DRD:
        ys = ysb
    elif DR3:
        ys = ys3
    else:
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
