"""Job-list prefill MoE for small-T, null-routed MoE prefill (action path, T=512).

Why: nkilib bwmm_shard_on_block at T=512 / block 128 sets up ~36k token slots for ~1.75k real routes (5 % fill)
and spends most of its time on per-block gathers, memsets and the gather-add-scatter chain (profiling).
Here every used expert is processed ONCE per layer for all of its tokens (<= 128 per job), unused experts move no
weight bytes, and the top-k combine is a gather-accumulate over a route-compact staging buffer (no RMW chain).

Per rank (TP over the expert intermediate, ep=1): Wgu [E, H, 2I] (gate | up), Wd [E, I, H], I = 128.

A JOB is (expert, tile of <= 128 tokens). The torch prep (moe_job_prep below; compares + one-hot matmuls, no
scatters: a scatter into the 36k-slot lane table costs ~2 ms as a graph op and ~0.75 ms as serialized indirect DMAs)
gives, for J = E + OV jobs:
  tokT [128, J] int32 token id of lane l of job j (T = out of bounds = padding lane).
  wT   [128, J] fp32  routing weight of that lane (0 for padding).
  zidT [128, J] int32 staging row of that lane = its route id t * K + k (T * K + 128 = past Z for padding lanes).
  jexp [1, J] int32   expert of job j (j < E: expert j; j >= E: overflow tiles of hot experts). E (out of bounds) =
                      empty job: its weight DMAs are skipped (oob_mode.skip) and none of its lanes is scattered.
  zrow [T, K] int32   staging row t * K + k of a real pick, T * K (a zero row) for null / padding picks.
Kernel (grid = 2 pcores): pcore c
  1. runs jobs c, c + 2, ...: gather the job's token rows -> PE transposes to the pmaj column layout (contraction
     row h = p * KT + kt, so an expert's [H, 2I] slice is 8 KB contiguous per partition) -> gu = x Wgu[e] ->
     h = silu(g) * u * w (bf16, score after SwiGLU and before down, HF order) -> y = h Wd[e] -> scatter the lanes to
     Z[route id] (padding lanes are out of bounds: skipped, so their garbage never reaches the output).
  2. core barrier on Z, then combines token tiles c, c + 2, ...: out[t] = sum_k Z[zrow[t, k]] (fp32 accumulate).
Output: routed partial [T, H] bf16 (the caller adds the shared expert and all-reduces over TP).
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl

F32, BF, I32 = nl.float32, nl.bfloat16, nl.int32
P = 128
# Rows of a job are independent (row l of every matmul output depends only on lane l), and empty / padding lanes
# are never scattered (their staging row is out of bounds), so their garbage never reaches the output: the weight
# and token-row memsets are only needed for the simulator (it NaN-initializes SBUF). MOEJ_MEMSET=1 restores them.
_MEMSET = os.environ.get("MOEJ_MEMSET", "0") == "1"
_NBUF = int(os.environ.get("MOEJ_NBUF", "3"))  # rotating weight / token buffer sets (1 = no prefetch)
_ILV = os.environ.get("MOEJ_ILV", "1") == "1"  # 2-job software pipeline (needs MOEJ_NBUF >= 3); 988 -> 901 us
_FUSE = os.environ.get("MOEJ_FUSE", "1") == "1"  # SwiGLU * w straight from PSUM
_EVSPLIT = os.environ.get("MOEJ_EVSPLIT", "0") == "1"  # down-proj PSUM evictions half on ScalarE
# PSUM evictions of the x transposes on VectorE (pe) or ScalarE (act). MOEJ_XT=dma (DMA gather-transpose
# from HBM, no PE transposes) was measured 4.05 ms vs 1.07 ms (DMA-bound on 2-byte-row descriptors) and removed.
_XT = os.environ.get("MOEJ_XT", "pe")
# gate_up orientation. xs = x^T stationary [128 k, 128 tokens], W moving [128 k, 256] -> PSUM [tokens, 2I]
# (16 LDWEIGHTS of x + 16 matmuls N=256). ws = W stationary, x^T moving -> PSUM [2I as 2x128, tokens] (32 LDWEIGHTS
# of W, 32 matmuls N=128) but h comes out transposed [I, tokens], which is the down-projection's stationary layout:
# the h transpose + its eviction disappear, and the routing weight becomes a free-dim (per-token) broadcast.
_GU = os.environ.get("MOEJ_GU", "xs")  # ws measured 1.108 vs 1.059 ms (more LDWEIGHTS): kept off
# dge_mode of the dynamic-offset (scalar_offset) weight DMAs. sw = GpSimd generates the descriptors
# (default), hw = the issuing engine's hardware DGE (frees GpSimd for the token gathers / Z scatters).
_WDGE = os.environ.get("MOEJ_WDGE", "sw")
# where the lane tables are built. torch = moe_job_prep one-hot matmuls ([R, J] fp32 graph tensors:
# ~240 us/layer, spill-bound); kernel = torch passes only job/lane/weight per pick [T, K] and each core builds the
# tables of ITS jobs on chip (iota compares + one-hot matmuls accumulated in PSUM).
_PREP = os.environ.get("MOEJ_PREP", "kernel")
_COMPACT = os.environ.get("MOEJ_COMPACT", "0") == "1"  # torch-prep form of the compaction (A/B only)
# static job cap (kernel prep only; jobs are compactly numbered there). Jobs past the cap run in a runtime
# tail loop with the same trip count on both cores (lock-step for core_barrier), so any cap is CORRECT. 224 covers
# every layer of the action prefill (137-219 jobs at L=462, route_stats.py): the tail runs zero trips there.
# 0 = static loop over all J jobs.
# "auto" = 224 at T <= 512 (the B=1 action prefill), static over all J for larger packed batches (at
# T = B * 512 most of the J = 256 + T*K/128 jobs are used, and the runtime tail has no prefetch).
_CAP_ENV = os.environ.get("MOEJ_CAP", "auto")


def _cap_for(T, J):
    if _CAP_ENV == "auto":
        return 224 if T <= 512 else 0
    return int(_CAP_ENV)


_CAP = 224  # legacy name (tests / docs); the kernel and the prep use _cap_for(T, J)


def _sb(shape, dt=F32):
    return nl.ndarray(shape, dtype=dt, buffer=nl.sbuf)


def _lane_tables(jl, T, K, J, NC, pid):
    """On-chip lane tables of this core's jobs (columns c <-> global job NC * c + pid). jl [T, K, 3] fp32 HBM:
    (job id or -1, lane, routing weight) per pick. Route tile (token tile tt, pick k): partitions = tokens.
    L[l, c] += sum_r A[r, l] * B[r, c] * val_r with A = onehot(lane), B = onehot(job) -- every (l, c) gets at most
    one route, so the fp32 PSUM sums are exact. val = (token + 1, weight, route id + 1); 0 marks an empty lane."""
    JL = J // NC
    NT = T // P
    iJ = _sb((P, JL))
    nisa.iota(iJ, [[NC, JL]], offset=pid, channel_multiplier=0)  # global job id of local column c
    iL = _sb((P, P))
    nisa.iota(iL, [[1, P]], offset=0, channel_multiplier=0)  # lane id
    W3 = 3 * JL
    NBK = -(-W3 // 512)  # PSUM banks of <= 512 fp32 columns (matmul moving free <= 512)
    pLs = [nl.ndarray((P, min(512, W3 - bk * 512)), dtype=F32, buffer=nl.psum) for bk in range(NBK)]
    for tt in range(NT):
        jt = _sb((P, K, 3))
        nisa.dma_copy(dst=jt, src=jl[tt * P:(tt + 1) * P, :, :])
        for k in range(K):
            B = _sb((P, JL))
            nisa.tensor_scalar(dst=B, data=iJ, op0=nl.equal, operand0=jt[:, k, 0:1])
            A = _sb((P, P))
            nisa.tensor_scalar(dst=A, data=iL, op0=nl.equal, operand0=jt[:, k, 1:2])
            mv = _sb((P, 3 * JL))
            t1 = _sb((P, 1))
            nisa.iota(t1, [[0, 1]], offset=tt * P + 1, channel_multiplier=1)
            r1 = _sb((P, 1))
            nisa.iota(r1, [[0, 1]], offset=tt * P * K + k + 1, channel_multiplier=K)
            nisa.tensor_scalar(dst=mv[:, 0:JL], data=B, op0=nl.multiply, operand0=t1)
            nisa.tensor_scalar(dst=mv[:, JL:2 * JL], data=B, op0=nl.multiply, operand0=jt[:, k, 2:3],
                               engine=nisa.gpsimd_engine)
            nisa.tensor_scalar(dst=mv[:, 2 * JL:3 * JL], data=B, op0=nl.multiply, operand0=r1)
            for bk in range(NBK):
                nisa.nc_matmul(pLs[bk], A, mv[:, bk * 512:bk * 512 + pLs[bk].shape[1]])
    Lt = _sb((P, W3))
    for bk in range(NBK):
        nisa.tensor_copy(dst=Lt[:, bk * 512:bk * 512 + pLs[bk].shape[1]], src=pLs[bk])
    occ = _sb((P, JL))
    nisa.tensor_scalar(dst=occ, data=Lt[:, 0:JL], op0=nl.greater, operand0=0.0)
    # empty lane -> token T (out of bounds) and route T * K + 128 (past Z); real lane -> value - 1
    tf = _sb((P, JL))
    nisa.tensor_scalar(dst=tf, data=Lt[:, 0:JL], op0=nl.add, operand0=float(T))
    tf2 = _sb((P, JL))
    nisa.scalar_tensor_tensor(dst=tf2, data=occ, op0=nl.multiply, operand0=-float(T + 1), op1=nl.add, operand1=tf)
    tok = _sb((P, JL), I32)
    nisa.tensor_copy(dst=tok, src=tf2)
    zf = _sb((P, JL))
    nisa.tensor_scalar(dst=zf, data=Lt[:, 2 * JL:3 * JL], op0=nl.add, operand0=float(T * K + P))
    zf2 = _sb((P, JL))
    nisa.scalar_tensor_tensor(dst=zf2, data=occ, op0=nl.multiply, operand0=-float(T * K + P + 1), op1=nl.add,
                              operand1=zf)
    zid = _sb((P, JL), I32)
    nisa.tensor_copy(dst=zid, src=zf2)
    wts = _sb((P, JL))
    nisa.tensor_copy(dst=wts, src=Lt[:, JL:2 * JL])
    return tok, wts, zid


@nki.jit
def moe_prefill_jobs(x, tokT, wT, zidT, jexp, zrow, Wgu, Wd, jl=None, ntail=None):
    T, H = x.shape
    E, _, I2 = Wgu.shape
    I = I2 // 2
    J = jexp.shape[1]
    K = zrow.shape[1]
    KT = H // P
    NB = H // 512  # down-projection PSUM banks
    NC = nl.num_programs(0)
    pid = nl.program_id(0) if NC > 1 else 0

    out = nl.ndarray((T, H), dtype=BF, buffer=nl.shared_hbm)
    Z = nl.ndarray((T * K + P, H), dtype=BF, buffer=nl.shared_hbm)  # route staging; rows >= T*K are zero (null picks)

    ident = _sb((P, P), BF)
    nisa.memset(ident, 0.0)
    iot = _sb((P, P))
    nisa.iota(iot, [[1, P]], offset=0, channel_multiplier=-1)  # iot[p, f] = f - p
    nisa.tensor_scalar(dst=ident, data=iot, op0=nl.equal, operand0=0.0)

    identf = _sb((P, P))
    nisa.tensor_copy(dst=identf, src=ident)
    onesf = _sb((P, I))
    nisa.memset(onesf, 1.0)
    zt = _sb((P, H), BF)
    nisa.memset(zt, 0.0)
    nisa.dma_copy(dst=Z[T * K:T * K + P, :], src=zt)  # both pcores write the same zeros (no ordering needed)

    if jl is None:
        tok = _sb((P, J), I32)
        nisa.dma_copy(dst=tok, src=tokT)
        wts = _sb((P, J))
        nisa.dma_copy(dst=wts, src=wT)
        zid = _sb((P, J), I32)
        nisa.dma_copy(dst=zid, src=zidT)
        jcol = lambda i: NC * i + pid  # noqa: E731  (lane-table column of local job i)
    else:
        tok, wts, zid = _lane_tables(jl, T, K, J, NC, pid)
        jcol = lambda i: i  # noqa: E731  (this core's tables hold only its own jobs)
    js = _sb((1, J), I32)
    nisa.dma_copy(dst=js, src=jexp)

    _cp = _cap_for(T, J)
    CAPJ = min(J, _cp) if (_cp and jl is not None) else J  # static jobs (compact numbering: kernel prep only)
    NJ = CAPJ // NC
    NBUF = _NBUF
    dk = {"dge_mode": nisa.dge_mode.hwdge} if _WDGE == "hw" else {}

    def load_w(eoff, wgu, wd, xg, tokc):  # weights of expert eoff (scalar_offset AP) + the job's token rows
        # pmaj layout: wgu[p, kt * I2 + c] = Wgu[e, p * KT + kt, c] (8 KB contiguous per partition)
        nisa.dma_copy(dst=wgu, src=Wgu.ap(pattern=[[KT * I2, P], [1, KT * I2]], offset=0, scalar_offset=eoff,
                                          indirect_dim=0), oob_mode=nisa.oob_mode.skip, **dk)
        nisa.dma_copy(dst=wd, src=Wd.ap(pattern=[[H, I], [1, H]], offset=0, scalar_offset=eoff, indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip, **dk)
        nisa.dma_copy(dst=xg, src=x.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=tokc, indirect_dim=0),
                      oob_mode=nisa.oob_mode.skip)

    def stage_a(wgu, xg, wcol):
        # pmaj columns: xP[p, kt * P + l] = x[tok_l, p * KT + kt]
        # is_transpose: bit-accurate data movement, so a garbage (unwritten) padding lane stays in its own column
        # instead of leaking NaN * 0 into every real lane as a plain identity matmul would
        xP = _sb((P, KT * P), BF)
        for g in range(KT // 4):
            tp = nl.ndarray((P, 4 * P), dtype=BF, buffer=nl.psum)
            for q in range(4):
                kt = 4 * g + q
                nisa.nc_matmul(tp[:, q * P:(q + 1) * P], xg.ap(pattern=[[H, P], [KT, P]], offset=kt), ident,
                               is_transpose=True)
            if _XT == "act":
                nisa.activation(dst=xP[:, g * 4 * P:(g + 1) * 4 * P], data=tp, op=nl.copy)
            else:
                nisa.tensor_copy(dst=xP[:, g * 4 * P:(g + 1) * 4 * P], src=tp)
        if _GU == "ws":
            # routing weights as a row broadcast over the I partitions: wrow[i, l] = w_l = (ones[128,128]^T diag(w))
            wd_ = _sb((P, P))
            nisa.tensor_scalar(dst=wd_, data=identf, op0=nl.multiply, operand0=wcol)
            pw = nl.ndarray((I, P), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(pw, onesf, wd_)
            wrow = _sb((I, P))
            nisa.tensor_copy(dst=wrow, src=pw)
            # gT[i, l] / uT[i, l] = sum_h W[h, i] x[l, h]; one PSUM tile [128, 2 * 128]: cols 0:128 = g, 128:256 = u
            pT = nl.ndarray((I, 2 * P), dtype=F32, buffer=nl.psum)
            for half in range(2):
                for kt in range(KT):
                    nisa.nc_matmul(pT[:, half * P:(half + 1) * P], wgu[:, kt * I2 + half * I:kt * I2 + (half + 1) * I],
                                   xP[:, kt * P:(kt + 1) * P])
            sgT = _sb((I, P), BF)
            nisa.activation(dst=sgT, data=pT[:, 0:P], op=nl.silu)
            hs = _sb((I, P), BF)
            # (u * w_l) * silu(g): w is per token = per free column here -> broadcast row wrow [I, 128]
            nisa.tensor_tensor(dst=hs, data1=pT[:, P:2 * P], data2=sgT, op=nl.multiply)
            hsw = _sb((I, P), BF)
            nisa.tensor_tensor(dst=hsw, data1=hs, data2=wrow, op=nl.multiply)
            hs = hsw
        else:
            pgu = nl.ndarray((P, I2), dtype=F32, buffer=nl.psum)
            for kt in range(KT):
                nisa.nc_matmul(pgu, xP[:, kt * P:(kt + 1) * P], wgu[:, kt * I2:(kt + 1) * I2])
            hs = _gu_tail(pgu, wcol)
        return hs

    def stage_b(hs, wd, zcol):
        y = _sb((P, H), BF)
        for b in range(NB):
            py = nl.ndarray((P, 512), dtype=F32, buffer=nl.psum)
            nisa.nc_matmul(py, hs, wd[:, b * 512:(b + 1) * 512])
            if _EVSPLIT and b % 2 == 0:  # split the PSUM evictions over ScalarE and VectorE
                nisa.activation(dst=y[:, b * 512:(b + 1) * 512], data=py, op=nl.copy)
            else:
                nisa.tensor_copy(dst=y[:, b * 512:(b + 1) * 512], src=py)
        nisa.dma_copy(dst=Z.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=zcol, indirect_dim=0),
                      src=y, oob_mode=nisa.oob_mode.skip)

    def compute(wgu, wd, xg, wcol, zcol):
        stage_b(stage_a(wgu, xg, wcol), wd, zcol)

    def _gu_tail(pgu, wcol):
        hw = _sb((P, I), BF)
        if _FUSE:
            # silu straight from PSUM (ScalarE), then (up * w) * silu in one DVE op. Rounding: silu(g) and the
            # product round once each to bf16 (unfused: gu, silu, silu*u, *w each round) -- within bf16 noise.
            sg = _sb((P, I), BF)
            nisa.activation(dst=sg, data=pgu[:, 0:I], op=nl.silu)
            nisa.scalar_tensor_tensor(dst=hw, data=pgu[:, I:I2], op0=nl.multiply, operand0=wcol,
                                      op1=nl.multiply, operand1=sg)
        else:
            gu = _sb((P, I2), BF)
            nisa.tensor_copy(dst=gu, src=pgu)
            sg = _sb((P, I), BF)
            nisa.activation(dst=sg, data=gu[:, 0:I], op=nl.silu)
            hh = _sb((P, I), BF)
            nisa.tensor_tensor(dst=hh, data1=sg, data2=gu[:, I:I2], op=nl.multiply)
            nisa.tensor_scalar(dst=hw, data=hh, op0=nl.multiply, operand0=wcol)
        ph = nl.ndarray((I, P), dtype=BF, buffer=nl.psum)
        nisa.nc_matmul(ph, hw, ident, is_transpose=True)
        hs = _sb((I, P), BF)
        nisa.tensor_copy(dst=hs, src=ph)
        return hs

    def load(i):  # job (NC * i + pid) into rotating buffer set i % NBUF
        j = NC * i + pid
        c = jcol(i)
        wgu, wd, xg = bufs[i % NBUF]
        load_w(js.ap(pattern=[[J, 1], [1, 1]], offset=j), wgu, wd, xg, tok[:, c:c + 1])

    bufs = []
    for b_ in range(NBUF):
        wgu_b = _sb((P, KT * I2), BF)
        wd_b = _sb((I, H), BF)
        xg_b = _sb((P, KT * P), BF)
        if _MEMSET:  # simulator only (it NaN-initializes SBUF); on HW empty / padding lanes are never scattered
            nisa.memset(wgu_b, 0.0)
            nisa.memset(wd_b, 0.0)
            nisa.memset(xg_b, 0.0)
        bufs.append((wgu_b, wd_b, xg_b))
    for i in range(min(NBUF - 1, NJ)):
        load(i)
    if _ILV and NBUF >= 3 and NJ > 0:
        # 2-job software pipeline -- stage A (transposes + gate_up + SwiGLU) of job i+1 is issued before
        # stage B (down + scatter) of job i, so the PE has independent work while job i's SwiGLU runs on ACT/DVE.
        # Needs NBUF >= 3 (job i's wd, job i+1's wgu/xg and job i+2's prefetch are live at once).
        c0 = jcol(0)
        hs_prev = stage_a(bufs[0][0], bufs[0][2], wts[:, c0:c0 + 1])
        for i in range(NJ):
            c = jcol(i)
            if i + NBUF - 1 < NJ:
                load(i + NBUF - 1)
            hs_next = None
            if i + 1 < NJ:
                cn = jcol(i + 1)
                bn = bufs[(i + 1) % NBUF]
                hs_next = stage_a(bn[0], bn[2], wts[:, cn:cn + 1])
            stage_b(hs_prev, bufs[i % NBUF][1], zid[:, c:c + 1])
            hs_prev = hs_next
    else:
        for i in range(NJ):
            c = jcol(i)
            if i + NBUF - 1 < NJ:
                load(i + NBUF - 1)  # prefetch: overlaps this job's compute
            wgu, wd, xg = bufs[i % NBUF]
            compute(wgu, wd, xg, wts[:, c:c + 1], zid[:, c:c + 1])

    if CAPJ < J:
        # Runtime tail for jobs CAPJ.. (compact numbering: non-empty only when a layer has > CAPJ jobs). Trip count
        # ntail = ceil(max(nj - CAPJ, 0) / NC) is the same on both cores (lock-step for core_barrier); job past nj
        # are empty (expert E: weight DMAs skipped, lanes out of bounds). Tail columns are first compacted into
        # offset-0 tiles (static offset + register scalar_offset was mis-lowered on HW, mk016).
        NX = (J - CAPJ) // NC
        c0 = CAPJ // NC
        jtl = _sb((1, NX), I32)
        nisa.tensor_copy(dst=jtl, src=js.ap(pattern=[[J, 1], [NC, NX]], offset=CAPJ + pid))
        tkl = _sb((P, NX), I32)
        nisa.tensor_copy(dst=tkl, src=tok[:, c0:c0 + NX])
        wtl = _sb((P, NX))
        nisa.tensor_copy(dst=wtl, src=wts[:, c0:c0 + NX])
        zdl = _sb((P, NX), I32)
        nisa.tensor_copy(dst=zdl, src=zid[:, c0:c0 + NX])
        nt_s = _sb((1, 1), I32)
        nisa.dma_copy(dst=nt_s, src=ntail)
        nt = nisa.scalar_load(nt_s)
        twgu, twd, txg = bufs[0]

        def tail(t):
            e1 = _sb((1, 1), I32)
            nisa.dma_copy(dst=e1, src=jtl.ap(pattern=[[NX, 1], [1, 1]], offset=0, scalar_offset=t, indirect_dim=1))
            tc = _sb((P, 1), I32)
            nisa.dma_copy(dst=tc, src=tkl.ap(pattern=[[NX, P], [1, 1]], offset=0, scalar_offset=t, indirect_dim=1))
            wc = _sb((P, 1))
            nisa.dma_copy(dst=wc, src=wtl.ap(pattern=[[NX, P], [1, 1]], offset=0, scalar_offset=t, indirect_dim=1))
            zc = _sb((P, 1), I32)
            nisa.dma_copy(dst=zc, src=zdl.ap(pattern=[[NX, P], [1, 1]], offset=0, scalar_offset=t, indirect_dim=1))
            load_w(e1, twgu, twd, txg, tc)
            compute(twgu, twd, txg, wc, zc)

        nl.fori_loop(0, nt, tail)

    if NC > 1:
        nisa.core_barrier(Z, (0, 1))

    for i in range(T // P // NC):
        tt = NC * i + pid
        zr = _sb((P, K), I32)
        nisa.dma_copy(dst=zr, src=zrow[tt * P:(tt + 1) * P, :])
        acc = _sb((P, H))
        nisa.memset(acc, 0.0)
        for k in range(K):
            g = _sb((P, H), BF)
            nisa.dma_copy(dst=g, src=Z.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=zr[:, k:k + 1],
                                          indirect_dim=0))
            nisa.tensor_tensor(dst=acc, data1=acc, data2=g, op=nl.add)
        ob = _sb((P, H), BF)
        nisa.tensor_copy(dst=ob, src=acc)
        nisa.dma_copy(dst=out[tt * P:(tt + 1) * P, :], src=ob)
    return out


def num_jobs(T, E, K=8):
    """J = E primary jobs + OV overflow tiles. OV = T * K // 128 covers every routing at T tokens (an expert with
    c tokens needs ceil(c / 128) - 1 overflow tiles; the sum over experts is <= T * K / 128). Rounded to even."""
    ov = (T * K) // P
    j = E + ov
    return j + (j % 2)


def moe_job_lanes(topv, topi, valid, E, J):
    """MOEJ_PREP=kernel: routing [T, K] -> (jl [T, K, 3] fp32 = (job or -1, lane, weight), jexp [1, J], zrow [T, K],
    ntail [1, 1] int32 = runtime tail trips beyond MOEJ_CAP static jobs; jobs are numbered compactly, see below).
    Only [T, E]-sized torch ops; the kernel builds the [128, J] lane tables itself."""
    import torch

    T, K = topi.shape
    dev = topi.device
    f = torch.float32
    ar_e = torch.arange(E, device=dev, dtype=topi.dtype)
    oh = (topi.unsqueeze(-1) == ar_e) & valid.unsqueeze(-1)  # [T, K, E]
    m = oh.to(f).sum(1)  # [T, E] 0/1
    tril = torch.ones(T, T, dtype=f, device=dev).tril()
    pos = tril @ m - m
    cnt = m.sum(0)
    extra = torch.clamp(torch.ceil(cnt / P) - 1, min=0)
    trile = torch.ones(E, E, dtype=f, device=dev).tril()
    cum = (trile @ extra.unsqueeze(1)).squeeze(1)
    ovb = cum - extra
    p_k = pos.gather(1, topi.long())
    tile = torch.floor(p_k / P)
    lane = p_k - tile * P
    ovb_k = ovb.index_select(0, topi.long().reshape(-1)).reshape(T, K)
    # COMPACT job numbering -- used experts get jobs 0..U-1 (in expert order), overflow tiles follow at
    # U + ovb + tile - 1, and every job >= U + cum[-1] is empty. The kernel's static loop covers the first
    # MOEJ_CAP jobs and a runtime tail covers the rest (normally zero trips).
    used = (cnt > 0).to(f)
    rank = (trile @ used.unsqueeze(1)).squeeze(1) - used  # exclusive: job of a used expert
    U = used.sum()
    rank_k = rank.index_select(0, topi.long().reshape(-1)).reshape(T, K)
    job = torch.where(tile == 0, rank_k, U + ovb_k + tile - 1)
    job = torch.where(valid, job, torch.full_like(job, -1.0))
    jl = torch.stack([job, lane, topv.to(f) * valid.to(f)], -1).contiguous()
    rid = torch.arange(T * K, device=dev, dtype=f).view(T, K)
    zrow = torch.where(valid, rid, torch.full_like(rid, float(T * K))).to(torch.int32)
    ar_jf = torch.arange(J, device=dev, dtype=f)
    # expert of job j: primary (j < U) -> the used expert of rank j; overflow (U <= j < U + cum[-1]) -> the expert
    # whose overflow range holds j - U; else E (empty). Both as one-hot sums over E (exact).
    ar_ef = torch.arange(E, device=dev, dtype=f)
    pri = ((rank.unsqueeze(0) == ar_jf.unsqueeze(1)) & (used.unsqueeze(0) > 0)).to(f)  # [J, E]
    so = ar_jf - U
    ovm = ((so.unsqueeze(1) >= ovb.unsqueeze(0)) & (so.unsqueeze(1) < cum.unsqueeze(0))).to(f)  # [J, E]
    sel = pri + ovm
    jexp = torch.where(sel.sum(1) > 0, (sel * ar_ef.unsqueeze(0)).sum(1), torch.full_like(ar_jf, float(E)))
    nj = U + cum[-1]  # number of non-empty jobs
    _cp = _cap_for(T, J)
    capj = min(J, _cp) if _cp else J
    ntail = torch.clamp(torch.ceil((nj - capj) / 2), min=0)  # uniform runtime tail trips (NC = 2)
    return jl, jexp.to(torch.int32).unsqueeze(0), zrow, ntail.to(torch.int32).reshape(1, 1)


def moe_job_prep(topv, topi, valid, E, J):
    """torch (device or CPU): routing [T, K] -> (tokT, wT, zidT [128, J], jexp [1, J], zrow [T, K]).
    topv/topi: renormed top-K weights / expert ids, valid [T, K] bool (real pick of a real token).
    Every lane-table entry is a sum with exactly one (or no) nonzero term, so the fp32 matmuls are exact."""
    import torch

    T, K = topi.shape
    dev = topi.device
    f = torch.float32
    R = T * K
    ar_e = torch.arange(E, device=dev, dtype=topi.dtype)
    m = ((topi.unsqueeze(-1) == ar_e) & valid.unsqueeze(-1)).to(f).sum(1)  # [T, E] 0/1
    tril = torch.ones(T, T, dtype=f, device=dev).tril()
    pos = tril @ m - m  # exclusive running count of each expert's tokens (exact in fp32)
    cnt = m.sum(0)  # [E]
    extra = torch.clamp(torch.ceil(cnt / P) - 1, min=0)  # overflow tiles per expert
    trile = torch.ones(E, E, dtype=f, device=dev).tril()
    cum = (trile @ extra.unsqueeze(1)).squeeze(1)  # inclusive
    ovb = cum - extra
    p_k = pos.gather(1, topi.long())  # [T, K] position of the token in its expert's list
    tile = torch.floor(p_k / P)
    lane = (p_k - tile * P).reshape(R)
    ovb_k = ovb.index_select(0, topi.long().reshape(-1)).reshape(T, K)
    job = torch.where(tile == 0, topi.to(f), E + ovb_k + tile - 1).reshape(R)
    vr = valid.reshape(R).to(f)
    A = (lane.unsqueeze(0) == torch.arange(P, device=dev, dtype=f).unsqueeze(1)).to(f)  # [128, R] lane one-hot
    B = (job.unsqueeze(1) == torch.arange(J, device=dev, dtype=f).unsqueeze(0)).to(f) * vr.unsqueeze(1)  # [R, J]
    rid = torch.arange(R, device=dev, dtype=f)
    tvals = torch.floor(rid / K)
    occ = A @ B
    tok = A @ (B * tvals.unsqueeze(1))
    wt = A @ (B * (topv.to(f).reshape(R) * vr).unsqueeze(1))
    zid = A @ (B * rid.unsqueeze(1))
    tokT = torch.where(occ > 0, tok, torch.full_like(tok, float(T))).to(torch.int32)
    zidT = torch.where(occ > 0, zid, torch.full_like(zid, float(R + P))).to(torch.int32)
    ridm = rid.view(T, K)
    zrow = torch.where(valid, ridm, torch.full_like(ridm, float(R))).to(torch.int32)
    s = torch.arange(J - E, device=dev, dtype=f)
    jov = (cum.unsqueeze(0) <= s.unsqueeze(1)).to(f).sum(1)  # expert of overflow slot s
    jov = torch.where(s < cum[-1], jov, torch.full_like(jov, float(E)))
    jpr = torch.where(cnt > 0, torch.arange(E, device=dev, dtype=f), torch.full_like(cnt, float(E)))
    jexp = torch.cat([jpr, jov]).to(torch.int32).unsqueeze(0)
    if _COMPACT:
        # used jobs first (stable), empty jobs last: new column c = old column perm[c]; one [J, J] permutation matmul
        used = (jexp[0] < E).to(f)
        ar_j = torch.arange(J, device=dev, dtype=f)
        nused = used.sum()
        rk = torch.where(used > 0, (torch.ones(J, J, dtype=f, device=dev).tril() @ used.unsqueeze(1)).squeeze(1) - 1,
                         nused + (torch.ones(J, J, dtype=f, device=dev).tril() @ (1 - used).unsqueeze(1)).squeeze(1) - 1)
        Pm = (rk.unsqueeze(1) == ar_j.unsqueeze(0)).to(f)  # [old, new]
        tokT = (tokT.to(f) @ Pm).to(torch.int32)
        wt = wt @ Pm
        zidT = (zidT.to(f) @ Pm).to(torch.int32)
        jexp = (jexp.to(f) @ Pm).to(torch.int32)
    return tokT, wt, zidT, jexp, zrow
