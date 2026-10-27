# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the query-split sparse prefill (``HY4_PREFILL_SPARSE_NKI=1``).

``select_to_rows`` -- per query, the first ``ksel`` positions of a 0/1
selection, ascending, padded with the selection width (out of range). The
torch reference is ``HYV4Attention._prefill_rows_ref`` in ``model.py``.

``sparse_prefill_attn`` -- the sparse prefill attention with batched gathers
(``HY4_PREFILL_SPARSE_BACKEND=cte``), laid out after nkilib's
``mla_sparse_attention_cte`` (DMA path). Same outputs as
``sparse_decode_attn`` in position mode; the torch reference is
``sparse_prefill_cte_ref`` in ``model.py``.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128


def _kassert(c, msg=""):
    assert c, msg


def _programs():
    """``(n_programs, program_id)`` on grid axis 0; ``(1, 0)`` without a grid."""
    if nl.program_ndim() == 0:
        return 1, 0
    return nl.num_programs(axes=0), nl.program_id(axis=0)

_GPSIMD_STRIDE = 16  # nonzero_with_count reads/writes partitions 0, 16, ..., 112 only
_GPSIMD_CORES = 8


@nki.jit
def select_to_rows(
    sel,  # [Q, P] fp32 0/1 selection over the prefix keys
    sel2=None,  # [Q, C] fp32 0/1 selection over the chunk's keys (positions P ..)
    ksel=2176,  # slots per query that the caller keeps
    chunk=2048,  # keys per nonzero_with_count call
):
    """``rows[q, :ksel]`` = the first ``ksel`` positions selected in
    ``[sel | sel2]`` (``sel2`` offset by ``P``; never concatenated), ascending,
    then ``P + C`` (out of range for the attention kernel).

    Output is ``[Q, ksel + chunk]``; the caller keeps ``[:, :ksel]`` (the tail
    absorbs a chunk written at a count already clamped to ``ksel``).

    The GpSimd ``nonzero_with_count`` compacts one query's chunk per GpSimd
    core (8 queries per call, on partitions 0, 16, ..., 112), and each chunk's
    indices are written into the query's output row at its running count,
    clamped to ``ksel`` (the pattern of nkilib's ``find_nonzero_indices``).
    Launched at the logical NC config: queries are split across programs.
    """
    Q, P = sel.shape
    C = 0 if sel2 is None else sel2.shape[1]
    K = P + C
    n_prg, prg = _programs()
    _kassert(Q % n_prg == 0, "queries must divide by the number of programs")
    QP = Q // n_prg
    q_base = prg * QP
    CH = min(chunk, max(P, C))
    # (source, first column, width, position offset) per nonzero_with_count call
    pieces = []
    for c0 in range(0, P, CH):
        pieces.append((sel, c0, min(CH, P - c0), c0))
    for c0 in range(0, C, CH):
        pieces.append((sel2, c0, min(CH, C - c0), P + c0))
    WS = ksel + CH
    rows_out = nl.ndarray((Q, WS), dtype=nl.int32, buffer=nl.shared_hbm)

    fill = nl.ndarray((QP, WS), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=fill, value=K)
    nisa.dma_copy(dst=rows_out[q_base : q_base + QP, 0:WS], src=fill[0:QP, 0:WS])

    n_rounds = (QP + _GPSIMD_CORES - 1) // _GPSIMD_CORES
    for rd in nl.static_range(n_rounds):
        q0 = q_base + rd * _GPSIMD_CORES
        nq = min(_GPSIMD_CORES, QP - rd * _GPSIMD_CORES)
        offs = nl.ndarray((1, _GPSIMD_CORES), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=offs, value=0)
        for src_t, c0, cw, pos0 in pieces:
            src = nl.ndarray((P_MAX, cw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=src, value=0.0)
            for j in nl.static_range(nq):
                p = j * _GPSIMD_STRIDE
                nisa.dma_copy(dst=src[p : p + 1, 0:cw], src=src_t[q0 + j : q0 + j + 1, c0 : c0 + cw])
            dst = nl.ndarray((P_MAX, cw + 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.nonzero_with_count(dst=dst, src=src, index_offset=pos0, padding_val=K)
            for j in nl.static_range(nq):
                p = j * _GPSIMD_STRIDE
                off_j = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=off_j, src=offs[0:1, j : j + 1])
                nisa.dma_copy(
                    dst=rows_out.ap(
                        pattern=[[WS, 1], [1, cw]],
                        offset=(q0 + j) * WS,
                        scalar_offset=off_j,
                        indirect_dim=1,
                    ),
                    src=dst[p : p + 1, 0:cw],
                )
                # Count to partition 0 by DMA, then off = min(off + n, ksel).
                cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.dma_copy(dst=cnt, src=dst[p : p + 1, cw : cw + 1])
                nisa.tensor_tensor(
                    dst=offs[0:1, j : j + 1], data1=offs[0:1, j : j + 1], data2=cnt, op=nl.add
                )
                nisa.tensor_scalar(
                    dst=offs[0:1, j : j + 1], data=offs[0:1, j : j + 1],
                    op0=nl.minimum, operand0=float(ksel),
                )
    return rows_out


_PSUM_F32 = 512  # fp32 elements per PSUM bank (one matmul output)
_SM_TILE = 1024  # free-axis tile of the fused exp + sum
_PT_BATCH = 8  # p transposes staged per PSUM eviction


@nki.jit
def sparse_prefill_attn(
    q_lat,  # [Q, H, L] fp32 or bf16 (query latent part; sets the score precision)
    q_rope,  # [Q, H, R]
    kv_cache,  # [..., W] bf16 paged cache; leading dims flatten to rows
    rows_cache,  # [Q, KSEL] int32 flat cache row per slot; out of range (>= n_rows) = none
    own_lat,  # [C, L] the chunk's own latent keys / values
    own_rope,  # [C, R] their rope keys
    rows_own,  # [Q, KSEL] int32 own row per slot; out of range (>= C) = none
    bound,  # [Q, KSEL] fp32 +BIG keep / -BIG drop
    scale=1.0,
    col_off=0,  # first cache column of the attended [latent | rope] slice
    src_sel=None,  # [Q, KSEL] fp32 1 = cache, 0 = own: clamp mode (all rows in range)
):
    """Sparse prefill attention, one batched indirect DMA per query and source.

    For query ``q`` and all ``H`` heads, over its ``KSEL`` slots: slot ``k``
    reads its key from cache row ``rows_cache[q, k]`` (the whole row is
    gathered, columns ``col_off .. col_off + L + R`` used) and/or own row ``rows_own[q, k]``; a row out of range is
    skipped (zeros), so a slot reads at most one source. Then

        s[h, k]  = min(scale * <q[h], key_k>, bound[q, k])
        m[q, h]  = max_k s       den[q, h] = sum_k exp(s - m)
        num[q,h] = sum_k exp(s - m) * latent_k     (unnormalised, fp32)

    Clamp mode (``src_sel`` given): every index is in range (the caller
    clamps unused ones to a valid row) and ``src_sel`` picks the source per
    slot; no out-of-range DMA at all.

    Layout after nkilib's ``mla_sparse_attention_cte`` (DMA path): the slot
    indices load column-major ``[128, KT]`` in one DMA and feed a 2-D
    ``vector_offset`` -- ONE indirect DMA gathers all ``KSEL`` rows of a source
    (``dst[p, m]`` = slot ``p * KT + m``). Queries are split across programs;
    every program attends all heads. Scores are ``[H, KSEL]`` with keys on the
    free axis (score column ``m * 128 + p`` = slot ``p * KT + m``; the bound is
    permuted to match on chip).
    """
    Q, H, L = q_lat.shape
    R = q_rope.shape[2]
    D = L + R
    KSEL = rows_cache.shape[1]
    W = kv_cache.shape[-1]
    n_rows = 1
    for d in kv_cache.shape[:-1]:
        n_rows = n_rows * d
    cache = kv_cache.reshape((n_rows, W))
    n_prg, prg = _programs()
    _kassert(Q % n_prg == 0, "queries must divide by the number of programs")
    QP = Q // n_prg
    q_base = prg * QP
    _kassert(H <= P_MAX, "heads must be <= 128")
    _kassert(L % P_MAX == 0 and L <= _PSUM_F32, "latent must be a multiple of 128 and <= 512")
    _kassert(R <= P_MAX, "rope must be <= 128")
    _kassert(KSEL % P_MAX == 0, "KSEL must be a multiple of 128")
    _kassert(col_off + D <= W, "attended slice exceeds the cache row")
    _kassert(own_lat.shape[1] == L and own_rope.shape[1] == R, "own widths")
    KT = KSEL // P_MAX
    n_l = L // P_MAX
    clamp = src_sel is not None
    # Whole cache rows are gathered: a batched (2-D vector_offset) DMA needs a
    # contiguous source, which a column slice of the row is not.
    GW = W
    g_off = col_off
    qk_dt = q_lat.dtype
    kv_dt = kv_cache.dtype

    m_out = nl.ndarray((Q, H), dtype=nl.float32, buffer=nl.shared_hbm)
    den_out = nl.ndarray((Q, H), dtype=nl.float32, buffer=nl.shared_hbm)
    num_out = nl.ndarray((Q, H, L), dtype=nl.float32, buffer=nl.shared_hbm)

    for i in nl.sequential_range(QP):
        q = q_base + i
        # ---- slot indices, column-major [128, KT]: idx[p, m] = rows[q, m * 128 + p] ----
        idx_c = nl.ndarray((P_MAX, KT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_c, src=rows_cache[q].reshape((KT, P_MAX)).ap(pattern=[[1, P_MAX], [P_MAX, KT]], offset=0))
        idx_o = nl.ndarray((P_MAX, KT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_o, src=rows_own[q].reshape((KT, P_MAX)).ap(pattern=[[1, P_MAX], [P_MAX, KT]], offset=0))

        # ---- bound in score-column order, broadcast over heads: bnd[h, m*128+p] = bound[q, p*KT+m] ----
        bnd_nat = nl.ndarray((H, KSEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bnd_nat, src=bound[q].reshape((1, KSEL)).ap(pattern=[[0, H], [1, KSEL]], offset=0))
        bnd = nl.ndarray((H, KSEL), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=bnd.ap(pattern=[[KSEL, H], [P_MAX, KT], [1, P_MAX]], offset=0),
            src=bnd_nat.ap(pattern=[[KSEL, H], [1, KT], [KT, P_MAX]], offset=0),
            engine=nisa.engine.vector,
        )

        # ---- queries feature-major: qT[d, c, h] (chunk n_l = rope) ----
        q_sb = nl.ndarray((H, D), dtype=qk_dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb[0:H, 0:L], src=q_lat[q, 0:H, 0:L])
        nisa.dma_copy(dst=q_sb[0:H, L:D], src=q_rope[q, 0:H, 0:R])
        qT = nl.ndarray((P_MAX, n_l + 1, H), dtype=qk_dt, buffer=nl.sbuf)
        for c in nl.static_range(n_l):
            tp = nl.ndarray((P_MAX, H), dtype=qk_dt, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P_MAX, 0:H], data=q_sb[0:H, c * P_MAX : (c + 1) * P_MAX])
            nisa.tensor_copy(dst=qT[0:P_MAX, c, 0:H], src=tp[0:P_MAX, 0:H])
        tpr = nl.ndarray((P_MAX, H), dtype=qk_dt, buffer=nl.psum)
        nisa.nc_transpose(dst=tpr[0:R, 0:H], data=q_sb[0:H, L:D])
        nisa.tensor_copy(dst=qT[0:R, n_l, 0:H], src=tpr[0:R, 0:H])

        # ---- ONE batched indirect DMA per source: all KSEL slots ----
        c_nat = nl.ndarray((P_MAX, KT, GW), dtype=kv_dt, buffer=nl.sbuf)
        o_lat = nl.ndarray((P_MAX, KT, L), dtype=kv_dt, buffer=nl.sbuf)
        o_rope = nl.ndarray((P_MAX, KT, R), dtype=kv_dt, buffer=nl.sbuf)
        if not clamp:
            nisa.memset(dst=c_nat, value=0.0)
            nisa.memset(dst=o_lat, value=0.0)
            nisa.memset(dst=o_rope, value=0.0)
        oob = nisa.oob_mode.error if clamp else nisa.oob_mode.skip
        nisa.dma_copy(
            dst=c_nat,
            src=cache.ap(
                pattern=[[W, P_MAX * KT], [1, GW]],
                offset=0,
                vector_offset=idx_c.view(nl.uint32),
                indirect_dim=0,
            ),
            oob_mode=oob,
            dge_mode=nisa.dge_mode.swdge,
        )
        nisa.dma_copy(
            dst=o_lat,
            src=own_lat.ap(pattern=[[L, P_MAX * KT], [1, L]], offset=0, vector_offset=idx_o.view(nl.uint32),
                           indirect_dim=0),
            oob_mode=oob,
            dge_mode=nisa.dge_mode.swdge,
        )
        nisa.dma_copy(
            dst=o_rope,
            src=own_rope.ap(pattern=[[R, P_MAX * KT], [1, R]], offset=0, vector_offset=idx_o.view(nl.uint32),
                            indirect_dim=0),
            oob_mode=oob,
            dge_mode=nisa.dge_mode.swdge,
        )

        # ---- merge the sources into o_lat / o_rope (key-major [p, m, :]) ----
        if clamp:
            # src_sel in gather order: [p, m] = slot p * KT + m (a plain row-major load).
            sel_c2 = nl.ndarray((P_MAX, KT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=sel_c2, src=src_sel[q].reshape((P_MAX, KT)))
            sel_o2 = nl.ndarray((P_MAX, KT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sel_o2, data=sel_c2, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            for m in nl.static_range(KT):
                nisa.tensor_scalar(
                    dst=o_lat[0:P_MAX, m, 0:L], data=o_lat[0:P_MAX, m, 0:L],
                    op0=nl.multiply, operand0=sel_o2[0:P_MAX, m : m + 1],
                )
                nisa.scalar_tensor_tensor(
                    dst=o_lat[0:P_MAX, m, 0:L], data=c_nat[0:P_MAX, m, g_off : g_off + L],
                    op0=nl.multiply, operand0=sel_c2[0:P_MAX, m : m + 1],
                    op1=nl.add, operand1=o_lat[0:P_MAX, m, 0:L],
                )
                nisa.tensor_scalar(
                    dst=o_rope[0:P_MAX, m, 0:R], data=o_rope[0:P_MAX, m, 0:R],
                    op0=nl.multiply, operand0=sel_o2[0:P_MAX, m : m + 1],
                )
                nisa.scalar_tensor_tensor(
                    dst=o_rope[0:P_MAX, m, 0:R], data=c_nat[0:P_MAX, m, g_off + L : g_off + D],
                    op0=nl.multiply, operand0=sel_c2[0:P_MAX, m : m + 1],
                    op1=nl.add, operand1=o_rope[0:P_MAX, m, 0:R],
                )
        else:
            # A slot reads one source; the other stayed zero (skipped).
            nisa.tensor_tensor(
                dst=o_lat, data1=o_lat, data2=c_nat[0:P_MAX, 0:KT, g_off : g_off + L], op=nl.add
            )
            nisa.tensor_tensor(
                dst=o_rope, data1=o_rope, data2=c_nat[0:P_MAX, 0:KT, g_off + L : g_off + D], op=nl.add
            )

        # ---- keys feature-major: kT[d, c, m*128 + p] ----
        kT = nl.ndarray((P_MAX, n_l + 1, KSEL), dtype=qk_dt, buffer=nl.sbuf)
        for m in nl.static_range(KT):
            tk = nl.ndarray((P_MAX, L), dtype=kv_dt, buffer=nl.psum)
            for c in nl.static_range(n_l):
                nisa.nc_transpose(
                    dst=tk[0:P_MAX, c * P_MAX : (c + 1) * P_MAX],
                    data=o_lat[0:P_MAX, m, c * P_MAX : (c + 1) * P_MAX],
                )
            for c in nl.static_range(n_l):
                nisa.tensor_copy(
                    dst=kT[0:P_MAX, c, m * P_MAX : (m + 1) * P_MAX],
                    src=tk[0:P_MAX, c * P_MAX : (c + 1) * P_MAX],
                    engine=nisa.engine.vector if c % 2 == 0 else nisa.engine.scalar,
                )
            tkr = nl.ndarray((P_MAX, P_MAX), dtype=kv_dt, buffer=nl.psum)
            nisa.nc_transpose(dst=tkr[0:R, 0:P_MAX], data=o_rope[0:P_MAX, m, 0:R])
            nisa.tensor_copy(dst=kT[0:R, n_l, m * P_MAX : (m + 1) * P_MAX], src=tkr[0:R, 0:P_MAX])

        # ---- scores [H, KSEL] (keys on the free axis), scaled and bounded ----
        s_sb = nl.ndarray((H, KSEL), dtype=nl.float32, buffer=nl.sbuf)
        for off in nl.static_range(0, KSEL, _PSUM_F32):
            w = min(_PSUM_F32, KSEL - off)
            sp = nl.ndarray((H, _PSUM_F32), dtype=nl.float32, buffer=nl.psum)
            for c in nl.static_range(n_l):
                nisa.nc_matmul(
                    dst=sp[0:H, 0:w], stationary=qT[0:P_MAX, c, 0:H], moving=kT[0:P_MAX, c, off : off + w],
                    accumulate=(c > 0),
                )
            nisa.nc_matmul(
                dst=sp[0:H, 0:w], stationary=qT[0:R, n_l, 0:H], moving=kT[0:R, n_l, off : off + w],
                accumulate=True,
            )
            nisa.scalar_tensor_tensor(
                dst=s_sb[0:H, off : off + w], data=sp[0:H, 0:w],
                op0=nl.multiply, operand0=float(scale),
                op1=nl.minimum, operand1=bnd[0:H, off : off + w],
            )

        # ---- softmax along the free axis: max, then fused exp + sum ----
        mx = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=s_sb, axis=1)
        neg_m = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_m, data=mx, op0=nl.multiply, operand0=-1.0)
        nisa.dma_copy(dst=m_out[q : q + 1, 0:H], src=mx)
        p_bf = nl.ndarray((H, KSEL), dtype=kv_dt, buffer=nl.sbuf)
        den = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        n_sm = (KSEL + _SM_TILE - 1) // _SM_TILE
        for t in nl.static_range(n_sm):
            so = t * _SM_TILE
            sw = min(_SM_TILE, KSEL - so)
            nisa.activation(
                dst=p_bf[0:H, so : so + sw], op=nl.exp, data=s_sb[0:H, so : so + sw], bias=neg_m, scale=1.0,
                reduce_op=nl.add,
                reduce_res=den if t == n_sm - 1 else None,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if t == 0 else nisa.reduce_cmd.reduce,
            )
        nisa.dma_copy(dst=den_out[q : q + 1, 0:H], src=den)

        # ---- p -> keys on partitions [128, m, H], batched per PSUM eviction ----
        p_t = nl.ndarray((P_MAX, KT, H), dtype=kv_dt, buffer=nl.sbuf)
        for b0 in nl.static_range(0, KT, _PT_BATCH):
            nb = min(_PT_BATCH, KT - b0)
            ptp = nl.ndarray((P_MAX, _PT_BATCH * H), dtype=kv_dt, buffer=nl.psum)
            for j in nl.static_range(nb):
                ks = (b0 + j) * P_MAX
                nisa.nc_transpose(dst=ptp[0:P_MAX, j * H : (j + 1) * H], data=p_bf[0:H, ks : ks + P_MAX])
            nisa.tensor_copy(
                dst=p_t[0:P_MAX, b0 : b0 + nb, 0:H],
                src=ptp[0:P_MAX, 0 : nb * H],
                engine=nisa.engine.scalar,
            )

        # ---- num [H, L] = sum_m p_t[:, m]^T @ latent[:, m] ----
        num_p = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.psum)
        for m in nl.static_range(KT):
            nisa.nc_matmul(
                dst=num_p[0:H, 0:L], stationary=p_t[0:P_MAX, m, 0:H], moving=o_lat[0:P_MAX, m, 0:L],
                accumulate=(m > 0),
            )
        num_sb = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=num_sb, src=num_p)
        nisa.dma_copy(dst=num_out[q, 0:H, 0:L], src=num_sb)

    return m_out, den_out, num_out
