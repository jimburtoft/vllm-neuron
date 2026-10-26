# SPDX-License-Identifier: Apache-2.0
"""NKI kernels for the sparse key-sharded decode attention (``HY4_DEC_SPARSE_NKI=1``).

Adapted from the workspace's absorbed-MLA DCP decode kernel
(``sarvam_mla/mla_dcp_decode_nki.py``, ``mla_dcp_decode_attn``): same algorithm
(keys on partitions, one indirect DMA per (token, 128-key tile) with a ``[KS, 1]``
index column, PE transposes for the feature-major key copy, clamp masking with
finite sentinels), generalised to a cache row that is wider than the attended
slice and to Hy4's ``[latent | rope]`` order.

``sparse_decode_attn`` -- per request ``b`` and all ``H`` heads, over the rows in
``rows[b]`` (``D = q_all.shape[2]`` columns from ``col_off`` of each cache row):

    s[k, h]  = scale * <q_all[b, h, :], cache[rows[b, k], col_off : col_off + D]>
    s[k, h]  = min(s, bound[b, k])        bound = +BIG keep / -BIG drop
    m[b, h]  = max_k s[k, h]
    p[k, h]  = exp(s - m)
    den[b,h] = sum_k p
    num[b,h] = sum_k p[k, h] * cache[rows[b, k], col_off + lat_off : ... + L]   (fp32)

A row index ``>= cache.shape[0]`` is out of range: the DMA skips it
(``oob_mode.skip``), so its bytes are never read; the zero-initialised tile
stands in and its bound must be ``-BIG``. That is how unselected keys cost no
cache traffic.

``gather_rows_cols`` -- ``out[r, k, :] = cache[rows[r, k], col_off : col_off + width]``,
one indirect DMA per 128 rows (the DSA indexer keys only, not whole entries).

Both are imported only when ``HY4_DEC_SPARSE_NKI=1`` runs on a Neuron device
(or the NKI simulator); the torch references in ``model.py`` implement the
same contracts for CPU.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
PSUM_F32 = 512


def _kassert(c, msg=""):
    assert c, msg


@nki.jit
def sparse_decode_attn(
    q_all,  # [B, H, D] fp32 or bf16, same column order as the attended cache slice
    kv_cache,  # [..., W] bf16 paged cache; leading dims flatten to rows
    rows,  # [B, n_keys] int32 flat cache row per key; >= n_rows = skipped
    bound,  # [B, n_keys] fp32 +BIG keep / -BIG drop
    scale=1.0,
    col_off=0,  # first cache column of the attended slice
    lat_off=0,  # value (latent) offset inside the slice
    lat_dim=512,  # value width
    rope_off=512,  # rope offset inside the slice
    rope_dim=64,
    tg=0,  # tokens per group (0 = auto: PSUM_F32 // H, reduced to divide B)
):
    B, H, D = q_all.shape
    W = kv_cache.shape[-1]
    n_rows = 1
    for d in kv_cache.shape[:-1]:
        n_rows = n_rows * d
    cache = kv_cache.reshape((n_rows, W))
    n_keys = rows.shape[1]
    L = lat_dim
    _kassert(H <= P_MAX, "n_heads must be <= 128")
    _kassert(L % P_MAX == 0 and L <= 512, "latent must be a multiple of 128 and <= 512")
    _kassert(rope_dim <= P_MAX, "rope_dim must be <= 128")
    _kassert(L + rope_dim == D, "query width must be latent + rope")
    _kassert(col_off + D <= W, "attended slice exceeds the cache row")
    _kassert(n_keys % P_MAX == 0 or n_keys <= P_MAX, "n_keys must be <=128 or a multiple of 128")
    KT = (n_keys + P_MAX - 1) // P_MAX
    KS = min(n_keys, P_MAX)
    n_lat = L // P_MAX
    TG = tg if tg > 0 else max(1, PSUM_F32 // H)
    if TG > B:
        TG = B
    while B % TG != 0:
        TG -= 1
    NG = B // TG
    TH = TG * H

    # Score precision follows the query: an fp32 query scores against fp32 copies
    # of the keys (the torch path's ``q.float() @ lat.float()``); a bf16 query
    # against the bf16 keys. The value product is bf16 either way.
    qk_dt = q_all.dtype

    m_out = nl.ndarray((B, H), dtype=nl.float32, buffer=nl.shared_hbm)
    den_out = nl.ndarray((B, H), dtype=nl.float32, buffer=nl.shared_hbm)
    num_out = nl.ndarray((B, H, L), dtype=nl.float32, buffer=nl.shared_hbm)

    ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_row, value=1.0)
    ones_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_col, value=1.0)

    for g in nl.sequential_range(NG):
        b0 = g * TG
        # ---- queries of the group, feature-major: qT[d, chunk, t*H + h] (chunk n_lat = rope) ----
        qT = nl.ndarray((P_MAX, n_lat + 1, TH), dtype=qk_dt, buffer=nl.sbuf)
        for t in nl.static_range(TG):
            q_sb = nl.ndarray((H, D), dtype=qk_dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_sb, src=q_all[b0 + t, 0:H, 0:D])
            for c in nl.static_range(n_lat):
                tp = nl.ndarray((P_MAX, H), dtype=qk_dt, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=tp[0:P_MAX, 0:H],
                    data=q_sb[0:H, lat_off + c * P_MAX : lat_off + (c + 1) * P_MAX],
                )
                nisa.tensor_copy(dst=qT[0:P_MAX, c, t * H : (t + 1) * H], src=tp[0:P_MAX, 0:H])
            tpr = nl.ndarray((P_MAX, H), dtype=qk_dt, buffer=nl.psum)
            nisa.nc_transpose(dst=tpr[0:rope_dim, 0:H], data=q_sb[0:H, rope_off : rope_off + rope_dim])
            nisa.tensor_copy(dst=qT[0:rope_dim, n_lat, t * H : (t + 1) * H], src=tpr[0:rope_dim, 0:H])

        # ---- gather each token's rows key-major: kv_rows[k, t, kt, D]; bounds bnd[k, t, kt] ----
        kv_rows = nl.ndarray((KS, TG, KT, D), dtype=kv_cache.dtype, buffer=nl.sbuf)
        nisa.memset(dst=kv_rows, value=0.0)
        bnd = nl.ndarray((KS, TG, KT), dtype=nl.float32, buffer=nl.sbuf)
        idx_g = nl.ndarray((KS, TG, KT), dtype=nl.int32, buffer=nl.sbuf)
        for t in nl.static_range(TG):
            nisa.dma_copy(
                dst=idx_g[0:KS, t, 0:KT],
                src=rows[b0 + t].reshape((KT, KS)).ap(pattern=[[1, KS], [KS, KT]], offset=0),
            )
            nisa.dma_copy(
                dst=bnd[0:KS, t, 0:KT],
                src=bound[b0 + t].reshape((KT, KS)).ap(pattern=[[1, KS], [KS, KT]], offset=0),
            )
            for kt in nl.static_range(KT):
                nisa.dma_copy(
                    dst=kv_rows[0:KS, t, kt, 0:D],
                    src=cache.ap(
                        pattern=[[W, KS], [1, D]],
                        offset=col_off,
                        vector_offset=idx_g.view(nl.uint32)[0:KS, t, kt : kt + 1],
                        indirect_dim=0,
                    ),
                    oob_mode=nisa.oob_mode.skip,
                )

        # ---- scores key-major: s[k, kt, t*H + h] ----
        s_sb = nl.ndarray((KS, KT, TH), dtype=nl.float32, buffer=nl.sbuf)
        for kt in nl.static_range(KT):
            for t in nl.static_range(TG):
                kvT = nl.ndarray((P_MAX, n_lat + 1, KS), dtype=qk_dt, buffer=nl.sbuf)
                for c in nl.static_range(n_lat):
                    tk = nl.ndarray((P_MAX, KS), dtype=kv_cache.dtype, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=tk[0:P_MAX, 0:KS],
                        data=kv_rows[0:KS, t, kt, lat_off + c * P_MAX : lat_off + (c + 1) * P_MAX],
                    )
                    nisa.tensor_copy(dst=kvT[0:P_MAX, c, 0:KS], src=tk[0:P_MAX, 0:KS])
                tkr = nl.ndarray((P_MAX, KS), dtype=kv_cache.dtype, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=tkr[0:rope_dim, 0:KS],
                    data=kv_rows[0:KS, t, kt, rope_off : rope_off + rope_dim],
                )
                nisa.tensor_copy(dst=kvT[0:rope_dim, n_lat, 0:KS], src=tkr[0:rope_dim, 0:KS])

                sp = nl.ndarray((KS, H), dtype=nl.float32, buffer=nl.psum)
                for c in nl.static_range(n_lat):
                    nisa.nc_matmul(
                        dst=sp[0:KS, 0:H],
                        stationary=kvT[0:P_MAX, c, 0:KS],
                        moving=qT[0:P_MAX, c, t * H : (t + 1) * H],
                        accumulate=(c > 0),
                    )
                nisa.nc_matmul(
                    dst=sp[0:KS, 0:H],
                    stationary=kvT[0:rope_dim, n_lat, 0:KS],
                    moving=qT[0:rope_dim, n_lat, t * H : (t + 1) * H],
                    accumulate=True,
                )
                nisa.tensor_scalar(
                    dst=s_sb[0:KS, kt, t * H : (t + 1) * H],
                    data=sp[0:KS, 0:H],
                    op0=nl.multiply,
                    operand0=float(scale),
                    op1=nl.minimum,
                    operand1=bnd[0:KS, t, kt : kt + 1],
                )

        # ---- max over keys for all TG*H columns at once ----
        pmax = nl.ndarray((KS, TH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pmax[0:KS, 0:TH], src=s_sb[0:KS, 0, 0:TH])
        for kt in nl.static_range(1, KT):
            nisa.tensor_tensor(
                dst=pmax[0:KS, 0:TH], data1=pmax[0:KS, 0:TH], data2=s_sb[0:KS, kt, 0:TH], op=nl.maximum
            )
        # cross-partition max per column: transpose 128-column chunks to [cols, KS], reduce free axis
        mrow = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
        for c0 in nl.static_range(0, TH, P_MAX):
            cw = min(P_MAX, TH - c0)
            pmT = nl.ndarray((P_MAX, KS), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pmT[0:cw, 0:KS], data=pmax[0:KS, c0 : c0 + cw])
            mcol = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mcol[0:cw, 0:1], op=nl.maximum, data=pmT[0:cw, 0:KS], axis=1)
            mrp = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=mrp[0:1, 0:cw], data=mcol[0:cw, 0:1])
            nisa.tensor_copy(dst=mrow[0:1, c0 : c0 + cw], src=mrp[0:1, 0:cw])
        nisa.dma_copy(dst=m_out[b0 : b0 + TG].reshape((1, TH)), src=mrow[0:1, 0:TH])
        negrow = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=negrow[0:1, 0:TH], data=mrow[0:1, 0:TH], op0=nl.multiply, operand0=-1.0)
        negm_p = nl.ndarray((KS, TH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=negm_p[0:KS, 0:TH], stationary=ones_row[0:1, 0:KS], moving=negrow[0:1, 0:TH])
        negm = nl.ndarray((KS, TH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=negm[0:KS, 0:TH], src=negm_p[0:KS, 0:TH])

        # ---- p = exp(s - m), den over keys ----
        p_bf = nl.ndarray((KS, KT, TH), dtype=kv_cache.dtype, buffer=nl.sbuf)
        psum_k = nl.ndarray((KS, TH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=psum_k, value=0.0)
        for kt in nl.static_range(KT):
            e = nl.ndarray((KS, TH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=e[0:KS, 0:TH], data1=s_sb[0:KS, kt, 0:TH], data2=negm[0:KS, 0:TH], op=nl.add)
            nisa.activation(dst=e[0:KS, 0:TH], op=nl.exp, data=e[0:KS, 0:TH])
            nisa.tensor_tensor(dst=psum_k[0:KS, 0:TH], data1=psum_k[0:KS, 0:TH], data2=e[0:KS, 0:TH], op=nl.add)
            nisa.tensor_copy(dst=p_bf[0:KS, kt, 0:TH], src=e[0:KS, 0:TH])
        den_p = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=den_p[0:1, 0:TH], stationary=ones_col[0:KS, 0:1], moving=psum_k[0:KS, 0:TH])
        den_sb = nl.ndarray((1, TH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=den_sb[0:1, 0:TH], src=den_p[0:1, 0:TH])
        nisa.dma_copy(dst=den_out[b0 : b0 + TG].reshape((1, TH)), src=den_sb[0:1, 0:TH])

        # ---- num per token: [H, L] = p[:, t]^T @ latent ----
        for t in nl.static_range(TG):
            num_p = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.static_range(KT):
                nisa.nc_matmul(
                    dst=num_p[0:H, 0:L],
                    stationary=p_bf[0:KS, kt, t * H : (t + 1) * H],
                    moving=kv_rows[0:KS, t, kt, lat_off : lat_off + L],
                    accumulate=(kt > 0),
                )
            num_sb = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=num_sb[0:H, 0:L], src=num_p[0:H, 0:L])
            nisa.dma_copy(dst=num_out[b0 + t], src=num_sb[0:H, 0:L])

    return m_out, den_out, num_out


@nki.jit
def gather_rows_cols(
    kv_cache,  # [..., W] paged cache; leading dims flatten to rows
    rows,  # [R, n] int32 flat cache rows; >= n_rows = skipped (zeros)
    col_off=0,
    width=128,
):
    R, n = rows.shape
    W = kv_cache.shape[-1]
    n_rows = 1
    for d in kv_cache.shape[:-1]:
        n_rows = n_rows * d
    cache = kv_cache.reshape((n_rows, W))
    _kassert(col_off + width <= W, "column slice exceeds the cache row")
    _kassert(n % P_MAX == 0 or n <= P_MAX, "n must be <=128 or a multiple of 128")
    KT = (n + P_MAX - 1) // P_MAX
    KS = min(n, P_MAX)
    out = nl.ndarray((R, n, width), dtype=kv_cache.dtype, buffer=nl.shared_hbm)
    for r in nl.sequential_range(R):
        idx = nl.ndarray((KS, KT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx[0:KS, 0:KT], src=rows[r].reshape((KT, KS)).ap(pattern=[[1, KS], [KS, KT]], offset=0))
        for kt in nl.static_range(KT):
            t = nl.ndarray((KS, width), dtype=kv_cache.dtype, buffer=nl.sbuf)
            nisa.memset(dst=t, value=0.0)
            nisa.dma_copy(
                dst=t[0:KS, 0:width],
                src=cache.ap(
                    pattern=[[W, KS], [1, width]],
                    offset=col_off,
                    vector_offset=idx.view(nl.uint32)[0:KS, kt : kt + 1],
                    indirect_dim=0,
                ),
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.dma_copy(dst=out[r, kt * KS : (kt + 1) * KS, 0:width], src=t[0:KS, 0:width])
    return out
