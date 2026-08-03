# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fused Whisper-large-v3 decoder MLP sublayer for the M=1 decode step (NKI 0.5.0).

Task 008a feasibility spike: prove/disprove that reading the MLP weight matrices
(fc1 [5120,1280], fc2 [1280,5120]) as LARGE CONTIGUOUS SBUF blocks raises HBM MBU
above the ~30% / 4-8 KB-fragment baseline of the plain-torch XLA lowering.

Math (bf16, M=1):
    h = x @ fc1_W^T + fc1_b     # [1,1280] @ [5120,1280]^T -> [1,5120]
    g = gelu(h)                 # exact erf GELU
    y = g @ fc2_W^T + fc2_b     # [1,5120] @ [1280,5120]^T -> [1,1280]

Weights stored [out, in] (PyTorch nn.Linear). Each weight row is `in` contiguous
bf16, so a [<=128, in] block DMA is one large contiguous burst. Transposes are
done on the PE (tensor) engine via PSUM (nc_transpose requires PSUM dst), then
copied to SBUF. The DMA reads -- what we measure for MBU -- are large contiguous.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


P_MAX = 128
GELU_INV_SQRT2 = 0.7071067811865476


@nki.jit
def decode_mlp(x, fc1_w, fc1_b, fc2_w, fc2_b):
    """Whisper decoder MLP for one token. See module docstring for shapes/math."""
    D = x.shape[1]           # 1280
    I = fc1_w.shape[0]       # 5120
    kernel_assert(fc1_w.shape[1] == D, "fc1_w in-dim mismatch")
    kernel_assert(fc2_w.shape == (D, I), "fc2_w shape mismatch")

    out = nl.ndarray((1, D), dtype=x.dtype, buffer=nl.shared_hbm)

    n_dk = div_ceil(D, P_MAX)   # 10 : D contraction chunks
    n_ik = div_ceil(I, P_MAX)   # 40 : I contraction chunks

    # ---- x [1,D] -> xT [D,1] chunks (contraction D on partition) ----
    x_sb = nl.ndarray((1, D), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x[0:1, 0:D])
    xT = nl.ndarray((P_MAX, n_dk), dtype=x.dtype, buffer=nl.sbuf)
    for dk in nl.affine_range(n_dk):
        d0 = dk * P_MAX
        dsz = min(P_MAX, D - d0)
        ps = nl.ndarray((P_MAX, 1), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=ps[0:dsz, 0:1], data=x_sb[0:1, d0:d0 + dsz])
        nisa.tensor_copy(dst=xT[0:dsz, dk:dk + 1], src=ps[0:dsz, 0:1])

    # ===================== fc1: h[1,I] = x @ W1^T =====================
    F_N = 512
    n_it = div_ceil(I, F_N)     # 10 output tiles
    h_sb = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
    for it in nl.affine_range(n_it):
        i0 = it * F_N
        isz = min(F_N, I - i0)
        n_rsub = div_ceil(isz, P_MAX)   # 4
        # LARGE CONTIGUOUS load: W1[i0:i0+isz, 0:D] as [<=128, n_rsub, D]
        w_blk = nl.ndarray((P_MAX, n_rsub, D), dtype=fc1_w.dtype, buffer=nl.sbuf)
        for rs in nl.affine_range(n_rsub):
            r0 = i0 + rs * P_MAX
            rsz = min(P_MAX, i0 + isz - r0)
            nisa.dma_copy(dst=w_blk[0:rsz, rs, 0:D], src=fc1_w[r0:r0 + rsz, 0:D])

        psum = nl.ndarray((1, isz), dtype=nl.float32, buffer=nl.psum)
        for dk in nl.affine_range(n_dk):
            d0 = dk * P_MAX
            dsz = min(P_MAX, D - d0)
            wt = nl.ndarray((P_MAX, F_N), dtype=fc1_w.dtype, buffer=nl.sbuf)
            for rs in nl.affine_range(n_rsub):
                c0 = rs * P_MAX
                csz = min(P_MAX, isz - c0)
                tps = nl.ndarray((P_MAX, P_MAX), dtype=fc1_w.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tps[0:dsz, 0:csz], data=w_blk[0:csz, rs, d0:d0 + dsz])
                nisa.tensor_copy(dst=wt[0:dsz, c0:c0 + csz], src=tps[0:dsz, 0:csz])
            nisa.nc_matmul(dst=psum[0:1, 0:isz],
                           stationary=xT[0:dsz, dk:dk + 1],
                           moving=wt[0:dsz, 0:isz])
        nisa.tensor_copy(dst=h_sb[0:1, i0:i0 + isz], src=psum[0:1, 0:isz])

    # ---- fc1 bias + GELU ----
    b1_sb = nl.ndarray((1, I), dtype=fc1_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b1_sb, src=fc1_b[0:1, 0:I])
    b1_f = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b1_f, src=b1_sb)
    nisa.tensor_tensor(dst=h_sb, data1=h_sb, data2=b1_f, op=nl.add)
    erf_sb = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=erf_sb, data=h_sb, op=nl.erf, scale=GELU_INV_SQRT2)
    nisa.tensor_scalar(dst=erf_sb, data=erf_sb, op0=nl.add, operand0=1.0)
    g_sb = nl.ndarray((1, I), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g_sb, data1=h_sb, data2=erf_sb, op=nl.multiply)
    nisa.tensor_scalar(dst=g_sb, data=g_sb, op0=nl.multiply, operand0=0.5)
    g_bf = nl.ndarray((1, I), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=g_bf, src=g_sb)
    gT = nl.ndarray((P_MAX, n_ik), dtype=x.dtype, buffer=nl.sbuf)
    for ik in nl.affine_range(n_ik):
        k0 = ik * P_MAX
        ksz = min(P_MAX, I - k0)
        ps = nl.ndarray((P_MAX, 1), dtype=x.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=ps[0:ksz, 0:1], data=g_bf[0:1, k0:k0 + ksz])
        nisa.tensor_copy(dst=gT[0:ksz, ik:ik + 1], src=ps[0:ksz, 0:1])

    # ===================== fc2: y[1,D] = g @ W2^T =====================
    n_ot = div_ceil(D, F_N)     # 3
    y_sb = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
    for ot in nl.affine_range(n_ot):
        o0 = ot * F_N
        osz = min(F_N, D - o0)
        n_rsub = div_ceil(osz, P_MAX)
        # LARGE CONTIGUOUS load: W2[o0:o0+osz, 0:I] as [<=128, n_rsub, I]
        w_blk = nl.ndarray((P_MAX, n_rsub, I), dtype=fc2_w.dtype, buffer=nl.sbuf)
        for rs in nl.affine_range(n_rsub):
            r0 = o0 + rs * P_MAX
            rsz = min(P_MAX, o0 + osz - r0)
            nisa.dma_copy(dst=w_blk[0:rsz, rs, 0:I], src=fc2_w[r0:r0 + rsz, 0:I])

        psum = nl.ndarray((1, osz), dtype=nl.float32, buffer=nl.psum)
        for ik in nl.affine_range(n_ik):
            k0 = ik * P_MAX
            ksz = min(P_MAX, I - k0)
            wt = nl.ndarray((P_MAX, F_N), dtype=fc2_w.dtype, buffer=nl.sbuf)
            for rs in nl.affine_range(n_rsub):
                c0 = rs * P_MAX
                csz = min(P_MAX, osz - c0)
                tps = nl.ndarray((P_MAX, P_MAX), dtype=fc2_w.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tps[0:ksz, 0:csz], data=w_blk[0:csz, rs, k0:k0 + ksz])
                nisa.tensor_copy(dst=wt[0:ksz, c0:c0 + csz], src=tps[0:ksz, 0:csz])
            nisa.nc_matmul(dst=psum[0:1, 0:osz],
                           stationary=gT[0:ksz, ik:ik + 1],
                           moving=wt[0:ksz, 0:osz])
        nisa.tensor_copy(dst=y_sb[0:1, o0:o0 + osz], src=psum[0:1, 0:osz])

    # ---- fc2 bias (fp32) + cast ----
    b2_sb = nl.ndarray((1, D), dtype=fc2_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b2_sb, src=fc2_b[0:1, 0:D])
    nisa.tensor_tensor(dst=y_sb, data1=y_sb, data2=b2_sb, op=nl.add)
    y_out = nl.ndarray((1, D), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=y_out, src=y_sb)
    nisa.dma_copy(dst=out[0:1, 0:D], src=y_out)
    return out
