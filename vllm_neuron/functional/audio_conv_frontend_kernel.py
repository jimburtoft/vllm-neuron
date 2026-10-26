# SPDX-License-Identifier: Apache-2.0
"""NKI kernel: fused 3-layer audio conv frontend (conv1 -> GELU -> conv2 -> GELU -> conv3 -> GELU).

Replaces, per base chunk (the conv batch), the stock frontend of a Whisper-style 2-D conv encoder

    x = gelu(conv2d1(mel))     # 1   -> C, 3x3, stride 2, pad 1   (+ bias)
    x = gelu(conv2d2(x))       # C   -> C, 3x3, stride 2, pad 1   (+ bias)
    x = gelu(conv2d3(x))       # C   -> C, 3x3, stride 2, pad 1   (+ bias)

with one implicit-GEMM kernel that keeps every intermediate activation in SBUF. Only the mel (in)
and the conv3 output (out) touch HBM. GELU is the exact (erf) form, computed in fp32 on ScalarE
straight from PSUM with the bias fused (``gelu(acc + bias)``), then rounded to bf16.

Inputs (HBM):
  xpad [n, H0 + 2, W0 + 2] bf16   mel chunks with a 1-element zero border (H0 = mel bins, W0 = frames)
  w1   [9, C]          bf16       conv1 weight, tap-major (t = 3 * kr + kc)
  w2   [C, 9, C]       bf16       conv2 weight as [ci, tap, co]
  w3   [C, 9, C]       bf16       conv3 weight as [ci, tap, co]
  b1, b2, b3 [C]       bf16
Output:
  out  [n, C, H3, W3]  bf16       NCHW, the stock conv3 output layout (H3 = H0 / 8, W3 = ceil(W0 / 8))

Layout (channels on partitions, C tiled by 128; the last tile may be partial, e.g. 480 = 3 x 128 + 96):
  conv1   moving = 9 tap partitions; tap t = (kr, kc) holds xpad[c, 2 i + kr, f + kc] (f full-res); the
          matmul reads f = 0, 2, 4, ... (stride-2 free AP), so no im2col copy. stationary = w1[9, co tile].
  conv2/3 stationary = w[ci tile, tap, co tile] (weights resident in SBUF for the whole call); moving =
          the previous activation [ci tile, rows 2 i + kr (stride 2), cols 2 j + kc (stride 2)] read in
          place from a zero-bordered SBUF buffer. 4 ci tiles x 9 taps = 36 accumulating matmuls per
          PSUM accumulation group, all issued back-to-back inside one group (no PSUM state crosses a
          group; every group starts with accumulate=False).
  Evictions: ScalarE gelu(psum + bias) -> bf16, written into the interior of the next layer's
  zero-bordered buffer. Borders are memset once per buffer set (2 sets, ping-pong over chunks) and
  never written again.
Schedule: a 3-stage software pipeline over chunks. Step s issues, per output-channel tile k,
  conv1(chunk s, k), conv2(chunk s - 1, k), conv3(chunk s - 2, k), so the PE always has matmuls
  whose inputs were evicted one step earlier, and ScalarE evictions overlap PE work.
LNC: a 1-D grid of n_prg programs splits the chunks (n % n_prg == 0); each writes its own slice.
No PSUM accumulation across loop iterations, no ring/modulo PSUM buffers, no DMA oob-skip / indirect DMA.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert

P = 128
PSUM_FP32 = 512  # fp32 elements per partition in one PSUM bank


def _tiles(C):
    return [(s, min(P, C - s)) for s in range(0, C, P)]


def _pow2_at_least(w):
    v = 1
    while v < w:
        v *= 2
    return v


def _out_hw(h, w):
    return (h + 1) // 2, (w + 1) // 2


def _border_memset(buf, csz, h_in, w_in):
    """Zero the border of a [P, Hb, Wb] buffer whose interior is [1:1+h_in, 1:1+w_in]."""
    _, hb, wb = buf.shape
    nisa.memset(dst=buf[0:csz, 0:1, :], value=0.0, engine=nisa.engine.gpsimd)
    nisa.memset(dst=buf[0:csz, :, 0:1], value=0.0, engine=nisa.engine.gpsimd)
    if hb > 1 + h_in:
        nisa.memset(dst=buf[0:csz, 1 + h_in:hb, :], value=0.0, engine=nisa.engine.gpsimd)
    if wb > 1 + w_in:
        nisa.memset(dst=buf[0:csz, :, 1 + w_in:wb], value=0.0, engine=nisa.engine.gpsimd)


@nki.jit
def qwen3_asr_conv_frontend_kernel(xpad, w1, b1, w2, b2, w3, b3, banks1=2):
    n, Hp, Wp = xpad.shape
    H0, W0 = Hp - 2, Wp - 2
    C = w1.shape[1]
    kernel_assert(w1.shape[0] == 9, "w1 must be [9, C]")
    kernel_assert(H0 % 2 == 0 and W0 % 2 == 0, "mel bins and frames per chunk must be even")
    kernel_assert(w2.shape[0] == C and w2.shape[1] == 9 and w2.shape[2] == C, "w2 must be [C, 9, C]")
    kernel_assert(w3.shape[0] == C and w3.shape[1] == 9 and w3.shape[2] == C, "w3 must be [C, 9, C]")
    dt = xpad.dtype
    H1, W1 = _out_hw(H0, W0)
    H2, W2 = _out_hw(H1, W1)
    H3, W3 = _out_hw(H2, W2)
    WB1, WB2, WB3 = _pow2_at_least(W1), _pow2_at_least(W2), _pow2_at_least(W3)
    kernel_assert(WB1 <= PSUM_FP32 and WB2 <= PSUM_FP32 and WB3 <= PSUM_FP32, "row too wide for a PSUM bank")
    R1, R2, R3 = PSUM_FP32 // WB1, PSUM_FP32 // WB2, PSUM_FP32 // WB3   # output rows per PSUM bank
    kernel_assert(H2 <= 2 * R2 and H3 <= R3, "conv2 must fit 2 PSUM banks, conv3 one bank")
    RG1 = banks1 * R1                                                   # conv1 rows per PSUM tile
    tiles = _tiles(C)
    NT = len(tiles)

    out = nl.ndarray((n, C, H3, W3), dtype=dt, buffer=nl.shared_hbm)
    out2 = out.reshape((n * C, H3, W3))

    n_prg = 1
    base = 0
    if nl.program_ndim() > 0:
        n_prg = nl.num_programs(0)
    kernel_assert(n % n_prg == 0, "chunks must split evenly across the LNC grid")
    cpp = n // n_prg
    if n_prg > 1:
        base = nl.program_id(0) * cpp

    # ---- resident weights and biases ---------------------------------------------------------
    w1s = nl.ndarray((9, C), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1s, src=w1)
    w2s, w3s, bias = [], [], []
    for j, (cs, csz) in enumerate(tiles):
        t2 = nl.ndarray((P, 9, C), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=t2[0:csz, :, :], src=w2[cs:cs + csz, :, :])
        t3 = nl.ndarray((P, 9, C), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=t3[0:csz, :, :], src=w3[cs:cs + csz, :, :])
        w2s.append(t2)
        w3s.append(t3)
    for li, b in enumerate((b1, b2, b3)):
        b2d = b.reshape((C, 1))
        row = []
        for k, (ks, ksz) in enumerate(tiles):
            braw = nl.ndarray((P, 1), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=braw[0:ksz, :], src=b2d[ks:ks + ksz, :])
            bf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bf[0:ksz, :], src=braw[0:ksz, :], engine=nisa.engine.vector)
            row.append(bf)
        bias.append(row)

    # ---- ping-pong activation buffers (zero borders written once) ----------------------------
    NBUF = 2
    xs = [nl.ndarray((9, H1, 2 * W1), dtype=dt, buffer=nl.sbuf) for _ in range(NBUF)]
    y1 = [[nl.ndarray((P, 2 * H2 + 1, 2 * W2 + 1), dtype=dt, buffer=nl.sbuf) for _ in range(NT)]
          for _ in range(NBUF)]
    y2 = [[nl.ndarray((P, 2 * H3 + 1, 2 * W3 + 1), dtype=dt, buffer=nl.sbuf) for _ in range(NT)]
          for _ in range(NBUF)]
    y3 = [[nl.ndarray((P, H3, W3), dtype=dt, buffer=nl.sbuf) for _ in range(NT)] for _ in range(NBUF)]
    for s in range(NBUF):
        for k, (ks, ksz) in enumerate(tiles):
            _border_memset(y1[s][k], ksz, H1, W1)
            _border_memset(y2[s][k], ksz, H2, W2)

    def load_x(i):
        c = base + i
        xb = xs[i % NBUF]
        for kr in range(3):
            # partitions 3 kr + kc (kc = 0, 1, 2) <- xpad[c, 2 r + kr, kc : kc + 2 W1 - 1]
            nisa.dma_copy(dst=xb[3 * kr:3 * kr + 3, :, 0:2 * W1 - 1],
                          src=xpad.ap(pattern=[[1, 3], [2 * Wp, H1], [1, 2 * W1 - 1]],
                                      offset=c * (Hp * Wp) + kr * Wp))

    def conv1(i, k):
        ks, ksz = tiles[k]
        xb = xs[i % NBUF]
        dst = y1[i % NBUF][k]
        for r0 in range(0, H1, RG1):
            rows = min(RG1, H1 - r0)
            ps = nl.ndarray((P, RG1, WB1), dtype=nl.float32, buffer=nl.psum)
            for b0 in range(0, rows, R1):
                rr = min(R1, rows - b0)
                nisa.nc_matmul(dst=ps[0:ksz, b0:b0 + rr, 0:W1], stationary=w1s[0:9, ks:ks + ksz],
                               moving=xb[0:9, r0 + b0:r0 + b0 + rr, 0:2 * W1 - 1:2], accumulate=False)
            nisa.activation(dst=dst[0:ksz, 1 + r0:1 + r0 + rows, 1:1 + W1], op=nl.gelu,
                            data=ps[0:ksz, 0:rows, 0:W1], bias=bias[0][k][0:ksz, :], scale=1.0)

    def conv3x3(src_set, wts, k, Hout, Wout, R, WB, nbank, evict):
        """One output-channel tile of a 3x3 / stride-2 conv over all ci tiles and taps."""
        ks, ksz = tiles[k]
        ps = nl.ndarray((P, nbank * R, WB), dtype=nl.float32, buffer=nl.psum)
        groups = [(r0, min(R, Hout - r0)) for r0 in range(0, Hout, R)]
        for j, (cs, csz) in enumerate(tiles):
            xin = src_set[j]
            for t in range(9):
                kr, kc = t // 3, t % 3
                # one stationary load, reused for every PSUM-bank row group (each group = its own
                # accumulation group over the 36 (ci tile, tap) steps, opened by step 0)
                for (r0, rr) in groups:
                    nisa.nc_matmul(
                        dst=ps[0:ksz, r0:r0 + rr, 0:Wout],
                        stationary=wts[j][0:csz, t, ks:ks + ksz],
                        moving=xin[0:csz, 2 * r0 + kr:2 * r0 + kr + 2 * rr - 1:2, kc:kc + 2 * Wout - 1:2],
                        accumulate=(j > 0 or t > 0))
        evict(ps[0:ksz, 0:Hout, 0:Wout], k, ksz)

    def conv2(i, k):
        dst = y2[i % NBUF][k]

        def ev(src, k_, ksz):
            nisa.activation(dst=dst[0:ksz, 1:1 + H2, 1:1 + W2], op=nl.gelu, data=src,
                            bias=bias[1][k_][0:ksz, :], scale=1.0)
        conv3x3(y1[i % NBUF], w2s, k, H2, W2, R2, WB2, 2, ev)

    def conv3(i, k):
        dst = y3[i % NBUF][k]
        c = base + i

        def ev(src, k_, ksz):
            nisa.activation(dst=dst[0:ksz, :, :], op=nl.gelu, data=src, bias=bias[2][k_][0:ksz, :], scale=1.0)
            ks_ = tiles[k_][0]
            nisa.dma_copy(dst=out2[nl.ds(c * C + ks_, ksz), :, :], src=dst[0:ksz, :, :])
        conv3x3(y2[i % NBUF], w3s, k, H3, W3, R3, WB3, 1, ev)

    load_x(0)
    for s in range(cpp + 2):
        if s + 1 < cpp:
            load_x(s + 1)
        for k in range(NT):
            if s < cpp:
                conv1(s, k)
            if 0 <= s - 1 < cpp:
                conv2(s - 1, k)
            if 0 <= s - 2 < cpp:
                conv3(s - 2, k)
    return out
