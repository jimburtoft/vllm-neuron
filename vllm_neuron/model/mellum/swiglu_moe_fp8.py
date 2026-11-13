"""swiglu_moe_fp8 (fp8-moe-decode-neuron-kernels): a fused routed-expert SwiGLU MoE for decode (token generation) on Trainium2, in NKI.

out[t] = sum_k w[t, k] * down_e( silu(gate_e(x[t])) * up_e(x[t]) ),  e = idx[t, k]

with each expert's gate / up / down weights in FP8 (e4m3) with 128 x 128 block scales (the DeepSeek-V3 /
Qwen3-FP8 / GLM-FP8 checkpoint format), kept as FP8 matmul operands; the scales are applied to the fp32 tile
products after the matmul (one per 128-column tile, so a tile's PSUM column is already its scaled dot product).

The design follows TrnRT's decode FFN half (github.com/RunanywhereAI/trnrt, kernels/moe_dedupe.py
moe_tiles_pairs_v1 and kernels/ffn_half_pb.py, Apache-2.0), generalised to any intermediate size:

- one packed blob per (expert, 64-row intermediate block) "unit": [128 partitions, F bytes], gate_up FP8,
  down FP8, gate_up tile scales and down column scales, so one DMA per unit fetches everything a unit needs
  (pack(), on the host, once at load);
- weights stationary [128 x 128] FP8 tiles, the token as a one-column moving operand: the PSUM column of a
  matmul is the tile's dot product; a 0/1 "fold" matmul brings the up rows (partitions 64-127) onto the gate
  rows (0-63);
- a static ring of RING expert buffers; step i issues the loads of group i + RING - 2, stage A (gate_up, scales)
  of group i, stages B / C (SwiGLU, down, accumulate) of group i - 1, so the loads run ahead of the compute and
  each buffer's next load waits only on the tensor engine's down matmuls that read it;
- G (token, expert, block) work items share each vector / scalar instruction;
- routing ids drive DMA addresses only (dma_copy src=blob.select(0, e)): nki 0.6 compute instructions cannot
  take runtime offsets, so everything else is static code;
- at LNC=2 (grid 2) each program takes a contiguous half of the work groups, and the two fp32 partials are
  exchanged by column halves (nisa.sendrecv) and added, so the output is written once;
- x is read once and laid out k-major on chip by identity matmuls (exact), and the [128, T, C] accumulator is
  transposed back to [T, H] on chip, so the caller's tensors are plain [T, H].

Interface (all HBM):
  x     bf16  [T, H]            T <= 128, H % 256 == 0
  idx   int32 [T, K]            expert ids in [0, E)
  w     bf16  [T, K]            routing weights (already normalised / scaled)
  blob  uint8 [E * NB, 128, F]  pack(): unit u = e * NB + b, NB = I / 64 blocks of 64 intermediate rows
  -> out bf16 [T, H]
Arithmetic: fp32 accumulation; g (gate_up) and a (silu(g) * u) rounded to bf16 as matmul operands; act 1 adds the
clamp-limit SwiGLU (gate clamped from above, up into [-limit, limit]); the sum over k in fp32; out rounded to bf16.
emulate() is that arithmetic in torch.
"""

from __future__ import annotations

import torch

P = 128
IB = 64  # intermediate rows per unit (gate 64 + up 64 = one 128-partition tile)
FP8 = torch.float8_e4m3fn
RING = 4
GROUP = 4


def auto_group(T: int, K: int, NB: int) -> int:
    """Work items per shared vector instruction: 1 when a call has at most 8 items (measured: GLM T=1 40.4 vs
    43.8 us), else GROUP (4 was best or tied at every other shape measured, T 1-32, two model shapes)."""
    return 1 if T * K * NB <= 8 else GROUP
# expert-unit loads: 0 software DGE, 1 hardware DGE (sync / scalar queues), 2 half and half
LOAD = int(__import__('os').environ.get('SWM_LOAD', '0'))
# PSUM accumulate flag on every matmul: False (explicit overwrite; SDK 2.32 neuronx-cc 2.27) or None (compiler decides;
# SDK 2.33 neuronx-cc 2.28 rejects a kernel whose flags mix set and unset: NCC_ILMM003). Each PSUM region here is
# written by exactly one matmul, so the two are the same arithmetic.
ACC = False
import os as _os
# neuronx-cc 2.28 (SDK 2.33) rejects the fold matmul's explicit flag at some shapes (NCC_ILMM003: "psum accumulation
# flags need to be all set or all unset"; seen at H 4096 / T 8, not at H 512): it is left to the compiler (None, the
# same overwrite: its PSUM region is written once). SWM_ACC_NONE=<site,...> overrides, for experiments.
_ACCN = set(_os.environ.get("SWM_ACC_NONE", "fold").split(","))


def _acc(site):
    return None if site in _ACCN else ACC


# --- host: layout ---------------------------------------------------------------------------

def blob_cols(H: int) -> int:
    """Bytes per partition of one unit: gate_up fp8 [H], down fp8 [H/2], gate_up tile scales fp32 [H/128],
    down column scales fp32 [H/128]."""
    C = H // P
    return H + H // 2 + 4 * C + 4 * C


def pack(w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, s_gate: torch.Tensor, s_up: torch.Tensor,
         s_down: torch.Tensor) -> torch.Tensor:
    """Checkpoint-layout FP8 experts -> unit blobs uint8 [E * NB, 128, blob_cols(H)].

    w_gate, w_up: fp8 [E, I, H] (out, in); w_down: fp8 [E, H, I] (out, in);
    s_gate, s_up: fp32 [E, I/128, H/128]; s_down: fp32 [E, H/128, I/128] (128 x 128 block scales,
    weight_scale_inv of the HF FP8 checkpoints: dequant = w * s per block).
    Within a unit (64 intermediate rows i0 .. i0 + 63 of expert e), partition p holds
      [0, H)            gate_up codes: [p, c * 128 + o] = W_gu[o, c * 128 + p], o < 64 gate row i0 + o, o >= 64 up row
      [H, 3H/2)         down codes:    [q, h'] = W_down[(q // 64) * H/2 + h', i0 + q % 64]
      [3H/2, +4C)       gate_up scales fp32 [o, c]  (row o's 128-column tile c)
      [.., +4C)         down scales fp32 [p, 2 c' + half] = s_down of output column half * H/2 + c' 128 + p,
                        input block i0 // 128
    """
    E, I, H = w_gate.shape
    assert I % IB == 0 and H % (2 * P) == 0, (I, H)
    NB, C, C2 = I // IB, H // P, H // 2 // P
    u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    gu = torch.cat([w_gate.view(E, NB, IB, H), w_up.view(E, NB, IB, H)], dim=2)  # [E, NB, 128 o, H]
    gu = gu.view(E, NB, P, C, P).permute(0, 1, 4, 3, 2).reshape(E * NB, P, C * P)  # [u, p, (c, o)]
    wd = w_down.view(E, H, NB, IB).permute(0, 2, 3, 1)  # [E, NB, i 64, H]
    wd = wd.reshape(E, NB, IB, 2, H // 2).permute(0, 1, 3, 2, 4).reshape(E * NB, P, H // 2)  # [u, (half, i), h']
    # per-row scales of the unit's gate and up rows (s_gate / s_up: per row [E, I, C], or per 128-row block)
    def rows(sc):
        return sc if sc.shape[1] == I else sc.repeat_interleave(P, dim=1)[:, :I]
    sgr, sur = rows(s_gate).view(E, NB, IB, C), rows(s_up).view(E, NB, IB, C)
    rb = torch.arange(NB) * IB // P
    sg = torch.cat([sgr, sur], dim=2).reshape(E * NB, P, C).float()  # [u, o, c]
    sdc = s_down[:, :, rb].permute(0, 2, 1)  # [E, NB, C (output tile)]
    sdh = sdc.repeat_interleave(P, dim=2)  # [E, NB, H] the scale of each output column
    sd = sdh.view(E, NB, 2, C2, P).permute(0, 1, 4, 3, 2).reshape(E * NB, P, 2 * C2).float()  # [u, p, 2 c' + half]
    return torch.cat([u8(gu), u8(wd), u8(sg), u8(sd)], dim=-1)


def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """fp8 [.., R, Cc] with block scales [.., ceil(R/128), ceil(Cc/128)] -> fp32 (a partial last block allowed: a
    TP shard of 64 rows keeps its 128-block's scale)."""
    R, Cc = w.shape[-2:]
    sf = s.float().repeat_interleave(P, dim=-2)[..., :R, :].repeat_interleave(P, dim=-1)[..., :Cc]
    return w.float() * sf


def emulate(x, idx, w, w_gate, w_up, w_down, s_gate, s_up, s_down, act: int = 0, limit: float = 0.0):
    """The kernel's arithmetic in torch (exact products, fp32 sums; g and a rounded to bf16)."""
    T, H = x.shape
    K = idx.shape[1]
    out = torch.zeros(T, H)
    for t in range(T):
        for k in range(K):
            e = int(idx[t, k])
            g = (dequant(w_gate[e], s_gate[e]) @ x[t].float()).bfloat16().float()
            u = (dequant(w_up[e], s_up[e]) @ x[t].float()).bfloat16().float()
            if act == 1:
                g, u = g.clamp(max=limit), u.clamp(-limit, limit)
            a = (torch.nn.functional.silu(g) * u).bfloat16().float()
            out[t] += float(w[t, k]) * (dequant(w_down[e], s_down[e]) @ a)
    return out.bfloat16()


def reference(x, idx, w, w_gate, w_up, w_down, s_gate, s_up, s_down, act: int = 0, limit: float = 0.0):
    """The plain fp32 MoE (the accuracy reference)."""
    T, H = x.shape
    out = torch.zeros(T, H)
    for t in range(T):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            g = dequant(w_gate[e], s_gate[e]) @ x[t].float()
            u = dequant(w_up[e], s_up[e]) @ x[t].float()
            if act == 1:
                g, u = g.clamp(max=limit), u.clamp(-limit, limit)
            out[t] += float(w[t, k]) * (dequant(w_down[e], s_down[e]) @ (torch.nn.functional.silu(g) * u))
    return out


def units(idx: torch.Tensor, NB: int) -> torch.Tensor:
    """int32 [1, T K NB]: work item n = (t K + k) NB + b -> unit idx[t, k] NB + b (host or graph; the kernel can
    also derive it, see unit_ids_in_kernel)."""
    return (idx.to(torch.int32).reshape(-1, 1) * NB + torch.arange(NB, dtype=torch.int32)).reshape(1, -1)


# --- kernel -----------------------------------------------------------------------------------

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def swiglu_moe_fp8(x, idx, w, blob, NB: int, group: int = GROUP, ring: int = RING, act: int = 0,
                       limit: float = 0.0, split: int = 1):
        """x bf16 [T, H]; idx int32 [T, K] expert ids; w bf16 [T, K]; blob uint8 [E NB, 128, F] -> bf16 [T, H].
        NB = I / 64 (units per expert). split 1: at grid 2 each program takes half the work groups (exchange by
        column halves at the end)."""
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        T, H = x.shape
        C, DW = H // P, H // 2
        C2 = DW // P
        F = blob.shape[2]
        o_dw, o_sg, o_sd = H, H + DW, H + DW + 4 * C
        K = idx.shape[1]
        NW = T * K * NB  # work items
        Q = group
        NQ = -(-NW // Q)
        NPAD = NQ * Q
        NR = ring
        PD = NR - 2
        out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        spl = split and npg == 2 and NQ >= 2
        q_lo, q_hi = (0, NQ) if not spl else ((0, (NQ + 1) // 2) if pid == 0 else ((NQ + 1) // 2, NQ))
        NL = q_hi - q_lo
        hw = C // 2
        if spl:
            rcv = nl.ndarray((P, T, hw), dtype=f32, buffer=nl.sbuf)

        # --- inputs: x rows, ids, weights; x -> k-major xc[i, c, t] = x[t, c 128 + i] by identity matmuls
        xt = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x, dge_mode=nisa.dge_mode.none)
        ipi = nl.ndarray((P, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ip, src=ipi)
        jpi = nl.ndarray((P, P), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jpi, pattern=[[1, P]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dd, src=jpi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        eye = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)  # eye[p, j] = [j == p]
        nisa.tensor_scalar(dst=eye, data=dd, op0=nl.equal, operand0=0.0)
        xc = nl.ndarray((P, C, T), dtype=bf16, buffer=nl.sbuf)
        CB = max(1, 512 // T)  # tiles c per PSUM bank (2 KB fp32 per partition)
        for c0 in range(0, C, CB):
            cn = min(CB, C - c0)
            xps = nl.ndarray((P, CB, T), dtype=f32, buffer=nl.psum)
            for c in range(cn):
                nisa.nc_matmul(dst=xps[:, c, :], stationary=xt[0:T, (c0 + c) * P:(c0 + c + 1) * P],
                               moving=eye[0:T, 0:T], accumulate=_acc('xT'))
            nisa.tensor_copy(dst=xc[:, c0:c0 + cn, :], src=xps[:, 0:cn, :], engine=nisa.vector_engine)

        # unit ids: work item n = (t K + k) NB + b -> idx[t, k] NB + b, on partition 0 (exact in fp32: < 2^24)
        N0 = T * K
        ii = nl.ndarray((1, N0), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ii, src=idx.reshape((1, N0)), dge_mode=nisa.dge_mode.none)
        idf = nl.ndarray((1, N0), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=idf, src=ii)
        e_sb = nl.ndarray((1, NPAD), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=e_sb, value=0)
        if NB == 1:
            nisa.tensor_copy(dst=e_sb[:, 0:NW], src=idf)
        else:
            bi = nl.ndarray((1, N0, NB), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=bi, pattern=[[0, N0], [1, NB]], offset=0, channel_multiplier=0)
            bfv = nl.ndarray((1, N0, NB), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=bfv, src=bi)
            ef = nl.ndarray((1, N0, NB), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=ef, data=idf.expand_dim(2).broadcast(2, NB), op0=nl.multiply,
                                      operand0=float(NB), op1=nl.add, operand1=bfv)
            nisa.tensor_copy(dst=e_sb[:, 0:NW], src=ef.flatten_dims(1, 2))
        # pair weights on every partition: wv[:, t K + k] (a [1, T K] row broadcast by a ones matmul)
        N = T * K
        wr = nl.ndarray((1, N), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wr, src=w.reshape((1, N)), dge_mode=nisa.dge_mode.none)
        one_r = nl.ndarray((1, P), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=one_r, value=1.0)
        wv = nl.ndarray((P, N), dtype=f32, buffer=nl.sbuf)
        for n0 in range(0, N, 512):  # one PSUM bank per 512 columns
            n1 = min(N, n0 + 512)
            pwb = nl.ndarray((P, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pwb, stationary=one_r, moving=wr[:, n0:n1], accumulate=_acc('wb'))
            nisa.tensor_copy(dst=wv[:, n0:n1], src=pwb)

        # --- constants: fold[o, q] = [o % 64 == q % 64]; mh = [p < 64, p >= 64]
        fd = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f1 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f2 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fd, data=dd, op0=nl.equal, operand0=0.0)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((P, 2), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        acc = nl.ndarray((P, T, C), dtype=f32, buffer=nl.sbuf)  # [p, t, g]: column h = (g % 2) H/2 + (g // 2) 128 + p
        nisa.memset(dst=acc, value=0.0)

        bufs = (nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        sg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        q_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        g_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        mm_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        sl_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        au_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        a2_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        y_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum))
        pf_r = (nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum))
        pd_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum))

        for it in range(-PD, NL + 1):
            gl = it + PD
            if 0 <= gl < NL:  # loads of group gl (runtime unit ids: the only runtime addresses in the kernel)
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[NPAD, 1], [1, 1]], offset=(q_lo + gl) * Q + s)
                    if LOAD == 0:  # software DGE (GpSimd), the compiler's choice for blob.select
                        nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e))
                    elif LOAD == 1:  # hardware DGE: unit e's [128, F] rows at a run-time offset on blob's axis 0
                        nisa.dma_copy(dst=bq[:, s, :], src=blob.ap(pattern=[[F, P], [1, F]], offset=0, scalar_offset=e,
                                                                   indirect_dim=0), dge_mode=nisa.dge_mode.hwdge,
                                      engine=(nisa.engine.sync if s % 2 == 0 else nisa.engine.scalar))
                    else:  # LOAD 2: half the units by software DGE, half by hardware DGE (two queue families)
                        if s % 2 == 0:
                            nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e))
                        else:
                            nisa.dma_copy(dst=bq[:, s, :], src=blob.ap(pattern=[[F, P], [1, F]], offset=0,
                                                                       scalar_offset=e, indirect_dim=0),
                                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            if 0 <= it < NL:  # stage A: gate_up tile products, scales, per-row sums
                gq = it
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    t = min(T - 1, ((q_lo + gq) * Q + s) // (K * NB))
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c:c + 1], stationary=wg[:, c * P:(c + 1) * P],
                                       moving=xc[:, c, t:t + 1], accumulate=_acc('gu'))
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(f32))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(f32))
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q, data1=pg, data2=sg_r[gq % 3], op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=2)
                nisa.tensor_tensor(dst=mm_r[gq % 3], data1=g.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
            if 1 <= it <= NL:  # stages B, C of group it - 1: fold, SwiGLU, down, weighted accumulate
                gq = it - 1
                if 'fold' in _ACCN:  # a fresh PSUM tile per group, compiler-inferred flag (its first and only write)
                    pf = nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum)
                else:
                    pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 2), stationary=fd, moving=mm_r[gq % 3].flatten_dims(1, 2),
                               accumulate=_acc('fold'))
                sl, au = sl_r[gq % 3], au_r[gq % 3]
                if act == 0:
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, 1], op=nl.multiply)
                else:
                    gc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    nisa.activation(dst=sl, op=nl.silu, data=gc)
                    nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                nisa.tensor_tensor(dst=a2_r[gq % 3], data1=au.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :], stationary=wd[:, c * P:(c + 1) * P],
                                       moving=a2_r[gq % 3][:, s, :], accumulate=_acc('dn'))
                y = y_r[gq % 3]
                nisa.tensor_tensor(dst=y.flatten_dims(2, 3), data1=pd.flatten_dims(2, 3), data2=sd_r[gq % 3],
                                   op=nl.multiply)
                for s in range(Q):
                    n = (q_lo + gq) * Q + s
                    if n < NW:
                        u = n // NB  # the pair t K + k
                        t = u // K
                        nisa.scalar_tensor_tensor(dst=acc[:, t, :], data=y[:, s].flatten_dims(1, 2), op0=nl.multiply,
                                                  operand0=wv[:, u:u + 1], op1=nl.add, operand1=acc[:, t, :])

        # --- combine the programs' partials (column halves), transpose [p, t, g] -> [t, h], write
        if spl:
            oth = 1 - pid
            nisa.sendrecv(src=acc[:, :, oth * hw:(oth + 1) * hw], dst=rcv, send_to_rank=oth, recv_from_rank=oth,
                          pipe_id=0)
            mine = acc[:, :, pid * hw:(pid + 1) * hw]
            nisa.tensor_tensor(dst=mine, data1=mine, data2=rcv, op=nl.add)
            g_lo, g_n = pid * hw, hw
        else:
            g_lo, g_n = 0, C
        if spl or pid == 0:
            accb = nl.ndarray((P, T, C), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=accb[:, :, g_lo:g_lo + g_n], src=acc[:, :, g_lo:g_lo + g_n])
            ot = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
            for gg in range(g_lo, g_lo + g_n):  # column block g: h = (g % 2) H/2 + (g // 2) 128 + p
                h0 = (gg % 2) * DW + (gg // 2) * P
                pt = nl.ndarray((T, P), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=pt, stationary=accb[:, :, gg], moving=eye, accumulate=_acc('oT'))
                nisa.tensor_copy(dst=ot[:, h0:h0 + P], src=pt, engine=nisa.vector_engine)
            if spl:  # this program's columns: g = 2 c' + half for g in [g_lo, g_lo + hw)
                for gg in range(g_lo, g_lo + g_n):
                    h0 = (gg % 2) * DW + (gg // 2) * P
                    nisa.dma_copy(dst=out[:, h0:h0 + P], src=ot[:, h0:h0 + P], dge_mode=nisa.dge_mode.none)
            else:
                nisa.dma_copy(dst=out, src=ot, dge_mode=nisa.dge_mode.none)
        if npg > 1:
            nisa.core_barrier(data=out, cores=(0, 1))
        return out
else:
    swiglu_moe_fp8 = None
