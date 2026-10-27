# SPDX-License-Identifier: Apache-2.0
"""NKI kernel for the query-split sparse prefill (``HY4_PREFILL_SPARSE_NKI=1``).

``select_to_rows`` -- per query, the first ``ksel`` positions of a 0/1
selection, ascending, padded with the selection width (out of range). The
torch reference is ``HYV4Attention._prefill_rows_ref`` in ``model.py``.
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
