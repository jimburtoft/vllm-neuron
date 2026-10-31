# SPDX-License-Identifier: Apache-2.0
"""One decode step (one token per request) of Qwen3.5-35B-A3B's 40 decoder layers, plus the final norm and the
vocab-parallel lm_head, as one NKI kernel per TP rank (TP=4, LNC=2).

Layer pattern [DeltaNet, DeltaNet, DeltaNet, attention] x 10, each followed by the sparse MoE block. Every
per-layer tensor is its own kernel argument (see entry.py), passed as the model holds it: weights (or a load-time
copy in the kernel's layout, see ``Qwen3_5DecodeMegakernel``), the DeltaNet state pages and the attention KV caches,
which are updated in place and returned so the compiler aliases them to the inputs. The TP all-reduces run inside
the kernel over ``ranks``.
"""
import nki.isa as nisa
import nki.language as nl

from .gdn_block import dims, gdn_mixer, gdn_win_load, page_tiles
from .mk_common import BF, H, KT, reduce_residual, row_rstd, rows_to_std, rs_cols, sb, store_rows, w_cols
from .mk_gqa import gqa_mixer
from .mk_lmhead import lm_head_tail, lm_tile_load
from .moe_block import TOPK, moe_block

PATTERN = "GGGA" * 10
WIN_PREFETCH = 2   # in_proj k-groups of the next DeltaNet layer loaded before the MoE all-reduce
LM_PREFETCH = 4    # lm_head tiles loaded before the last MoE all-reduce


def step_body(x_rows, cosb, sinb, bt, slot, pos, pages, ident_in, mk_in, bmask_in, W, gmap, pattern=PATTERN,
              ranks=None, lm=None):
    """W: dict group -> tuple of per-layer tensors (entry.py). gmap[g] = state page group of DeltaNet layer g
    (row of ``pages``). Returns (h_out, [logits,] *state_a, *state_b, *k_caches, *v_caches); the state / KV
    outputs ARE the inputs (in-place writes)."""
    NC = nl.num_programs(0)
    pid = nl.program_id(0) if NC > 1 else 0
    B = x_rows.shape[0]
    HV_C = dims(NC, B)[1]
    HW = H // NC
    h_out = nl.ndarray((B, H), dtype=BF, buffer=nl.shared_hbm)

    ident = sb((128, 128), BF)
    nisa.dma_copy(dst=ident, src=ident_in)
    ones = sb((128, 128), BF)
    nisa.memset(ones, 1.0)
    ones_f = sb((B, 128))
    nisa.memset(ones_f, 1.0)
    J = HV_C * B
    mk = sb((J, J * 128), BF)
    nisa.dma_copy(dst=mk, src=mk_in)
    bmask = sb((B, TOPK * B))
    nisa.dma_copy(dst=bmask, src=bmask_in)
    hR = sb((B, H), BF)
    nisa.dma_copy(dst=hR, src=x_rows)
    pgs = [page_tiles(pages, g, B) for g in range(max(gmap) + 1)]

    ig = ia = 0
    wnext = ()
    lm_pre = ()
    for l, kind in enumerate(pattern):
        # input RMSNorm: rsqrt on the rows, the scale applied in the transposed column layout
        rk1 = rs_cols(row_rstd(hR, B), ident, B)
        xT = rows_to_std(hR, KT, ident, B, wcol=w_cols(W["ln1"][l], B), rk=rk1)
        if kind == "G":
            gsc = nl.ndarray((NC * B * 2 * HV_C,), dtype=BF, buffer=nl.shared_hbm)
            y = gdn_mixer(xT, W["Win"][ig], W["Wout"][ig], W["cw"][ig], W["sa"][ig], W["sb"][ig], W["alog"][ig],
                          W["dtb"][ig], W["nw"][ig], ident, ones, mk, gsc, pgs[gmap[ig]], B, NC, pid, w_pre=wnext)
            wnext = ()
            ig += 1
        else:
            y = gqa_mixer(xT, W["Wqkv"][ia], W["Wo"][ia], W["qnw"][ia], W["knw"][ia], cosb, sinb, W["kc"][ia],
                          W["vc"][ia], bt, slot, pos, ident, B, NC, pid)
            ia += 1
        hR = reduce_residual(y, hR, B, NC, pid, ranks, f"x{l}")
        rk2 = rs_cols(row_rstd(hR, B), ident, B)
        ym = moe_block(hR, W["Wr"][l], W["gu"][l], W["dn"][l], W["Wsgu"][l], W["Wsd"][l], W["Wsg"][l], ident,
                       ones_f, bmask, B, NC, pid, wcol=w_cols(W["ln2"][l], B, pmaj=True), rk=rk2)
        ymb = sb((B, HW), BF)
        nisa.tensor_copy(dst=ymb, src=ym)
        if l + 1 < len(pattern) and pattern[l + 1] == "G":
            wnext = tuple(gdn_win_load(W["Win"][ig], g, NC, pid, B) for g in range(WIN_PREFETCH))
        if lm is not None and l == len(pattern) - 1:
            lm_pre = tuple(lm_tile_load(lm[1], t, pid) for t in range(LM_PREFETCH))
        hR = reduce_residual(ymb, hR, B, NC, pid, ranks, f"m{l}")

    state = tuple(W["sa"]) + tuple(W["sb"]) + tuple(W["kc"]) + tuple(W["vc"])
    if lm is not None:  # final norm + vocab-parallel lm_head; h_out = normed hidden
        logits = nl.ndarray((B, lm[2]), dtype=BF, buffer=nl.shared_hbm)
        lm_head_tail(hR, lm[0], lm[1], logits, h_out, ident, B, NC, pid, lm[2], w_pre=lm_pre)
        return (h_out, logits) + state
    store_rows(h_out, hR, NC, pid)
    return (h_out,) + state
