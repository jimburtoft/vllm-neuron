"""40-layer MTP verify step (B requests x T tokens, R = B*T rows b-major) in the port's layouts.
Same argument contract as mk_port.port_body except: x_rows/cosb/sinb/slot have R rows, pos0 [B] = position of each
request's first verify token, ridx [B] = read checkpoint rows, widx [T*B] (token-major) = per-token write rows,
bmask [R, 8R], cm [16, T] fp32 causal mask (row h*T + t, NEG where t' > t). mk is the GDN block mask for J = HV_C*B.
"""
import os

import nki.isa as nisa
import nki.language as nl

from .gdn_block import dims, gdn_win_load, load_mk
from .gdn_t4 import gdn_mixer_t4, slot_io_t4
from .mk_common import BF, H, KT, mk_fold_t4, mul_w_rows, rs_cols, w_cols, reduce_residual, rmsnorm_rows, row_rstd, rows_to_std, sb, store_rows
from .mk_gqa_t4 import gqa_mixer_t4
from .mk_lmhead import lm_head_tail, lm_tile_load
from .moe_block import TOPK, moe_block

PATTERN = "GGGA" * 10


def port_body_t4(x_rows, cosb, sinb, bt, slot, pos0, ridx, widx, ident_in, mk_in, bmask_in, cm_in, W,
                 pattern=PATTERN, ranks=None, lm=None):
    NC = nl.num_programs(0)
    pid = nl.program_id(0) if NC > 1 else 0
    R = x_rows.shape[0]
    B = bt.shape[0]
    T = R // B
    HV_C = dims(NC, B)[1]
    HW = H // NC
    h_out = nl.ndarray((R, H), dtype=BF, buffer=nl.shared_hbm)
    ident = sb((128, 128), BF)
    nisa.dma_copy(dst=ident, src=ident_in)
    ones = sb((128, 128), BF)
    nisa.memset(ones, 1.0)
    ones_f = sb((R, 128))
    nisa.memset(ones_f, 1.0)
    J = HV_C * B
    mk = load_mk(mk_in)
    bmask = sb((R, TOPK * R))
    nisa.dma_copy(dst=bmask, src=bmask_in)
    hR = sb((R, H), BF)
    nisa.dma_copy(dst=hR, src=x_rows)
    sio = slot_io_t4(ridx, widx, B, T, NC, pid)

    ig = ia = 0
    wnext = ()
    LMPF = int(os.environ.get("MK_LMPF", "0"))  # first LMPF lm_head tiles before the last MoE AR
    lm_pre = ()
    PFG = int(os.environ.get("MK_PFG", "0"))  # next GDN layer's first PFG in_proj k-groups early
    for l, kind in enumerate(pattern):
        FOLD = mk_fold_t4()
        rsc1 = row_rstd(hR, R) if FOLD else None
        rk1 = rs_cols(rsc1, ident, R) if FOLD == 4 else None
        x1 = rmsnorm_rows(hR, W["ln1"][l], R) if not FOLD else (mul_w_rows(hR, W["ln1"][l], R) if FOLD == 2 else hR)
        xT = rows_to_std(x1, KT, ident, R, wcol=w_cols(W["ln1"][l], R) if FOLD in (3, 4) else None, rk=rk1)
        rsc1 = None if FOLD == 4 else rsc1
        if kind == "G":
            gsc = nl.ndarray((NC * R * 2 * HV_C,), dtype=BF, buffer=nl.shared_hbm)
            y = gdn_mixer_t4(xT, W["Win"][ig], W["Wout"][ig], W["cw"][ig], W["cs"][ig], W["rs"][ig], W["alog"][ig],
                             W["dtb"][ig], W["nw"][ig], ident, ones, mk, gsc, B, T, NC, pid, sio, rsc=rsc1,
                             w_pre=wnext)
            wnext = ()
            ig += 1
        else:
            y = gqa_mixer_t4(xT, W["Wqkv"][ia], W["Wo"][ia], W["qnw"][ia], W["knw"][ia], cosb, sinb, W["kc"][ia],
                             W["vc"][ia], bt, slot, pos0, cm_in, ident, B, T, NC, pid, rsc=rsc1)
            ia += 1
        hR = reduce_residual(y, hR, R, NC, pid, ranks, f"x{l}")
        x2 = rmsnorm_rows(hR, W["ln2"][l], R) if not FOLD else (mul_w_rows(hR, W["ln2"][l], R) if FOLD == 2 else hR)
        fold = (row_rstd(hR, R), None if FOLD == 2 else W["ln2"][l]) if FOLD else None
        rk2 = rs_cols(fold[0], ident, R) if FOLD == 4 else None
        fold = None if FOLD == 4 else fold
        ym = moe_block(x2, W["Wr"][l], W["gu"][l], W["dn"][l], W["Wsgu"][l], W["Wsd"][l], W["Wsg"][l], ident,
                       ones_f, bmask, R, NC, pid, fold=fold,
                       wcol=w_cols(W["ln2"][l], R, pmaj=True) if FOLD in (3, 4) else None, rk=rk2)
        ymb = sb((R, HW), BF)
        nisa.tensor_copy(dst=ymb, src=ym)
        if PFG and l + 1 < len(pattern) and pattern[l + 1] == "G":
            wnext = tuple(gdn_win_load(W["Win"][ig], g, NC, pid, B) for g in range(PFG))
        if lm is not None and LMPF and l == len(pattern) - 1:
            lm_pre = tuple(lm_tile_load(lm[1], t, pid) for t in range(LMPF))
        hR = reduce_residual(ymb, hR, R, NC, pid, ranks, f"m{l}")

    if lm is not None:  # final norm + vocab-parallel lm_head in-kernel; h_out = normed hidden
        logits = nl.ndarray((R, lm[2]), dtype=BF, buffer=nl.shared_hbm)
        lm_head_tail(hR, lm[0], lm[1], logits, h_out, ident, R, NC, pid, lm[2], w_pre=lm_pre)
        return (h_out, logits) + tuple(W["cs"]) + tuple(W["rs"]) + tuple(W["kc"]) + tuple(W["vc"])
    store_rows(h_out, hR, NC, pid)
    return (h_out,) + tuple(W["cs"]) + tuple(W["rs"]) + tuple(W["kc"]) + tuple(W["vc"])
