# SPDX-License-Identifier: Apache-2.0
"""Final RMSNorm + vocab-parallel lm_head inside the decode megakernel (one TP rank, LNC=2).

Weights: lnf [H] fp32 (1 + w of the model's final norm), lmw = this rank's vocab shard in the kernel tile layout
  [NC, NT, 128, KT, 512] bf16, element (c, t, p, kt, j) = W[v = c*VC + t*512 + j, k = kt*128 + p] (zero rows past VC),
  VC = V_rank / NC columns per pcore. One [128, KT, 512] tile = 16 KB contiguous per partition (one 2 MB DMA).
Outputs: hn [R, H] bf16 (normed hidden) and logits [R, V_rank] bf16 (pcore c writes columns [c*VC, (c+1)*VC)), the
same rank-local logits shard the model's ColumnParallelLinear lm_head produces, so the on-device sampler runs
unchanged.
"""
import nki.isa as nisa
import nki.language as nl

from .mk_common import BF, F32, H, KT, rmsnorm_rows, rows_to_std, sb

NTW = 512


def lm_layout(W, NC):
    """torch: this rank's lm_head shard W [V_rank, H] -> [NC, NT, 128, KT, 512] (zero padded), contiguous."""
    import torch
    V = W.shape[0]
    VC = V // NC
    NT = -(-VC // NTW)
    out = torch.zeros(NC, NT * NTW, H, dtype=W.dtype)
    out[:, :VC] = W.reshape(NC, VC, H)
    return out.reshape(NC, NT, NTW, KT, 128).permute(0, 1, 4, 3, 2).contiguous()


def lm_tile_load(lmw, t, pid):
    NT, KT_ = lmw.shape[1], lmw.shape[3]
    w = sb((128, KT_, NTW), BF)
    nisa.dma_copy(dst=w, src=lmw.ap(pattern=[[KT_ * NTW, 128], [NTW, KT_], [1, NTW]],
                                     offset=(pid * NT + t) * 128 * KT_ * NTW))
    return w


def lm_head_tail(hR, lnf, lmw, logits, hn_out, ident, R, NC, pid, V, w_pre=()):
    """hR [R, H] bf16 SBUF residual after the last layer (identical on both pcores). Writes hn_out [R, H] (each pcore
    its H half) and this pcore's logits columns."""
    NT = lmw.shape[1]
    VC = V // NC
    hw = H // NC
    xn = rmsnorm_rows(hR, lnf, R)
    nisa.dma_copy(dst=hn_out[:, nl.ds(pid * hw, hw)], src=xn[:, nl.ds(pid * hw, hw)])
    xT = rows_to_std(xn, KT, ident, R)
    TSZ = 128 * KT * NTW
    for t in range(NT):
        w = w_pre[t] if t < len(w_pre) else lm_tile_load(lmw, t, pid)
        ps = nl.ndarray((R, NTW), dtype=F32, buffer=nl.psum)
        for kt in range(KT):
            nisa.nc_matmul(ps, xT[:, kt * R:(kt + 1) * R], w[:, kt, :])
        nn = min(NTW, VC - t * NTW)
        lg = sb((R, NTW), BF)
        nisa.tensor_copy(dst=lg[:, 0:nn], src=ps[:, 0:nn])
        nisa.dma_copy(dst=logits[:, nl.ds(pid * VC + t * NTW, nn)], src=lg[:, 0:nn])

