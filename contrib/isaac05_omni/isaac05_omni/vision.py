# SPDX-License-Identifier: Apache-2.0
"""Isaac-0.5 vision tower on Neuron for the action path: the checkpoint's Qwen3.5-MoE ViT (patch embed, Isaac05
pos-embed interpolation, 27 blocks, merger) on N images of the same (H, W) patch grid, as one compiled graph.

Images attend only within themselves (HF ``cu_seqlens`` per image), so the batch is [N, S, D] with per-image SDPA.
The host precomputes the grid-dependent tables once per (H, W): the 4-tap pos-embed indices + weights (Isaac05 order:
interpolate, one bf16 4-way sum, merge-permute) and the 2-D rotary cos/sin, bit-identical to the HF module's math.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

# vision attention kernel. sdpa = torch SDPA; nki = nkilib attention_cte flash kernel.
_ATTN = os.environ.get("ISAAC05_VIS_ATTN", "sdpa")  # nki = 1.05x tower speed at 6 images but action cos 0.9981 -> 0.9958 on one obs, no end-to-end gain: off
# vision MLP. torch = HF module; nki = nkilib mlp (fc1 + tanh-GELU + fc2 with both biases, LayerNorm fused),
# the vllm-neuron Qwen3-VL serving-encoder pattern (QWEN3VL_VIS_MLP=nki).
_MLP = os.environ.get("ISAAC05_VIS_MLP", "torch")  # nki: nkilib mlp (LN fused), same verdict as attention: off
_KT = os.environ.get("ISAAC05_VIS_KLAYOUT", "sd")  # K layout for attention_cte: sd = [B, S, D]; ds = [B, D, S] (119 vs 114 ms at 24 images, off)


def _mlp_nki(blk, x):
    """norm2 + fc1 + GELU(tanh) + fc2 via nkilib mlp, weights pre-transposed once per block ([H, I] / [I, H])."""
    import vllm_neuron.functional as NF
    from nkilib.core.utils.common_types import ActFnType, NormType

    m = blk.mlp
    if not hasattr(m, "_w1t"):
        m._w1t = m.linear_fc1.weight.t().contiguous()
        m._w2t = m.linear_fc2.weight.t().contiguous()
    shp = x.shape
    out = NF.mlp(x.reshape(-1, shp[-1]), m._w1t, m._w1t, m._w2t, skip_gate=True,
                 act_fn=ActFnType.GELU_Tanh_Approx, up_b=m.linear_fc1.bias.unsqueeze(0),
                 ln_w=blk.norm2.weight.unsqueeze(0), norm_b=blk.norm2.bias.unsqueeze(0),
                 norm_type=NormType.LAYER_NORM, eps=blk.norm2.eps)
    return out.reshape(shp) + m.linear_fc2.bias


class IsaacVisionTower(nn.Module):
    def __init__(self, hf_visual: nn.Module):
        super().__init__()
        import copy

        self.v = hf_visual  # Isaac05Qwen36VisionModel (weights loaded); moved to the device by the pipeline
        # CPU twin for the grid tables (pos-embed interpolation + rotary use host ops / .tolist())
        self._cpu = copy.deepcopy(hf_visual).float()
        self._cpu.blocks = nn.ModuleList()
        self._tables = {}

    @torch.no_grad()
    def tables(self, h: int, w: int, device):
        key = (h, w)
        if key not in self._tables:
            g = torch.tensor([[1, h, w]], dtype=torch.long)
            pos = self._cpu.fast_pos_embed_interpolate(g)            # [S, D] (Isaac05 order, bf16 sum)
            rot = self._cpu.rot_pos_emb(g)                            # [S, dim/2]
            emb = torch.cat((rot, rot), dim=-1)
            self._tables[key] = (pos.to(device), emb.cos().to(device), emb.sin().to(device))
        return self._tables[key]

    def forward(self, patches: torch.Tensor, pos: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """patches [N, S, C*T*P*P] (temporal tubelets), pos [S, D], cos/sin [S, hd] -> merged [N * S / 4, out_hidden]."""
        v = self.v
        N, S, _ = patches.shape
        x = v.patch_embed(patches.reshape(N * S, -1)).view(N, S, -1) + pos.unsqueeze(0).to(patches.dtype)
        nh = v.blocks[0].attn.num_heads
        c, s = cos.float().view(1, S, 1, -1), sin.float().view(1, S, 1, -1)
        for blk in v.blocks:
            y = blk.norm1(x)
            qkv = blk.attn.qkv(y).view(N, S, 3, nh, -1)
            q, k, val = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            qd, kd = q.dtype, k.dtype
            qf, kf = q.float(), k.float()
            h2 = qf.shape[-1] // 2
            rq = torch.cat((-qf[..., h2:], qf[..., :h2]), dim=-1)
            rk = torch.cat((-kf[..., h2:], kf[..., :h2]), dim=-1)
            q = (qf * c + rq * s).to(qd)
            k = (kf * c + rk * s).to(kd)
            if _ATTN == "nki":
                # nkilib attention_cte (flash, bidirectional): [N*nh, S, hd] blocks, no [N, nh, S, S] score tensor in
                # HBM (the torch SDPA path spills it: ~7.8 GB reload at 24 images, profiling).
                # K goes in [B, D, S] (tp_k=False): the kernel then needs no per-block K transpose (the [B, S, D] K
                # cost 33k 72x72 PE transposes per call at 24 images).
                import vllm_neuron.functional as NF

                hd = q.shape[-1]
                qb = q.permute(0, 2, 1, 3).reshape(N * nh, S, hd)
                kb = (k.permute(0, 2, 1, 3).reshape(N * nh, S, hd) if _KT == "sd" else
                      k.permute(0, 2, 3, 1).reshape(N * nh, hd, S))
                vb = val.permute(0, 2, 1, 3).reshape(N * nh, S, hd)
                o = NF.flash_attention(qb, kb, vb, scale=blk.attn.scaling, causal_mask=False, tp_q=True,
                                       tp_k=(_KT == "sd"))
                o = o.view(N, nh, S, hd)
            else:
                o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), val.transpose(1, 2),
                                                   scale=blk.attn.scaling, is_causal=False)
            x = x + blk.attn.proj(o.transpose(1, 2).reshape(N, S, -1))
            if _MLP == "nki":
                x = x + _mlp_nki(blk, x)
            else:
                x = x + blk.mlp(blk.norm2(x))
        return v.merger(x.reshape(N * S, -1))
