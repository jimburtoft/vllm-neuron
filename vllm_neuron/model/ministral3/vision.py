# SPDX-License-Identifier: Apache-2.0
"""Pixtral vision tower + Mistral3 multimodal projector on Neuron (BF16).

Block-packed layout (like the Qwen3-VL encoder): every tensor is
``[num_blocks, block_size, ...]``, one image per block. Attention is
bidirectional within an image (``bound_min`` / ``bound_max``), and pad rows
attend to nothing. The CPU side (``Ministral3ForConditionalGeneration``)
patchifies the image, computes the 2D rotary cos/sin, and builds the 2x2
spatial-merge gather index, so the device graph has static shapes.

Architecture (HF ``PixtralVisionModel`` + ``Mistral3MultiModalProjector``)::

    patches [*, C*p*p] -> patch_conv (linear, no bias) -> ln_pre (RMSNorm)
    -> N x [RMSNorm -> attention (2D RoPE) -> + ; RMSNorm -> SiLU-GLU MLP -> +]
    -> projector RMSNorm -> 2x2 merge (unfold order) -> merging_layer
    -> linear_1 -> GELU -> linear_2 -> encoder cache rows

Parallelism: encoder TP group (heads and MLP / projector intermediate sharded),
encoder DP across blocks, as in the Qwen3-VL encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import get_neuron_encoder_tp_group
from vllm_neuron.utils.dynamo_graph import allow_in_graph_cacheable
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

_ATTN_SEQ_ALIGN = 128


def _use_nki_mlp() -> bool:
    import os

    return os.environ.get("VLLM_NEURON_PIXTRAL_NKI_MLP") == "1"


def _use_fused_attn() -> bool:
    """VLLM_NEURON_PIXTRAL_FUSED_QKV=1: one NF.qkv_proj with the RMSNorm fused,
    flash attention with tp_out, NF.o_proj; one gate/up matmul in the MLP."""
    import os

    return os.environ.get("VLLM_NEURON_PIXTRAL_FUSED_QKV") == "1"


@allow_in_graph_cacheable
def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, traceable on Neuron."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * w.float()).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _param(*shape, dtype):
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def _row_shard(shard, n):
    """Checkpoint [out, in] -> stored [in, out/n] (column parallel)."""
    return sharding_weight_loader(
        shard_dim=1, shard_size=shard, num_shards=n, is_storage_transposed=True
    )


def _col_shard(shard, n):
    """Checkpoint [out, in] -> stored [in/n, out] (row parallel)."""
    return sharding_weight_loader(
        shard_dim=0, shard_size=shard, num_shards=n, is_storage_transposed=True
    )


def _replicated_transposed():
    return SafetensorsWeightLoader(transform=lambda s, r: s[0][:].t().contiguous())


class PixtralAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int, dtype: torch.dtype):
        super().__init__()
        self.tp_group = get_neuron_encoder_tp_group()
        self.tp = self.tp_group.world_size
        assert num_heads % self.tp == 0
        self.head_dim = hidden // num_heads
        self.heads = num_heads // self.tp
        self.scale = self.head_dim**-0.5
        local = self.heads * self.head_dim
        self.q_weight = _param(hidden, local, dtype=dtype)
        self.k_weight = _param(hidden, local, dtype=dtype)
        self.v_weight = _param(hidden, local, dtype=dtype)
        self.o_weight = _param(local, hidden, dtype=dtype)
        for w in (self.q_weight, self.k_weight, self.v_weight):
            set_weight_loader(w, _row_shard(local, self.tp))
        set_weight_loader(self.o_weight, _col_shard(local, self.tp))

    def fuse_weights(self) -> None:
        """Build the concatenated [H, 3*local] QKV weight (after loading)."""
        self.qkv_weight = nn.Parameter(
            torch.cat([self.q_weight, self.k_weight, self.v_weight], dim=1).contiguous(),
            requires_grad=False,
        )

    def forward_fused(self, x, ln_w, eps, cos, sin, bound_min, bound_max):
        """x: residual stream [nb, bs, H]; the RMSNorm runs inside NF.qkv_proj."""
        from nkilib.core.utils.common_types import NormType

        nb, bs, _ = x.shape
        local = self.heads * self.head_dim
        qkv = NF.qkv_proj(
            hidden=x,
            qkv_weights=self.qkv_weight,
            norm_weights=ln_w.view(1, -1),
            eps=eps,
            norm_type=NormType.RMS_NORM,
        )
        q = qkv[..., :local].view(nb, bs, self.heads, self.head_dim)
        k = qkv[..., local : 2 * local].view(nb, bs, self.heads, self.head_dim)
        v = qkv[..., 2 * local :].view(nb, bs, self.heads, self.head_dim)
        c = cos.to(x.dtype).unsqueeze(2)
        s = sin.to(x.dtype).unsqueeze(2)
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s

        def heads_first(t):
            return t.permute(0, 2, 1, 3).reshape(nb * self.heads, bs, self.head_dim)

        out = NF.flash_attention(
            heads_first(q),
            heads_first(k),
            heads_first(v),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=True,  # [nb*N, D, S]: the NF.o_proj layout, no permute
            bound_min=bound_min.repeat_interleave(self.heads, dim=0),
            bound_max=bound_max.repeat_interleave(self.heads, dim=0),
        )
        out = NF.o_proj(out.view(nb, self.heads, self.head_dim, bs), self.o_weight)
        if self.tp > 1:
            self.tp_group.all_reduce(out)
        return out

    def forward(self, x, cos, sin, bound_min, bound_max):
        nb, bs, _ = x.shape
        q = (x @ self.q_weight).view(nb, bs, self.heads, self.head_dim)
        k = (x @ self.k_weight).view(nb, bs, self.heads, self.head_dim)
        v = (x @ self.v_weight).view(nb, bs, self.heads, self.head_dim)
        c = cos.to(x.dtype).unsqueeze(2)
        s = sin.to(x.dtype).unsqueeze(2)
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s

        def heads_first(t):
            return t.permute(0, 2, 1, 3).reshape(nb * self.heads, bs, self.head_dim)

        out = NF.flash_attention(
            heads_first(q),
            heads_first(k),
            heads_first(v),
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=bound_min.repeat_interleave(self.heads, dim=0),
            bound_max=bound_max.repeat_interleave(self.heads, dim=0),
        )
        out = out.view(nb, self.heads, bs, self.head_dim).permute(0, 2, 1, 3)
        out = out.reshape(nb, bs, self.heads * self.head_dim) @ self.o_weight
        if self.tp > 1:
            self.tp_group.all_reduce(out)
        return out


class PixtralMLP(nn.Module):
    def __init__(self, hidden: int, inter: int, dtype: torch.dtype, act: str = "silu"):
        super().__init__()
        if act not in ("silu", "gelu"):
            raise NotImplementedError(f"Pixtral hidden_act={act!r} is not supported")
        self.act = F.silu if act == "silu" else gelu
        self.tp_group = get_neuron_encoder_tp_group()
        self.tp = self.tp_group.world_size
        self.inter = inter // self.tp
        self.gate_weight = _param(hidden, self.inter, dtype=dtype)
        self.up_weight = _param(hidden, self.inter, dtype=dtype)
        self.down_weight = _param(self.inter, hidden, dtype=dtype)
        set_weight_loader(self.gate_weight, _row_shard(self.inter, self.tp))
        set_weight_loader(self.up_weight, _row_shard(self.inter, self.tp))
        set_weight_loader(self.down_weight, _col_shard(self.inter, self.tp))

    def fuse_weights(self) -> None:
        self.gate_up_weight = nn.Parameter(
            torch.cat([self.gate_weight, self.up_weight], dim=1).contiguous(), requires_grad=False
        )

    def forward(self, x, ln_w=None, eps=1e-5):
        """x: [nb, bs, H]. With ln_w (RMSNorm gamma), the norm is applied first;
        VLLM_NEURON_PIXTRAL_NKI_MLP=1 fuses it into the nkilib MLP kernel."""
        if _use_nki_mlp() and self.act is F.silu:
            from nkilib.core.utils.common_types import NormType

            nb, bs, h = x.shape
            out = NF.mlp(
                x.reshape(nb * bs, h),
                self.gate_weight,
                self.up_weight,
                self.down_weight,
                eps=eps,
                ln_w=ln_w.view(1, -1) if ln_w is not None else None,
                norm_type=NormType.RMS_NORM if ln_w is not None else NormType.NO_NORM,
            ).view(nb, bs, h)
        elif getattr(self, "gate_up_weight", None) is not None:
            if ln_w is not None:
                x = _rms_norm(x, ln_w, eps)
            gu = x @ self.gate_up_weight
            out = (self.act(gu[..., : self.inter]) * gu[..., self.inter :]) @ self.down_weight
        else:
            if ln_w is not None:
                x = _rms_norm(x, ln_w, eps)
            out = (self.act(x @ self.gate_weight) * (x @ self.up_weight)) @ self.down_weight
        if self.tp > 1:
            self.tp_group.all_reduce(out)
        return out


class PixtralLayer(nn.Module):
    def __init__(self, cfg, dtype):
        super().__init__()
        self.eps = 1e-5
        self.attention_norm = _param(cfg.hidden_size, dtype=dtype)
        self.ffn_norm = _param(cfg.hidden_size, dtype=dtype)
        self.attention = PixtralAttention(cfg.hidden_size, cfg.num_attention_heads, dtype)
        self.feed_forward = PixtralMLP(
            cfg.hidden_size, cfg.intermediate_size, dtype, getattr(cfg, "hidden_act", "silu")
        )

    def forward(self, x, cos, sin, bmin, bmax):
        if getattr(self.attention, "qkv_weight", None) is not None:
            x = x + self.attention.forward_fused(x, self.attention_norm, self.eps, cos, sin, bmin, bmax)
            return x + self.feed_forward(x, self.ffn_norm, self.eps)
        x = x + self.attention(_rms_norm(x, self.attention_norm, self.eps), cos, sin, bmin, bmax)
        return x + self.feed_forward(x, self.ffn_norm, self.eps)


class Mistral3Projector(nn.Module):
    """RMSNorm -> 2x2 patch merge -> merging_layer -> linear_1 -> GELU -> linear_2."""

    def __init__(self, vision_hidden, text_hidden, merge, eps, dtype):
        super().__init__()
        self.tp_group = get_neuron_encoder_tp_group()
        self.tp = self.tp_group.world_size
        self.merge_unit = merge * merge
        self.eps = eps
        self.norm = _param(vision_hidden, dtype=dtype)
        # merging_layer: [d, d*4] in the checkpoint; kept replicated.
        self.merging_weight = _param(vision_hidden * self.merge_unit, vision_hidden, dtype=dtype)
        set_weight_loader(self.merging_weight, _replicated_transposed())
        self.text_local = text_hidden // self.tp
        self.linear_1 = _param(vision_hidden, self.text_local, dtype=dtype)
        self.linear_2 = _param(self.text_local, text_hidden, dtype=dtype)
        set_weight_loader(self.linear_1, _row_shard(self.text_local, self.tp))
        set_weight_loader(self.linear_2, _col_shard(self.text_local, self.tp))

    def forward(self, x, merge_idx):
        """x: [nb, bs, d]; merge_idx: [nb, merged_bs, 4] (block-local row ids, unfold order)."""
        nb, bs, d = x.shape
        x = _rms_norm(x, self.norm, self.eps)
        mb = merge_idx.shape[1]
        flat = (merge_idx + torch.arange(nb, device=x.device, dtype=merge_idx.dtype).view(nb, 1, 1) * bs).reshape(-1)
        g = x.reshape(nb * bs, d).index_select(0, flat).view(nb, mb, self.merge_unit, d)
        # torch unfold orders channels d-major: feature index = c * 4 + k.
        g = g.transpose(2, 3).reshape(nb, mb, d * self.merge_unit)
        h = g @ self.merging_weight
        h = gelu(h @ self.linear_1) @ self.linear_2
        if self.tp > 1:
            self.tp_group.all_reduce(h)
        return h


class PixtralVisionTower(nn.Module):
    """Vision tower + projector; writes merged rows into the encoder cache."""

    def __init__(self, hf_config, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        vc = hf_config.vision_config
        self.config = hf_config
        self.dtype = dtype
        self.patch_dim = vc.num_channels * vc.patch_size * vc.patch_size
        self.merge_unit = hf_config.spatial_merge_size**2
        from vllm_neuron.parallel.neuron_parallel_state import get_neuron_encoder_dp_group

        self.dp_group = get_neuron_encoder_dp_group()
        self.dp_size = self.dp_group.world_size if self.dp_group else 1
        self.patch_weight = _param(self.patch_dim, vc.hidden_size, dtype=dtype)
        set_weight_loader(
            self.patch_weight,
            SafetensorsWeightLoader(
                transform=lambda s, r: s[0][:].reshape(s[0].get_shape()[0], -1).t().contiguous()
            ),
        )
        self.ln_pre = _param(vc.hidden_size, dtype=dtype)
        n_layers = vc.num_hidden_layers
        layer = hf_config.vision_feature_layer
        if isinstance(layer, int) and layer != -1:
            n_layers = layer + 1 if layer >= 0 else n_layers + layer + 1
        self.layers = nn.ModuleList([PixtralLayer(vc, dtype) for _ in range(n_layers)])
        text_hidden = hf_config.text_config.hidden_size
        self.projector = Mistral3Projector(
            vc.hidden_size,
            text_hidden,
            hf_config.spatial_merge_size,
            hf_config.text_config.rms_norm_eps,
            dtype,
        )

    @staticmethod
    def build_weight_mappings(n_layers: int) -> dict[str, str]:
        m = {
            "patch_weight": "vision_tower.patch_conv.weight",
            "ln_pre": "vision_tower.ln_pre.weight",
            "projector.norm": "multi_modal_projector.norm.weight",
            "projector.merging_weight": "multi_modal_projector.patch_merger.merging_layer.weight",
            "projector.linear_1": "multi_modal_projector.linear_1.weight",
            "projector.linear_2": "multi_modal_projector.linear_2.weight",
        }
        for i in range(n_layers):
            p, c = f"layers.{i}", f"vision_tower.transformer.layers.{i}"
            m[f"{p}.attention_norm"] = f"{c}.attention_norm.weight"
            m[f"{p}.ffn_norm"] = f"{c}.ffn_norm.weight"
            for n in ("q", "k", "v", "o"):
                m[f"{p}.attention.{n}_weight"] = f"{c}.attention.{n}_proj.weight"
            for n in ("gate", "up", "down"):
                m[f"{p}.feed_forward.{n}_weight"] = f"{c}.feed_forward.{n}_proj.weight"
        return m

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp = get_neuron_encoder_tp_group()
        sd = SafetensorsCheckpoint(checkpoint_path).load_sharded(
            rank=tp.rank_in_group,
            world_size=tp.world_size,
            model=self,
            mappings=self.build_weight_mappings(len(self.layers)),
            device=device,
        ).state_dict
        sd = {k: v.to(self.dtype) for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"Pixtral weights mismatch. missing={sorted(missing)} unexpected={sorted(unexpected)}")
        if _use_fused_attn():
            for layer in self.layers:
                layer.attention.fuse_weights()
                layer.feed_forward.fuse_weights()

    def forward(
        self,
        pixel_patches: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
        merge_idx: torch.Tensor,
        encoder_cache_buffer: torch.Tensor,
        write_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        """pixel_patches [nb, bs, C*p*p]; cos/sin [nb, bs, hd]; bounds [nb, bs, 1] int32;
        merge_idx [nb, bs // 4, 4] int64. Writes [nb, bs // 4, text_hidden] into the cache."""
        if self.dp_size > 1:
            per = pixel_patches.shape[0] // self.dp_size
            idx = (
                torch.arange(per, device=pixel_patches.device, dtype=torch.int64)
                + self.dp_group.rank_in_group_tensor.to(torch.int64) * per
            )
            pixel_patches, cos, sin = (t.index_select(0, idx) for t in (pixel_patches, cos, sin))
            bound_min, bound_max, merge_idx = (t.index_select(0, idx) for t in (bound_min, bound_max, merge_idx))

        real_bs = pixel_patches.shape[1]
        pad = (-real_bs) % _ATTN_SEQ_ALIGN
        if pad:
            pixel_patches = F.pad(pixel_patches, (0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad))
            sin = F.pad(sin, (0, 0, 0, pad))
            bound_min = F.pad(bound_min, (0, 0, 0, pad))
            bound_max = F.pad(bound_max, (0, 0, 0, pad))

        x = pixel_patches.to(self.dtype) @ self.patch_weight
        x = _rms_norm(x, self.ln_pre, 1e-5)
        for layer in self.layers:
            x = layer(x, cos, sin, bound_min, bound_max)
        out = self.projector(x, merge_idx)
        if self.dp_size > 1:
            out = self.dp_group.all_gather(out, dim=0)
        # A block of fewer rows than the cache block (small-image bucket) fills its head.
        cache_rows = encoder_cache_buffer.shape[1]
        if out.shape[1] < cache_rows:
            out = F.pad(out, (0, 0, 0, cache_rows - out.shape[1]))
        encoder_cache_buffer.index_put_((write_block_ids,), out.to(encoder_cache_buffer.dtype))
        return encoder_cache_buffer


# ---------------------------------------------------------------------------
# CPU preprocessing
# ---------------------------------------------------------------------------


def axial_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    return torch.cat([inv[0::2], inv[1::2]])  # [hd/2]: first half rows, second half cols


def pixtral_cos_sin(rows: int, cols: int, head_dim: int, theta: float):
    """HF PixtralVisionRotaryEmbedding for one image grid -> cos/sin [rows*cols, head_dim]."""
    inv = axial_inv_freq(head_dim, theta)
    h, w = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    pos = torch.stack([h.flatten(), w.flatten()], -1).float()  # [n, 2]
    freqs = pos[:, :, None] * inv[None, None, :]  # [n, 2, hd/2]
    q = head_dim // 4
    fh = freqs[:, 0, :q]
    fw = freqs[:, 1, q : 2 * q]
    hw = torch.cat([fh, fw], -1)
    emb = torch.cat([hw, hw], -1)
    return emb.cos(), emb.sin()


def patchify(img: torch.Tensor, patch: int, rows: int, cols: int) -> torch.Tensor:
    """[C, H, W] -> [rows*cols, C*patch*patch] in patch_conv weight order (c, kh, kw)."""
    c = img.shape[0]
    img = img[:, : rows * patch, : cols * patch]
    x = img.reshape(c, rows, patch, cols, patch).permute(1, 3, 0, 2, 4)
    return x.reshape(rows * cols, c * patch * patch)


def merge_index(rows: int, cols: int, merge: int) -> torch.Tensor:
    """Block-local row ids of each merged token's merge x merge group, in unfold order (ki, kj)."""
    out = []
    for r in range(rows // merge):
        for cc in range(cols // merge):
            out.append([(r * merge + ki) * cols + cc * merge + kj for ki in range(merge) for kj in range(merge)])
    return torch.tensor(out, dtype=torch.int64).reshape(-1, merge * merge)
