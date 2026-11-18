# SPDX-License-Identifier: Apache-2.0
"""Several llama3 decoder layers per NKI call (task 035, opt-in ``VLLM_NEURON_DECODE_LAYERS_MK``).

One call runs ``len(layers)`` decoder layers of an S=1 BF16 decode step on one TP rank:

    for each layer:
        a   = attention_block_tkg(rmsnorm(h), ...)   # fused RMSNorm + QKV + RoPE + paged attention + KV write + O-proj
        h   = h + all_reduce(a)                      # in-kernel TP all-reduce
        m   = mlp(rmsnorm(h), ...)                   # fused RMSNorm + SwiGLU (TKG)
        h   = h + all_reduce(m)

It composes the same nkilib kernels the per-layer plugin path calls (``NF.attention_decode`` and ``NF.mlp``), with
the same arguments, so the numerics match. What it removes is the framework boundary between them: the two
collectives and the two residual adds per layer stop being separate graph ops.

``ranks=None`` makes the all-reduce an identity (single rank / NKI simulator).

Per-layer tensors are separate kernel arguments (NKI kernels cannot take Python lists): the plugin's own
parameters and vLLM's per-layer KV caches are passed as they are (zero-copy). The KV caches are written in place by
attention_block_tkg and returned so wrap_nki aliases them back to the inputs.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nkilib.core.mlp.mlp import mlp as _mlp
from nkilib.core.utils.kernel_helpers import get_verified_program_sharding_info
from nkilib.core.utils.common_types import ActFnType, ComputationMode, DtypeMode, NormType, QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg as _attn


def _all_reduce_add(partial, residual, ranks, B, H, tag):
    """residual + sum over TP ranks of ``partial`` ([B, 1, H] or [B, H] @ HBM) -> new [B, 1, H] @ shared HBM."""
    _, NC, pid = get_verified_program_sharding_info("layers_tkg", (0, 1), 2)
    src = partial.reshape((B, H))
    if ranks is not None:
        if NC > 1:
            nisa.core_barrier(data=partial, cores=(0, 1))
        red = nl.ndarray((B, H), dtype=partial.dtype, buffer=nl.shared_hbm, name=f"ar_{tag}")
        ncc.all_reduce(dsts=[red], srcs=[src], op=nl.add, replica_group=ncc.ReplicaGroup([list(ranks)]))
        src = red
    out = nl.ndarray((B, 1, H), dtype=partial.dtype, buffer=nl.shared_hbm, name=f"h_{tag}")
    # Each logical core adds its half of H (columns), then both halves are visible in shared HBM.
    hw = H // NC
    a_sb = nl.ndarray((B, hw), dtype=partial.dtype, buffer=nl.sbuf)
    r_sb = nl.ndarray((B, hw), dtype=partial.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_sb, src=src[:, nl.ds(pid * hw, hw)])
    nisa.dma_copy(dst=r_sb, src=residual.reshape((B, H))[:, nl.ds(pid * hw, hw)])
    nisa.tensor_tensor(dst=a_sb, data1=a_sb, data2=r_sb, op=nl.add)
    nisa.dma_copy(dst=out.reshape((B, H))[:, nl.ds(pid * hw, hw)], src=a_sb)
    if NC > 1:
        nisa.core_barrier(data=out, cores=(0, 1))
    return out


def layers_body(X, cos, sin, attention_mask, active_blocks_table, kv_cache_update_idx, softmax_scale, eps,
                ln1, wqkv, wo, ln2, wg, wu, wd, kc, vc, ranks=None):
    """X [B, 1, H] @ HBM. Per-layer tuples: ln1/ln2 [1, H], wqkv [H, Dqkv], wo [Nq*D, H], wg/wu [H, I], wd [I, H],
    kc/vc paged caches [blocks, kv_heads, block, D]. Returns (h_out, *kc, *vc)."""
    B, _, H = X.shape
    h = X
    for l in range(len(wqkv)):
        a = _attn(
            X=h, X_hidden_dim_actual=H,
            rmsnorm_X_enabled=True, rmsnorm_X_eps=eps, rmsnorm_X_gamma=ln1[l],
            W_qkv=wqkv[l], bias_qkv=None, quantization_type_qkv=QuantizationType.NONE,
            weight_dequant_scale_qkv=None, input_dequant_scale_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False, rmsnorm_QK_pre_rope_eps=1e-6,
            rmsnorm_QK_pre_rope_W_Q=None, rmsnorm_QK_pre_rope_W_K=None,
            cos=cos, sin=sin, rope_contiguous_layout=True,
            rmsnorm_QK_post_rope_enabled=False, rmsnorm_QK_post_rope_eps=1e-6,
            rmsnorm_QK_post_rope_W_Q=None, rmsnorm_QK_post_rope_W_K=None,
            K_cache_transposed=False, active_blocks_table=active_blocks_table,
            K_cache=kc[l], V_cache=vc[l], attention_mask=attention_mask, sink=None,
            update_cache=True, kv_cache_update_idx=kv_cache_update_idx,
            W_out=wo[l], bias_out=None, quantization_type_out=QuantizationType.NONE,
            weight_dequant_scale_out=None, input_dequant_scale_out=None,
            transposed_out=False, out_in_sb=False, softmax_scale=softmax_scale,
        )[0]
        h = _all_reduce_add(a, h, ranks, B, H, f"a{l}")
        m = _mlp(
            hidden_tensor=h, gate_proj_weights_tensor=wg[l], up_proj_weights_tensor=wu[l],
            down_proj_weights_tensor=wd[l], normalization_weights_tensor=ln2[l],
            activation_fn=ActFnType.SiLU, normalization_type=NormType.RMS_NORM,
            quantization_type=QuantizationType.NONE, eps=eps, store_output_in_sbuf=False,
            use_tkg_gate_up_proj_column_tiling=True, use_tkg_down_proj_column_tiling=True,
            use_tkg_down_proj_optimized_layout=False, mode=ComputationMode.DECODE, force_cte_mode=False,
            dtype_mode=DtypeMode.AUTO,
        )
        m = m[0] if isinstance(m, (tuple, list)) else m
        h = _all_reduce_add(m, h, ranks, B, H, f"m{l}")
    return (h,) + tuple(kc) + tuple(vc)


_GROUPS = ("ln1", "wqkv", "wo", "ln2", "wg", "wu", "wd", "kc", "vc")
_SHARED = "X, cos, sin, attention_mask, active_blocks_table, kv_cache_update_idx"
SUPPORTED_N = (1, 2, 4, 17, 34)


def layers_tkg_entry(n: int):
    """The ``@nki.jit`` entry for ``n`` layers per call (pre-generated in layers_tkg_entries.py)."""
    from vllm_neuron.model.llama3 import layers_tkg_entries as _e

    if n not in SUPPORTED_N:
        raise ValueError(f"VLLM_NEURON_DECODE_LAYERS_MK={n}: supported layer counts are {SUPPORTED_N}")
    return getattr(_e, f"layers_tkg_{n}")


def _generate(path):
    """Write layers_tkg_entries.py (one explicit argument per per-layer tensor; Dynamo cannot trace exec)."""
    out = ['"""GENERATED by layers_tkg._generate -- do not edit."""', "import nki", "",
           "from vllm_neuron.model.llama3.layers_tkg import layers_body", ""]
    for n in SUPPORTED_N:
        args = ", ".join(f"{g}_{i}" for g in _GROUPS for i in range(n))
        tup = ", ".join(f"{g}=(" + ", ".join(f"{g}_{i}" for i in range(n)) + ",)" for g in _GROUPS)
        out += ["", "@nki.jit", f"def layers_tkg_{n}({_SHARED}, {args}, softmax_scale=1.0, eps=1e-5, ranks=None):",
                f"    return layers_body({_SHARED}, softmax_scale, eps, ranks=ranks, {tup})", ""]
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    import os

    _generate(os.path.join(os.path.dirname(os.path.abspath(__file__)), "layers_tkg_entries.py"))
