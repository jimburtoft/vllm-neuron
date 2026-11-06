# SPDX-License-Identifier: Apache-2.0
"""Device test: fused GQA decode attention at a speculative verify shape.

With more than one KV head per rank, ``attention_block_tkg`` folds the KV heads into the
batch and resizes the block-KV mask layout for the folded shape. A mask built by
``gen_attention_decode_mask(bs=B, q_head=q_heads)`` must be moved to that layout
(``_fold_gqa_block_kv_mask``). Without the move, Qwen3 at TP=1 (8 KV heads per rank)
on a 16-token verify step gives an attention output with cosine ~0.5 against the torch
reference, and greedy DFlash / Eagle3 output degenerates. A one-token decode step is not
affected at these shapes, so plain serving looks correct.

Runs on a Neuron device (skipped elsewhere):

    pytest test/vllm_neuron/functional/attention/test_attention_decode_gqa_verify.py -q
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "localhost:61239")

import vllm_neuron.functional.attention.attention_decode as ad  # noqa: E402
from vllm_neuron import envs  # noqa: E402
from vllm_neuron.envs import get_compile_backend_name  # noqa: E402
from vllm_neuron.functional.attention.attention_decode_mask import (  # noqa: E402
    gen_attention_decode_mask,
)
from vllm_neuron.model.qwen3_vl.utils.decode_kv import (  # noqa: E402
    build_per_head_block_table,
)


def _has_device():
    if envs.VLLM_NEURON_CPU_MODE:
        return False
    import glob

    return bool(glob.glob("/dev/neuron*"))


pytestmark = pytest.mark.skipif(not _has_device(), reason="needs a Neuron device")

_MASK_KW = (
    {"use_kernel": False}
    if "use_kernel" in inspect.signature(gen_attention_decode_mask).parameters
    else {}
)


@pytest.mark.parametrize(
    "H,q_heads,kv_heads,B,S,num_blocks",
    [
        (4096, 32, 8, 1, 16, 32),  # Qwen3-8B TP1 verify (K=15), S_ctx 1024
        (2048, 16, 8, 1, 16, 32),  # Qwen3-ASR-1.7B TP1 verify
        (2048, 16, 8, 1, 16, 16),  # S_ctx 512
        (2048, 4, 2, 1, 16, 32),  # 2 KV heads per rank
        (4096, 32, 8, 1, 1, 32),  # one-token decode
        (2048, 16, 8, 3, 1, 32),  # odd batch decode
    ],
)
def test_gqa_verify_matches_reference(H, q_heads, kv_heads, B, S, num_blocks):
    torch.manual_seed(0)
    d, bl = 128, 32
    I = d * (q_heads + 2 * kv_heads)
    s_ctx = num_blocks * bl
    X = torch.randn(B, S, H, dtype=torch.bfloat16)
    W = (torch.randn(H, I) * H**-0.5).to(torch.bfloat16)
    K = torch.randn(B * num_blocks + 1, kv_heads, bl, d, dtype=torch.bfloat16)
    V = torch.randn(B * num_blocks + 1, kv_heads, bl, d, dtype=torch.bfloat16)
    table = (torch.arange(B * num_blocks, dtype=torch.int32) + 1).view(B, num_blocks)
    pos = (torch.arange(S) + s_ctx // 2).repeat(B).view(1, B * S).float()
    mask = gen_attention_decode_mask(
        pos_ids=pos, bs=B, q_head=q_heads, s_active=S, s_prior=s_ctx, block_len=bl, **_MASK_KW
    )
    ref, _, _ = ad._torch_attention_decode_impl(
        X=X, W_qkv=W, K_cache=K, V_cache=V, active_blocks_table=table, attention_mask=mask
    )
    dev = torch.device("neuron:0")
    fn = torch.compile(ad.attention_decode, backend=get_compile_backend_name(), fullgraph=True)
    out = fn(
        X=X.to(dev),
        W_qkv=W.to(dev),
        K_cache=K.to(dev),
        V_cache=V.to(dev),
        active_blocks_table=build_per_head_block_table(table, kv_heads).to(dev),
        attention_mask=mask.to(dev),
        update_cache=False,
    )[0].cpu()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    assert cos > 0.9995, f"cos={cos:.5f}"
