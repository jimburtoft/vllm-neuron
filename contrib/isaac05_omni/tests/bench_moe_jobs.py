"""HW A/B, one MoE layer, one rank shape (TP=4: I=128), T=512 bucket, real-like routing (null drops ~50 %,
one hot expert > 128 tokens): job-list kernel vs nkilib bwmm shard_on_block (block 128 / 256), both under
torch.compile(backend="neuron") on one logical core (LNC=2). Accuracy vs a CPU fp32 reference; device timing = median
of 20 runs (synchronized) after warmup.
usage: python bench_moe_jobs.py   (env: T=512 E=256 K=8 L=462 NULLP=0.5 HOT=200)"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm-neuron-vl"))
from libtorch_neuronx_lite.compile.native_backend import register  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

import vllm_neuron.functional as NF  # noqa: E402
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation  # noqa: E402
from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode  # noqa: E402
import nki.language as nl  # noqa: E402
from vllm_neuron.model.qwen3_5_moe.kernels.moe_prefill_jobs import _PREP, moe_job_lanes, moe_job_prep, moe_prefill_jobs, num_jobs  # noqa: E402

T, E, K, H, I = int(os.environ.get("T", 512)), int(os.environ.get("E", 256)), 8, 2048, 128
L, NULLP, HOT = int(os.environ.get("L", 462)), float(os.environ.get("NULLP", 0.5)), int(os.environ.get("HOT", 200))
dev = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
x = torch.randn(T, H, generator=g).to(torch.bfloat16)
lg = torch.randn(T, E, generator=g)
lg[:HOT, 3] += 6.0  # expert 3 hot: HOT tokens (> 128 -> overflow job)
USED = int(os.environ.get("USED", E))  # restrict routing to the first USED experts (real layers use 137-219 of 256)
lg[:, USED:] = -1e9
topl, topi = lg.topk(K, -1)
w = torch.exp(topl - topl[:, :1])
keep = torch.rand(T, K, generator=g) >= NULLP
keep[:, 0] |= torch.arange(T) < HOT
keep[L:] = False
w = w * keep
s = w.sum(-1, keepdim=True)
topv = torch.where(s > 0, w / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(w))
Wgu = (torch.randn(E, H, 2 * I, generator=g) * 0.02).to(torch.bfloat16)
Wd = (torch.randn(E, I, H, generator=g) * 0.02).to(torch.bfloat16)
cnt = torch.bincount(topi[keep], minlength=E)
print(f"T={T} L={L} routes={int(keep.sum())} experts_used={int((cnt > 0).sum())} max/expert={int(cnt.max())}", flush=True)

# CPU reference (fp32 math on bf16 inputs, per-route bf16 roundings as the kernels)
xf = x.float()
ref = torch.zeros(T, H)
for e in torch.nonzero(cnt).flatten().tolist():
    tt, kk = torch.nonzero((topi == e) & keep, as_tuple=True)
    gu = (xf[tt] @ Wgu[e].float()).bfloat16().float()
    hh = (torch.nn.functional.silu(gu[:, :I]).bfloat16().float() * gu[:, I:]).bfloat16().float()
    hw = (hh * topv[tt, kk].unsqueeze(1)).bfloat16().float()
    ref.index_add_(0, tt, (hw @ Wd[e].float()).bfloat16().float())

xd, Wgud, Wdd = x.to(dev), Wgu.to(dev), Wd.to(dev)
topvd, topid, validd = topv.to(dev), topi.to(torch.int32).to(dev), (keep & (topv > 0)).to(dev)
J = num_jobs(T, E, K)


def jobs(x_, tv, ti, va):
    if _PREP == "kernel":
        jl, jexp, zrow, ntail = moe_job_lanes(tv, ti, va, E, J)
        return wrap_nki(moe_prefill_jobs)[2](x=x_, tokT=zrow, wT=zrow, zidT=zrow, jexp=jexp, zrow=zrow, Wgu=Wgud,
                                             Wd=Wdd, jl=jl, ntail=ntail)
    tokT, wT, zidT, jexp, zrow = moe_job_prep(tv, ti, va, E, J)
    return wrap_nki(moe_prefill_jobs)[2](x=x_, tokT=tokT, wT=wT, zidT=zidT, jexp=jexp, zrow=zrow, Wgu=Wgud, Wd=Wdd)


class _G:  # minimal moe_group for build_blockwise_mapping (single rank, no collectives)
    world_size = 1
    rank_in_group = 0
    rank_in_group_tensor = torch.zeros(1, dtype=torch.int32, device=dev)


def bwmm(bs):
    def f(x_, tv, ti, va):
        aff = torch.zeros(T, E, dtype=torch.float32, device=dev).scatter(1, ti.long(), tv * va.to(tv.dtype))
        pm = torch.arange(T, device=dev) < L
        aff_m, tpid, b2e, cond = NF.build_blockwise_mapping(expert_affinities=aff, num_local_experts=E,
                                                            num_experts_per_token=K, block_size=bs, moe_group=_G(),
                                                            tp_degree=1, padding_mask=pm)
        tpid = tpid.clamp(min=-1, max=T - 1)
        return NF.moe_cte(implementation=MoECTEImplementation.shard_on_block, conditions=cond, hidden_states=x_,
                          expert_affinities_masked=aff_m, gate_up_proj_weight=Wgud.reshape(E, H, 2, I),
                          down_proj_weight=Wdd, activation_function=ActFnType.SiLU, block_size=bs,
                          token_position_to_id=tpid.to(torch.int32), block_to_expert=b2e.to(torch.int32),
                          expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE, skip_token=True,
                          skip_weight=True, is_tensor_update_accumulating=True, compute_dtype=nl.bfloat16)
    return f


def bench(name, fn):
    try:
        c = torch.compile(fn, backend=register(), fullgraph=True, dynamic=False)
        t0 = time.time()
        y = c(xd, topvd, topid, validd).cpu().float()
        tc = time.time() - t0
        ts = []
        for _ in range(25):
            t1 = time.time(); c(xd, topvd, topid, validd).cpu(); ts.append(time.time() - t1)
        y = y.reshape(T, -1)[:, :H] if y.numel() == T * H else y.reshape(T, -1, H)[:, 0]
        cos = torch.nn.functional.cosine_similarity(y[:L].flatten(), ref[:L].flatten(), 0).item()
        print(f"RESULT {name}: median {np.median(ts[5:]) * 1e3:.2f} ms (min {min(ts) * 1e3:.2f}) compile {tc:.0f}s "
              f"cos {cos:.6f} max|d| {(y[:L] - ref[:L]).abs().max():.4e} pad_zero {bool((y[L:] == 0).all())}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {name}: EXC {type(e).__name__}: {str(e)[:600]}", flush=True)


for nm in os.environ.get("RUN", "jobs,bwmm128,bwmm256").split(","):
    bench(nm, jobs if nm == "jobs" else bwmm(int(nm[4:])))
