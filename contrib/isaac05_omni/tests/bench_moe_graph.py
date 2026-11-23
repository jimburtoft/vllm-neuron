"""the job-list MoE inside a 40-layer-like graph (the action VLM calls it 40x per prefill with other ops in
between): NL back-to-back layers x = x + moe(x) / 4, one compiled graph, device time per layer from the host wall of
a synchronized loop (median of 10). The NEFF also gets profiled by the caller. Same routing generator as
bench_moe_jobs.py; each layer gets its own random weights (no cross-layer weight reuse in SBUF)."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/work/vllm-neuron-vl"))
from libtorch_neuronx_lite.compile.native_backend import register  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

from vllm_neuron.model.qwen3_5_moe.kernels.moe_prefill_jobs import (  # noqa: E402
    _PREP, moe_job_lanes, moe_job_prep, moe_prefill_jobs, num_jobs)

T, E, K, H, I = 512, 256, 8, 2048, 128
NL, L = int(os.environ.get("NL", 8)), 462
dev = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
Wr = [(torch.randn(H, E, generator=g) * 0.05).to(dev) for _ in range(NL)]
Wgu = [(torch.randn(E, H, 2 * I, generator=g) * 0.02).to(torch.bfloat16).to(dev) for _ in range(NL)]
Wd = [(torch.randn(E, I, H, generator=g) * 0.02).to(torch.bfloat16).to(dev) for _ in range(NL)]
x0 = torch.randn(T, H, generator=g).to(torch.bfloat16).to(dev)
J = num_jobs(T, E, K)
pm = (torch.arange(T) < L).to(dev)


def layer(x, n):
    lg = x.float() @ Wr[n]
    lv, ti = torch.topk(lg, K, -1)
    w = torch.exp(lv - lv[:, :1])
    w = w * (torch.rand_like(w) >= 0.5).to(w.dtype) if False else w * (lv >= lv[:, 3:4]).to(w.dtype)  # ~half null
    s = w.sum(-1, keepdim=True)
    tv = w / torch.where(s > 0, s, torch.ones_like(s))
    va = (tv > 0) & pm.unsqueeze(1)
    if _PREP == "kernel":
        jl, jexp, zrow, ntail = moe_job_lanes(tv, ti, va, E, J)
        y = wrap_nki(moe_prefill_jobs)[2](x=x, tokT=zrow, wT=zrow, zidT=zrow, jexp=jexp, zrow=zrow, Wgu=Wgu[n],
                                          Wd=Wd[n], jl=jl, ntail=ntail)
    else:
        tokT, wT, zidT, jexp, zrow = moe_job_prep(tv, ti, va, E, J)
        y = wrap_nki(moe_prefill_jobs)[2](x=x, tokT=tokT, wT=wT, zidT=zidT, jexp=jexp, zrow=zrow, Wgu=Wgu[n], Wd=Wd[n])
    return (x.float() + 0.25 * y.float()).to(torch.bfloat16)


def model(x):
    for n in range(NL):
        x = layer(x, n)
    return x


c = torch.compile(model, backend=register(), fullgraph=True, dynamic=False)
t0 = time.time()
c(x0).cpu()
tc = time.time() - t0
ts = []
for _ in range(12):
    t1 = time.time(); c(x0).cpu(); ts.append(time.time() - t1)
print(f"RESULT graph NL={NL} prep={_PREP}: median {np.median(ts[2:]) * 1e3:.2f} ms total, "
      f"{np.median(ts[2:]) * 1e3 / NL:.3f} ms/layer (host wall incl. dispatch), compile {tc:.0f}s", flush=True)
