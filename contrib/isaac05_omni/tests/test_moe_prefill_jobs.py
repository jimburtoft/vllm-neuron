"""simulator / HW test of kernels/moe_prefill_jobs.py vs a NumPy reference (bf16 HF order per route,
fp32 sum over picks). Usage: T=256 E=8 K=4 SKEW=1 NULLP=0.3 python test_moe_prefill_jobs.py [sim]
SKEW=1 makes expert 0 hot (> 128 tokens -> overflow jobs); NULLP drops that fraction of picks (null routes);
the last PADT tokens are padding (no routes)."""
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nki  # noqa: E402

sys.path.insert(0, os.environ.get("KDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "../../vllm-neuron/vllm_neuron/model/qwen3_5_moe/kernels")))
from moe_prefill_jobs import _PREP, moe_job_lanes, moe_job_prep, moe_prefill_jobs, num_jobs  # noqa: E402

b16 = ml_dtypes.bfloat16
bf = lambda a: np.asarray(a, np.float32).astype(b16).astype(np.float32)
T, E, K = int(os.environ.get("T", 256)), int(os.environ.get("E", 8)), int(os.environ.get("K", 4))
H, I = int(os.environ.get("H", 2048)), 128
NULLP, PADT = float(os.environ.get("NULLP", 0.3)), int(os.environ.get("PADT", 16))
r = np.random.default_rng(0)
x = bf(r.standard_normal((T, H)))
lg = r.standard_normal((T, E))
if os.environ.get("SKEW", "1") == "1":
    lg[:, 0] += 3.0
topi = np.argsort(-lg, 1)[:, :K]
topv = np.exp(np.take_along_axis(lg, topi, 1))
valid = r.random((T, K)) >= NULLP
valid[T - PADT:] = False
topv = topv * valid
s = topv.sum(1, keepdims=True)
topv = np.where(s > 0, topv / np.where(s > 0, s, 1), 0).astype(np.float32)
Wgu = bf(r.standard_normal((E, H, 2 * I)) * 0.02)
Wd = bf(r.standard_normal((E, I, H)) * 0.02)
J = num_jobs(T, E, K)
cnt = np.bincount(topi[valid], minlength=E)
print(f"T={T} E={E} K={K} J={J} routes={valid.sum()} per-expert={cnt.tolist()}", flush=True)
tokT, wT, zidT, jexp, zrow = [a.numpy() for a in moe_job_prep(torch.from_numpy(topv), torch.from_numpy(topi),
                                                                torch.from_numpy(valid), E, J)]
# --- prep self-check (numpy semantics of the job tensors)
for e in range(E):
    lanes = [(j, l) for j in range(J) for l in range(128) if jexp[0, j] == e and tokT[l, j] < T]
    assert sorted(tokT[l, j] for j, l in lanes) == sorted(np.nonzero((topi == e) & valid)[0].tolist()), e
assert ((zrow < T * K) == valid).all() and (zidT[tokT >= T] >= T * K + 128).all()
lv = tokT < T
assert np.allclose(wT[lv], topv.reshape(-1)[zidT[lv]]) and (zidT[lv] // K == tokT[lv]).all()
print("prep OK; jobs used", int((jexp < E).sum()), flush=True)


def silu(v):
    return v / (1 + np.exp(-v))


ref = np.zeros((T, H))
for t in range(T):
    for k in range(K):
        if not valid[t, k]:
            continue
        e = topi[t, k]
        gu = bf(x[t] @ Wgu[e])
        hh = bf(bf(silu(gu[:I])) * gu[I:])
        hw = bf(hh * topv[t, k])
        ref[t] += bf(hw @ Wd[e])
ref = bf(ref)
# Wgu in the pmaj contraction order the kernel uses: row h of the [H, 2I] slice = p * KT + kt is natural order,
# so no host reorder is needed (the kernel's xP layout matches it).
sim = len(sys.argv) > 1 and sys.argv[1] == "sim"
NC = int(os.environ.get("NC", 2))
k_ = nki.simulate(moe_prefill_jobs[NC]) if sim else moe_prefill_jobs[NC]
t0 = time.time()
if _PREP == "kernel":
    jl, jexp2, zrow2, ntail = [a.numpy() for a in moe_job_lanes(torch.from_numpy(topv), torch.from_numpy(topi),
                                                                  torch.from_numpy(valid), E, J)]
    assert (zrow2 == zrow).all() and sorted(jexp2[jexp2 < E].tolist()) == sorted(jexp[jexp < E].tolist())
    print("ntail", int(ntail[0, 0]), "used jobs", int((jexp2 < E).sum()), flush=True)
    y = np.asarray(k_(x.astype(b16), zrow, zrow, zrow, jexp2, zrow, Wgu.astype(b16), Wd.astype(b16), jl, ntail),
                   np.float32)
else:
    y = np.asarray(k_(x.astype(b16), tokT, wT, zidT, jexp, zrow, Wgu.astype(b16), Wd.astype(b16)), np.float32)
dt = time.time() - t0
d = y - ref
cos = float((y * ref).sum() / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-30))
print(f"RESULT sim={sim} t={dt:.1f}s cos={cos:.6f} max|d|={np.abs(d).max():.4e} ref_absmax={np.abs(ref).max():.3f} "
      f"pad_rows_zero={bool((y[T - PADT:] == 0).all())} finite={bool(np.isfinite(y).all())}", flush=True)
