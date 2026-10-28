# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator check of ``topk_rows`` (cte backend's direct top-k).

SIMULATOR ONLY: a pass here is a development milestone, not device evidence.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn3 GRID=2 python test/sim/sim_topk_rows.py [Q P ...]

Indexer-like scores (relu-weighted sum over 32 heads of bf16 q . k, fp32), the
prefix partly invisible (-BIG), causal over the chunk; one query with fewer
than ``k`` visible keys. Compares the selected SET per query to fp32
``torch.topk`` over the concatenation (exact match expected: no ties in these
fp32 scores) and checks the bound (+BIG iff visible).
"""
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
GRID = int(os.environ.get("GRID", "1"))
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", str(GRID))
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))
import nki  # noqa: E402
from nki_sparse_prefill import topk_rows  # noqa: E402

BIG = 1.0e30
K = int(os.environ.get("K", "2048"))
WIDTH = int(os.environ.get("WIDTH", "0"))
C = 2048
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]


def scores(g, Q, N, H=32, D=128):
    q = torch.randn(Q, H, D, generator=g).to(torch.bfloat16).float()
    kk = torch.randn(N, D, generator=g).to(torch.bfloat16).float()
    w = torch.randn(Q, H, generator=g) * (H * D) ** -0.5
    out = torch.zeros(Q, N)
    for i in range(0, N, 4096):
        s = torch.relu(torch.einsum("qhd,nd->qhn", q, kk[i : i + 4096]))
        out[:, i : i + 4096] = (s * w.unsqueeze(-1)).sum(1)
    return out


def case(Q, P, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = P + C
    s = scores(g, Q, N)
    prior = int(P * 0.75)
    first = Q * 0 + 100  # chunk offset of the local queries (causal)
    vis = torch.zeros(Q, N, dtype=torch.bool)
    vis[:, :prior] = True
    vis[:, P:] = torch.arange(C).view(1, C) <= (first + torch.arange(Q)).view(Q, 1)
    vis[0] = False  # few visible keys
    vis[0, P : P + 50] = True
    vis[0, :30] = True
    sm = torch.where(vis, s, torch.full_like(s, -BIG))
    t0 = time.time()
    rows, bound = nki.simulate(topk_rows)[GRID](
        sm[:, :P].contiguous().numpy(), sm[:, P:].contiguous().numpy(), k=K, width=WIDTH
    )
    t = time.time() - t0
    rows = torch.as_tensor(np.asarray(rows)).long()
    bound = torch.as_tensor(np.asarray(bound))
    if WIDTH > K:
        assert (rows[:, K:] == N).all() and (bound[:, K:] < 0).all(), "pad slots"
        rows, bound = rows[:, :K], bound[:, :K]
    ref = sm.topk(K, dim=1).indices
    bad_set, bad_bound, n_diff = 0, 0, 0
    for i in range(Q):
        got = set(rows[i][bound[i] > 0].tolist())
        want = set(ref[i][vis[i, ref[i]]].tolist())
        d = len(got ^ want)
        n_diff += d
        bad_set += d > 0
        bad_bound += int(not torch.equal(bound[i] > 0, vis[i, rows[i]]))
        assert len(set(rows[i].tolist())) == K, "duplicate positions"
    return {"queries_set_mismatch": bad_set, "keys_differing": n_diff, "bound_mismatch": bad_bound,
            "q0_visible_selected": int((bound[0] > 0).sum()), "t_sim": t}


if __name__ == "__main__":
    shapes = [(32, 6144), (32, 34816)]
    if len(sys.argv) > 1:
        a = [int(x) for x in sys.argv[1:]]
        shapes = list(zip(a[0::2], a[1::2]))
    ok = True
    for (Q, P), seed in [(sh, sd) for sh in shapes for sd in SEEDS]:
        r = case(Q, P, seed)
        good = r["queries_set_mismatch"] == 0 and r["bound_mismatch"] == 0 and r["q0_visible_selected"] == 80
        ok &= good
        print(f"target={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']} grid={GRID} Q={Q} P={P} C={C} k={K}: "
              f"seed={seed} {'PASS' if good else 'FAIL'} (simulator only) "
              + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
    sys.exit(0 if ok else 1)
