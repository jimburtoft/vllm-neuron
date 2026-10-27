# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator check of ``sparse_decode_attn`` as the sparse prefill uses it.

SIMULATOR ONLY: a pass here is a development milestone, not device evidence.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 GRID=2 python test/sim/sim_sparse_prefill.py [Q K NSEL ...]

Prefill shapes (``HY4_PREFILL_SPARSE_NKI``): ``Q`` local queries (C / world)
are the kernel's "requests"; the cache is the chunk's flat ``keys [K, 704]``
(row = [index 128 | latent 512 | rope 64]); each query has ``KSEL`` slots
(``HY4_PREFILL_KSEL``, 2176) holding its selected rows ascending, padded with
``K`` (skipped by the DMA, bound -BIG). The query is fp32, ``tg`` as the model
picks it. Also checks the sink fold against an fp32 dense reference.
"""
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
GRID = int(os.environ.get("GRID", "1"))
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", str(GRID))
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))
import nki  # noqa: E402
from nki_sparse_decode import sparse_decode_attn  # noqa: E402

H, W, IDX, L, R = 64, 704, 128, 512, 64
D = L + R
BIG = 1.0e30
SCALE = 192 ** -0.5
KSEL = int(os.environ.get("KSEL", "2176"))


def bf(t):
    return t.float().numpy().astype(ml_dtypes.bfloat16)


def model_tg(ksel):
    kt = -(-ksel // 128)
    return max(1, min(8, 49152 // (kt * D * 2)))


def case(Q, K, nsel, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = (torch.randn(K, W, generator=g) * 0.5).to(torch.bfloat16)
    q = torch.randn(Q, H, D, generator=g) * 0.5
    sinks = torch.randn(H, generator=g)
    rows = torch.full((Q, KSEL), K, dtype=torch.int32)
    sel = torch.zeros(Q, K, dtype=torch.bool)
    for i in range(Q):
        n = nsel if i % 3 else max(1, nsel // 7)  # some queries see few keys
        n = min(n, KSEL, K)
        pick = torch.randperm(K, generator=g)[:n].sort().values
        rows[i, :n] = pick.to(torch.int32)
        sel[i, pick] = True
    bound = torch.where(rows < K, torch.tensor(BIG), torch.tensor(-BIG)).float()

    kf = keys.float()
    s = torch.einsum("qhd,kd->qhk", q, kf[:, IDX : IDX + D]) * SCALE
    s = s.masked_fill(~sel[:, None, :], float("-inf"))
    full = torch.cat((s, sinks.view(1, H, 1).expand(Q, H, 1)), dim=-1)
    p = torch.softmax(full, dim=-1)[..., :-1]
    ref = torch.einsum("qhk,kc->qhc", p, kf[:, IDX : IDX + L])

    tg = model_tg(KSEL)
    t0 = time.time()
    km, kd, kn = nki.simulate(sparse_decode_attn)[GRID](
        q.numpy().astype(np.float32), bf(keys), rows.numpy(), bound.numpy(), scale=SCALE,
        col_off=IDX, lat_off=0, lat_dim=L, rope_off=L, rope_dim=R, tg=tg,
    )
    t = time.time() - t0
    m, den, num = [torch.as_tensor(np.asarray(x, dtype=np.float32)) for x in (km, kd, kn)]
    sink = sinks.view(1, H)
    m_f = torch.maximum(m, sink)
    sc = torch.exp(m - m_f)
    o = num * sc[..., None] / (den * sc + torch.exp(sink - m_f))[..., None]
    err = ((o - ref).abs().max() / ref.abs().max()).item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    nan = bool(torch.isnan(o).any())
    return {"tg": tg, "max_rel": err, "cos": cos, "nan": nan, "t_sim": t}


if __name__ == "__main__":
    shapes = [(32, 8192, 2048), (32, 34816, 2048)]
    if len(sys.argv) > 1:
        a = [int(x) for x in sys.argv[1:]]
        shapes = list(zip(a[0::3], a[1::3], a[2::3]))
    ok = True
    for Q, K, nsel in shapes:
        r = case(Q, K, nsel)
        good = r["max_rel"] < 2e-2 and r["cos"] > 0.9999 and not r["nan"]
        ok &= good
        print(f"target={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']} grid={GRID} Q={Q} K={K} "
              f"nsel={nsel} ksel={KSEL}: {'PASS' if good else 'FAIL'} (simulator only) "
              + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
    sys.exit(0 if ok else 1)
