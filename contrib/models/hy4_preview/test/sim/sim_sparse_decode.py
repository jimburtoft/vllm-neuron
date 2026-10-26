# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator check of ``nki_sparse_decode`` against fp32 torch references.

SIMULATOR ONLY: a pass here is a development milestone, not device evidence.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 GRID=2 python test/sim/sim_sparse_decode.py [B NK ...]

``GRID`` (default 1) is the launch grid; the model launches at the logical NC
config (2 on trn2/trn3 under vllm-neuron).

Hy4 shapes: cache [pages, 1, 32, 704] bf16 with row = [index 128 | latent 512 |
rope 64], H = 64 heads, attended slice = columns 128:704, value = 128:640.
Rows: a random selection of real rows, the rest one past the last row (skipped
by the DMA, bound -BIG); one request may have nothing selected.
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
from nki_sparse_decode import gather_rows_cols, sparse_decode_attn  # noqa: E402

H, BS, W, IDX, L, R = 64, 32, 704, 128, 512, 64
D = L + R
BIG = 1.0e30
SCALE = 192 ** -0.5


def bf(t):
    return t.float().numpy().astype(ml_dtypes.bfloat16)


def case(B, NK, q_dtype, empty_req, seed=0):
    g = torch.Generator().manual_seed(seed)
    pages = max(8, (B * NK) // BS + 4)
    n_rows = pages * BS
    cache = (torch.randn(pages, 1, BS, W, generator=g) * 0.5).to(torch.bfloat16)
    q = torch.randn(B, H, D, generator=g) * 0.5
    if q_dtype == "bf16":
        q = q.to(torch.bfloat16).float()
    rows = torch.stack([torch.randperm(n_rows, generator=g)[:NK] for _ in range(B)]).to(torch.int32)
    keep = torch.rand(B, NK, generator=g) < 0.6
    keep[:, 0] = True
    if empty_req and B > 1:
        keep[1] = False
    rows = torch.where(keep, rows, torch.full_like(rows, n_rows))
    bound = torch.where(keep, torch.tensor(BIG), torch.tensor(-BIG)).float()

    flat = cache.float().reshape(n_rows, W)
    safe = torch.where(keep, rows, torch.zeros_like(rows)).long()
    keys = flat[safe] * keep.unsqueeze(-1)  # [B, NK, W]; skipped rows are zeros
    sl = keys[..., IDX : IDX + D]
    s = torch.einsum("bhd,bkd->bhk", q, sl) * SCALE
    s = torch.minimum(s, bound[:, None, :])
    m = s.amax(-1)
    p = torch.exp(s - m[..., None])
    den = p.sum(-1)
    num = torch.einsum("bhk,bkc->bhc", p.to(torch.bfloat16).float(), sl[..., :L])

    qin = q.numpy().astype(np.float32) if q_dtype == "fp32" else bf(q)
    t0 = time.time()
    km, kd, kn = nki.simulate(sparse_decode_attn)[GRID](
        qin, bf(cache), rows.numpy(), bound.numpy(), scale=SCALE, col_off=IDX,
        lat_off=0, lat_dim=L, rope_off=L, rope_dim=R,
    )
    t_attn = time.time() - t0
    km, kd, kn = [torch.as_tensor(np.asarray(x, dtype=np.float32)) for x in (km, kd, kn)]

    t0 = time.time()
    gk = nki.simulate(gather_rows_cols)[GRID](bf(cache), rows.numpy(), col_off=0, width=IDX)
    t_gather = time.time() - t0
    gk = torch.as_tensor(np.asarray(gk, dtype=np.float32))

    def rel(a, b):
        return ((a - b).abs().max() / b.abs().max().clamp_min(1e-9)).item()

    live = keep.any(-1)
    res = {
        "m_abs": (km - m).abs().max().item(),
        "den_rel": rel(kd[live], den[live]),
        "num_rel": rel(kn[live], num[live]),
        "num_cos": torch.nn.functional.cosine_similarity(kn[live].flatten(), num[live].flatten(), dim=0).item(),
        "gather_exact": bool(torch.equal(gk, keys[..., :IDX])),
        "nan": bool(torch.isnan(km).any() or torch.isnan(kd).any() or torch.isnan(kn).any()),
        "t_attn": t_attn, "t_gather": t_gather,
    }
    if empty_req and B > 1:
        res["empty_m"] = km[1].max().item()  # -BIG: the merge scales it by exp(-BIG - m) = 0
    return res


if __name__ == "__main__":
    shapes = [(1, 128), (4, 128), (1, 512), (4, 512)]
    if len(sys.argv) > 1:
        a = [int(x) for x in sys.argv[1:]]
        shapes = list(zip(a[0::2], a[1::2]))
    ok = True
    for B, NK in shapes:
        for qd in ("fp32", "bf16"):
            r = case(B, NK, qd, empty_req=True)
            good = (r["m_abs"] < 1e-3 and r["den_rel"] < 1e-4 and r["num_rel"] < 1e-2
                    and r["num_cos"] > 0.9999 and r["gather_exact"] and not r["nan"])
            ok &= good
            print(f"grid={GRID} B={B} NK={NK} q={qd}: {'PASS' if good else 'FAIL'} (simulator only) "
                  + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
                  flush=True)
    sys.exit(0 if ok else 1)
