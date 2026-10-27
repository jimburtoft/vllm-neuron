# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator check of ``sparse_prefill_attn`` (``HY4_PREFILL_SPARSE_BACKEND=cte``).

SIMULATOR ONLY: a pass here is a development milestone, not device evidence.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn3 GRID=2 MODE=skip python test/sim/sim_sparse_prefill_cte.py [Q K NSEL ...]

Same cases as ``sim_sparse_prefill.py`` (prefix from a paged cache through a
position table, positions past the cached length unmapped; the 2048 chunk rows
from separate latent / rope inputs; ``KSEL`` slots per query, ascending, padded
with ``P + C``; some queries with few selected, one with chunk keys only), but
the kernel takes per-slot flat rows: ``rows_cache = pos_full[rows]``,
``rows_own = own_map[rows]`` (tables of length ``P + C + 1``, the pad out of
range in both). ``MODE=skip`` (out-of-range rows skipped by the DMA) or
``MODE=clamp`` (every row clamped in range, source picked by ``src_sel``).
Checks the sink-folded output against an fp32 dense reference.
"""
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
GRID = int(os.environ.get("GRID", "1"))
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", str(GRID))
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))
import nki  # noqa: E402
from nki_sparse_prefill import sparse_prefill_attn  # noqa: E402

H, W, IDX, L, R = 64, 704, 128, 512, 64
D = L + R
BIG = 1.0e30
SCALE = 192 ** -0.5
KSEL = int(os.environ.get("KSEL", "2176"))
MODE = os.environ.get("MODE", "skip")


def bf(t):
    return t.float().numpy().astype(ml_dtypes.bfloat16)


def case(Q, K, nsel, seed=0, cached_frac=0.75):
    g = torch.Generator().manual_seed(seed)
    C = 2048
    P = K - C
    BS = 32
    pages = P // BS + 8
    n_rows = pages * BS
    cache = (torch.randn(pages, 1, BS, W, generator=g) * 0.5).to(torch.bfloat16)
    cached = int(P * cached_frac)
    perm = torch.randperm(pages - 1, generator=g)[: P // BS] + 1
    flat_pos = (perm.view(-1, 1) * BS + torch.arange(BS).view(1, -1)).reshape(-1)
    pos_map = torch.where(torch.arange(P) < cached, flat_pos, torch.full_like(flat_pos, n_rows))
    own_lat = (torch.randn(C, L, generator=g) * 0.5).to(torch.bfloat16)
    own_rope = (torch.randn(C, R, generator=g) * 0.5).to(torch.bfloat16)
    q = torch.randn(Q, H, D, generator=g) * 0.5
    sinks = torch.randn(H, generator=g)
    vis = torch.cat((torch.arange(P) < cached, torch.ones(C, dtype=torch.bool)))
    rows = torch.full((Q, KSEL), K, dtype=torch.int32)
    sel = torch.zeros(Q, K, dtype=torch.bool)
    cand = torch.nonzero(vis).flatten()
    for i in range(Q):
        n = nsel if i % 3 else max(1, nsel // 7)
        n = min(n, KSEL, cand.numel())
        pick = cand[torch.randperm(cand.numel(), generator=g)[:n]].sort().values
        if i == 0:  # chunk keys only
            pick = torch.arange(P, P + min(n, C))
        if i == 1:  # a single key
            pick = pick[:1]
        rows[i, : pick.numel()] = pick.to(torch.int32)
        sel[i, pick] = True
    bound = torch.where(rows < K, torch.tensor(BIG), torch.tensor(-BIG)).float()

    flat = cache.float().reshape(n_rows, W)
    keys = torch.zeros(K, D)
    keys[:cached] = flat[flat_pos[:cached]][:, IDX : IDX + D]
    keys[P:, :L] = own_lat.float()
    keys[P:, L:] = own_rope.float()
    s = torch.einsum("qhd,kd->qhk", q, keys) * SCALE
    s = s.masked_fill(~sel[:, None, :], float("-inf"))
    full = torch.cat((s, sinks.view(1, H, 1).expand(Q, H, 1)), dim=-1)
    p = torch.softmax(full, dim=-1)[..., :-1]
    ref = torch.einsum("qhk,kc->qhc", p, keys[:, :L])

    # Per-slot flat rows, through tables of length P + C + 1 (the pad P + C out of range in both).
    posr = torch.arange(P + C + 1)
    pos_full = torch.where(posr < P, pos_map[posr.clamp(max=P - 1)], torch.full_like(posr, n_rows))
    own_map = torch.where((posr >= P) & (posr < P + C), posr - P, torch.full_like(posr, C))
    r_l = rows.long()
    rows_cache = pos_full[r_l].to(torch.int32)
    rows_own = own_map[r_l].to(torch.int32)
    kw = {}
    if MODE == "clamp":
        is_c = rows_cache < n_rows
        kw["src_sel"] = is_c.float().numpy()
        rows_cache = torch.where(is_c, rows_cache, torch.zeros_like(rows_cache))
        rows_own = torch.where(rows_own < C, rows_own, torch.zeros_like(rows_own))
    t0 = time.time()
    km, kd, kn = nki.simulate(sparse_prefill_attn)[GRID](
        q[..., :L].contiguous().numpy().astype(np.float32),
        q[..., L:].contiguous().numpy().astype(np.float32),
        bf(cache), rows_cache.numpy(), bf(own_lat), bf(own_rope), rows_own.numpy(), bound.numpy(),
        scale=SCALE, col_off=IDX, **kw,
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
    return {"max_rel": err, "cos": cos, "nan": nan, "t_sim": t}


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
        print(f"target={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']} grid={GRID} mode={MODE} "
              f"Q={Q} K={K} nsel={nsel} ksel={KSEL}: "
              f"{'PASS' if good else 'FAIL'} (simulator only) "
              + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()),
              flush=True)
    sys.exit(0 if ok else 1)
