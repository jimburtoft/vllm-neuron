# SPDX-License-Identifier: Apache-2.0
"""NKI-simulator check of ``select_to_rows`` against the model's torch ``_prefill_rows_ref``.

SIMULATOR ONLY: a pass here is a development milestone, not device evidence.

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 GRID=2 python test/sim/sim_select_rows.py [Q K ...]

Selections come from the model's own ``select_topk`` (top-2048 by bisection)
over prefill-shaped visibility (prefix + causal), on three score patterns:
random, tie-heavy (scores quantised to a few levels, so many keys tie with
the threshold), and constant (every visible key ties). Some rows see fewer
than 2048 keys. Checks: per query the same SET of rows (and the same order,
ascending) and the same bound sign as the torch reference, ``ksel = 2176``.
"""
import os
import sys
import time

os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
GRID = int(os.environ.get("GRID", "1"))
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", str(GRID))
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "unit")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src")))
import nki  # noqa: E402

try:
    import transformers  # noqa: F401
except ImportError:  # the simulator venv: config.py only subclasses PretrainedConfig
    import types as _types

    sys.modules["transformers"] = _types.SimpleNamespace(PretrainedConfig=object)
from hy4_stubs import Group, model, set_group  # noqa: E402
from nki_sparse_decode import select_to_rows  # noqa: E402

KSEL, TOPK, CHUNK = 2176, 2048, 2048


def selection(Q, K, pattern, seed):
    g = torch.Generator().manual_seed(seed)
    ctx = K - CHUNK
    q_pos = torch.arange(Q) * (CHUNK // Q)  # local queries spread over the chunk
    # Prefix visible up to a per-case cached length; rows 0/1 see very few keys.
    cached = ctx if seed % 2 == 0 else ctx // 3
    k = torch.arange(K)
    vis = (k.view(1, K) < cached) | (
        (k.view(1, K) >= ctx) & (k.view(1, K) - ctx <= q_pos.view(Q, 1))
    )
    vis[0] = False
    vis[0, ctx] = True  # one visible key
    vis[1] = False
    vis[1, ctx : ctx + 700] = True  # fewer than top-k
    s = torch.randn(Q, K, generator=g)
    if pattern == "ties":
        s = torch.round(s * 2) / 2  # ~13 levels: heavy ties at the threshold
    elif pattern == "const":
        s = torch.ones(Q, K)
    return model.select_topk(s, vis, TOPK)


def case(Q, K, pattern, seed, attn):
    sel = selection(Q, K, pattern, seed)
    ref_rows, ref_bound = attn._prefill_rows_ref(sel)
    t0 = time.time()
    rows = np.asarray(nki.simulate(select_to_rows)[GRID](sel.float().numpy(), ksel=KSEL))
    t = time.time() - t0
    rows = torch.as_tensor(rows.astype(np.int64))[:, :KSEL]
    bound_pos = rows < K
    n_sel = sel.sum(-1)
    same_set = all(
        set(rows[i][bound_pos[i]].tolist()) == set(ref_rows[i][ref_bound[i] > 0].tolist())
        for i in range(Q)
    )
    return {
        "same_set": same_set,
        "same_order": bool(torch.equal(rows, ref_rows.long())),
        "same_bound": bool(torch.equal(bound_pos, ref_bound > 0)),
        "max_sel": int(n_sel.max()), "min_sel": int(n_sel.min()),
        "t_sim": t,
    }


if __name__ == "__main__":
    shapes = [(32, 8192), (32, 34816)]
    if len(sys.argv) > 1:
        a = [int(x) for x in sys.argv[1:]]
        shapes = list(zip(a[0::2], a[1::2]))
    set_group(Group())
    import types
    cfg = types.SimpleNamespace(
        torch_dtype=torch.float32, hidden_size=64, num_attention_heads=2, q_lora_rank=32,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        index_topk=TOPK, index_n_heads=4, index_head_dim=16, indexer_types=("full",),
        indexer_rope_first=False, rms_norm_eps=1e-6, gated_mla=True, learnable_sink=True,
        rope_theta=10000.0, neuron_config=None, qk_head_dim=32, kv_latent_dim=48,
    )
    model.PREFILL_KSEL = KSEL
    attn = model.HYV4Attention(cfg, 0)
    ok = True
    for Q, K in shapes:
        for si, pattern in enumerate(("random", "ties", "const")):
            for seed in (2 * si, 2 * si + 1):
                r = case(Q, K, pattern, seed, attn)
                good = r["same_set"] and r["same_order"] and r["same_bound"]
                ok &= good
                print(f"target={os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']} grid={GRID} Q={Q} "
                      f"K={K} {pattern} seed={seed}: {'PASS' if good else 'FAIL'} (simulator only) "
                      + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in r.items()), flush=True)
    sys.exit(0 if ok else 1)
