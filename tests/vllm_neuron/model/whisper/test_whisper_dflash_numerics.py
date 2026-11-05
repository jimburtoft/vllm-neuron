# SPDX-License-Identifier: Apache-2.0
"""CPU numerics check: the Whisper DFlash draft vs a reference DFlash implementation.

usage: PYTHONPATH=<reference root>:<vllm-neuron root> python test_whisper_dflash_numerics.py CKPT_DIR

The reference package provides ``model.release.load_release(path, dtype)``, which returns a
module ``ref(noise_emb, target_feats, q_pos, ctx_pos, mask, enc_out)`` with the DFlash block
forward and the audio context (for example the training code of the draft).

Checks (same weights, same inputs):
 1. block_hidden (audio K/V + context K/V in the slot buffers + non-causal block) vs the
    reference ``DFlashDraft.forward`` with full context visibility, fp32 and bf16, TP=1.
 2. TP=4 emulation (4 rank shards, all-reduce = sum over ranks) vs TP=1, fp32.
 3. forward(): after a "prefill" step and after a "verify" step with ``a`` accepted drafts, the
    draft tokens equal the argmax of the reference logits for the context ``j < seq_len``.
"""

import sys

import torch
import torch.nn.functional as F

from model.release import load_release
from vllm_neuron.model.whisper.dflash_draft import WhisperDFlashDraft, load_draft_config

torch.manual_seed(0)
CK = sys.argv[1]
cfg = load_draft_config(CK)
V, H = cfg["vocab_size"], cfg["hidden_size"]
NF = len(cfg["dflash_config"]["target_layer_ids"]) * cfg["target_hidden_size"]
SLOTS, C, S = 3, 448, cfg["block_size"]


def mk(dtype, world=1, rank=0):
    m = WhisperDFlashDraft(cfg, num_slots=SLOTS, max_ctx=C, world=world, rank=rank, dtype=dtype,
                           on_device_sampling=False)
    m.load_weights(CK)
    return m


def ref_hidden(ref, emb, feats, enc, q_pos, n_ctx, dtype):
    ctx_pos = torch.arange(n_ctx).view(1, -1)
    return ref(emb.view(1, S, H).to(dtype), feats[:n_ctx].view(1, n_ctx, NF).to(dtype), q_pos.view(1, S),
               ctx_pos, None, enc.to(dtype))[0]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


enc = torch.randn(1, 1500, cfg["target_hidden_size"]) * 0.5
n_tot = 30
feats = torch.randn(n_tot, NF)
emb_table = torch.randn(V, H) * 0.02
lm = torch.randn(V, H) * 0.02
slot = 1

ref32 = load_release(CK, dtype=torch.float32).eval()
gold = None
for dtype in (torch.float32, torch.bfloat16):
    ref = ref32 if dtype == torch.float32 else load_release(CK, dtype=dtype).eval()
    m = mk(dtype)
    m.write_audio_kv(enc, torch.tensor([slot]))
    n = 12
    m.write_context_kv(feats[:n].to(dtype), torch.arange(n), slot * C + torch.arange(n))
    ids = torch.tensor([123] + [cfg["dflash_config"]["mask_token_id"]] * (S - 1))
    emb = emb_table[ids].to(dtype)
    q_pos = torch.arange(n, n + S)
    with torch.no_grad():
        mine = m.block_hidden(ids, q_pos, torch.tensor([slot]), torch.tensor([n]), h=emb)
        theirs = ref_hidden(ref, emb, feats, enc, q_pos, n, dtype)
    if gold is None:
        gold = theirs.float()
    r = rel(mine, theirs)
    agree = (F.linear(mine.float(), lm).argmax(-1) == F.linear(theirs.float(), lm).argmax(-1)).float().mean().item()
    print(f"[1] {dtype}: ours vs ref same dtype rel {r:.2e}; vs fp32 ref: ours {rel(mine, gold):.2e}, "
          f"ref {rel(theirs, gold):.2e}; argmax agree {agree:.3f}")
    if dtype == torch.float32:
        assert r < 1e-4, r
    else:
        # bf16-level: our bf16 error vs the fp32 reference is no worse than the reference's own.
        assert rel(mine, gold) < 1.5 * rel(theirs, gold) + 1e-3

# Exact TP emulation via a shared all-reduce: run the 4 ranks in lockstep threads.
import threading

barrier = threading.Barrier(4)
buf = [None] * 4
res = [None] * 4


def make_ar(r):
    def ar(x):
        buf[r] = x
        barrier.wait()
        s = sum(buf)
        barrier.wait()
        return s
    return ar


ranks = [mk(torch.float32, world=4, rank=r) for r in range(4)]
m1 = mk(torch.float32)
n = 20
for m in ranks + [m1]:
    m.write_audio_kv(enc, torch.tensor([slot]))
    m.write_context_kv(feats[:n], torch.arange(n), slot * C + torch.arange(n))
ids = torch.tensor([7] + [cfg["dflash_config"]["mask_token_id"]] * (S - 1))
emb = emb_table[ids]
q_pos = torch.arange(n, n + S)


def run(r):
    ranks[r]._all_reduce = make_ar(r)
    with torch.no_grad():
        res[r] = ranks[r].block_hidden(ids, q_pos, torch.tensor([slot]), torch.tensor([n]), h=emb)


ths = [threading.Thread(target=run, args=(r,)) for r in range(4)]
[t.start() for t in ths]
[t.join() for t in ths]
with torch.no_grad():
    h1 = m1.block_hidden(ids, q_pos, torch.tensor([slot]), torch.tensor([n]), h=emb)
r = max(rel(x, h1) for x in res)
print(f"[2] TP4 emulation vs TP1 (fp32): rel err {r:.2e}")
assert r < 1e-5, r

# [3] forward(): prefill then verify, vs reference argmax with ctx j < seq_len.
ref = load_release(CK, dtype=torch.float32).eval()
m = mk(torch.float32)


class T:
    pass


tgt = T()
tgt.decoder = T()
tgt.decoder.embed_tokens = torch.nn.Embedding.from_pretrained(emb_table)
tgt.lm_head = torch.nn.Linear(H, V, bias=False)
tgt.lm_head.weight = torch.nn.Parameter(lm)
tgt.lm_mode = "fp32"
tgt.per_rank, tgt.vocab_pad = V, 0
tgt.config = T()
tgt.config.vocab_size = V
m.bind_target(tgt)
m.write_audio_kv(enc, torch.tensor([slot]))
rank = torch.tensor(0, dtype=torch.int32)
K = S - 1
mask_id = cfg["dflash_config"]["mask_token_id"]


def ref_drafts(bonus, seq_len):
    ids = torch.tensor([bonus] + [mask_id] * K)
    q_pos = torch.arange(seq_len, seq_len + S)
    with torch.no_grad():
        h = ref_hidden(ref, emb_table[ids], feats, enc, q_pos, seq_len, torch.float32)
    return F.linear(h[1:], lm).argmax(-1)


def noop(T_, B=1):
    return dict(prev_sampled_token_ids=torch.zeros(B, K + 1, dtype=torch.int32),
                prev_num_draft_tokens=torch.full((B,), K, dtype=torch.int32),
                req_indices_per_token=torch.zeros(T_, dtype=torch.int64),
                cur_num_draft_tokens=torch.full((B,), K, dtype=torch.int32))


# prefill: 4 real prompt rows in a 16-row bucket
Tp = 16
pos = torch.cat([torch.arange(4), torch.zeros(12, dtype=torch.long)])
fe = torch.zeros(Tp, NF)
fe[:4] = feats[:4]
with torch.no_grad():
    st, dr = m(pos, fe, torch.tensor([3]), torch.tensor([[555]], dtype=torch.int32),
               torch.tensor([slot, 2], dtype=torch.int32), rank, **noop(Tp))
exp = ref_drafts(555, 4)
print(f"[3a] prefill: drafts match ref {int((dr[0] == exp).sum())}/{K}; bonus {int(st[0,0])}")
assert torch.equal(dr[0].long(), exp)

# verify: target rows at positions 4..19 (bonus 555 + 15 drafts); accept a = 6 drafts.
a = 6
pos = torch.arange(4, 4 + S)
raw = torch.full((1, S), -1, dtype=torch.int32)
raw[0, : a + 1] = torch.arange(100, 101 + a, dtype=torch.int32)
with torch.no_grad():
    st, dr = m(pos, feats[4:4 + S], torch.tensor([S - 1]), raw, torch.tensor([slot, 2], dtype=torch.int32),
               rank, **noop(S))
seq_len = 4 + 1 + a  # rows 4..4+a are verified; the new bonus sits at 5 + a
exp = ref_drafts(int(raw[0, a]), seq_len)
print(f"[3b] verify a={a}: drafts match ref {int((dr[0] == exp).sum())}/{K}; bonus {int(st[0,0])}")
assert int(st[0, 0]) == int(raw[0, a])
assert torch.equal(dr[0].long(), exp)

# async: the scheduler assumed full acceptance of the previous step (rows at 4 + 16 ..),
# prev step accepted a; the corrected positions must give the same as an exact schedule.
m2 = mk(torch.float32)
m2.bind_target(tgt)
m2.write_audio_kv(enc, torch.tensor([slot]))
with torch.no_grad():
    m2(torch.cat([torch.arange(4), torch.zeros(12, dtype=torch.long)]), fe, torch.tensor([3]),
       torch.tensor([[555]], dtype=torch.int32), torch.tensor([slot, 2], dtype=torch.int32), rank, **noop(Tp))
    m2(torch.arange(4, 4 + S), feats[4:4 + S], torch.tensor([S - 1]), raw,
       torch.tensor([slot, 2], dtype=torch.int32), rank, **noop(S))
    # next verify: true positions start at seq_len (5 + a); the optimistic ones at 4 + S.
    true0 = seq_len
    raw2 = torch.full((1, S), -1, dtype=torch.int32)
    raw2[0, :3] = torch.tensor([900, 901, 902], dtype=torch.int32)
    kw = noop(S)
    kw["prev_sampled_token_ids"] = raw
    st2, dr2 = m2(torch.arange(4 + S, 4 + 2 * S), feats[true0:true0 + S], torch.tensor([S - 1]), raw2,
                  torch.tensor([slot, 2], dtype=torch.int32), rank, **kw)
exp2 = ref_drafts(902, true0 + 3)
print(f"[3c] async corrected verify: drafts match ref {int((dr2[0] == exp2).sum())}/{K}")
assert torch.equal(dr2[0].long(), exp2)
print("ALL OK")
