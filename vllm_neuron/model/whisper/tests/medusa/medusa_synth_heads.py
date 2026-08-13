# SPDX-License-Identifier: Apache-2.0
"""
whisper-xla Task 020 Milestone 3 -- SYNTHETIC partially-correct Medusa head builder.

Purpose
-------
Construct a Medusa head checkpoint whose heads are *partially correct* -- good
enough that some drafts match the target's greedy continuation, so the Medusa
accept path (>1 token accepted / verify) is EXERCISED. This is the strongest
correctness evidence for M3: it proves the framework stays byte-identical to
greedy even when tokens are ACTUALLY accepted (not just all-rejected).

This is NOT training a real speculative model. It is a crude, fast fit designed
only to raise acceptance above zero so the multi-token-accepted branch of the
rejection sampler is driven. Acceptance rate here is meaningless for latency --
that is the customer's job with real trained heads.

How the fit works
-----------------
The Medusa-1 heads are tied to the model's (frozen) lm_head:
    head_j(h) = argmax( lm_head( ResBlock_j(h) ) )
where ResBlock_j(h) = h + SiLU(Linear_j(h)). Head j is supposed to predict the
token at anchor+1+j.

We harvest, from a plain-greedy decode of the calibration clips, pairs
(h_t, y_{t+1+j}) where h_t is the decoder's LAST hidden state at position t and
y_{t+1+j} is the greedy token j+1 steps later. Then we fit each head's Linear_j
by a few Adam steps minimizing cross-entropy between lm_head(ResBlock_j(h_t)) and
y_{t+1+j}, with the lm_head FROZEN. The projection is done in fp32 to match
compute_logits.

For head 0 the target y_{t+1} is exactly argmax(lm_head(h_t)) -- so a zero
ResBlock already gets it right; the fit just sharpens it. For heads 1..K-1 the
target is further ahead and the fit yields partial accuracy -> a realistic
descending acceptance profile that exercises 2..K-token accepts.

Output
------
A checkpoint at ``--out`` (default /large/work/ref/synth_medusa_heads.pt) in the
loader's internal layout ``medusa_heads.{i}.{j}.linear.{weight,bias}``, loadable
via ``load_medusa_heads(init='load', heads_path=...)`` /
``medusa_config={'init':'load','heads_path':...}``.

Launch (TP=1 is enough to build the checkpoint; heads are TP-agnostic params):
    python medusa_synth_heads.py --clips jfk,ls1,ls2 --steps 300
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/large/work")
sys.path.insert(0, "/large/work/whisper_pkg_parent")

MODEL_DIR = os.environ.get("WX_MODEL", "/large/work/whisper-large-v3")
REF = "/large/work/ref"
DTYPE = torch.bfloat16
SOT = [50258, 50259, 50360, 50364]
EOS = 50257
BLOCK_SIZE = 32
MAX_BLOCKS = 16


def build_attn_metadata(n_layers, positions_list, block_size, max_blocks, device,
                        is_prefill, K):
    n = len(positions_list)
    pos_t = torch.tensor(positions_list, dtype=torch.long, device=device)
    slot_mapping = pos_t.clone().to(torch.long)
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device).view(1, max_blocks)
    meta = {}
    for i in range(n_layers):
        meta[f"decoder.layers.{i}.self_attn"] = {
            "slot_mapping": slot_mapping,
            "block_size": block_size,
            "block_table_tensor": block_table,
            "max_query_len": (n if is_prefill else 1),
            "decode_token_threshold": (0 if is_prefill else K + 1),
        }
    return meta


def alloc_self_kv(model, n_layers, max_blocks, block_size, device):
    spec = model.get_kv_spec()
    kv = {}
    for ls in spec.layers:
        k = torch.zeros(max_blocks, ls.num_kv_heads, block_size, ls.head_size,
                        dtype=ls.dtype, device=device)
        v = torch.zeros(max_blocks, ls.num_kv_heads, block_size, ls.head_size,
                        dtype=ls.dtype, device=device)
        kv[ls.name] = [k, v]
    model.bind_kv_cache(kv)
    return kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="jfk,ls1,ls2,ls3")
    ap.add_argument("--K", type=int, default=int(os.environ.get("WX_K", "5")))
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--out", default=os.path.join(REF, "synth_medusa_heads.pt"))
    args = ap.parse_args()
    K = args.K

    import vllm_neuron  # noqa: F401
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("WX_PORT", "29572"))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from transformers import AutoConfig
    from whisper_pkg.config import WhisperConfig
    from whisper_pkg.model_bf16 import WhisperForConditionalGeneration
    from vllm_neuron.model.neuron_config import NeuronConfig

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    nc = NeuronConfig(on_device_sampling_config=None)
    wc = WhisperConfig.from_configs(hf_cfg, nc)
    # No medusa heads needed for harvesting the greedy hidden states.
    device = "neuron:0"
    torch.manual_seed(0)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(wc)
    model.load_weights(MODEL_DIR, torch.device("cpu"), None)
    model = model.to(device)
    model.eval()
    n_layers = wc.decoder_layers
    d_model = wc.d_model if hasattr(wc, "d_model") else 1280
    print(f"model loaded d_model={d_model} K={K}", flush=True)

    # Compile the decoder wrapper (returns the last hidden state directly) so we
    # can both harvest hidden states AND derive the greedy token via
    # compute_logits on the host. One NEFF, self-consistent greedy.
    dec_fn = torch.compile(model.decoder, backend="vllm_neuron", fullgraph=True)
    model.visual = torch.compile(model.visual, backend="vllm_neuron", fullgraph=True)

    kv = alloc_self_kv(model, n_layers, MAX_BLOCKS, BLOCK_SIZE, device)

    # Extract the fp32 real-vocab lm_head weight on CPU up front. We project the
    # decoder's (CPU-moved) hidden state on the host with this, avoiding an
    # eager device-side compute_logits (which hits a dtype-cast issue outside a
    # compiled graph). At TP=1 the sharded lm_head owns the full padded vocab.
    W = model.lm_head.weight.detach().cpu().float()            # [padded_vocab, d]
    W = W[: wc.vocab_size]                                     # [vocab, d]
    vocab = W.shape[0]

    def cpu_project(hidden_cpu):  # [.., d] fp32 cpu -> logits [.., vocab]
        return hidden_cpu.float() @ W.t()

    def prefill_and_hidden():
        ids = torch.tensor(SOT, dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device)
        am = build_attn_metadata(n_layers, list(range(len(SOT))), BLOCK_SIZE,
                                 MAX_BLOCKS, device, True, K)
        with torch.no_grad():
            hidden = dec_fn(ids, pos, am, True)  # [len(SOT), d]
        h_cpu = hidden[len(SOT) - 1:].cpu().float()  # [1, d]
        logits = cpu_project(h_cpu)  # [1, vocab]
        return logits

    def step_hidden(tok, base_pos):
        """Run one decode token; return (next_tok, hidden[1,d] cpu at this pos)."""
        ids = torch.tensor([tok], dtype=torch.long, device=device)
        pos = torch.tensor([base_pos], dtype=torch.long, device=device)
        am = build_attn_metadata(n_layers, [base_pos], BLOCK_SIZE, MAX_BLOCKS,
                                 device, False, K)
        with torch.no_grad():
            hidden = dec_fn(ids, pos, am, False)  # [1, d]
        h_cpu = hidden.cpu().float()  # [1, d]
        logits = cpu_project(h_cpu)  # [1, vocab]
        nxt = int(torch.argmax(logits[0]).item())
        return nxt, h_cpu

    # ---- harvest greedy hidden states + tokens for each clip ----
    hiddens = []  # h_t   (post-final-LN last hidden at position t)
    tokens = []   # y_{t+1} = greedy token AFTER position t (== argmax lm_head(h_t))
    for clip in args.clips.split(","):
        clip = clip.strip()
        mel = np.load(os.path.join(REF, f"{clip}_mel.npy"))
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)
        # reset cross-KV by re-running the encoder for this clip
        model.embed_multimodal(input_features=mel_t)
        first_logits = prefill_and_hidden()
        first = int(torch.argmax(first_logits[0]).item())
        seq = [first]
        pos = len(SOT)
        cur = first
        clip_h, clip_y = [], []
        while cur != EOS and len(seq) < args.max_new:
            nxt, h = step_hidden(cur, pos)
            clip_h.append(h[0])       # hidden at position `pos` -> predicts token `nxt`
            clip_y.append(nxt)
            seq.append(nxt)
            pos += 1
            cur = nxt
        hiddens.append(torch.stack(clip_h))  # [T, d]
        tokens.append(clip_y)
        print(f"harvested clip={clip} T={len(clip_y)} (tokens incl EOS in seq={len(seq)})",
              flush=True)

    # Build per-head (h, target) training tensors.
    # Head j predicts token at anchor+1+j. If h_t is the hidden that predicts
    # y_{t+1} (the immediate next token), then from anchor position a the head
    # reads h_a and must predict y_{a+1+j}. In our harvest, hiddens[c][i] is h at
    # position (len(SOT)+i) and tokens[c][i] is the greedy token at that position
    # + 1. So the token j+1 steps ahead of hiddens[c][i] is tokens[c][i + j].
    H_all = {j: [] for j in range(K)}
    Y_all = {j: [] for j in range(K)}
    for c in range(len(hiddens)):
        Hc = hiddens[c]      # [T, d]
        Yc = tokens[c]       # list length T
        T = len(Yc)
        for j in range(K):
            for i in range(T - j):
                H_all[j].append(Hc[i])
                Y_all[j].append(Yc[i + j])
    for j in range(K):
        H_all[j] = torch.stack(H_all[j])                       # [Nj, d]
        Y_all[j] = torch.tensor(Y_all[j], dtype=torch.long)    # [Nj]
        print(f"head {j}: {H_all[j].shape[0]} training pairs", flush=True)

    # ---- fit each head's ResBlock (frozen lm_head projection) on CPU ----
    # W (fp32 real-vocab lm_head weight) was extracted above.
    def project(x):  # x:[N,d] fp32 -> logits [N, vocab]
        return x @ W.t()

    def silu(x):
        return x * torch.sigmoid(x)

    synth_state = {}
    acc_report = {}
    for j in range(K):
        lin = torch.nn.Linear(d_model, d_model)
        torch.nn.init.zeros_(lin.weight)   # start from identity ResBlock (head-0 correct)
        torch.nn.init.zeros_(lin.bias)
        lin.train()
        opt = torch.optim.Adam(lin.parameters(), lr=args.lr)
        Hj = H_all[j]        # [Nj, d] fp32
        Yj = Y_all[j]        # [Nj]
        loss_fn = torch.nn.CrossEntropyLoss()
        for step in range(args.steps):
            opt.zero_grad()
            res = Hj + silu(lin(Hj))       # ResBlock
            logits = project(res)          # [Nj, vocab]
            loss = loss_fn(logits, Yj)
            loss.backward()
            opt.step()
        # measure argmax accuracy after fit
        with torch.no_grad():
            res = Hj + silu(lin(Hj))
            pred = project(res).argmax(dim=-1)
            acc = float((pred == Yj).float().mean().item())
        acc_report[j] = acc
        print(f"head {j}: fit done loss={loss.item():.3f} argmax_acc={acc:.3f}", flush=True)
        synth_state[f"medusa_heads.{j}.0.linear.weight"] = lin.weight.detach().to(DTYPE)
        synth_state[f"medusa_heads.{j}.0.linear.bias"] = lin.bias.detach().to(DTYPE)

    torch.save({"heads": synth_state, "acc": acc_report, "K": K}, args.out)
    print(f"SAVED synthetic heads -> {args.out}", flush=True)
    print(f"HEAD_ARGMAX_ACC (per-head, offline, one-step): {acc_report}", flush=True)
    print("SUCCESS", flush=True)


if __name__ == "__main__":
    main()
