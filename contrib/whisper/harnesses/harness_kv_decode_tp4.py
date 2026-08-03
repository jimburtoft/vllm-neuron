# SPDX-License-Identifier: Apache-2.0
"""
Task 004: KV-cache decode at TP=4 LNC=2 on the vllm_neuron native backend.

Launched via torchrun --nproc_per_node=4. Same CCOM/rendezvous machinery as the
Task 003 encoder TP=4 harness. Confirms the full KV-cache decode pipeline
(precompute_cross_kv -> prefill -> decode_step loop) works end-to-end at TP=4 and
produces the byte-identical transcript on the JFK clip.

At TP=4 each rank holds n_heads_local = 20/4 = 5 heads; the cross-KV and self-KV
buffers are sized per-rank ([1,5,*,64]); RowParallelLinear all-reduces the
attention output across the 4 logical cores.
"""
import json
import os
import sys
import time
# contrib layout: whisper_neuron.py / weight_loaders.py live one dir up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

MODEL_DIR = sys.argv[1]
REF = "/large/work/ref"
DTYPE = torch.bfloat16
MAX_SELF = 448
SOT = [50258, 50259, 50360, 50364]
EOS = 50257
MAX_NEW = 200
CLIP = os.environ.get("WX_CLIP", "jfk")


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)

    import vllm_neuron  # noqa: F401
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()

    from transformers import AutoConfig
    from whisper_neuron import WhisperEncoder, WhisperDecoder, WhisperConfigLite
    from weight_loaders import load_hf_state, build_state

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg = WhisperConfigLite(hf_cfg)
    hf_sd = load_hf_state(MODEL_DIR)

    enc = WhisperEncoder(cfg, DTYPE)
    enc.load_state_dict(build_state(hf_sd, "model.encoder.", DTYPE), strict=False)
    enc.eval()
    dec = WhisperDecoder(cfg, DTYPE, max_self_len=MAX_SELF)
    dec.load_state_dict(build_state(hf_sd, "model.decoder.", DTYPE), strict=False)
    dec.eval()

    device = "neuron:0"
    enc = enc.to(device)
    dec = dec.to(device)
    if rank == 0:
        print(f"[r{rank}] n_heads_local={dec.n_heads_local} head_dim={dec.head_dim}",
              flush=True)

    enc_fn = torch.compile(enc, backend="vllm_neuron", fullgraph=True)
    precompute_fn = torch.compile(dec.precompute_cross_kv, backend="vllm_neuron", fullgraph=True)
    prefill_fn = torch.compile(dec.prefill, backend="vllm_neuron", fullgraph=True)
    step_fn = torch.compile(dec.decode_step, backend="vllm_neuron", fullgraph=True)

    refs = json.load(open(os.path.join(REF, "refs_all.json")))
    mel = np.load(os.path.join(REF, f"{CLIP}_mel.npy"))
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)

    with torch.no_grad():
        enc_hidden = enc_fn(mel_t)
        precompute_fn(enc_hidden)

    ids = torch.tensor([SOT], dtype=torch.long, device=device)
    pos = torch.arange(len(SOT), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        pf = prefill_fn(ids, pos)
    last = pf[0, len(SOT) - 1].to("cpu").float()
    nxt = int(torch.argmax(last).item())
    gen = [nxt]
    tokens = list(SOT) + [nxt]
    step_times = []
    while nxt != EOS and len(gen) < MAX_NEW:
        cur_pos = len(tokens) - 1
        input_id = torch.tensor([[nxt]], dtype=torch.long, device=device)
        position = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        cur = torch.tensor(cur_pos, dtype=torch.long, device=device)
        ts = time.time()
        with torch.no_grad():
            sl = step_fn(input_id, position, cur)
        last = sl[0, -1].to("cpu").float()
        step_times.append(time.time() - ts)
        nxt = int(torch.argmax(last).item())
        gen.append(nxt)
        tokens.append(nxt)

    if rank == 0:
        ref_tokens = refs[CLIP]["tokens"]
        gen_content = gen[:-1] if gen and gen[-1] == EOS else gen
        match = gen_content == ref_tokens
        med = float(np.median(step_times)) * 1000 if step_times else 0.0
        print(f"[r{rank}] CLIP={CLIP}", flush=True)
        print(f"[r{rank}] GEN ({len(gen)}): {gen}", flush=True)
        print(f"[r{rank}] REF ({len(ref_tokens)}): {ref_tokens}", flush=True)
        print(f"TP4_BYTE_IDENTICAL={match}", flush=True)
        print(f"TP4_MEDIAN_STEP_MS {med:.1f}", flush=True)
        print("SUCCESS" if match else "FAIL", flush=True)

    # Note: no dist.barrier()/destroy_process_group() here -- in the bare direct
    # harness those touch the PrivateUse1 backend at teardown and raise
    # "RegisterPrivateUse1HooksInterface" AFTER the result is already printed.
    # The decode result above is unaffected; ranks exit independently.


if __name__ == "__main__":
    main()
