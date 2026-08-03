# SPDX-License-Identifier: Apache-2.0
"""Task 005: 30-s TP=4 LNC=2 latency baseline + encoder/prefill/decode phase split.

Reuses the Task 004 KV-cache pipeline (precompute_cross_kv -> prefill -> decode_step).
Warm system: discards first WARMUP end-to-end runs, then REPEATS measured runs.
Times encoder, prefill, and the per-token decode loop separately. The .to("cpu")
of each phase output acts as the device-sync boundary for accurate wall-clock.

Launch: torchrun --nproc_per_node=4 harness_bench_tp4.py <MODEL_DIR>
Primary clip: libri_3 (29.4 s, 84 ref tokens) -- the 30-s sample.
"""
import json, os, sys, time
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
MAX_NEW = 250
CLIP = os.environ.get("WX_CLIP", "libri_3")
WARMUP = 2
REPEATS = 12


def pct(a, p):
    return float(np.percentile(a, p))


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
    enc = enc.to(device); dec = dec.to(device)

    enc_fn = torch.compile(enc, backend="vllm_neuron", fullgraph=True)
    precompute_fn = torch.compile(dec.precompute_cross_kv, backend="vllm_neuron", fullgraph=True)
    prefill_fn = torch.compile(dec.prefill, backend="vllm_neuron", fullgraph=True)
    step_fn = torch.compile(dec.decode_step, backend="vllm_neuron", fullgraph=True)

    refs = json.load(open(os.path.join(REF, "refs_all.json")))
    ref_tokens = refs[CLIP]["tokens"]
    mel = np.load(os.path.join(REF, f"{CLIP}_mel.npy"))
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)

    def one_run(collect_phase=False):
        # ENCODER
        t = time.time()
        with torch.no_grad():
            enc_hidden = enc_fn(mel_t)
        _ = enc_hidden[0, 0, 0].to("cpu")  # sync
        t_enc = time.time() - t
        # PRECOMPUTE cross-KV (part of "prefill" cost, one-time per clip)
        t = time.time()
        with torch.no_grad():
            precompute_fn(enc_hidden)
        _ = getattr(dec, "cross_k_0")[0, 0, 0, 0].to("cpu")  # sync
        t_precompute = time.time() - t
        # PREFILL (SOT prompt)
        ids = torch.tensor([SOT], dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device).unsqueeze(0)
        t = time.time()
        with torch.no_grad():
            pf = prefill_fn(ids, pos)
        last = pf[0, len(SOT) - 1].to("cpu").float()
        t_prefill = time.time() - t
        nxt = int(torch.argmax(last).item())
        gen = [nxt]; tokens = list(SOT) + [nxt]
        step_times = []
        t_dec0 = time.time()
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
            gen.append(nxt); tokens.append(nxt)
        t_decode = time.time() - t_dec0
        total = t_enc + t_precompute + t_prefill + t_decode
        gen_content = gen[:-1] if gen and gen[-1] == EOS else gen
        return {
            "enc": t_enc, "precompute": t_precompute, "prefill": t_prefill,
            "decode": t_decode, "total": total, "nsteps": len(step_times),
            "step_med": float(np.median(step_times)) if step_times else 0.0,
            "match": gen_content == ref_tokens, "ntok": len(gen_content),
        }

    # warmup
    for _ in range(WARMUP):
        one_run()
    runs = [one_run() for _ in range(REPEATS)]

    if rank == 0:
        totals = np.array([r["total"] for r in runs]) * 1000
        encs = np.array([r["enc"] for r in runs]) * 1000
        pcs = np.array([r["precompute"] for r in runs]) * 1000
        pfs = np.array([r["prefill"] for r in runs]) * 1000
        decs = np.array([r["decode"] for r in runs]) * 1000
        steps = np.array([r["step_med"] for r in runs]) * 1000
        r0 = runs[0]
        print(f"BENCH_CLIP {CLIP} ref_tokens={len(ref_tokens)} gen_tokens={r0['ntok']} "
              f"decode_steps={r0['nsteps']} all_match={all(r['match'] for r in runs)}", flush=True)
        print(f"TOTAL_MS mean={totals.mean():.1f} p50={pct(totals,50):.1f} "
              f"p90={pct(totals,90):.1f} p99={pct(totals,99):.1f}", flush=True)
        print(f"ENCODER_MS mean={encs.mean():.1f} p50={pct(encs,50):.1f}", flush=True)
        print(f"PRECOMPUTE_CROSSKV_MS mean={pcs.mean():.1f} p50={pct(pcs,50):.1f}", flush=True)
        print(f"PREFILL_MS mean={pfs.mean():.1f} p50={pct(pfs,50):.1f}", flush=True)
        print(f"DECODE_LOOP_MS mean={decs.mean():.1f} p50={pct(decs,50):.1f}", flush=True)
        print(f"DECODE_STEP_MED_MS mean={steps.mean():.2f}", flush=True)
        tot = totals.mean()
        print(f"PHASE_PCT enc={100*encs.mean()/tot:.1f} precompute={100*pcs.mean()/tot:.1f} "
              f"prefill={100*pfs.mean()/tot:.1f} decode={100*decs.mean()/tot:.1f}", flush=True)
        json.dump({"clip": CLIP, "total_ms": totals.tolist(),
                   "enc_ms": encs.tolist(), "precompute_ms": pcs.tolist(),
                   "prefill_ms": pfs.tolist(), "decode_ms": decs.tolist(),
                   "step_med_ms": steps.tolist(), "nsteps": int(r0["nsteps"]),
                   "gen_tokens": int(r0["ntok"])},
                  open("/large/work/task005_bench_tp4.json", "w"), indent=2)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
