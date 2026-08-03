# SPDX-License-Identifier: Apache-2.0
"""Task 007 optimization harness: correctness gate (5 clips) + warm 30-s bench.

Runs the KV-cache pipeline with the Lever-#1 greedy path (sharded LM head +
on-device argmax). Works at TP=1 (correctness gate) and TP=4 (bench).

Env:
  WX_MODE = gate | bench     (default gate)
  WX_CLIP = libri_3          (bench clip)
  WX_GREEDY = 1              (use sharded-LM-head greedy path; 0 = baseline full-vocab argmax)

Launch:
  TP=1 gate : torchrun --nproc_per_node=1 harness_opt.py <MODEL_DIR>  (WX_MODE=gate)
  TP=4 bench: torchrun --nproc_per_node=4 harness_opt.py <MODEL_DIR>  (WX_MODE=bench)
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
MODE = os.environ.get("WX_MODE", "gate")
CLIP = os.environ.get("WX_CLIP", "libri_3")
GREEDY = os.environ.get("WX_GREEDY", "1") == "1"
PRETRANSPOSE = os.environ.get("WX_PRETRANSPOSE", "0") == "1"
FOLD_AR = os.environ.get("WX_FOLD_AR", "0") == "1"
WARMUP = 2
REPEATS = 12
GATE_CLIPS = ["jfk", "libri_0", "libri_1", "libri_2", "libri_3"]


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
    dec.tie_lm_head()  # copy tied embed weight into the sharded LM head
    if PRETRANSPOSE:
        from whisper_neuron import enable_pretranspose
        n = enable_pretranspose(dec)
        if rank == 0:
            print(f"PRETRANSPOSE enabled on {n} linears", flush=True)

    enc_fn = torch.compile(enc, backend="vllm_neuron", fullgraph=True)
    precompute_fn = torch.compile(dec.precompute_cross_kv, backend="vllm_neuron", fullgraph=True)
    CCFLAGS = os.environ.get("WX_CCFLAGS", "")
    step_opts = {}
    if CCFLAGS:
        step_opts = {"compiler_args": CCFLAGS}
        if rank == 0:
            print(f"CCFLAGS={CCFLAGS}", flush=True)
    if GREEDY:
        prefill_fn = torch.compile(dec.prefill_greedy, backend="vllm_neuron", fullgraph=True)
        step_fn = torch.compile(dec.decode_step_greedy, backend="vllm_neuron",
                                fullgraph=True, options=step_opts)
    else:
        prefill_fn = torch.compile(dec.prefill, backend="vllm_neuron", fullgraph=True)
        step_fn = torch.compile(dec.decode_step, backend="vllm_neuron",
                                fullgraph=True, options=step_opts)

    refs = json.load(open(os.path.join(REF, "refs_all.json")))

    def gen_clip(mel_t, time_steps=False):
        with torch.no_grad():
            enc_hidden = enc_fn(mel_t)
            precompute_fn(enc_hidden)
        ids = torch.tensor([SOT], dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device).unsqueeze(0)
        t_enc = t_pc = t_pf = 0.0
        with torch.no_grad():
            pf = prefill_fn(ids, pos)
        if GREEDY:
            nxt = int(pf.to("cpu").item())
        else:
            nxt = int(torch.argmax(pf[0, len(SOT) - 1].to("cpu").float()).item())
        gen = [nxt]; tokens = list(SOT) + [nxt]
        step_times = []
        while nxt != EOS and len(gen) < MAX_NEW:
            cur_pos = len(tokens) - 1
            input_id = torch.tensor([[nxt]], dtype=torch.long, device=device)
            position = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
            cur = torch.tensor(cur_pos, dtype=torch.long, device=device)
            ts = time.time()
            with torch.no_grad():
                sl = step_fn(input_id, position, cur)
            if GREEDY:
                nxt = int(sl.to("cpu").item())
            else:
                nxt = int(torch.argmax(sl[0, -1].to("cpu").float()).item())
            step_times.append(time.time() - ts)
            gen.append(nxt); tokens.append(nxt)
        gen_content = gen[:-1] if gen and gen[-1] == EOS else gen
        return gen_content, step_times

    if MODE == "gate":
        n_pass = 0; n_tot = 0
        for clip in GATE_CLIPS:
            mp = os.path.join(REF, f"{clip}_mel.npy")
            if not os.path.exists(mp) or clip not in refs:
                continue
            mel = np.load(mp)
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)
            gen_content, _ = gen_clip(mel_t)
            ref_tokens = refs[clip]["tokens"]
            match = gen_content == ref_tokens
            n_tot += 1; n_pass += int(match)
            if rank == 0:
                print(f"[{clip}] BYTE_IDENTICAL={match} gen={len(gen_content)} ref={len(ref_tokens)}", flush=True)
                if not match:
                    print(f"   GEN={gen_content}", flush=True)
                    print(f"   REF={ref_tokens}", flush=True)
        if rank == 0:
            print(f"CORRECTNESS_GATE {n_pass}/{n_tot} byte-identical", flush=True)
            print("SUCCESS" if n_pass >= 5 else "PARTIAL", flush=True)
        return

    # bench mode
    mel = np.load(os.path.join(REF, f"{CLIP}_mel.npy"))
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)
    ref_tokens = refs[CLIP]["tokens"]

    def one_run():
        t = time.time()
        with torch.no_grad():
            enc_hidden = enc_fn(mel_t)
        _ = enc_hidden[0, 0, 0].to("cpu")
        t_enc = time.time() - t
        t = time.time()
        with torch.no_grad():
            precompute_fn(enc_hidden)
        _ = getattr(dec, "cross_k_0")[0, 0, 0, 0].to("cpu")
        t_pc = time.time() - t
        ids = torch.tensor([SOT], dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device).unsqueeze(0)
        t = time.time()
        with torch.no_grad():
            pf = prefill_fn(ids, pos)
        if GREEDY:
            nxt = int(pf.to("cpu").item())
        else:
            nxt = int(torch.argmax(pf[0, len(SOT) - 1].to("cpu").float()).item())
        t_pf = time.time() - t
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
            if GREEDY:
                nxt = int(sl.to("cpu").item())
            else:
                nxt = int(torch.argmax(sl[0, -1].to("cpu").float()).item())
            step_times.append(time.time() - ts)
            gen.append(nxt); tokens.append(nxt)
        t_decode = time.time() - t_dec0
        gen_content = gen[:-1] if gen and gen[-1] == EOS else gen
        return {"enc": t_enc, "pc": t_pc, "pf": t_pf, "decode": t_decode,
                "total": t_enc + t_pc + t_pf + t_decode,
                "nsteps": len(step_times),
                "step_med": float(np.median(step_times)) if step_times else 0.0,
                "match": gen_content == ref_tokens, "ntok": len(gen_content)}

    for _ in range(WARMUP):
        one_run()
    runs = [one_run() for _ in range(REPEATS)]
    if rank == 0:
        totals = np.array([r["total"] for r in runs]) * 1000
        encs = np.array([r["enc"] for r in runs]) * 1000
        pcs = np.array([r["pc"] for r in runs]) * 1000
        pfs = np.array([r["pf"] for r in runs]) * 1000
        decs = np.array([r["decode"] for r in runs]) * 1000
        steps = np.array([r["step_med"] for r in runs]) * 1000
        r0 = runs[0]
        print(f"BENCH_CLIP {CLIP} ref_tokens={len(ref_tokens)} gen_tokens={r0['ntok']} "
              f"decode_steps={r0['nsteps']} greedy={GREEDY} all_match={all(r['match'] for r in runs)}", flush=True)
        print(f"TOTAL_MS mean={totals.mean():.1f} p50={pct(totals,50):.1f} "
              f"p90={pct(totals,90):.1f} p99={pct(totals,99):.1f} min={totals.min():.1f}", flush=True)
        print(f"ENCODER_MS mean={encs.mean():.1f}", flush=True)
        print(f"PRECOMPUTE_CROSSKV_MS mean={pcs.mean():.1f}", flush=True)
        print(f"PREFILL_MS mean={pfs.mean():.1f}", flush=True)
        print(f"DECODE_LOOP_MS mean={decs.mean():.1f}", flush=True)
        print(f"DECODE_STEP_MED_MS mean={steps.mean():.2f}", flush=True)
        tot = totals.mean()
        print(f"PHASE_PCT enc={100*encs.mean()/tot:.1f} precompute={100*pcs.mean()/tot:.1f} "
              f"prefill={100*pfs.mean()/tot:.1f} decode={100*decs.mean()/tot:.1f}", flush=True)
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
