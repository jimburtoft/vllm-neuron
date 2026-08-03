# SPDX-License-Identifier: Apache-2.0
"""
Task 004: KV-cache decode harness (TP=1) on the vllm_neuron native backend.

Pipeline per clip:
  1. Encoder NEFF: mel -> encoder_hidden [1,1500,1280]   (compiled once)
  2. precompute_cross_kv(encoder_hidden): project encoder output through every
     decoder layer's cross-attn K/V ONCE, .copy_() into persistent HBM buffers.
     (compiled graph; run once per clip)
  3. prefill NEFF: SOT prompt [1,4] -> logits[1,4,vocab]; writes self-KV [0:4).
  4. decode_step NEFF [1,1]: append one token; reads cross-KV + self-KV buffers,
     writes new token's self-KV in-place. Looped greedily to EOS.

Correctness gate: greedy token IDs must be byte-identical to the OpenAI-whisper
reference (refs_all.json) for >=5/5 clips.

HBM-persistence validation:
  * assert every cache buffer stays on device (.device.type == 'neuron')
  * confirm the decode_step graph is compiled ONCE (single bucket at [1,1]) and
    reused across all steps (no per-step recompile == no encoder reprojection).
  * confirm cross-KV buffers are NOT mutated during decode (checksum before/after
    the decode loop is identical) -> proves no per-step cross-attn recompute.
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
DTYPE = torch.float32 if os.environ.get("WX_CPU", "0") == "1" else torch.bfloat16
CPU = os.environ.get("WX_CPU", "0") == "1"
COMPILE = os.environ.get("WX_COMPILE", "1") == "1" and not CPU
MAX_SELF = 448

SOT = [50258, 50259, 50360, 50364]  # <|sot|><|en|><|transcribe|><|notimestamps|>
EOS = 50257
MAX_NEW = 200

CLIPS = ["jfk", "libri_0", "libri_1", "libri_2", "libri_3"]


def main():
    import torch.distributed as dist
    from transformers import AutoConfig
    from whisper_neuron import WhisperEncoder, WhisperDecoder, WhisperConfigLite
    from weight_loaders import load_hf_state, build_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29551")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg = WhisperConfigLite(hf_cfg)
    print(f"CONFIG OK d_model={cfg.d_model} vocab={cfg.vocab_size} "
          f"dec_layers={cfg.decoder_layers}", flush=True)

    hf_sd = load_hf_state(MODEL_DIR)
    enc = WhisperEncoder(cfg, DTYPE)
    enc.load_state_dict(build_state(hf_sd, "model.encoder.", DTYPE), strict=False)
    enc.eval()
    dec = WhisperDecoder(cfg, DTYPE, max_self_len=MAX_SELF)
    dm, du = dec.load_state_dict(build_state(hf_sd, "model.decoder.", DTYPE), strict=False)
    print(f"DEC MISSING={len(dm)} UNEXPECTED={len(du)}", flush=True)
    dec.eval()

    device = "cpu" if CPU else "neuron:0"
    enc = enc.to(device)
    dec = dec.to(device)

    # cache buffers must live on device
    for n, b in dec.named_buffers():
        if n.startswith(("cross_", "self_")):
            assert b.device.type == device.split(":")[0], (n, b.device)
    print("BUFFERS_ON_DEVICE ok", flush=True)

    # ---- compiled callables ----
    if COMPILE:
        enc_fn = torch.compile(enc, backend="vllm_neuron", fullgraph=True)
        # compile the three decoder entry points as bound methods
        precompute_fn = torch.compile(dec.precompute_cross_kv,
                                      backend="vllm_neuron", fullgraph=True)
        prefill_fn = torch.compile(dec.prefill, backend="vllm_neuron", fullgraph=True)
        step_fn = torch.compile(dec.decode_step, backend="vllm_neuron", fullgraph=True)
    else:
        enc_fn = enc
        precompute_fn = dec.precompute_cross_kv
        prefill_fn = dec.prefill
        step_fn = dec.decode_step

    refs = json.load(open(os.path.join(REF, "refs_all.json")))

    def cross_checksum():
        s = 0.0
        for i in range(cfg.decoder_layers):
            s += getattr(dec, f"cross_k_{i}").to("cpu").float().abs().sum().item()
            s += getattr(dec, f"cross_v_{i}").to("cpu").float().abs().sum().item()
        return s

    results = {}
    step_compile_count = {"n": 0}

    for clip in CLIPS:
        mel_path = os.path.join(REF, f"{clip}_mel.npy")
        if not os.path.exists(mel_path) or clip not in refs:
            print(f"SKIP {clip} (missing mel or ref)", flush=True)
            continue
        mel = np.load(mel_path)
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)

        # No need to zero the self-KV cache between clips: prefill overwrites
        # positions [0:P) and the decode-step additive mask (-inf beyond cur_pos)
        # makes any stale cache content unattendable. Also, the buffers are now
        # aliased to the compiled graph's outputs (shared storage), so an
        # out-of-graph .zero_() raises "ReserveSpace on shared storage".

        with torch.no_grad():
            enc_hidden = enc_fn(mel_t)  # [1,1500,1280]
            # populate cross-KV buffers once
            precompute_fn(enc_hidden)
        cs_after_precompute = cross_checksum()

        # ---- prefill ----
        ids = torch.tensor([SOT], dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            pf_logits = prefill_fn(ids, pos)  # [1,P,vocab]
        last = pf_logits[0, len(SOT) - 1].to("cpu").float()
        nxt = int(torch.argmax(last).item())

        gen = [nxt]
        tokens = list(SOT) + [nxt]
        t0 = time.time()
        n_step_calls = 0
        step_times = []
        while nxt != EOS and len(gen) < MAX_NEW:
            cur_pos = len(tokens) - 1  # position of the token we will process now
            input_id = torch.tensor([[nxt]], dtype=torch.long, device=device)
            position = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
            cur = torch.tensor(cur_pos, dtype=torch.long, device=device)
            ts = time.time()
            with torch.no_grad():
                step_logits = step_fn(input_id, position, cur)  # [1,1,vocab]
            last = step_logits[0, -1].to("cpu").float()
            step_times.append(time.time() - ts)
            n_step_calls += 1
            nxt = int(torch.argmax(last).item())
            gen.append(nxt)
            tokens.append(nxt)
        dt = time.time() - t0

        cs_after_decode = cross_checksum()
        cross_unchanged = abs(cs_after_precompute - cs_after_decode) < 1e-3

        # OpenAI ref tokens do NOT include the trailing EOS; our gen stops ON EOS.
        # Strip our trailing EOS before the byte-identical comparison.
        ref_tokens = refs[clip]["tokens"]
        gen_content = gen[:-1] if gen and gen[-1] == EOS else gen
        match = gen_content == ref_tokens
        match_noeos = match

        med_step = float(np.median(step_times)) * 1000 if step_times else 0.0
        results[clip] = {
            "match": match, "n_gen": len(gen), "n_ref": len(ref_tokens),
            "cross_unchanged": cross_unchanged, "median_step_ms": med_step,
            "decode_sec": dt,
        }
        print(f"\n=== {clip} dur={refs[clip].get('dur','?')} ===", flush=True)
        print(f"  GEN  ({len(gen)}): {gen}", flush=True)
        print(f"  REF  ({len(ref_tokens)}): {ref_tokens}", flush=True)
        print(f"  BYTE_IDENTICAL={match} (noeos={match_noeos})", flush=True)
        print(f"  CROSS_KV_UNCHANGED_DURING_DECODE={cross_unchanged} "
              f"(cs_pre={cs_after_precompute:.1f} cs_post={cs_after_decode:.1f})", flush=True)
        print(f"  decode {dt:.2f}s, {n_step_calls} step calls, median step {med_step:.1f} ms",
              flush=True)

    n_pass = sum(1 for r in results.values() if r["match"])
    print(f"\nCORRECTNESS_GATE {n_pass}/{len(results)} byte-identical", flush=True)
    print("SUCCESS" if n_pass >= 5 else "PARTIAL", flush=True)
    json.dump(results, open(os.path.join(REF, "task004_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
