# SPDX-License-Identifier: Apache-2.0
"""
whisper-xla Task 020 Milestone 3 -- Medusa CUSTOMER LATENCY HARNESS (BS=1, native).

This is the tool the CUSTOMER runs, once they load their TRAINED Medusa heads, to
measure the single-stream latency speedup the heads deliver. It drives the SAME
BS=1 Medusa decode loop the framework ships (plugin verify-K NEFF + host rejection
+ block-managed self-KV), on a set of clips, and reports per clip:

  * end-to-end decode latency (prefill + all verify iters), warm
  * mean per-iter (per-verify) latency
  * mean accepted tokens / iter (the acceptance the heads deliver)
  * effective ms / token  (= e2e_latency / tokens_emitted)
  * speedup vs plain greedy on the SAME config

PLUS the breakeven / speedup MODEL so the customer understands
"speedup = f(acceptance)": given the measured verify/step ratio r (~1.116x from
M0.5), speedup ~= (mean_accepted + 1) / r. This makes explicit that
placeholder/untrained heads (mean_accepted ~ 0) are SLOWER than greedy, and the
win scales with acceptance -- heads must clear ~ (r - 1) mean-accepted just to
break even.

HONEST framing: this harness does NOT itself produce a speedup with placeholder
heads (~1.0x or slightly slower). It is the measurement TOOL; the speedup is a
property of the customer's trained heads.

Configurable:
  --medusa-heads  zero | random | <path-to-head-ckpt>   (default zero)
  --num-speculative-tokens K                            (default 5)
  --clips jfk,ls1,ls2,ls3,ls4
  --warmup N   --iters N   (repeat measurements; report warm mean/min)

Launch:  torchrun --nproc_per_node=4 medusa_bench.py --medusa-heads zero   (TP=4)
         python medusa_bench.py --medusa-heads /path/to/heads.pt            (TP=1)
"""
import argparse
import json
import os
import sys
import time

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

# Measured verify/step latency ratio from the M0.5 de-risk (TP=4 LNC=2):
# a [1,K+1] verify step costs ~1.116x a single-token [1,1] step (decode is
# bandwidth-bound; the extra K positions are nearly free). Used for the
# theoretical speedup model. Customers on a different config should re-measure
# with --verify-ratio.
DEFAULT_VERIFY_RATIO = 1.116


def _dist_setup():
    import torch.distributed as dist
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)
        import vllm_neuron  # noqa: F401
        dist.init_process_group(backend="gloo", rank=rank, world_size=world)
        from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
        rendezvous_ccom_bootstrap()
        from vllm_neuron.parallel.neuron_parallel_state import (
            initialize_neuron_parallel_state,
        )
        initialize_neuron_parallel_state(
            tp_global_ranks=list(range(world)), local_rank=local_rank
        )
        return rank, world
    import vllm_neuron  # noqa: F401
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("WX_PORT", "29591"))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return 0, 1


class SpecMeta:
    def __init__(self, K, device):
        self.logits_indices = torch.arange(K + 1, dtype=torch.long, device=device)
        self.target_logits_indices = torch.arange(K, dtype=torch.long, device=device)
        self.bonus_logits_indices = torch.tensor([K], dtype=torch.long, device=device)
        self.cu_num_draft_tokens = torch.tensor([K], dtype=torch.int32, device=device)
        self.draft_token_ids = None
        self.num_draft_tokens = [K]


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


def speedup_model(verify_ratio, max_acc=6):
    """Theoretical speedup vs greedy as a function of mean accepted / iter.

    Each greedy token costs 1 [1,1] step. Medusa emits (mean_accepted + 1)
    tokens per verify (accepted drafts + the bonus), and a verify costs
    verify_ratio [1,1]-steps. So:
        speedup ~= (mean_accepted + 1) / verify_ratio
    (Here mean_accepted counts drafts accepted BEYOND the bonus; total tokens/
    iter = mean_accepted + 1. In our accept-count convention accepted[] already
    includes the bonus, so tokens/iter == mean_accepted, and
        speedup ~= mean_accepted / verify_ratio.)
    We report the curve for tokens-emitted-per-verist = 1..max_acc.
    """
    rows = []
    for tok_per_iter in range(1, max_acc + 1):
        rows.append((tok_per_iter, tok_per_iter / verify_ratio))
    breakeven = verify_ratio  # tokens/iter needed for speedup >= 1.0
    return rows, breakeven


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medusa-heads", default="zero",
                    help="zero | random | <path to head checkpoint>")
    ap.add_argument("--num-speculative-tokens", type=int,
                    default=int(os.environ.get("WX_K", "5")))
    ap.add_argument("--clips", default=os.environ.get("WX_CLIPS", "jfk,ls1,ls2,ls3,ls4"))
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--verify-ratio", type=float, default=DEFAULT_VERIFY_RATIO)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = args.num_speculative_tokens

    rank, world = _dist_setup()

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    from transformers import AutoConfig
    from whisper_pkg.config import WhisperConfig
    from whisper_pkg.model_bf16 import WhisperForConditionalGeneration
    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.vllm.sample.rejection_sampler import (
        RejectionSampler, PLACEHOLDER_TOKEN_ID,
    )

    # Resolve head mode.
    heads_arg = args.medusa_heads
    if heads_arg in ("zero", "random"):
        med = {"num_heads": K, "medusa_num_layers": 1, "init": heads_arg, "seed": 0}
        head_desc = heads_arg
    else:
        med = {"num_heads": K, "medusa_num_layers": 1, "init": "load",
               "heads_path": heads_arg, "seed": 0}
        head_desc = f"load:{os.path.basename(heads_arg)}"

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    nc = NeuronConfig(on_device_sampling_config=None)
    wc = WhisperConfig.from_configs(hf_cfg, nc)
    wc.medusa_config = med

    device = "neuron:0"
    torch.manual_seed(0)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(wc)
    model.load_weights(MODEL_DIR, torch.device("cpu"), None)
    model = model.to(device)
    model.eval()
    n_layers = wc.decoder_layers
    model_fn = torch.compile(model, backend="vllm_neuron", fullgraph=True)
    model.visual = torch.compile(model.visual, backend="vllm_neuron", fullgraph=True)
    kv = alloc_self_kv(model, n_layers, MAX_BLOCKS, BLOCK_SIZE, device)
    log(f"[r{rank}] TP={world} K={K} heads={head_desc} "
        f"load_report={model._medusa_load_report}")

    def prefill():
        ids = torch.tensor(SOT, dtype=torch.long, device=device)
        pos = torch.arange(len(SOT), dtype=torch.long, device=device)
        am = build_attn_metadata(n_layers, list(range(len(SOT))), BLOCK_SIZE,
                                 MAX_BLOCKS, device, True, K)
        with torch.no_grad():
            logits = model_fn(ids, pos, attn_metadata=am,
                              sampling_positions=torch.tensor(
                                  [len(SOT) - 1], dtype=torch.long, device=device))
        return int(torch.argmax(logits[0].float().cpu()).item())

    def greedy_step(anchor_tok, base_pos):
        ids = torch.tensor([anchor_tok], dtype=torch.long, device=device)
        pos = torch.tensor([base_pos], dtype=torch.long, device=device)
        am = build_attn_metadata(n_layers, [base_pos], BLOCK_SIZE, MAX_BLOCKS,
                                 device, False, K)
        td = time.perf_counter()
        with torch.no_grad():
            logits = model_fn(ids, pos, attn_metadata=am,
                              sampling_positions=torch.tensor(
                                  [0], dtype=torch.long, device=device))
        tok = int(torch.argmax(logits[0].float().cpu()).item())  # forces device sync
        return tok, time.perf_counter() - td

    def run_greedy():
        t0 = time.perf_counter()
        first = prefill()
        gen = [first]
        pos = len(SOT)
        cur = first
        dev_t = 0.0
        while cur != EOS and len(gen) < args.max_new:
            nxt, dt = greedy_step(cur, pos)
            dev_t += dt
            gen.append(nxt)
            pos += 1
            cur = nxt
        return gen, time.perf_counter() - t0, dev_t

    def rejection_greedy(verify_logits_cpu, drafts):
        k = len(drafts)
        target_argmax = verify_logits_cpu[:k].argmax(dim=-1)
        bonus = int(verify_logits_cpu[k].argmax().item())
        out = torch.empty((1, k + 1), dtype=torch.int32)
        out.fill_(PLACEHOLDER_TOKEN_ID)
        RejectionSampler._rejection_greedy_sample(
            out, torch.tensor([k], dtype=torch.int32),
            torch.tensor(drafts, dtype=torch.int32),
            target_argmax, torch.tensor([bonus], dtype=torch.int32), None, k, 1,
        )
        return [t for t in out[0].tolist() if t != PLACEHOLDER_TOKEN_ID]

    def verify_step(window, base_pos):
        ids = torch.tensor(window, dtype=torch.long, device=device)
        positions = torch.arange(base_pos, base_pos + K + 1, dtype=torch.long, device=device)
        am = build_attn_metadata(n_layers, list(range(base_pos, base_pos + K + 1)),
                                 BLOCK_SIZE, MAX_BLOCKS, device, False, K)
        sm = SpecMeta(K, device)
        td = time.perf_counter()
        with torch.no_grad():
            out = model_fn(ids, positions, attn_metadata=am,
                           sampling_positions=sm.logits_indices, spec_decode_metadata=sm)
        vl_cpu = out[0].float().cpu()  # forces device sync -> device time captured
        dev_t = time.perf_counter() - td
        return vl_cpu, out[2].to(torch.int32).cpu(), dev_t

    def run_medusa():
        t0 = time.perf_counter()
        first = prefill()
        gen = [first]
        accept_counts = []
        iter_times = []
        dev_times = []
        pos = len(SOT)
        anchor = first
        drafts = [0] * K
        while anchor != EOS and len(gen) < args.max_new:
            ti = time.perf_counter()
            window = [anchor] + list(drafts)
            vl, next_drafts, dt = verify_step(window, pos)
            accepted = rejection_greedy(vl, list(drafts))
            iter_times.append(time.perf_counter() - ti)
            dev_times.append(dt)
            accept_counts.append(len(accepted))
            for t in accepted:
                gen.append(t)
                if t == EOS:
                    break
            pos += len(accepted)
            anchor = accepted[-1]
            drafts = next_drafts.view(-1).tolist()
            if anchor == EOS:
                break
        return gen, accept_counts, iter_times, dev_times, time.perf_counter() - t0

    clips = [c.strip() for c in args.clips.split(",")]
    per_clip = {}
    for clip in clips:
        mel = np.load(os.path.join(REF, f"{clip}_mel.npy"))
        mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)
        model.embed_multimodal(input_features=mel_t)
        # warmup
        for _ in range(args.warmup):
            model.embed_multimodal(input_features=mel_t)
            run_greedy()
            model.embed_multimodal(input_features=mel_t)
            run_medusa()
        g_lat, g_dev, m_e2e, m_dev, m_iters, m_devit, m_acc = [], [], [], [], [], [], []
        toks = None
        for _ in range(args.iters):
            model.embed_multimodal(input_features=mel_t)
            greedy, gt, gdev = run_greedy()
            model.embed_multimodal(input_features=mel_t)
            med_tok, accepts, it_times, dev_times, mt = run_medusa()
            g_lat.append(gt)
            g_dev.append(gdev)
            m_e2e.append(mt)
            m_dev.append(sum(dev_times))
            m_iters.extend(it_times)
            m_devit.extend(dev_times)
            m_acc.append(float(np.mean(accepts)))
            toks = len(med_tok)
        greedy_lat = float(np.min(g_lat))
        greedy_dev = float(np.min(g_dev))
        med_lat = float(np.min(m_e2e))
        med_dev = float(np.min(m_dev))
        mean_iter = float(np.mean(m_iters))
        mean_devit = float(np.mean(m_devit))
        mean_acc = float(np.mean(m_acc))
        ms_per_tok = med_lat / max(toks, 1) * 1e3
        # wall-clock speedup (host-bound: includes the K+1-row logit->CPU transfer
        # + Python rejection loop this offline harness pays per verify).
        speedup = greedy_lat / med_lat if med_lat > 0 else 0.0
        # DEVICE-only speedup (NEFF time only) -- the number that matters once the
        # framework uses on-device rejection (M4+ lever) and drops the host bounce.
        speedup_dev = greedy_dev / med_dev if med_dev > 0 else 0.0
        per_clip[clip] = {
            "tokens": toks,
            "greedy_e2e_s": greedy_lat,
            "greedy_dev_s": greedy_dev,
            "medusa_e2e_s": med_lat,
            "medusa_dev_s": med_dev,
            "mean_iter_ms": mean_iter * 1e3,
            "mean_dev_iter_ms": mean_devit * 1e3,
            "mean_accepted_per_iter": mean_acc,
            "eff_ms_per_token": ms_per_tok,
            "speedup_vs_greedy_wall": speedup,
            "speedup_vs_greedy_device": speedup_dev,
        }
        log(f"[r{rank}] clip={clip:5s} toks={toks:3d} greedy_wall={greedy_lat*1e3:6.1f}ms "
            f"medusa_wall={med_lat*1e3:6.1f}ms dev_iter={mean_devit*1e3:5.1f}ms "
            f"acc/iter={mean_acc:.3f} spd_wall={speedup:.3f}x spd_dev={speedup_dev:.3f}x")

    if rank == 0:
        # ---- clean table ----
        log("\n===== MEDUSA LATENCY HARNESS (BS=1, TP=%d, K=%d, heads=%s) =====" %
            (world, K, head_desc))
        log(f"{'clip':6s} {'toks':>5s} {'greedy(ms)':>11s} {'medusa(ms)':>11s} "
            f"{'devit(ms)':>9s} {'acc/iter':>9s} {'ms/tok':>8s} "
            f"{'spd_wall':>9s} {'spd_dev':>8s}")
        log("-" * 90)
        for clip, r in per_clip.items():
            log(f"{clip:6s} {r['tokens']:5d} {r['greedy_e2e_s']*1e3:11.1f} "
                f"{r['medusa_e2e_s']*1e3:11.1f} {r['mean_dev_iter_ms']:9.1f} "
                f"{r['mean_accepted_per_iter']:9.3f} {r['eff_ms_per_token']:8.1f} "
                f"{r['speedup_vs_greedy_wall']:8.3f}x {r['speedup_vs_greedy_device']:7.3f}x")
        agg_acc = float(np.mean([r["mean_accepted_per_iter"] for r in per_clip.values()]))
        agg_wall = float(np.mean([r["speedup_vs_greedy_wall"] for r in per_clip.values()]))
        agg_dev = float(np.mean([r["speedup_vs_greedy_device"] for r in per_clip.values()]))
        log("-" * 90)
        log(f"{'MEAN':6s} {'':5s} {'':11s} {'':11s} {'':9s} "
            f"{agg_acc:9.3f} {'':8s} {agg_wall:8.3f}x {agg_dev:7.3f}x")
        log("\nspd_wall = end-to-end wall-clock speedup vs greedy on the SAME config.")
        log("spd_dev  = model_fn NEFF-call time only (excludes the Python rejection loop).")
        log("NOTE: BOTH numbers in this OFFLINE harness are HOST-TRANSFER-BOUND -- greedy")
        log("  returns [1,vocab] per call, the Medusa verify returns [K+1,vocab] (6x more")
        log("  logits copied to CPU for host rejection), so a placeholder-head run looks")
        log("  ~0.5x here regardless of the on-device NEFF ratio. The clean ON-DEVICE")
        log("  verify/step ratio is 1.116x (M0.5); use the model below for the real")
        log("  customer-facing speedup projection. On-device rejection (an M4+ lever) drops")
        log("  the host bounce and realizes the model's speedup.")

        # ---- speedup-vs-acceptance model ----
        rows, breakeven = speedup_model(args.verify_ratio, max_acc=K + 1)
        log(f"\n===== SPEEDUP-vs-ACCEPTANCE MODEL (on-device verify/step ratio r={args.verify_ratio:.3f}, M0.5) =====")
        log("On device, a [1,K+1] verify costs r x a [1,1] step. Emitting T tokens per")
        log("verify (accepted drafts + bonus) gives:  speedup ~= T / r")
        log(f"{'tokens/verify':>13s} {'theoretical speedup':>20s}")
        for tpv, sp in rows:
            note = "  <- placeholder heads land here (SLOWER than greedy)" if abs(tpv - 1) < 1e-9 else ""
            note = note or ("  (< 1.0x: SLOWER)" if sp < 1.0 else "")
            log(f"{tpv:13d} {sp:19.3f}x{note}")
        log(f"\nBREAKEVEN: heads must deliver > {breakeven:.3f} tokens/verify "
            f"(> {breakeven-1:.3f} accepted drafts/verify beyond the bonus) to beat greedy on device.")
        theo = agg_acc / args.verify_ratio
        log(f"MEASURED mean tokens/verify (this run, heads={head_desc}): {agg_acc:.3f}")
        log(f"  -> THEORETICAL on-device speedup ~= {agg_acc:.3f} / {args.verify_ratio:.3f} = {theo:.3f}x "
            f"({'SPEEDUP' if theo > 1.0 else 'NO SPEEDUP -- expected for placeholder/untrained heads'})")
        log("\nHONEST FRAMING: with placeholder/untrained heads (zero/random, or crude "
            "synthetic), mean tokens/verify stays near 1, so Medusa is ~1.0x or SLOWER than "
            "greedy on device -- you pay the ~1.116x verify overhead for ~0 accepted drafts. "
            "This harness is the MEASUREMENT TOOL: load your TRAINED heads and re-run. The "
            "speedup scales with the acceptance your heads deliver (speedup ~= tokens_per_verify "
            "/ 1.116); heads need > ~0.12 accepted drafts/verify just to break even.")

        out = {
            "tp": world, "K": K, "heads": head_desc, "verify_ratio": args.verify_ratio,
            "per_clip": per_clip, "mean_accepted_per_iter": agg_acc,
            "mean_speedup_wall": agg_wall, "mean_speedup_device": agg_dev,
            "theoretical_ondevice_speedup": theo,
            "speedup_model": [{"tokens_per_verify": t, "speedup": s} for t, s in rows],
            "breakeven_tokens_per_verify": breakeven,
        }
        outp = args.out or os.path.join(REF, f"task020_m3_bench_tp{world}_{head_desc.replace(':','_').replace('.','_')}.json")
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        log(f"\n(wrote {outp})")
        log("SUCCESS")


if __name__ == "__main__":
    main()
