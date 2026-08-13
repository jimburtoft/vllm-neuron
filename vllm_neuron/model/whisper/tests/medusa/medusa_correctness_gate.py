# SPDX-License-Identifier: Apache-2.0
"""
whisper-xla Task 020 Milestone 3 -- Medusa CORRECTNESS GATE (BS=1, native plugin).

The formal correctness gate: prove the Medusa framework's accepted-token output
is BYTE-IDENTICAL to plain greedy decode of the SAME Whisper target on the SAME
served config (TP/precision), for EVERY head mode -- INCLUDING when tokens are
genuinely accepted (the multi-token accept path, not just all-rejected).

This drives the FULL BS=1 Medusa decode loop through the REAL M2 artifacts (the
plugin WhisperForConditionalGeneration.forward verify-K path, the plugin
MedusaHeads, the plugin CPU RejectionSampler._rejection_greedy_sample, the
block-managed self-KV) exactly as medusa_m2_test.py does, but sweeps THREE head
modes over FIVE clips:

  (a) zero    -- ResBlock identity heads (M0 spec §2a; head_j == argmax(lm_head(h))).
  (b) random  -- garbage ResBlock heads (drafts rejected -> ~0 accept).
  (c) load    -- a SYNTHETIC partially-correct head set (built by
                 medusa_synth_heads.py) that yields >0 acceptance so the
                 MULTI-TOKEN ACCEPT PATH is exercised. THIS is the strongest
                 correctness evidence: byte-identical to greedy WHILE tokens are
                 actually being accepted.

GATE (M3): for every (mode, clip), content(medusa) == content(greedy). The
greedy reference is computed IN-PROCESS on the same config (M0 spec §5: compare
Medusa-vs-greedy on the SAME TP/precision, not Medusa-vs-openai-whisper, to
isolate the framework). We ALSO report accepted/iter per mode so the accept path
is demonstrably exercised in mode (c).

Launch:  torchrun --nproc_per_node=4 medusa_correctness_gate.py    (TP=4, the serving config)
         python medusa_correctness_gate.py                          (TP=1)
Env:     WX_SYNTH=/large/work/ref/synth_medusa_heads.pt  (synthetic head ckpt)
         WX_CLIPS=jfk,ls1,ls2,ls3,ls4
"""
import json
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
MAX_NEW = int(os.environ.get("WX_MAXNEW", "160"))
K = int(os.environ.get("WX_K", "5"))
BLOCK_SIZE = 32
MAX_BLOCKS = 16  # 16*32 = 512 >= max target positions
CLIPS = os.environ.get("WX_CLIPS", "jfk,ls1,ls2,ls3,ls4").split(",")
SYNTH_PATH = os.environ.get("WX_SYNTH", os.path.join(REF, "synth_medusa_heads.pt"))


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
        os.environ.setdefault("MASTER_PORT", os.environ.get("WX_PORT", "29581"))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return 0, 1


class SpecMeta:
    """Minimal SpecDecodeMetadata stand-in (matches the runner-built fields)."""
    def __init__(self, K, device):
        self.logits_indices = torch.arange(K + 1, dtype=torch.long, device=device)
        self.target_logits_indices = torch.arange(K, dtype=torch.long, device=device)
        self.bonus_logits_indices = torch.tensor([K], dtype=torch.long, device=device)
        self.cu_num_draft_tokens = torch.tensor([K], dtype=torch.int32, device=device)
        self.draft_token_ids = None
        self.num_draft_tokens = [K]


def build_attn_metadata(n_layers, positions_list, block_size, max_blocks, device,
                        is_prefill):
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


def build_model(init, device, world):
    """Construct the plugin model with Medusa heads in `init` mode.
    init in {zero, random, load}. For load, WX_SYNTH is the checkpoint."""
    from transformers import AutoConfig
    from whisper_pkg.config import WhisperConfig
    from whisper_pkg.model_bf16 import WhisperForConditionalGeneration
    from vllm_neuron.model.neuron_config import NeuronConfig

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    nc = NeuronConfig(on_device_sampling_config=None)  # host rejection (Lever B)
    wc = WhisperConfig.from_configs(hf_cfg, nc)
    med = {"num_heads": K, "medusa_num_layers": 1, "init": init, "seed": 0}
    if init == "load":
        med["heads_path"] = SYNTH_PATH
    wc.medusa_config = med

    torch.manual_seed(0)
    with torch.device("meta"):
        model = WhisperForConditionalGeneration(wc)
    model.load_weights(MODEL_DIR, torch.device("cpu"), None)
    model = model.to(device)
    model.eval()
    model_fn = torch.compile(model, backend="vllm_neuron", fullgraph=True)
    model.visual = torch.compile(model.visual, backend="vllm_neuron", fullgraph=True)
    return model, model_fn, wc


def main():
    rank, world = _dist_setup()

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    from vllm_neuron.vllm.sample.rejection_sampler import (
        RejectionSampler,
        PLACEHOLDER_TOKEN_ID,
    )
    from medusa import MedusaProposer  # noqa: F401  (contract check)

    device = "neuron:0"

    def rejection_greedy(verify_logits_cpu, drafts):
        k = len(drafts)
        target_argmax = verify_logits_cpu[:k].argmax(dim=-1)
        bonus = int(verify_logits_cpu[k].argmax().item())
        out = torch.empty((1, k + 1), dtype=torch.int32)
        out.fill_(PLACEHOLDER_TOKEN_ID)
        RejectionSampler._rejection_greedy_sample(
            out, torch.tensor([k], dtype=torch.int32),
            torch.tensor(drafts, dtype=torch.int32),
            target_argmax, torch.tensor([bonus], dtype=torch.int32),
            None, k, 1,
        )
        accepted = [t for t in out[0].tolist() if t != PLACEHOLDER_TOKEN_ID]
        return accepted, target_argmax.tolist(), bonus

    def content(g):
        return g[:-1] if g and g[-1] == EOS else g

    results = {}
    gate_pass = True

    # Reuse ONE model per mode across all clips (re-run encoder per clip resets
    # cross-KV). Build modes in a fixed order.
    for mode in ("zero", "random", "load"):
        if mode == "load" and not os.path.exists(SYNTH_PATH):
            log(f"[r{rank}] SKIP mode=load (no synth ckpt at {SYNTH_PATH})")
            continue
        log(f"\n[r{rank}] ===== HEAD MODE: {mode} =====")
        model, model_fn, wc = build_model(mode, device, world)
        n_layers = wc.decoder_layers
        kv = alloc_self_kv(model, n_layers, MAX_BLOCKS, BLOCK_SIZE, device)
        proposer = MedusaProposer(_FakeVllmCfg(K), device, on_device_sampling=False)
        proposer.set_target_model(model)
        acc_rep = model._medusa_load_report
        log(f"[r{rank}] mode={mode} heads_loaded={model.medusa_heads is not None} "
            f"load_report={acc_rep}")

        def prefill():
            ids = torch.tensor(SOT, dtype=torch.long, device=device)
            pos = torch.arange(len(SOT), dtype=torch.long, device=device)
            am = build_attn_metadata(n_layers, list(range(len(SOT))), BLOCK_SIZE,
                                     MAX_BLOCKS, device, is_prefill=True)
            with torch.no_grad():
                logits = model_fn(ids, pos, attn_metadata=am,
                                  sampling_positions=torch.tensor(
                                      [len(SOT) - 1], dtype=torch.long, device=device))
            return int(torch.argmax(logits[0].float().cpu()).item())

        def greedy_step(anchor_tok, base_pos):
            ids = torch.tensor([anchor_tok], dtype=torch.long, device=device)
            pos = torch.tensor([base_pos], dtype=torch.long, device=device)
            am = build_attn_metadata(n_layers, [base_pos], BLOCK_SIZE, MAX_BLOCKS,
                                     device, is_prefill=False)
            with torch.no_grad():
                logits = model_fn(ids, pos, attn_metadata=am,
                                  sampling_positions=torch.tensor(
                                      [0], dtype=torch.long, device=device))
            return int(torch.argmax(logits[0].float().cpu()).item())

        def run_greedy():
            first = prefill()
            gen = [first]
            pos = len(SOT)
            cur = first
            while cur != EOS and len(gen) < MAX_NEW:
                nxt = greedy_step(cur, pos)
                gen.append(nxt)
                pos += 1
                cur = nxt
            return gen

        def verify_step(window, base_pos):
            ids = torch.tensor(window, dtype=torch.long, device=device)
            positions = torch.arange(base_pos, base_pos + K + 1, dtype=torch.long, device=device)
            am = build_attn_metadata(n_layers, list(range(base_pos, base_pos + K + 1)),
                                     BLOCK_SIZE, MAX_BLOCKS, device, is_prefill=False)
            sm = SpecMeta(K, device)
            with torch.no_grad():
                out = model_fn(ids, positions, attn_metadata=am,
                               sampling_positions=sm.logits_indices,
                               spec_decode_metadata=sm)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError(f"verify NEFF did not return 3-tuple: {type(out)}")
            verify_logits, last_hidden, drafts = out[0], out[1], out[2]
            return (verify_logits.float().cpu(), last_hidden, drafts.to(torch.int32).cpu())

        def run_medusa():
            first = prefill()
            gen = [first]
            accept_counts = []
            pos = len(SOT)
            anchor = first
            drafts = [0] * K
            while anchor != EOS and len(gen) < MAX_NEW:
                window = [anchor] + list(drafts)
                verify_logits_cpu, last_hidden, next_drafts = verify_step(window, pos)
                accepted, targ, bonus = rejection_greedy(verify_logits_cpu, list(drafts))
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
            return gen, accept_counts

        mode_res = {}
        for clip in CLIPS:
            clip = clip.strip()
            mel = np.load(os.path.join(REF, f"{clip}_mel.npy"))
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)
            model.embed_multimodal(input_features=mel_t)  # reset cross-KV for this clip
            greedy = run_greedy()
            med, accepts = run_medusa()
            match = content(med) == content(greedy)
            mean_acc = float(np.mean(accepts)) if accepts else 0.0
            max_acc = int(np.max(accepts)) if accepts else 0
            n_multi = int(sum(1 for c in accepts if c > 1))
            term = (med[-1] == EOS) if med else False
            if not match:
                gate_pass = False
            log(f"[r{rank}] mode={mode} clip={clip} byte_identical={match} "
                f"iters={len(accepts)} mean_acc/iter={mean_acc:.3f} max_acc={max_acc} "
                f"n_multi_accept_iters={n_multi} term_eos={term} "
                f"greedy_len={len(greedy)} medusa_len={len(med)}")
            mode_res[clip] = {
                "byte_identical": bool(match),
                "iters": len(accepts),
                "mean_accepted_per_iter": mean_acc,
                "max_accepted": max_acc,
                "n_multi_accept_iters": n_multi,
                "terminated_at_eos": bool(term),
                "greedy_len": len(greedy),
                "medusa_len": len(med),
                "accept_counts": accepts,
            }
        results[mode] = mode_res
        del model, model_fn
        import gc
        gc.collect()

    if rank == 0:
        n_clips = len(CLIPS)
        summary = {"tp": world, "K": K, "clips": CLIPS, "modes": {}}
        for mode, mr in results.items():
            n_pass = sum(1 for c in mr.values() if c["byte_identical"])
            accept_exercised = any(c["n_multi_accept_iters"] > 0 for c in mr.values())
            total_mean = float(np.mean([c["mean_accepted_per_iter"] for c in mr.values()]))
            summary["modes"][mode] = {
                "byte_identical": f"{n_pass}/{n_clips}",
                "accept_path_exercised": accept_exercised,
                "mean_accepted_per_iter_over_clips": total_mean,
            }
        summary["gate_pass"] = bool(gate_pass) and all(
            sum(1 for c in mr.values() if c["byte_identical"]) == n_clips
            for mr in results.values()
        )
        summary["detail"] = results
        outp = os.path.join(REF, f"task020_m3_gate_tp{world}.json")
        with open(outp, "w") as f:
            json.dump(summary, f, indent=2)
        log("\n[r0] ===== M3 CORRECTNESS GATE SUMMARY =====")
        for mode, s in summary["modes"].items():
            log(f"[r0] mode={mode}: byte_identical={s['byte_identical']} "
                f"accept_path_exercised={s['accept_path_exercised']} "
                f"mean_acc/iter={s['mean_accepted_per_iter_over_clips']:.3f}")
        log(f"[r0] GATE_PASS={summary['gate_pass']}  (wrote {outp})")
        log("SUCCESS" if summary["gate_pass"] else "FAIL")


class _FakeVllmCfg:
    def __init__(self, K):
        class _Spec:
            method = "medusa"
            num_speculative_tokens = K
        self.speculative_config = _Spec()


if __name__ == "__main__":
    main()
