# SPDX-License-Identifier: Apache-2.0
"""
Phase 1b: ENCODER correctness at TP=4 LNC=2 on the vllm_neuron native backend.

Launched via torchrun --nproc_per_node=4. Each rank pins one logical core
(NEURON_RT_VISIBLE_CORES=<rank>) on the single trn2.3xlarge chip, and uses the
plugin's init_neuron_distributed_environment to set up TP=4. CPL/RPL then shard
the projections and all-reduce across the 4 cores.

Only rank 0 compares to the reference (all ranks produce the same gathered
output because out_proj is RowParallel all-reduce, giving full [.,.,1280]).
"""
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


def load_hf_single(model_dir):
    from safetensors.torch import load_file
    return load_file(os.path.join(model_dir, "model.safetensors"))


def build_encoder_state(hf_sd):
    from weight_loaders import build_state
    return build_state(hf_sd, "model.encoder.", DTYPE)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    # Pin this rank to logical core == its rank index on the single chip.
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(local_rank)

    # Importing vllm_neuron runs _init_backend() which registers the "neuron"
    # privateuse1 backend + torch.compile backend.
    import vllm_neuron  # noqa: F401
    import torch.distributed as dist

    # Plain gloo process group. CPL/RPL default to dist.group.WORLD, so a
    # world_size=4 group gives them tp_size=4 with all_reduce across the 4
    # logical cores -- no need for vLLM's init_distributed_environment (which
    # requires a fully-registered accelerator the direct harness doesn't set up).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    # Neuron collectives (CCOM) rendezvous: rank 0 picks a free port, broadcasts
    # NEURON_RT_ROOT_COMM_ID to all ranks. Required before the first NEFF that
    # contains an all-reduce (RowParallelLinear). Mirrors the plugin worker's
    # rendezvous_ccom_bootstrap().
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()

    from whisper_neuron import WhisperEncoder, WhisperConfigLite
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg = WhisperConfigLite(hf_cfg)

    hf_sd = load_hf_single(MODEL_DIR)
    enc_sd = build_encoder_state(hf_sd)

    enc = WhisperEncoder(cfg, DTYPE)
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if rank == 0:
        print(f"[r{rank}] MISSING={len(missing)} UNEXPECTED={len(unexpected)}", flush=True)
    enc.eval()

    device = "neuron:0"
    enc = enc.to(device)

    mel = np.load(os.path.join(REF, "mel_ref.npy"))
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE).to(device)

    fn = torch.compile(enc, backend="vllm_neuron", fullgraph=True)

    t0 = time.time()
    with torch.no_grad():
        out = fn(mel_t)
    if rank == 0:
        print(f"[r{rank}] FORWARD_SEC {time.time()-t0:.2f}", flush=True)

    if rank == 0:
        out_cpu = out.to("cpu").float().squeeze(0).numpy()
        ref = np.load(os.path.join(REF, "enc_hidden_ref.npy"))
        a = out_cpu.reshape(-1)
        b = ref.reshape(-1)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print(f"[r{rank}] out mean {out_cpu.mean():.6f} std {out_cpu.std():.6f}", flush=True)
        print(f"ENC_TP4_COS_SIM {cos:.6f}", flush=True)
        print("SUCCESS" if cos > 0.99 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
