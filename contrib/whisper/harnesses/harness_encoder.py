# SPDX-License-Identifier: Apache-2.0
"""
Phase 1 harness: ENCODER correctness at TP=1 on the vllm_neuron native backend.

Loads openai/whisper-large-v3 HF weights, builds WhisperEncoder, compiles with
torch.compile(backend="vllm_neuron"), feeds mel_ref.npy, compares to
enc_hidden_ref.npy (OpenAI CPU reference). SUCCESS = cos-sim > 0.99.
"""
import os
import sys
import time
# contrib layout: whisper_neuron.py / weight_loaders.py live one dir up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else None
REF = "/large/work/ref"
DTYPE = torch.float32 if os.environ.get("WX_CPU", "0") == "1" else torch.bfloat16
COMPILE = os.environ.get("WX_COMPILE", "1") == "1"


def load_hf_state():
    from safetensors.torch import load_file
    import glob
    # Prefer the canonical model.safetensors (bf16/fp16); skip fp32 shards.
    single = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    files = sorted(
        f for f in glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))
        if "fp32" not in os.path.basename(f)
    )
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def build_encoder_state(hf_sd, cfg):
    from weight_loaders import build_state
    return build_state(hf_sd, "model.encoder.", DTYPE)


def main():
    import torch.distributed as dist
    from transformers import AutoConfig
    from whisper_neuron import WhisperEncoder, WhisperConfigLite

    # single-rank distributed (needed so CPL/RPL see tp_size=1 cleanly; harmless)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg = WhisperConfigLite(hf_cfg)  # asserts dims (Footgun 3)
    print(f"CONFIG OK d_model={cfg.d_model} vocab={cfg.vocab_size} "
          f"n_heads={cfg.encoder_attention_heads} enc_layers={cfg.encoder_layers}",
          flush=True)

    hf_sd = load_hf_state()
    enc_sd = build_encoder_state(hf_sd, cfg)

    enc = WhisperEncoder(cfg, DTYPE)
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    # embed_positions.weight is sinusoids (buffer-like); HF also stores it, ok to load
    print(f"MISSING={len(missing)} first={missing[:5]}", flush=True)
    print(f"UNEXPECTED={len(unexpected)} first={unexpected[:5]}", flush=True)
    enc.eval()

    mel = np.load(os.path.join(REF, "mel_ref.npy"))  # [128, 3000]
    print("mel shape", mel.shape, "mean", mel.mean(), "std", mel.std(), flush=True)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(DTYPE)  # [1,128,3000]

    ref = np.load(os.path.join(REF, "enc_hidden_ref.npy"))  # [1500,1280]
    print("ref shape", ref.shape, flush=True)

    device = "cpu" if os.environ.get("WX_CPU", "0") == "1" else "neuron:0"
    enc = enc.to(device)
    mel_dev = mel_t.to(device)

    fn = enc
    if COMPILE and device != "cpu":
        compiler_args = os.environ.get("WX_COMPILER_ARGS", "")
        opts = {}
        if compiler_args:
            opts["compiler_args"] = compiler_args
        print(f"COMPILING with backend=vllm_neuron opts={opts}", flush=True)
        fn = torch.compile(enc, backend="vllm_neuron", options=opts, fullgraph=True)

    t0 = time.time()
    with torch.no_grad():
        out = fn(mel_dev)
    print(f"FORWARD_SEC {time.time()-t0:.2f}", flush=True)

    out_cpu = out.to("cpu").float().squeeze(0).numpy()  # [1500,1280]
    print("out shape", out_cpu.shape, "mean", out_cpu.mean(), "std", out_cpu.std(),
          flush=True)

    a = out_cpu.reshape(-1)
    b = ref.reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    mae = float(np.abs(a - b).mean())
    print(f"ENC_COS_SIM {cos:.6f}", flush=True)
    print(f"ENC_MAE {mae:.6f}", flush=True)
    print("SUCCESS" if cos > 0.99 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
