# SPDX-License-Identifier: Apache-2.0
"""
Task 004 CPU-eager validation of the KV-cache decode algorithm.

Runs the decoder two ways on CPU (fp32) with random-but-fixed encoder output and
confirms they produce IDENTICAL logits/argmax at every step:

  (A) NAIVE Task-003 path: recompute full prefix each step (dec.forward).
  (B) KV-cache Task-004 path: precompute_cross_kv -> prefill -> decode_step loop.

This proves the KV-cache math is correct BEFORE we pay for a device compile.
No Neuron device needed; run in the container or any torch env.
"""
import os
import sys
# contrib layout: whisper_neuron.py / weight_loaders.py live one dir up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

MODEL_DIR = sys.argv[1]
DTYPE = torch.float32


def main():
    import torch.distributed as dist
    from transformers import AutoConfig
    from whisper_neuron import WhisperDecoder, WhisperConfigLite
    from weight_loaders import load_hf_state, build_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29541")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    cfg = WhisperConfigLite(hf_cfg)
    print(f"CONFIG OK d_model={cfg.d_model} dec_layers={cfg.decoder_layers}", flush=True)

    hf_sd = load_hf_state(MODEL_DIR)
    dec = WhisperDecoder(cfg, DTYPE)
    dm, du = dec.load_state_dict(build_state(hf_sd, "model.decoder.", DTYPE), strict=False)
    print(f"DEC MISSING={len(dm)} UNEXPECTED={len(du)}", flush=True)
    dec.eval()

    torch.manual_seed(0)
    # fixed pseudo encoder output [1,1500,1280]
    enc = torch.randn(1, cfg.max_source_positions, cfg.d_model, dtype=DTYPE) * 0.5

    SOT = [50258, 50259, 50360, 50364]
    lm_w = dec.embed_tokens.weight

    def naive_logits(tokens):
        ids = torch.tensor([tokens], dtype=torch.long)
        pos = torch.arange(len(tokens), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            h = dec.forward(ids, pos, enc)
            logits = F.linear(h, lm_w)
        return logits[0, -1]  # last position

    # ---- KV cache path ----
    with torch.no_grad():
        dec.precompute_cross_kv(enc)
        ids = torch.tensor([SOT], dtype=torch.long)
        pos = torch.arange(len(SOT), dtype=torch.long).unsqueeze(0)
        pf_logits = dec.prefill(ids, pos)  # [1, P, vocab]
        kv_last = pf_logits[0, -1]

    naive_last = naive_logits(SOT)
    cos = F.cosine_similarity(kv_last.unsqueeze(0), naive_last.unsqueeze(0)).item()
    mad = (kv_last - naive_last).abs().max().item()
    print(f"PREFILL step P={len(SOT)}: cos={cos:.8f} max_abs_diff={mad:.6f} "
          f"argmax_kv={kv_last.argmax().item()} argmax_naive={naive_last.argmax().item()}",
          flush=True)

    # ---- decode steps ----
    tokens = list(SOT)
    ok = (kv_last.argmax().item() == naive_last.argmax().item())
    for step in range(20):
        # feed argmax of naive as next token (deterministic teacher forcing on naive)
        nxt = naive_last.argmax().item()
        tokens.append(nxt)
        cur_pos = len(tokens) - 1  # index of the just-appended token
        with torch.no_grad():
            input_id = torch.tensor([[nxt]], dtype=torch.long)
            position = torch.tensor([[cur_pos]], dtype=torch.long)
            cur = torch.tensor(cur_pos, dtype=torch.long)
            step_logits = dec.decode_step(input_id, position, cur)  # [1,1,vocab]
        kv_last = step_logits[0, -1]
        naive_last = naive_logits(tokens)
        a_kv = kv_last.argmax().item()
        a_nv = naive_last.argmax().item()
        cos = F.cosine_similarity(kv_last.unsqueeze(0), naive_last.unsqueeze(0)).item()
        mad = (kv_last - naive_last).abs().max().item()
        match = a_kv == a_nv
        ok = ok and match
        print(f"step {step:2d} pos={cur_pos:3d}: cos={cos:.8f} max_abs_diff={mad:.6f} "
              f"argmax_kv={a_kv} argmax_naive={a_nv} {'OK' if match else 'MISMATCH'}",
              flush=True)

    print("VALIDATION", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
