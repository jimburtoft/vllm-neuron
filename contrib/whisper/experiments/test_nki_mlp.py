# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness test for nki_decode_mlp: NKI kernel vs plain-torch reference.

Runs under nki.simulate (CPU) for fast iteration, and optionally on device.
Usage:
    python test_nki_mlp.py sim      # nki.simulate
    python test_nki_mlp.py device   # on Neuron device
"""
import sys
import math
import numpy as np
import torch

import nki
from nki_decode_mlp import decode_mlp

D = 1280
I = 5120


def torch_ref(x, fc1_w, fc1_b, fc2_w, fc2_b):
    xf = x.float()
    h = xf @ fc1_w.float().t() + fc1_b.float()
    g = h * 0.5 * (1.0 + torch.erf(h / math.sqrt(2.0)))
    y = g @ fc2_w.float().t() + fc2_b.float()
    return y


def make_inputs(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(1, D, dtype=torch.bfloat16) * 0.1
    fc1_w = (torch.randn(I, D, dtype=torch.bfloat16) * 0.02)
    fc1_b = (torch.randn(1, I, dtype=torch.bfloat16) * 0.02)
    fc2_w = (torch.randn(D, I, dtype=torch.bfloat16) * 0.02)
    fc2_b = (torch.randn(1, D, dtype=torch.float32) * 0.02)
    return x, fc1_w, fc1_b, fc2_w, fc2_b


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def run(mode):
    x, fc1_w, fc1_b, fc2_w, fc2_b = make_inputs()
    ref = torch_ref(x, fc1_w, fc1_b, fc2_w, fc2_b)  # fp32 reference

    args = [x, fc1_w, fc1_b, fc2_w, fc2_b]
    if mode == "sim":
        np_args = [a.float().numpy().astype(np.float32) if a.dtype == torch.float32
                   else a.view(torch.int16).numpy() for a in args]
        # nki.simulate works with numpy; feed bf16 via ml_dtypes if available
        import ml_dtypes
        sim_args = []
        for a in args:
            if a.dtype == torch.bfloat16:
                sim_args.append(a.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
            else:
                sim_args.append(a.numpy())
        out = nki.simulate(decode_mlp)(*sim_args)
        out_t = torch.from_numpy(out.astype(np.float32))
    else:
        # Standalone NKI device execution: passing np.ndarray runs on the
        # NeuronDevice without a framework (torch_neuronx not present in this
        # vllm-neuron container). LNC=2 default on trn2.3xlarge.
        import ml_dtypes
        dev_args = []
        for a in args:
            if a.dtype == torch.bfloat16:
                dev_args.append(a.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
            else:
                dev_args.append(a.numpy())
        import os
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
        out = decode_mlp[2](*dev_args)
        out_t = torch.from_numpy(out.astype(np.float32))

    md = (out_t - ref).abs().max().item()
    cs = cos_sim(out_t, ref)
    rel = (out_t - ref).norm().item() / (ref.norm().item() + 1e-12)
    print(f"[{mode}] max_abs_diff={md:.6f}  cos_sim={cs:.8f}  rel_l2={rel:.6f}")
    print(f"       ref[:5]={ref.flatten()[:5].tolist()}")
    print(f"       out[:5]={out_t.flatten()[:5].tolist()}")
    ok = cs >= 0.9999
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    ok = run(mode)
    sys.exit(0 if ok else 1)
