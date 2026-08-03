# SPDX-License-Identifier: Apache-2.0
"""Shared weight loaders: HF whisper-large-v3 safetensors -> WhisperNeuron modules."""
import glob
import os
import re

import torch
from safetensors.torch import load_file


def load_hf_state(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    files = sorted(
        f for f in glob.glob(os.path.join(model_dir, "*.safetensors"))
        if "fp32" not in os.path.basename(f)
    )
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def _remap_key(nk):
    # HF encoder/decoder MLP: layers.N.fc1/fc2 -> layers.N.mlp.fc1/fc2
    nk = re.sub(r"(layers\.\d+)\.(fc1|fc2)\.", r"\1.mlp.\2.", nk)
    # RowParallelLinearFP32Bias wraps the RPL: <name>.weight -> <name>.rpl.weight
    #   applies to out_proj (self_attn + encoder_attn) and mlp.fc2
    nk = re.sub(r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.weight$",
                r"\1.rpl.weight", nk)
    return nk


def build_state(hf_sd, prefix, weight_dtype):
    """prefix in {'model.encoder.', 'model.decoder.'}.

    Weights -> weight_dtype; RowParallel fp32 biases stay fp32.
    """
    out = {}
    for k, v in hf_sd.items():
        if not k.startswith(prefix):
            continue
        nk = _remap_key(k[len(prefix):])
        # fp32 bias for the wrapped RPL layers (out_proj / fc2 bias)
        is_rpl_bias = bool(re.search(
            r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.bias$", nk))
        out[nk] = v.to(torch.float32 if is_rpl_bias else weight_dtype)
    return out
