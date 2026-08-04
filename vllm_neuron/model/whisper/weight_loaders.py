# SPDX-License-Identifier: Apache-2.0
"""Weight loaders: HF ``openai/whisper-large-v3`` safetensors -> native-plugin
Whisper modules.

Ported from contrib/whisper/weight_loaders.py (byte-identical remap logic) and
extended with:

  * HF-name / local-dir checkpoint resolution (reuses the plugin's
    ``_get_checkpoint_source`` so ``openai/whisper-large-v3`` resolves the same
    way it does for llama3/qwen3_vl).
  * per-module state-dict routing (encoder., decoder.) into the plugin module
    tree. TP sharding is delegated to ColumnParallelLinear / RowParallelLinear
    ``_load_from_state_dict`` (they shard the FULL weight per-rank when the
    loaded tensor shape != the local sharded shape), so this loader emits FULL
    (unsharded) tensors and never shards itself.

The RowParallelLinearFP32Bias wrapper (out_proj / mlp.fc2) keeps its bias in
fp32 (RowParallelLinear rejects non-fp32 bias at tp>1) and moves the wrapped
weight under ``<name>.rpl.weight`` -- the remap below rewrites those keys.
"""
import glob
import os
import re

import torch
from safetensors.torch import load_file

from vllm_neuron.utils.checkpoints import _get_checkpoint_source


def resolve_checkpoint_dir(model_name_or_path: str, cache_dir: str | None) -> str:
    """Return a local directory that contains the whisper .safetensors files.

    Works for both a local directory and an HF repo id (downloads/uses the HF
    cache). Mirrors how SafetensorsCheckpoint resolves files, but returns the
    directory so we can use the contrib raw-safetensors loader unchanged.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    source = _get_checkpoint_source(model_name_or_path, ".safetensors", cache_dir)
    file_names = source.get_file_names()
    assert len(file_names) > 0, (
        f"No .safetensors files found for {model_name_or_path}"
    )
    # Ensure every shard is present in the HF cache, then return its dir.
    dirs = set()
    for fn in file_names:
        source.download_file(fn)
        path = source.get_file_path(fn)
        dirs.add(os.path.dirname(path))
    # All shards of one repo live in the same snapshot dir.
    assert len(dirs) == 1, f"safetensors span multiple dirs: {dirs}"
    return next(iter(dirs))


def load_hf_state(model_dir: str) -> dict[str, torch.Tensor]:
    """Load the whisper HF safetensors state dict from a local directory."""
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    files = sorted(
        f
        for f in glob.glob(os.path.join(model_dir, "*.safetensors"))
        if "fp32" not in os.path.basename(f)
    )
    sd: dict[str, torch.Tensor] = {}
    for f in files:
        sd.update(load_file(f))
    return sd


def _remap_key(nk: str) -> str:
    """Remap an HF encoder/decoder-relative key to the plugin module path.

    (contrib weight_loaders.py:25-32, byte-identical)
      * HF MLP: layers.N.fc1/fc2 -> layers.N.mlp.fc1/fc2
      * RowParallelLinearFP32Bias wraps the RPL: <name>.weight -> <name>.rpl.weight
        (out_proj on self_attn + encoder_attn, and mlp.fc2)
    """
    nk = re.sub(r"(layers\.\d+)\.(fc1|fc2)\.", r"\1.mlp.\2.", nk)
    nk = re.sub(
        r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.weight$",
        r"\1.rpl.weight",
        nk,
    )
    return nk


def build_state(
    hf_sd: dict[str, torch.Tensor], prefix: str, weight_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Extract + remap the ``prefix`` (encoder./decoder.) sub-state-dict.

    ``prefix`` in {'model.encoder.', 'model.decoder.'}. Weights are cast to
    ``weight_dtype``; RowParallel fp32 biases (out_proj / fc2 bias) stay fp32.
    (contrib weight_loaders.py:35-48, byte-identical.)
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in hf_sd.items():
        if not k.startswith(prefix):
            continue
        nk = _remap_key(k[len(prefix):])
        is_rpl_bias = bool(
            re.search(
                r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.bias$", nk
            )
        )
        out[nk] = v.to(torch.float32 if is_rpl_bias else weight_dtype)
    return out
