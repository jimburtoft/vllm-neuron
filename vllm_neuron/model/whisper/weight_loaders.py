# SPDX-License-Identifier: Apache-2.0
"""Load the HF ``openai/whisper-large-v3`` safetensors into the native model.

The loader emits FULL (unsharded) tensors. ``ColumnParallelLinear`` and
``RowParallelLinear`` shard a full tensor in ``_load_from_state_dict`` when its
shape is different from the local shape. Thus this loader does not shard.

``RowParallelLinearFP32Bias`` (``out_proj`` and ``mlp.fc2``) keeps the bias in
fp32 and stores the weight as ``<name>.rpl.weight``. The key remap below adds
the ``.rpl`` level.
"""

import glob
import os
import re

import torch
from safetensors.torch import load_file

from vllm_neuron.utils.checkpoints import _get_checkpoint_source


def resolve_checkpoint_dir(model_name_or_path: str, cache_dir: str | None) -> str:
    """Return a local directory that holds the Whisper safetensors files.

    Accept a local directory or an HF repo id. For a repo id, download the
    files to the HF cache and return the snapshot directory.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    source = _get_checkpoint_source(model_name_or_path, ".safetensors", cache_dir)
    file_names = source.get_file_names()
    if not file_names:
        raise FileNotFoundError(
            f"No .safetensors file found for {model_name_or_path}. "
            "Use a Whisper checkpoint in safetensors format."
        )
    dirs = set()
    for fn in file_names:
        source.download_file(fn)
        dirs.add(os.path.dirname(source.get_file_path(fn)))
    if len(dirs) != 1:
        raise RuntimeError(
            f"The safetensors files are in more than one directory: {dirs}."
        )
    return next(iter(dirs))


def load_hf_state(model_dir: str) -> dict[str, torch.Tensor]:
    """Load the Whisper HF state dict from a local directory."""
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


_RPL_WEIGHT = re.compile(r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.weight$")
_RPL_BIAS = re.compile(r"(self_attn\.out_proj|encoder_attn\.out_proj|mlp\.fc2)\.bias$")


def remap_key(key: str) -> str:
    """Change an HF encoder or decoder key to the module path of the model.

    * ``layers.N.fc1`` and ``layers.N.fc2`` become ``layers.N.mlp.fc1`` and
      ``layers.N.mlp.fc2``.
    * The weight of a row-parallel layer with an fp32 bias moves to
      ``<name>.rpl.weight``.

    Example:
        >>> remap_key("layers.3.fc2.weight")
        'layers.3.mlp.fc2.rpl.weight'
        >>> remap_key("layers.0.self_attn.q_proj.bias")
        'layers.0.self_attn.q_proj.bias'
    """
    key = re.sub(r"(layers\.\d+)\.(fc1|fc2)\.", r"\1.mlp.\2.", key)
    return _RPL_WEIGHT.sub(r"\1.rpl.weight", key)


def build_state(
    hf_sd: dict[str, torch.Tensor], prefix: str, weight_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Extract and remap the sub-state-dict that starts with ``prefix``.

    ``prefix`` is ``"model.encoder."`` or ``"model.decoder."``. Cast each tensor
    to ``weight_dtype``. Keep the row-parallel biases in fp32.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in hf_sd.items():
        if not k.startswith(prefix):
            continue
        nk = remap_key(k[len(prefix) :])
        is_rpl_bias = bool(_RPL_BIAS.search(nk))
        out[nk] = v.to(torch.float32 if is_rpl_bias else weight_dtype)
    return out
