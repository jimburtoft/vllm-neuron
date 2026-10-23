# SPDX-License-Identifier: Apache-2.0
"""CPU unit-test harness for hy4_preview: no Neuron device, no vLLM needed.

If vLLM is not importable, the few ``vllm`` / ``vllm_neuron`` names the model
imports are stubbed; the plugin's own pure-python modules (``weight_loader``,
``bucket_utils``) are loaded from this checkout. ``hy4_preview`` is this
folder's ``src``. Tests swap the tensor-parallel group through ``set_group``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "src"))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "..", "scripts"))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
PLUGIN = os.path.join(REPO, "vllm_neuron")


class Group:
    """One-rank tensor-parallel group (collectives are the identity)."""

    world_size = 1
    rank_in_group = 0

    def all_reduce(self, t):
        return t

    def all_gather(self, t, dim=0):
        return torch.cat([t] * self.world_size, dim=dim)


_GROUP = [Group()]


def get_tp_group():
    return _GROUP[0]


def set_group(group) -> None:
    _GROUP[0] = group


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _paged_slot_indices(slot_mapping, block_size, num_blocks):
    """``vllm_neuron.model.kv_cache.paged_slot_indices`` (same contract)."""
    max_slot = num_blocks * block_size
    safe = torch.where(
        (slot_mapping < 0) | (slot_mapping >= max_slot),
        torch.zeros_like(slot_mapping),
        slot_mapping,
    )
    return safe // block_size, safe % block_size


def _install_stubs() -> None:
    try:
        import vllm  # noqa: F401

        return  # a real install: the tests patch get_tp_group on the modules
    except ImportError:
        pass
    _stub("vllm")
    _stub("vllm.distributed")
    _stub("vllm.distributed.parallel_state", get_tp_group=get_tp_group)
    vn = _stub("vllm_neuron")
    vn.__path__ = [PLUGIN]
    _stub("vllm_neuron.nn")
    _stub("vllm_neuron.nn.embedding", VocabDimShardedEmbedding=object)
    _stub("vllm_neuron.nn.sampler", Sampler=object)
    _stub("vllm_neuron.model")
    _stub(
        "vllm_neuron.model.kv_cache",
        KVSpec=object,
        LayerSpec=object,
        paged_slot_indices=_paged_slot_indices,
    )
    _stub("vllm_neuron.utils").__path__ = [os.path.join(PLUGIN, "utils")]
    _stub("vllm_neuron.utils.checkpoints", SafetensorsCheckpoint=object)
    _load(
        "vllm_neuron.utils.weight_loader",
        os.path.join(PLUGIN, "utils", "weight_loader.py"),
    )
    _load(
        "vllm_neuron.utils.bucket_utils",
        os.path.join(PLUGIN, "utils", "bucket_utils.py"),
    )


_install_stubs()
if "hy4_preview" not in sys.modules:
    # The package without its __init__ (which registers with vLLM).
    _stub("hy4_preview").__path__ = [SRC]

model = importlib.import_module("hy4_preview.model")
moe = importlib.import_module("hy4_preview.moe")
mx_experts = importlib.import_module("hy4_preview.mx_experts")
for _mod in (model, moe):
    _mod.get_tp_group = get_tp_group
