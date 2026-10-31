# SPDX-License-Identifier: Apache-2.0
"""Read routing and producer fan-in of ``Hy4NixlConnectorWorker`` (CPU, no NIXL).

The plugin's ``NeuronNixlConnectorWorker`` maps decode rank -> prefill rank through
``get_total_num_kv_heads()`` (8 for this checkpoint), so at TP64 only every 8th
prefill rank is read and notified. The Hy4 worker routes decode rank ``d`` to
prefill rank ``d`` for equal TP without DCP. These tests drive the real base
methods on a worker built without ``__init__``.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _ensure_module(name, **attrs):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_ensure_module(parent), child, mod)
    for k, v in attrs.items():
        if not hasattr(mod, k):
            setattr(mod, k, v)
    return mod


def _install_nixl_stubs():
    """Stub the vLLM NIXL names the plugin connector imports, if vLLM is absent."""
    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker  # noqa: F401

        return
    except ImportError:
        pass
    import logging

    class _ReadSpec:
        def __init__(self, remote_rank, local_block_ids, remote_block_ids):
            self.remote_rank = remote_rank
            self.local_block_ids = local_block_ids
            self.remote_block_ids = remote_block_ids

    nixl = "vllm.distributed.kv_transfer.kv_connector.v1.nixl"
    _ensure_module(
        nixl + ".metadata",
        NixlAgentMetadata=type("NixlAgentMetadata", (), {}),
        NixlConnectorMetadata=type("NixlConnectorMetadata", (), {}),
        NixlHandshakePayload=type("NixlHandshakePayload", (), {}),
    )
    _ensure_module(nixl + ".connector", NixlConnector=type("NixlConnector", (), {}))
    _ensure_module(
        nixl + ".scheduler",
        NixlConnectorScheduler=type("NixlConnectorScheduler", (), {}),
    )
    _ensure_module(
        nixl + ".worker", NixlConnectorWorker=type("NixlConnectorWorker", (), {})
    )
    _ensure_module(
        nixl + ".tp_mapping", ReadSpec=_ReadSpec, TPMapping=type("TPMapping", (), {})
    )
    _ensure_module("vllm.logger", init_logger=logging.getLogger)


@pytest.fixture(scope="module")
def mods():
    _install_nixl_stubs()
    base = importlib.import_module(
        "vllm_neuron.vllm.kv_connector.neuron_nixl_connector"
    )
    hy4 = importlib.import_module("hy4_preview.kv_connector")
    return base, hy4


TP = 64
CKPT_KV_HEADS = 8  # config.json num_key_value_heads
EID = "prefill-engine"


def _worker(cls, tp_rank=0, p_tp=TP, d_tp=TP, p_dcp=1, d_dcp=1, head_ratio=1.0):
    w = object.__new__(cls)
    model_config = SimpleNamespace(get_total_num_kv_heads=lambda: CKPT_KV_HEADS)
    w.vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    w.world_size = d_tp
    w.tp_rank = tp_rank
    w.block_size = 64
    w._local_dcp_size = d_dcp
    w._local_dcp_rank = tp_rank % d_dcp
    w._remote_tp_size = {EID: p_tp}
    w._remote_dcp = {EID: p_dcp}
    w._head_ratio = {EID: head_ratio}
    w._remote_prefill_bs = {EID: 64}
    w._producer_fanin = {}
    w.transfer_topo = SimpleNamespace(
        get_engine_info=lambda eid: SimpleNamespace(remote_block_size=64)
    )
    w.src_xfer_handles_by_block_size = {64: "local"}
    w.dst_xfer_side_handles = {EID: {r: f"remote{r}" for r in range(p_tp)}}
    w._recving_metadata = {}
    w._recving_transfers = {}
    reads = []

    def _read_blocks(
        read_spec,
        dst_engine_id,
        request_id,
        remote_request_id,
        local_xfer_side_handle,
        remote_xfer_side_handle,
        expected_consumers,
    ):
        reads.append(
            (read_spec.remote_rank, remote_xfer_side_handle, expected_consumers)
        )

    w._read_blocks = _read_blocks
    return w, reads


def _meta(num_blocks=3):
    # num_blocks=0: a full prefix-cache hit, no local groups (notif-only read).
    return SimpleNamespace(
        local_physical_block_ids=[list(range(num_blocks))] if num_blocks else [],
        remote=SimpleNamespace(
            engine_id=EID,
            request_id="req-0",
            block_ids=[list(range(100, 100 + num_blocks))],
        ),
    )


def test_base_routes_eight_to_one(mods):
    base, _ = mods
    w, _ = _worker(base.NeuronNixlConnectorWorker)
    fanin = w._producer_fanin_map(EID)
    assert fanin == {r: 8 for r in range(0, TP, 8)}
    targets = set()
    for d in range(TP):
        wd, reads = _worker(base.NeuronNixlConnectorWorker, tp_rank=d)
        wd._read_blocks_for_req("req", _meta())
        targets.update(r for r, _, _ in reads)
    assert targets == set(range(0, TP, 8))


def test_hy4_fanin_identity(mods):
    _, hy4 = mods
    w, _ = _worker(hy4.Hy4NixlConnectorWorker)
    real = w.vllm_config
    assert w._producer_fanin_map(EID) == {r: 1 for r in range(TP)}
    assert w.vllm_config is real
    assert w.vllm_config.model_config.get_total_num_kv_heads() == CKPT_KV_HEADS


@pytest.mark.parametrize("num_blocks", [3, 0])
def test_hy4_read_routing_identity(mods, num_blocks):
    _, hy4 = mods
    for d in range(TP):
        w, reads = _worker(hy4.Hy4NixlConnectorWorker, tp_rank=d)
        real = w.vllm_config
        w._read_blocks_for_req("req", _meta(num_blocks))
        assert reads == [(d, f"remote{d}", 1)]
        assert w.vllm_config is real


def test_hy4_other_topologies_keep_base_routing(mods):
    base, hy4 = mods
    # Prefill TP 32 -> decode TP 64: not one-to-one, base routing unchanged.
    w_b, _ = _worker(base.NeuronNixlConnectorWorker, p_tp=32)
    w_h, _ = _worker(hy4.Hy4NixlConnectorWorker, p_tp=32)
    assert w_h._producer_fanin_map(EID) == w_b._producer_fanin_map(EID)
