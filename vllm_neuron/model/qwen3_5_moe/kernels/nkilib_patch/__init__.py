# SPDX-License-Identifier: Apache-2.0
"""Optional vendored nkilib MoE CTE (shard_on_block) with optimization fixes.

ISAAC_BWMM_PATCH=1 swaps nkilib.core.moe.moe_cte.moe_cte.bwmm_shard_on_block (resolved from module globals when
the @nki.jit moe_cte traces) for bwmm_shard_on_block_isaac.bwmm_shard_on_block. Default off."""
import os


def install():
    if os.environ.get("ISAAC_BWMM_PATCH", "0") != "1":
        return False
    import importlib
    import sys

    importlib.import_module("nkilib.core.moe.moe_cte.moe_cte")
    M = sys.modules["nkilib.core.moe.moe_cte.moe_cte"]  # the package re-exports the Kernel under the same name

    from .bwmm_shard_on_block_isaac import bwmm_shard_on_block

    M.bwmm_shard_on_block = bwmm_shard_on_block
    return True
