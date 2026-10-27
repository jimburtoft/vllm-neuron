# SPDX-License-Identifier: Apache-2.0
"""How GPT-OSS FP8 runs prefill: from the FP8 expert weights, or from a BF16 copy.

With ``quantization="fp8"`` the expert weights are FP8 for decode. Prefill can either use
the same FP8 weights (``"fp8"``) or a BF16 copy of the experts (``"bf16_copy"``). The copy
costs about 5 GiB of HBM per logical core and halves the KV cache, and makes the platform
cap ``gpu_memory_utilization`` at 0.80. It used to be faster because the NKI Library's
``bwmm_shard_on_I`` upcast every FP8 weight tile to BF16 before the matmul.

Set it with ``additional_config={"neuron_config": {"gpt_oss_fp8_prefill": ...}}``:

``"auto"`` (default)
    ``"fp8"`` when the installed NKI Library prefill kernel uses FP8 weights directly
    (``bwmm_shard_on_I.FP8_DIRECT_MATMUL``) and reads the gate/up-major layout that the
    decode kernel uses (``bwmm_shard_on_I.SUPPORTS_GATE_UP_MAJOR``), and the per-rank
    expert intermediate size is at least ``MIN_FP8_PREFILL_INTERMEDIATE_PER_RANK`` (720:
    GPT-OSS at TP <= 4); ``"bf16_copy"`` otherwise (for example the bundled NKI Library,
    or TP=8). With the FP8 weights, the prefill MoE kernel does not compile at
    TP=8 (per-rank intermediate 360), so ``auto`` keeps the copy there.
``"fp8"``
    No copy: 2x the KV cache, same prefill and decode speed with a capable NKI Library.
``"bf16_copy"``
    Keep the BF16 prefill copy.

The legacy ``GPTOSS_HYBRID_PREFILL`` environment variable ("1" = copy, "0" = no copy) is
still honoured when the config key is not set.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

CONFIG_KEY = "gpt_oss_fp8_prefill"
_VALID = ("auto", "fp8", "bf16_copy")
# Smallest per-rank expert intermediate size for which "auto" runs prefill from the FP8
# weights. Measured: 720 (TP=4) compiles and runs; 360 (TP=8) fails to compile in the
# shard_on_i_hybrid prefill kernel on trn2 and trn3.
MIN_FP8_PREFILL_INTERMEDIATE_PER_RANK = 720


def nkilib_supports_fp8_prefill() -> bool:
    """True if the installed bwmm_shard_on_I runs per-channel FP8 at BF16 speed and
    accepts the gate/up-major weight layout."""
    try:
        from nkilib.core.moe.moe_cte import bwmm_shard_on_I as _bsi
    except Exception:
        return False
    return bool(
        getattr(_bsi, "FP8_DIRECT_MATMUL", False)
        and getattr(_bsi, "SUPPORTS_GATE_UP_MAJOR", False)
    )


def keep_bf16_prefill_copy(
    neuron_config, intermediate_size_per_rank: int | None = None
) -> bool:
    """Resolve whether GPT-OSS FP8 keeps a BF16 copy of the experts for prefill.

    Args:
        neuron_config: a dict (``additional_config["neuron_config"]``) or a NeuronConfig.
        intermediate_size_per_rank: expert intermediate size per tensor-parallel rank, if
            known. ``auto`` keeps the copy below ``MIN_FP8_PREFILL_INTERMEDIATE_PER_RANK``.
    """
    if isinstance(neuron_config, dict):
        mode = neuron_config.get(CONFIG_KEY)
    else:
        mode = getattr(neuron_config, CONFIG_KEY, None)
    if mode is None:
        env = os.environ.get("GPTOSS_HYBRID_PREFILL")
        if env in ("0", "1"):
            return env == "1"
        mode = "auto"
    if mode not in _VALID:
        raise ValueError(
            f"neuron_config[{CONFIG_KEY!r}] must be one of {_VALID}, got {mode!r}"
        )
    if mode == "auto":
        if (
            intermediate_size_per_rank is not None
            and intermediate_size_per_rank < MIN_FP8_PREFILL_INTERMEDIATE_PER_RANK
        ):
            return True
        return not nkilib_supports_fp8_prefill()
    return mode == "bf16_copy"
