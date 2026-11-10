# SPDX-License-Identifier: Apache-2.0
"""Packed prefill: several whole prompts in one prefill step (opt-in).

``VLLM_NEURON_MAX_PREFILLS_PER_BATCH=N`` (default 1) lets the scheduler admit up
to N waiting prompts into one prefill step. Every prompt is whole (never
chunked) and the step's total tokens fit one compiled prefill bucket. A prefill
step has a large fixed cost, so short prompts share it instead of paying it one
by one.

Layout of a packed step (T = the bucket for the total real token count)::

    [ req0 real | req1 real | ... | reqK real + all padding ]

* Each request keeps its own positions, slot mapping and last-token logit
  index (the runner already builds those per request). All padding goes to the
  last row, so the step needs exactly one bucket.
* Every prefill bucket up to ``VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS``
  (default 2048, rounded down to a bucket) compiles one packed graph with ``N``
  request rows (block table, sampling positions, sampling params), so the number
  of packed prompts does not change the graph. Unused rows repeat the last real
  logit index; their samples are dropped. Larger buckets keep the one-prompt
  graph, so long prompts run exactly as before.
* The model gets per-token segment bounds in the attention metadata
  (``packed_bound_min`` / ``packed_bound_max``, int32 ``[T]``). Token i attends
  KV j with ``bound_min[i] <= j < bound_max[i]`` on top of the causal and
  sliding-window masks, which makes attention block-diagonal. The padding
  tokens form their own segment, so they never change a real token's output.
  ``packed_real_tokens`` (int64 ``[1]``) is the number of real tokens, for
  models that mask padding by position (positions restart per prompt).

Only models that declare ``supports_packed_prefill = True`` take this path, and
only configurations where every prefill is a single-shot, position-0 prefill
(no prefix caching, no segmented prefill, no speculation, no KV transfer).
Requests with multimodal inputs, structured output, prompt embeddings or LoRA
are never packed with others.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vllm.config import VllmConfig

ENV = "VLLM_NEURON_MAX_PREFILLS_PER_BATCH"
BUDGET_ENV = "VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS"
BOUND_MIN_KEY = "packed_bound_min"
BOUND_MAX_KEY = "packed_bound_max"
REAL_TOKENS_KEY = "packed_real_tokens"


def requested_max_prefills() -> int:
    """``VLLM_NEURON_MAX_PREFILLS_PER_BATCH`` as an int >= 1 (unset, empty or bad: 1)."""
    raw = (os.getenv(ENV) or "").strip()
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


DEFAULT_MAX_TOKENS = 2048


def requested_token_budget() -> int:
    """``VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS`` (default 2048; 0: the largest bucket)."""
    raw = (os.getenv(BUDGET_ENV) or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_TOKENS
    except ValueError:
        return DEFAULT_MAX_TOKENS


def packed_bucket_limit(buckets: list[int], max_num_batched_tokens: int) -> int:
    """Largest prefill bucket that uses the packed graph (0: none).

    Buckets up to the limit compile the packed prefill graph; larger buckets keep
    the one-prompt graph. A packed step's real tokens never exceed the limit, so
    its bucket is always a packed one.

    Example:
        >>> packed_bucket_limit([128, 256, 512, 1024, 2048, 4096], 8192)
        2048
    """
    cap = requested_token_budget() or max(buckets)
    cap = min(cap, max_num_batched_tokens)
    fit = [b for b in buckets if b <= cap]
    return max(fit) if fit else 0


def _model_supports(vllm_config: "VllmConfig") -> bool:
    from vllm_neuron.model.registry import get_models

    archs = list(getattr(vllm_config.model_config, "architectures", None) or [])
    if not archs:
        hf = getattr(vllm_config.model_config, "hf_config", None)
        archs = list(getattr(hf, "architectures", None) or [])
    models = dict(get_models())
    return bool(archs) and all(
        getattr(models.get(a), "supports_packed_prefill", False) for a in archs
    )


# Architectures served by the llama3 model code.
_LLAMA3_ARCHS = (
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Mistral3" + "ForConditionalGeneration",
    "Ministral3" + "ForCausalLM",
)


def unsupported_reason(vllm_config: "VllmConfig") -> str | None:
    """Why packed prefill cannot run with this configuration, or None if it can."""
    mc = vllm_config.model_config
    pc = vllm_config.parallel_config
    sc = vllm_config.scheduler_config
    nc = (vllm_config.additional_config or {}).get("neuron_config", {}) or {}
    if not _model_supports(vllm_config):
        return "the model does not declare supports_packed_prefill"
    archs = list(getattr(mc, "architectures", None) or [])
    if "GptOssForCausalLM" in archs:
        # Only the bf16 GPT-OSS module honours the bounds (not mxfp4 / trn3 defaults).
        if nc.get("quantization") != "bf16":
            return "GPT-OSS packed prefill needs neuron_config quantization='bf16'"
    if set(_LLAMA3_ARCHS) & set(archs):
        # The bf16 and static-FP8 (ModelOpt) llama3 modules honour the bounds.
        hf = getattr(mc, "hf_config", None)
        qc = getattr(hf, "quantization_config", None)
        if isinstance(qc, dict) and qc.get("quant_method") not in (None, "modelopt"):
            return "llama3 packed prefill needs a bf16 or ModelOpt static-FP8 checkpoint"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is configured"
    if vllm_config.kv_transfer_config is not None:
        return "a KV connector is configured"
    if getattr(pc, "data_parallel_size", 1) > 1:
        return "data parallelism is enabled"
    if getattr(pc, "decode_context_parallel_size", 1) > 1:
        return "decode context parallelism is enabled"
    if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
        return "prefix caching is enabled"
    if getattr(vllm_config, "lora_config", None) is not None:
        return "LoRA is configured"
    if getattr(mc, "enable_prompt_embeds", False):
        return "prompt embeddings are enabled"
    if getattr(mc, "runner_type", "generate") != "generate":
        return "not a generate model"
    if nc.get("enable_structured_outputs", False):
        return "structured outputs are enabled"
    if nc.get("max_jump_forward_tokens", 0):
        return "jump-forward decoding is enabled"
    if nc.get("kv_segment_size_buckets") or sc.max_num_batched_tokens < mc.max_model_len:
        return "segmented (chunked) prefill is possible (max_num_batched_tokens < max_model_len)"
    return None


def resolve_max_prefills(vllm_config: "VllmConfig") -> tuple[int, str | None]:
    """Effective prefills per step and, when packing was asked for but is off, the reason.

    The scheduler and the model runner both call this, so they agree on N.
    """
    n = requested_max_prefills()
    if n <= 1:
        return 1, None
    reason = unsupported_reason(vllm_config)
    if reason is not None:
        return 1, reason
    return min(n, vllm_config.scheduler_config.max_num_seqs), None


def effective_rows(n: int, buckets: list[int]) -> int:
    """Clamp N so every prefill bucket keeps more than one token per request row.

    The runner and warmup classify a step as prefill when tokens per row exceed
    the decode threshold, so a bucket must hold at least two tokens per row.

    Example:
        >>> effective_rows(8, [128, 256])
        8
        >>> effective_rows(200, [128, 256])
        64
    """
    if n <= 1 or not buckets:
        return 1
    return max(1, min(n, min(buckets) // 2))


def bucket_for(total: int, buckets: list[int]) -> int:
    """Smallest bucket >= total (the caller guarantees one exists)."""
    for b in buckets:
        if b >= total:
            return b
    raise ValueError(f"packed prefill of {total} tokens exceeds the largest bucket {buckets[-1]}")


def pack_padding(actual: list[int], buckets: list[int]) -> list[int]:
    """Per-row padded token counts of a packed prefill: all padding on the last row.

    Example:
        >>> pack_padding([13, 40, 70], [128, 256, 512])
        [13, 40, 75]
        >>> pack_padding([100], [128, 256])
        [128]
    """
    if not actual:
        return []
    total = int(sum(actual))
    out = [int(a) for a in actual]
    out[-1] += bucket_for(total, buckets) - total
    return out


def segment_bounds(actual: list[int] | np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-token [start, end) of the token's prompt in the packed layout (int32).

    ``actual``: real token counts of the rows, back to back from 0. Tokens from
    ``sum(actual)`` to ``total`` are padding and form their own segment, so a
    prompt attends exactly its own real tokens wherever it sits in the step.

    Example:
        >>> lo, hi = segment_bounds([2, 3], 7)
        >>> lo.tolist(), hi.tolist()
        ([0, 0, 2, 2, 2, 5, 5], [2, 2, 5, 5, 5, 7, 7])
    """
    actual = np.asarray(actual, dtype=np.int64)
    pad = int(total) - int(actual.sum())
    if pad < 0:
        raise ValueError(f"{int(actual.sum())} real tokens exceed the step size {total}")
    sizes = np.concatenate([actual, [pad]]) if pad else actual
    ends = np.cumsum(sizes)
    starts = ends - sizes
    return (
        np.repeat(starts, sizes).astype(np.int32),
        np.repeat(ends, sizes).astype(np.int32),
    )
