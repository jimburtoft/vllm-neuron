# SPDX-License-Identifier: Apache-2.0
"""Ministral 3 config adapter: the multimodal ``mistral3`` config -> LlamaConfig.

<-- MODEL-SPECIFIC. Takes the nested ``text_config`` (``ministral3``), keeps
``rope_parameters`` (YaRN), and drops ``vision_config``. Every value that the
Llama 3 dataclass would otherwise default (rope_theta 5e5,
tie_word_embeddings True, ...) is taken from the checkpoint and checked by
:func:`validate_text_config`.
"""

import json
from typing import Any

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.llama3.config import LlamaConfig
from vllm_neuron.model.neuron_config import NeuronConfig


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, (str, bytes)):
        with open(cfg) as f:
            return json.load(f)
    if isinstance(cfg, PretrainedConfig):
        d = cfg.to_dict()
        dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
        if dtype is not None:
            d["torch_dtype"] = dtype
        return d
    return dict(cfg)


def _resolve_dtype(*candidates) -> torch.dtype:
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, torch.dtype):
            return c
        if isinstance(c, str):
            return getattr(torch, c.replace("torch.", ""))
    return torch.bfloat16


def text_config_dict(hf_config: Any) -> dict:
    """Flat text-decoder config dict from a ``mistral3`` (or flat) config."""
    top = _as_dict(hf_config)
    text = top.get("text_config")
    text = _as_dict(text) if text is not None else dict(top)
    text.pop("vision_config", None)

    # The top-level dtype is the checkpoint dtype. vLLM can infer a different
    # text_config dtype for a local checkpoint that has one fp32 tensor.
    text["torch_dtype"] = _resolve_dtype(
        top.get("torch_dtype"),
        top.get("dtype"),
        text.get("torch_dtype"),
        text.get("dtype"),
    )
    # tie_word_embeddings may be set on the wrapper only.
    if "tie_word_embeddings" not in text and "tie_word_embeddings" in top:
        text["tie_word_embeddings"] = top["tie_word_embeddings"]
    if "quantization_config" in top and "quantization_config" not in text:
        text["quantization_config"] = top["quantization_config"]

    # Older transformers keep rope_theta / rope_scaling at the top level.
    if "rope_parameters" not in text or not text["rope_parameters"]:
        rp = dict(text.get("rope_scaling") or {})
        if text.get("rope_theta") is not None:
            rp.setdefault("rope_theta", text["rope_theta"])
        if "type" in rp and "rope_type" not in rp:
            rp["rope_type"] = rp["type"]
        rp.setdefault("rope_type", "default")
        text["rope_parameters"] = rp
    else:
        rp = dict(text["rope_parameters"])
        if "type" in rp and "rope_type" not in rp:
            rp["rope_type"] = rp["type"]
        text["rope_parameters"] = rp
    return text


def llama4_scaling(text: dict) -> tuple[float | None, int | None]:
    """(beta, original_max_position) of the Llama 4 query scaling, if any."""
    rp = text.get("rope_parameters") or {}
    beta = rp.get("llama_4_scaling_beta")
    if beta is None:
        l4 = text.get("llama_4_scaling") or {}
        beta = l4.get("beta")
        orig = l4.get("original_max_position_embeddings")
    else:
        orig = rp.get("original_max_position_embeddings")
    return (float(beta) if beta is not None else None, int(orig) if orig else None)


def ministral3_llama_config(
    hf_config: Any, neuron_config: NeuronConfig | None
) -> LlamaConfig:
    text = text_config_dict(hf_config)
    quant = text.pop("quantization_config", None)
    if quant:
        method = str(quant.get("quant_method", "")).lower()
        if method == "fp8":
            raise NotImplementedError(
                "Ministral 3 FP8 checkpoints (quant_method 'fp8', "
                "weight_scale_inv / activation_scale) are not supported yet. "
                "Use the BF16 checkpoint (e.g. "
                "mistralai/Ministral-3-8B-Instruct-2512-BF16)."
            )
        text["quantization_config"] = quant
    config = LlamaConfig.from_configs(text, neuron_config)
    # The serving dtype is vLLM's resolved --dtype, not the checkpoint dtype.
    try:
        from vllm.config import get_current_vllm_config

        model_config = get_current_vllm_config().model_config
        if model_config is not None and isinstance(model_config.dtype, torch.dtype):
            config.torch_dtype = model_config.dtype
    except Exception:  # noqa: BLE001 - no vLLM context (unit tests)
        pass
    validate_text_config(config, text)
    return config


def validate_text_config(config: LlamaConfig, text: dict) -> None:
    """Load-time check: the Llama dataclass got every checkpoint value."""

    def check(name, got, want):
        if want is None:
            return
        if isinstance(want, float) or isinstance(got, float):
            ok = abs(float(got) - float(want)) <= 1e-12 * max(1.0, abs(float(want)))
        else:
            ok = got == want
        if not ok:
            raise ValueError(
                f"Ministral 3 config mismatch: {name}={got!r}, checkpoint has {want!r}"
            )

    check("hidden_size", config.hidden_size, text.get("hidden_size"))
    check("intermediate_size", config.intermediate_size, text.get("intermediate_size"))
    check("num_hidden_layers", config.num_hidden_layers, text.get("num_hidden_layers"))
    check(
        "num_attention_heads",
        config.num_attention_heads,
        text.get("num_attention_heads"),
    )
    check(
        "num_key_value_heads",
        config.num_key_value_heads,
        text.get("num_key_value_heads"),
    )
    check(
        "head_dim",
        config.head_dim,
        text.get("head_dim")
        or text["hidden_size"] // text["num_attention_heads"],
    )
    check("vocab_size", config.vocab_size, text.get("vocab_size"))
    check("rms_norm_eps", config.rms_norm_eps, text.get("rms_norm_eps"))
    check(
        "tie_word_embeddings",
        config.tie_word_embeddings,
        bool(text.get("tie_word_embeddings", False)),
    )
    rp_cfg, rp_ckpt = config.rope_parameters, text["rope_parameters"]
    for key in (
        "rope_type",
        "rope_theta",
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "attention_factor",
    ):
        if key in rp_ckpt:
            check(f"rope_parameters.{key}", rp_cfg.get(key), rp_ckpt[key])
    if text.get("sliding_window") is not None:
        raise NotImplementedError(
            f"sliding_window={text['sliding_window']} is not supported: the "
            "Llama 3 attention has no sliding window."
        )
    hidden_act = text.get("hidden_act", "silu")
    if hidden_act != "silu":
        raise NotImplementedError(f"hidden_act={hidden_act!r} is not supported")
