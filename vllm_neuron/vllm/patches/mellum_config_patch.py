# SPDX-License-Identifier: Apache-2.0
"""Use transformers' own MellumConfig instead of vLLM's ``MellumConfig(Qwen3MoeConfig)`` shim.

vLLM 0.29 registers ``vllm.transformers_utils.configs.mellum.MellumConfig``, a bare subclass of
``Qwen3MoeConfig``. For Mellum checkpoints, whose ``rope_parameters`` is nested by layer type
(``{"full_attention": {...yarn...}, "sliding_attention": {...}}``), Qwen3MoeConfig adds a flat
``rope_theta: 10000.0`` and ``rope_type: "default"`` next to the two sub-dicts. vLLM's
``patch_rope_parameters`` then fails with ``AttributeError: 'float' object has no attribute 'get'``.

transformers >= 5.17 ships ``MellumConfig`` with the nested RoPE handling. When it is importable,
point vLLM's shim module at it. vLLM resolves its config registry lazily by attribute
(``LazyConfigDict`` -> ``configs.<ClassName>`` -> module attribute), so replacing the module
attribute works in every process and does not depend on import order (the plugin can be imported
while ``vllm.transformers_utils.config`` is still initializing, e.g. in the engine-core process).
"""

import logging

logger = logging.getLogger(__name__)


def apply_mellum_config_patch() -> None:
    try:
        from transformers import MellumConfig as HFMellumConfig
    except ImportError:
        return
    try:
        import vllm.transformers_utils.configs.mellum as shim
    except Exception:  # pragma: no cover - vLLM without the shim
        shim = None
    if shim is not None and getattr(shim, "MellumConfig", None) is not HFMellumConfig:
        shim.MellumConfig = HFMellumConfig
    try:
        import vllm.transformers_utils.config as vcfg

        reg = getattr(vcfg, "_CONFIG_REGISTRY", None)
        if reg is not None and "mellum" in reg:
            dict.pop(reg, "mellum", None)
    except Exception:  # pragma: no cover - partially initialized module
        pass
