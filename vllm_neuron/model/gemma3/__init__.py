# SPDX-License-Identifier: Apache-2.0
"""Gemma3 text decoder for the Neuron backend.

NOTE: ``factory.Gemma3ForCausalLM`` and ``model.Gemma3ForCausalLM`` share a
name. The public export below is the **factory** -- it is what
``model/registry.py`` must receive, because only the factory's ``from_configs``
runs ``_validate_config`` before instantiating the implementation.
"""

from .factory import Gemma3ForCausalLM
from .config import Gemma3Config
from .model import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3MLP,
    Gemma3Model,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
)

__all__ = [
    "Gemma3Config",
    "Gemma3ForCausalLM",
    "Gemma3Attention",
    "Gemma3DecoderLayer",
    "Gemma3MLP",
    "Gemma3Model",
    "Gemma3RMSNorm",
    "Gemma3RotaryEmbedding",
]
