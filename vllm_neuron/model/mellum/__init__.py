# SPDX-License-Identifier: Apache-2.0
"""Mellum (MellumForCausalLM) for the Neuron backend."""

from .config import MellumConfig
from .model import MellumForCausalLM

__all__ = ["MellumConfig", "MellumForCausalLM"]
