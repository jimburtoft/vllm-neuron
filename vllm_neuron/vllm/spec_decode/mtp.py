# SPDX-License-Identifier: Apache-2.0
"""MTP proposer for Qwen3.5-MoE on Neuron (method="mtp").

Reuses the EAGLE fused-draft flow (one draft NEFF: KV fill pass over the target
tokens + K-1 recurrent passes), with the draft model being the checkpoint's own
``mtp.*`` head (Qwen3_5MoeMTPForCausalLM). The aux hidden state handed over by the
target is its final normed hidden state [T, H] (not 3 x H as in EAGLE3).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from vllm_neuron import envs
from vllm_neuron.model.neuron_config import NeuronConfig, OnDeviceSamplingConfig
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer

logger = logging.getLogger(__name__)


class MTPProposer(EagleProposer):
    def __init__(self, vllm_config, device, on_device_sampling=True):
        sc = vllm_config.speculative_config
        orig = sc.method
        sc.method = "eagle3"  # satisfy the base-class assert, restore below
        try:
            super().__init__(vllm_config, device, on_device_sampling)
        finally:
            sc.method = orig
        self.method = "mtp"

    def load_model(self, target_hidden_size, target_hidden_pad_groups=1, compile_options=None):
        tc = self.vllm_config.model_config.hf_config
        text = getattr(tc, "text_config", tc)
        n_layers = text.num_hidden_layers
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=n_layers, target_hidden_size=target_hidden_size,
            compile_options=compile_options,
        )
        self.attn_layer_names = [f"layers.{n_layers}.self_attn"]

    def compile_and_load_draft_model(self, start_layer_idx, target_hidden_size,
                                     target_hidden_pad_groups=1, compile_options=None):
        import time

        from libtorch_neuronx_lite import envs as libtorch_envs

        from vllm_neuron.model.qwen3_5_moe.mtp_model import Qwen3_5MoeMTPForCausalLM

        target_cfg = self.vllm_config.model_config.hf_config
        nc = NeuronConfig(
            on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
            if self.on_device_sampling else None
        )
        with torch.device("meta"):
            self.model = Qwen3_5MoeMTPForCausalLM.from_configs(target_cfg, start_layer_idx, nc)
        self.model.num_speculative_tokens = self.num_speculative_tokens
        self.model.kv_fill_only = self.kv_fill_only
        t0 = time.perf_counter()
        if not envs.VLLM_NEURON_CPU_COMPILE:
            self.model.load_weights(self.vllm_config.model_config.model, self.device,
                                    self.vllm_config.load_config.download_dir)
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to("meta")
        logger.info("MTP draft head loaded in %.1fs", time.perf_counter() - t0)
        fullgraph = not envs.VLLM_NEURON_DEBUG_MODE
        kw: dict = {"fullgraph": fullgraph}
        if compile_options is not None:
            kw["options"] = compile_options
        cpu = envs.VLLM_NEURON_CPU_MODE
        eager = self.vllm_config.model_config.enforce_eager
        if eager or cpu or envs.VLLM_NEURON_DEBUG_MODE or libtorch_envs.NEURON_LIBTORCH_DISABLE_GRAPH_CAPTURE_BACKEND:
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(self.model, backend="neuron_libtorch_graph_capture", **kw)
        self.compile_kwargs = {"backend": envs.get_compile_backend_name(), **kw}
        return torch.compile(self.model, **self.compile_kwargs)

    def alias_target_weights(self, target_model: nn.Module) -> None:
        """Share embed_tokens / lm_head with the target (same checkpoint tensors)."""
        base = getattr(self.model, "_orig_mod", self.model)
        tgt = getattr(target_model, "_orig_mod", target_model)
        base.embed_tokens.weight = tgt.embed_tokens.weight
        base.lm_head.weight = tgt.lm_head.weight
        logger.info("MTP draft: aliased embed_tokens and lm_head to the target")

    def _build_synthetic_inputs(self, num_tokens, num_reqs, device=None):
        d = super()._build_synthetic_inputs(num_tokens, num_reqs, device)
        H = self.model.config.hidden_size
        d["target_hidden_states"] = torch.ones(
            d["target_hidden_states"].shape[0], H, dtype=torch.bfloat16
        ).to(d["target_hidden_states"].device)
        return d
