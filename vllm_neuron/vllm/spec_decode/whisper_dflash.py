# SPDX-License-Identifier: Apache-2.0
"""DFlash proposer for a Whisper (encoder-decoder) target.

Same lifecycle as :class:`DFlashProposer` (``speculative_config.method = "dflash"``),
with the Whisper draft :class:`WhisperDFlashDraft`:

* the draft keeps its context K/V and audio K/V in model-owned per-request-slot buffers
  (the target's cross-KV slots), so it has no paged-KV layers;
* the target audio tower runs the draft audio K/V graph (:class:`DraftAudioKV`) after each
  encoder call, as the draft-model proposer does for its cross-KV;
* the token embedding and the LM head are the target's (tied embedding, vocab-parallel
  LM head with its pad columns).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_neuron import envs
from vllm_neuron.metrics import COMPILATION_TIME, NEFF_EXECUTION_COUNT
from vllm_neuron.utils.neuron_utils import model_forward_context
from vllm_neuron.vllm.spec_decode.dflash import DFlashProposer

logger = logging.getLogger(__name__)


def is_whisper_target(vllm_config: VllmConfig) -> bool:
    hf = vllm_config.model_config.hf_config
    return getattr(hf, "model_type", None) == "whisper"


class WhisperDFlashProposer(DFlashProposer):
    """DFlash proposer for ``WhisperForConditionalGeneration`` targets."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, on_device_sampling: bool = True):
        super().__init__(vllm_config, device, on_device_sampling)
        if not on_device_sampling:
            raise ValueError("Whisper DFlash needs on-device greedy sampling.")
        from vllm.distributed.parallel_state import get_tp_group

        self.rank_tensor = torch.tensor(get_tp_group().rank_in_group, dtype=torch.int32, device=device)
        self.audio_kv: nn.Module | None = None
        self._audio_kv_warm = False
        self.draft = None
        # WHISPER_DFLASH_TIMING=1: sync around the draft graph and log its mean time
        # (debug only: the sync breaks the async overlap).
        self._timing = os.environ.get("WHISPER_DFLASH_TIMING", "0") == "1"
        self._t_sum, self._t_n, self._w_sum = 0.0, 0, 0.0

    # ------------------------------------------------------------------ load
    def _draft_path(self) -> str:
        path = self.speculative_config.model
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                path, allow_patterns=["*.json", "*.safetensors"],
                cache_dir=self.vllm_config.load_config.download_dir,
            )
        return path

    def load_model(self, target_hidden_size=None, target_hidden_pad_groups=1, compile_options=None) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_neuron.model.whisper.dflash_draft import (
            DraftAudioKV,
            WhisperDFlashDraft,
            load_draft_config,
        )

        if target_hidden_pad_groups != 1:
            raise NotImplementedError("Whisper DFlash expects an unpadded target hidden dim.")
        path = self._draft_path()
        cfg = load_draft_config(path)
        tgt = self.vllm_config.model_config.hf_config
        d_model = getattr(tgt, "d_model", None)
        if cfg.get("target_hidden_size", cfg["hidden_size"]) != d_model:
            raise ValueError(
                f"Draft target_hidden_size {cfg.get('target_hidden_size')} != Whisper d_model {d_model}"
            )
        tp = get_tp_group()
        num_slots = self.vllm_config.scheduler_config.max_num_seqs + 1
        max_ctx = getattr(tgt, "max_target_positions", 448)
        start = time.perf_counter()
        draft = WhisperDFlashDraft(
            cfg, num_slots=num_slots, max_ctx=max_ctx, world=tp.world_size, rank=tp.rank_in_group,
            tp_group=tp.device_group, dtype=torch.bfloat16, on_device_sampling=True,
        )
        if not envs.VLLM_NEURON_CPU_COMPILE:
            draft.load_weights(path)
        draft = draft.to(self.device)
        self.draft = draft
        logger.info(
            "Whisper DFlash draft %s loaded in %.1f s: %d layers, block %d, audio_context=%s, "
            "target layers %s, TP %d",
            path, time.perf_counter() - start, draft.n_layers, draft.block_size, draft.audio_context,
            draft.target_layer_ids, tp.world_size,
        )
        compile_kwargs: dict = {"fullgraph": not envs.VLLM_NEURON_DEBUG_MODE}
        if compile_options is not None:
            compile_kwargs["options"] = compile_options
        self.compile_kwargs = {"backend": envs.get_compile_backend_name(), **compile_kwargs}
        self.model = torch.compile(draft, **self.compile_kwargs)
        self.audio_kv = torch.compile(DraftAudioKV(draft), **self.compile_kwargs) if draft.n_audio else None
        self.capture_backend_model = None
        self.attn_layer_names = []

    def recompile(self) -> None:
        """Wrap the audio K/V graph again after the runner re-wraps the draft graph."""
        if self.audio_kv is not None:
            self.audio_kv = torch.compile(
                getattr(self.audio_kv, "_orig_mod", self.audio_kv), **self.compile_kwargs
            )

    def attach_target(self, target_model: nn.Module) -> None:
        """Bind the target head and make the target audio tower write the draft audio K/V."""
        base = getattr(target_model, "_orig_mod", target_model)
        self.draft.bind_target(base)
        if self.audio_kv is not None:
            if not hasattr(base, "set_draft_cross_kv"):
                raise ValueError(f"The target {type(base).__name__} has no draft encoder hook.")
            base.set_draft_cross_kv(self.audio_kv)

    def alias_target_weights(self, target_model: nn.Module) -> None:
        self.attach_target(target_model)
        self._target_weights_aliased = True

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        # The draft keeps its K/V in model-owned buffers: no paged layers to check.
        return

    def get_target_aux_hidden_state_layers(self) -> tuple[int, ...]:
        ids = self._target_layer_ids_from_config(self.draft_model_config.hf_config)
        n = getattr(self.vllm_config.model_config.hf_config, "decoder_layers", None)
        layers = tuple(t + 1 for t in ids)
        if n is None or not all(0 <= x < n for x in layers):
            raise ValueError(f"DFlash target layers {ids} do not fit the Whisper decoder ({n} layers).")
        return layers

    def _num_aux_hidden_state_layers(self) -> int:
        return len(self._target_layer_ids_from_config(self.draft_model_config.hf_config))

    # ------------------------------------------------------------------ warmup
    def _warm_audio_kv(self) -> None:
        if self._audio_kv_warm or self.audio_kv is None:
            return
        h = torch.zeros(1, 1500, self.draft.target_hidden, dtype=torch.bfloat16, device=self.device)
        scratch = self.draft.num_slots - 1
        self.audio_kv(h, torch.tensor([scratch], dtype=torch.int64).to(self.device)).cpu()
        self._audio_kv_warm = True

    def warmup(self, num_tokens: int, num_reqs: int, attn_metadata: AttentionMetadata) -> None:
        K = self.num_speculative_tokens
        verify = num_tokens == num_reqs * (1 + K)
        logger.info("Whisper DFlash warmup: num_tokens=%d, num_reqs=%d", num_tokens, num_reqs)
        self._warm_audio_kv()
        per_req = num_tokens // num_reqs
        nf = self._num_aux_hidden_state_layers() * self.draft.target_hidden
        # Two shapes of the rejection output: [B, 1] after a prefill, [B, 1 + K] after a verify.
        cols = [1, 1 + K] if verify else [1]
        for c in cols:
            out = self.propose(
                target_token_ids=torch.ones(num_tokens, dtype=torch.int32, device=self.device),
                target_positions=torch.zeros(num_tokens, dtype=torch.int64, device=self.device),
                target_hidden_states=torch.zeros(num_tokens, nf, dtype=torch.bfloat16, device=self.device),
                last_token_indices=torch.tensor(
                    [(i + 1) * per_req - 1 for i in range(num_reqs)], dtype=torch.int64
                ).to(self.device),
                attn_metadata=attn_metadata,
                raw_sampled_token_ids=torch.zeros(num_reqs, c, dtype=torch.int32, device=self.device),
                is_warmup=True,
            )
            for t in out:
                t.cpu()

    def graph_extract(self, *args, **kwargs) -> None:
        return

    # ------------------------------------------------------------------ propose
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: AttentionMetadata,
        raw_sampled_token_ids: torch.Tensor,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
        cur_num_draft_tokens: torch.Tensor | None = None,
        is_warmup: bool = False,
        model_override: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        B = last_token_indices.shape[0]
        T = target_hidden_states.shape[0]
        K = self.num_speculative_tokens
        if prev_sampled_token_ids is None:
            noop = self._build_noop_async_spec_correction_kwargs(T, B, device=dev)
            prev_sampled_token_ids = noop["prev_sampled_token_ids"]
            prev_num_draft_tokens = noop["prev_num_draft_tokens"]
            req_indices_per_token = noop["req_indices_per_token"]
        if cur_num_draft_tokens is None:
            cur_num_draft_tokens = torch.full((B,), K, dtype=torch.int32, device=dev)
        meta = next(iter(attn_metadata.values()))
        slots = meta.get("request_slots")
        if slots is None:
            slots = torch.zeros(B, dtype=torch.int32, device=dev)
        timing = self._timing and not is_warmup
        if timing:
            w0 = time.perf_counter()
            target_hidden_states.cpu()  # wait for the target graph
            self._w_sum += time.perf_counter() - w0
        start = time.perf_counter()
        with contextlib.nullcontext() if is_warmup else model_forward_context(self.vllm_config):
            # Fixed input dtypes: a dtype change would be a Dynamo recompile.
            stacked, drafts = self.model(
                target_positions=target_positions.to(dev).to(torch.int64),
                target_hidden_states=target_hidden_states.to(torch.bfloat16),
                last_token_indices=last_token_indices.to(dev).to(torch.int64),
                raw_sampled_token_ids=raw_sampled_token_ids.to(dev).to(torch.int32),
                request_slots=slots.to(dev).to(torch.int32),
                rank=self.rank_tensor,
                prev_sampled_token_ids=prev_sampled_token_ids.to(dev).to(torch.int32),
                prev_num_draft_tokens=prev_num_draft_tokens.to(dev).to(torch.int32),
                req_indices_per_token=req_indices_per_token.to(dev).to(torch.int64),
                cur_num_draft_tokens=cur_num_draft_tokens.to(dev).to(torch.int32),
            )
        if timing:
            drafts.cpu()
            self._t_sum += time.perf_counter() - start
            self._t_n += 1
            if self._t_n % 100 == 0:
                logger.info(
                    "Whisper DFlash timing: draft graph %.2f ms, wait for target %.2f ms (mean of %d)",
                    1e3 * self._t_sum / self._t_n, 1e3 * self._w_sum / self._t_n, self._t_n,
                )
        bucket = f"whisper_dflash_block{K + 1}"
        if is_warmup:
            COMPILATION_TIME.labels(model_name=self.speculative_config.model, bucket_name=bucket).set(
                time.perf_counter() - start
            )
        else:
            NEFF_EXECUTION_COUNT.labels(model_name=self.speculative_config.model, bucket_name=bucket).inc()
        return stacked, drafts
