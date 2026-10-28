# SPDX-License-Identifier: Apache-2.0
"""Draft-model speculative decoding proposer (greedy) for encoder-decoder Whisper targets.

The draft is a small Whisper decoder (for example ``distil-whisper/distil-large-v3``,
2 decoder layers) that has the same encoder as the target. The proposer does not run a
second encoder. The target audio tower gives its encoder output to the draft, and the
draft projects only its own cross-attention K/V (``DraftCrossKV``).

One draft call (one NEFF) does all K proposals of a step:

1. The first pass reads the verified tokens of the target step and the bonus token.
   It writes the draft self-KV of these tokens and proposes draft token 1.
   * After a verify step: the block is the K + 1 target tokens plus one column. The
     bonus token goes to the column after the last accepted token. The causal mask
     hides the columns after the bonus token, and later steps write over their KV.
   * After a prefill step: the block is the padded prompt. The bonus token goes to
     the column after the last prompt token.
2. K - 1 single-token steps propose draft tokens 2 to K.

The target verifies the K tokens in one step and the rejection sampler accepts the
longest matching prefix, so the output is the same as greedy decode of the target.
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

logger = logging.getLogger(__name__)

DRAFT_LAYER_PREFIX = "draft"


class DraftCrossKV(nn.Module):
    """Project the target encoder output to the draft cross-KV and write the draft slots."""

    def __init__(self, owner: nn.Module):
        super().__init__()
        # Keep the owner out of the module tree: the draft model holds it.
        object.__setattr__(self, "_owner", owner)

    def forward(self, encoder_hidden: torch.Tensor, write_block_ids: torch.Tensor) -> torch.Tensor:
        # The owner writes the layout of its cross mode (plain, xkernel, fp8, ...).
        self._owner.write_cross_kv(encoder_hidden, write_block_ids)
        # Return a tensor, so that the warmup trace has an output.
        return encoder_hidden[:1, :1, :1]


def _slots_of(positions: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Return the paged-KV slot of each position, ``[B, S]`` from ``[B, S]`` positions.

    A position in a block that is not allocated (table entry -1) goes to the null
    block (slot 0). The write there is not read: the position mask hides it.
    """
    blocks = block_table.to(torch.int64).gather(1, positions // block_size)
    slots = blocks * block_size + positions % block_size
    return torch.where(blocks < 0, torch.zeros_like(slots), slots)


class WhisperDraftPropose(nn.Module):
    """The compiled draft graph: the first pass and K - 1 single-token steps."""

    def __init__(self, draft: nn.Module, num_speculative_tokens: int, vocab_size: int):
        super().__init__()
        self.draft = draft
        self.num_speculative_tokens = num_speculative_tokens
        self.vocab_size = vocab_size
        self.layer_names = [layer.self_attn.layer_name for layer in draft.decoder.layers]
        # The last position of the learned position table. A draft step after it is not
        # used (the runner stops proposing before max_model_len), so clamp to stay in range.
        self.max_pos = draft.config.max_target_positions - 1

    def get_kv_spec(self):
        return self.draft.get_kv_spec()

    def bind_kv_cache(self, kv_caches):
        self.draft.bind_kv_cache(kv_caches)

    def _meta(self, block_table, block_size, slot_mapping, max_query_len, threshold, request_slots):
        m = {
            "block_table_tensor": block_table,
            "slot_mapping": slot_mapping,
            "max_query_len": max_query_len,
            "block_size": block_size,
            "max_blocks_per_seq": block_table.shape[1],
            "decode_token_threshold": threshold,
        }
        if request_slots is not None:
            m["request_slots"] = request_slots
        # One dict for all draft layers: the decoder keeps per-step inputs in it.
        return {name: m for name in self.layer_names}

    def _sample(self, ids, positions, meta, sampling_positions, rank):
        out = self.draft(
            input_ids=ids,
            positions=positions.to(torch.int32),
            attn_metadata=meta,
            sampling_positions=sampling_positions,
            sampling_params=None,
            rank=rank,
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.to(torch.int32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        rank: torch.Tensor,
        request_slots: torch.Tensor | None = None,
        prev_sampled_token_ids: torch.Tensor | None = None,
        prev_num_draft_tokens: torch.Tensor | None = None,
        req_indices_per_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K = self.num_speculative_tokens
        if prev_sampled_token_ids is not None:
            # Async scheduling: correct the positions for the rejected drafts of the
            # previous step, as the target graph does.
            prev = prev_sampled_token_ids.to(torch.int32)
            valid = ((prev != -1) & (prev < self.vocab_size)).sum(dim=1).to(torch.int32)
            rejected = (1 + prev_num_draft_tokens.to(torch.int32)) - valid
            positions = positions.to(torch.int64) - rejected[req_indices_per_token].to(torch.int64)
        B = raw_sampled_token_ids.shape[0]
        T = input_ids.shape[0]
        S = T // B
        dev = input_ids.device
        ids = input_ids.to(torch.int32).view(B, S)
        raw = raw_sampled_token_ids.to(torch.int32)
        if raw.shape[1] > 1:
            # After a verify step: columns 0..a hold the verified tokens; the bonus is
            # the last valid entry of the rejection output.
            valid = (raw != -1) & (raw < self.vocab_size)
            accepted = (valid.sum(dim=1) - 1).clamp(min=0).to(torch.int64)
            bonus = raw.gather(1, accepted.view(B, 1)).view(B)
            col = accepted + 1
            ids = torch.cat([ids, torch.zeros(B, 1, dtype=ids.dtype, device=dev)], dim=1)
            pos = positions.to(torch.int64).view(B, S)
            pos = torch.cat([pos, (pos[:, -1:] + 1).clamp(max=self.max_pos)], dim=1)
            width = S + 1
            threshold = width
        else:
            # After a prefill step (one row): the prompt starts at position 0.
            bonus = raw.view(B)
            col = (last_token_indices.to(torch.int64).view(B) + 1).clamp(max=S - 1)
            pos = torch.arange(S, device=dev, dtype=torch.int64).view(1, S).expand(B, S)
            width = S
            threshold = 1
        ids = ids.scatter(1, col.view(B, 1), bonus.view(B, 1).to(ids.dtype))
        slots = _slots_of(pos, block_table, block_size)
        meta = self._meta(block_table, block_size, slots.view(-1), width, threshold, request_slots)
        rows = torch.arange(B, device=dev, dtype=torch.int64) * width + col
        tok = self._sample(ids.view(-1), pos.reshape(-1), meta, rows, rank)
        cur = pos.gather(1, col.view(B, 1)).view(B)
        drafts = [tok]
        step_rows = torch.arange(B, device=dev, dtype=torch.int64)
        for _ in range(K - 1):
            cur = (cur + 1).clamp(max=self.max_pos)
            s = _slots_of(cur.view(B, 1), block_table, block_size).view(B)
            meta = self._meta(block_table, block_size, s, 1, 1, request_slots)
            tok = self._sample(tok, cur, meta, step_rows, rank)
            drafts.append(tok)
        drafts_only = torch.stack(drafts, dim=1)
        stacked = torch.stack([bonus.to(torch.int32)] + drafts, dim=1)
        return stacked, drafts_only


class DraftModelProposer:
    """Greedy draft-model proposer (``speculative_config.method = "draft_model"``).

    Supports a Whisper target with a Whisper draft that has the same encoder.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, on_device_sampling: bool = True):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        if self.speculative_config.method != "draft_model":
            raise ValueError(
                f"DraftModelProposer needs method='draft_model'; got {self.speculative_config.method!r}"
            )
        if not on_device_sampling:
            raise ValueError("Draft-model speculative decoding needs on-device greedy sampling.")
        self.draft_model_config = self.speculative_config.draft_model_config
        self.device = device
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.model: nn.Module | None = None
        self.cross_kv: nn.Module | None = None
        self.capture_backend_model = None
        self.attn_layer_names: list[str] = []
        self._draft_tensor_capture = None
        self._cross_kv_warm = False
        from vllm.distributed.parallel_state import get_tp_group

        self.rank_tensor = torch.tensor(get_tp_group().rank_in_group, dtype=torch.int32, device=device)

    def load_model(
        self,
        target_hidden_size: int | None = None,
        target_hidden_pad_groups: int = 1,
        compile_options: dict | None = None,
    ) -> None:
        """Build, load and compile the draft decoder. The draft encoder weights are dropped."""
        arch = (self.draft_model_config.architectures or [""])[0]
        if arch != "WhisperForConditionalGeneration":
            raise ValueError(
                f"The draft-model proposer supports a Whisper draft only; got architecture {arch!r}."
            )
        from vllm.distributed.parallel_state import get_tp_group

        from vllm_neuron.model.neuron_config import (
            EncoderNeuronConfig,
            NeuronConfig,
            OnDeviceSamplingConfig,
        )
        from vllm_neuron.model.whisper.model_bf16 import WhisperForConditionalGeneration

        world = get_tp_group().world_size
        nc = NeuronConfig(quantization="bf16", on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True))
        ec = EncoderNeuronConfig(
            num_encoder_input_buckets=[1500], encoder_attention_block_size=1500, tp_size=world, dp_size=1
        )
        hf = self.draft_model_config.hf_config
        with torch.device("meta"):
            draft = WhisperForConditionalGeneration.from_configs(
                hf, text_neuron_config=nc, encoder_neuron_config=ec, layer_name_prefix=DRAFT_LAYER_PREFIX
            )
        load_start = time.perf_counter()
        path = self.speculative_config.model
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            # Resolve the repo id to the local snapshot. The fp32 safetensors files are not needed.
            path = snapshot_download(
                path,
                allow_patterns=["*.json", "model.safetensors", "*.txt"],
                cache_dir=self.vllm_config.load_config.download_dir,
            )
        draft.load_weights(path, torch.device("cpu"), self.vllm_config.load_config.download_dir)
        # The target encoder gives the encoder output. Free the draft encoder layers.
        draft.encoder.layers = nn.ModuleList()
        draft = draft.to(self.device)
        logger.info(
            "Draft model %s loaded in %.1f s (%d decoder layers)",
            self.speculative_config.model,
            time.perf_counter() - load_start,
            len(draft.decoder.layers),
        )
        self.draft = draft
        self.attn_layer_names = [layer.self_attn.layer_name for layer in draft.decoder.layers]
        compile_kwargs: dict = {"fullgraph": not envs.VLLM_NEURON_DEBUG_MODE}
        if compile_options is not None:
            compile_kwargs["options"] = compile_options
        self.compile_kwargs = {"backend": envs.get_compile_backend_name(), **compile_kwargs}
        propose = WhisperDraftPropose(draft, self.num_speculative_tokens, hf.vocab_size)
        self.model = torch.compile(propose, **self.compile_kwargs)
        self.cross_kv = torch.compile(DraftCrossKV(draft.audio_tower), **self.compile_kwargs)

    def attach_target(self, target_model: nn.Module) -> None:
        """Make the target audio tower write the draft cross-KV after each encoder call."""
        base = getattr(target_model, "_orig_mod", target_model)
        if not hasattr(base, "set_draft_cross_kv"):
            raise ValueError(
                f"The target {type(base).__name__} does not support a draft model with a shared encoder."
            )
        base.set_draft_cross_kv(self.cross_kv)

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        groups: dict[str, int] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                groups[name] = gid
        assert len({groups[name] for name in self.attn_layer_names}) == 1, (
            "All draft layers must be in one KV-cache group"
        )

    def _warm_cross_kv(self) -> None:
        if self._cross_kv_warm:
            return
        d = self.draft.config.d_model
        h = torch.zeros(1, 1500, d, dtype=self.draft.dtype, device=self.device)
        scratch = self.draft.num_request_slots - 1
        out = self.cross_kv(h, torch.tensor([scratch], dtype=torch.int64).to(self.device))
        out.cpu()
        self._cross_kv_warm = True

    def warmup(self, num_tokens: int, num_reqs: int, attn_metadata: AttentionMetadata) -> None:
        """Compile the draft graph of one target shape (prefill bucket or verify step)."""
        K = self.num_speculative_tokens
        verify = num_tokens == num_reqs * (1 + K)
        logger.info("Draft-model warmup: num_tokens=%d, num_reqs=%d, verify=%s", num_tokens, num_reqs, verify)
        if not verify:
            self._warm_cross_kv()
        per_req = num_tokens // num_reqs
        out = self.propose(
            target_token_ids=torch.ones(num_tokens, dtype=torch.int32, device=self.device),
            target_positions=torch.zeros(num_tokens, dtype=torch.int64, device=self.device),
            target_hidden_states=None,
            last_token_indices=torch.tensor(
                [(i + 1) * per_req - 1 for i in range(num_reqs)], dtype=torch.int64
            ).to(self.device),
            attn_metadata=attn_metadata,
            raw_sampled_token_ids=torch.zeros(
                num_reqs, (1 + K) if verify else 1, dtype=torch.int32, device=self.device
            ),
            is_warmup=True,
        )
        for t in out:
            t.cpu()

    def graph_extract(self, *args, **kwargs) -> None:
        """No parallel graph capture: the warmup compiles the draft graph."""
        return

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor | None,
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
        """Return ``(bonus + drafts [B, 1 + K], drafts [B, K])``."""
        assert self.model is not None
        meta = attn_metadata[self.attn_layer_names[0]]
        dev = self.device
        B = last_token_indices.shape[0]
        T = target_token_ids.shape[0]
        K = self.num_speculative_tokens
        if raw_sampled_token_ids.shape[1] > 1 and prev_sampled_token_ids is None:
            # A verify-shaped step without a correction (sync scheduling, warmup): give
            # tensors that make the correction a no-op, so one graph serves both.
            prev_sampled_token_ids = torch.zeros(B, 1 + K, dtype=torch.int32, device=dev)
            prev_num_draft_tokens = torch.full((B,), K, dtype=torch.int32, device=dev)
            req_indices_per_token = (
                torch.arange(B, dtype=torch.int64).repeat_interleave(T // B).to(dev)
            )
        corr = {}
        if raw_sampled_token_ids.shape[1] > 1:
            corr = {
                "prev_sampled_token_ids": prev_sampled_token_ids.to(dev),
                "prev_num_draft_tokens": prev_num_draft_tokens.to(dev),
                "req_indices_per_token": req_indices_per_token.to(dev),
            }
        start = time.perf_counter()
        with contextlib.nullcontext() if is_warmup else model_forward_context(self.vllm_config):
            stacked, drafts_only = self.model(
                input_ids=target_token_ids.to(dev),
                positions=target_positions.to(dev),
                last_token_indices=last_token_indices.to(dev),
                raw_sampled_token_ids=raw_sampled_token_ids.to(dev),
                block_table=meta["block_table_tensor"].to(dev),
                block_size=meta["block_size"],
                rank=self.rank_tensor,
                request_slots=meta.get("request_slots"),
                **corr,
            )
        bucket = f"draft_model_k{self.num_speculative_tokens}"
        if is_warmup:
            COMPILATION_TIME.labels(model_name=self.speculative_config.model, bucket_name=bucket).set(
                time.perf_counter() - start
            )
        else:
            NEFF_EXECUTION_COUNT.labels(model_name=self.speculative_config.model, bucket_name=bucket).inc()
        return stacked, drafts_only
