# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 BF16 Implementation (text decoder)
=========================================

Annotated implementation of the Gemma3 text decoder for the Neuron backend.

Checkpoints: ``google/gemma-3-12b-it`` and ``google/gemma-3-4b-it`` (and their
finetunes); all shapes come from the config. The checkpoint ships the multimodal
``Gemma3ForConditionalGeneration`` wrapper; the SigLIP vision tower is DROPPED
(see ``load_weights``).

Supported parallelism: TP, SP, DP.

Reference models this was ported from:
  * ``qwen3/model.py``   -- primary base. Dense GQA + per-head Q/K RMSNorm
                            applied BEFORE RoPE, which Gemma3 also has.
  * ``gpt_oss/model_bf16.py`` -- source of the per-layer sliding-window
                            machinery: the two-mask decode precompute and the
                            per-layer ``sliding_window_size`` in
                            ``get_kv_spec()``.

Key differences from Qwen3 (the base):
  1. 4-norm "sandwich" per layer: input / post-attention / pre-FFN / post-FFN,
     with the post-norms applied BEFORE the residual add.
  2. RMSNorm is ``(1 + gamma) * norm(x)``. Gamma is pre-incremented at load
     time so every consumer (this module AND the fused QK-norm kernel) uses the
     plain ``w * x`` form. See Section 1.
  3. DUAL RoPE: sliding layers use theta 1e4 (unscaled), global layers use
     theta 1e6 with linear x8 scaling. Selected per layer by the 5:1 schedule.
  4. Hybrid attention: every 6th layer is GLOBAL, the other five are sliding
     with window 1024 (``(layer_idx + 1) % sliding_window_pattern != 0`` is
     sliding). NOT gpt_oss's ``layer_idx % 2 == 0``.
  5. Softmax scale is ``query_pre_attn_scalar ** -0.5``, not ``head_dim**-0.5``.
  6. ``gelu_pytorch_tanh`` gated MLP, written out explicitly (see Section 4).
  7. Token embeddings are scaled by ``sqrt(hidden_size)`` (bf16-rounded).
  8. LM head is TIED to the input embedding.
  9. ``head_dim = 256`` with ``hidden_size = 2560``, so QKV is NON-SQUARE
     (8 * 256 = 2048 != 2560) and ``head_dim > 128`` -- which changes which
     kernel arguments are legal. See the HEAD_DIM notes in Section 3.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Gemma3-specific. Change when porting.
"""

import logging
import os

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

import vllm_neuron.functional as NF
from vllm_neuron.utils.neuron_utils import can_run_kernel
from nkilib.core.utils.common_types import ActFnType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, build_per_head_slot_mapping
from vllm_neuron.model.neuron_config import NeuronConfig

# Shared GQA decode helpers. The fused decode mega-kernel only accepts GQA
# (kv_heads_per_rank > 1) via a PER-HEAD 3D block table, and returns 4D
# per-head new-K/V that must be scattered back into the paged cache. Both
# operations already exist as shared, unit-tested, device-agnostic helpers --
# do NOT reimplement them here.
from vllm_neuron.model.qwen3_vl.utils.decode_kv import (
    build_per_head_block_table,
    scatter_new_kv,
)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

import vllm_neuron.nn as neuron_nn
from .config import Gemma3Config

logger = logging.getLogger(__name__)

# <-- MODEL-SPECIFIC: HF checkpoint key prefixes for the text decoder. The
# multimodal Gemma3 checkpoint nests the decoder under
# ``language_model.model.``; a text-only re-export would use a bare ``model.``.
# Probed in order by ``_detect_text_prefix``.
_TEXT_PREFIX_CANDIDATES = ("language_model.model.", "language_model.", "model.")

# <-- MODEL-SPECIFIC: vision-tower key prefixes that are intentionally NOT
# loaded. They are never referenced by the weight mappings, so they are simply
# dropped; these strings exist only so the load audit can report how many
# tensors were dropped on purpose vs. genuinely unexpected.
_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")

# <-- MODEL-SPECIFIC: parameter-name suffixes whose loaded gamma must be
# incremented by 1.0 (Gemma3's ``(1 + gamma)`` RMSNorm). Every RMSNorm in the
# model must appear here; a missing entry is a silent accuracy bug.
_RMSNORM_PARAM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.norm.weight",
)

# The head dimension above which the fused decode mega-kernel (nkilib
# ``attention_block_tkg``) can no longer fold the output projection OR the KV
# cache update into itself. Both are hard ``kernel_assert``s in the kernel:
#
#   "In-kernel output projection (W_out != None) is not supported for
#    d_head > 128 ...; pass W_out=None and project externally."
#   "In-kernel KV cache update (update_cache=True) is not supported for
#    d_head > 128 ...; pass update_cache=False and update the cache externally."
#
# (Attention / QKV / RoPE themselves ARE supported above 128 via the kernel's
# n_d_tiles d-tiling path.) Gemma3's head_dim is 256, so this always applies
# here; named so the branches read as kernel constraints, not magic numbers.
_MAX_FUSED_OUT_PROJ_HEAD_DIM = 128

# The NKI Library branch ``gemma3-trn2-perf-2.33`` (https://github.com/jimburtoft/nki-library) lifts both
# limits for d_head = n * 128 and exports these markers. With a stock NKI Library both are False and the
# model keeps the external output projection and the external (deferred) KV scatter.
try:
    import importlib

    _nkilib_abt = importlib.import_module("nkilib.experimental.transformer.attention_block_tkg")

    _NKILIB_D256_OUT_PROJ = bool(getattr(_nkilib_abt, "SUPPORTS_D256_OUT_PROJ", False))
    _NKILIB_D256_KV_UPDATE = bool(getattr(_nkilib_abt, "SUPPORTS_D256_BLOCK_KV_UPDATE", False))
except Exception:  # pragma: no cover - nkilib missing (CPU-only test environments)
    _NKILIB_D256_OUT_PROJ = _NKILIB_D256_KV_UPDATE = False


# =============================================================================
# HARDWARE-DISCOVERED FIXES (from an earlier hand-built Gemma3 port, measured on trn2)
#
# X4 -- DECODE ROUTES TO THE TORCH FALLBACK, NOT stock ``NF.attention_decode``.
#     The stock nkilib ``attention_block_tkg`` d_head>128 GQA decode path
#     COMPILES and is numerically EXACT in every ISOLATED reproducer on hardware
#     (cos>=0.999997 vs a dense fp32 CPU reference -- including
#     QK-norm, real dual-RoPE, real hidden=2560, per-head 3D block table,
#     garbage/zeroed cache, 2-step autoregressive, both the fused-pos_id and the
#     pre-built-mask paths) but is NUMERICALLY WRONG IN THE FULL 34-LAYER MODEL
#     on trn2 (neuronx-cc 2.27 / SDK 2.32): TP=1 (kv_heads=4/rank) collapses to a
#     repeated token from decode step 1; TP=4 (kv_heads=1/rank) degrades
#     gracefully. In this port the same defect surfaced as pure
#     whitespace output ('\n \n \n ...') despite a passing determinism check and
#     bit-exact static correctness (RoPE / RMSNorm / prefill attention).
#
#     A differential test localized it: swapping ONLY the attention CORE
#     kernel for torch -- keeping the SAME mask, scale, o_proj, scatter and cache
#     -- recovered 0/12 -> 10/12. Mask (fused vs pre-built), cache aliasing
#     (clone), scale, RoPE, QK-norm, o_proj and scatter are all RULED OUT. The
#     defect is the stock NKI kernel's in-graph d256 GQA-fold decode output for
#     the novel combination (update_cache=False + W_out=None + d_head>128 + GQA
#     per-head table) that no shipping model exercises.
#
#     Current default (``GEMMA3_STOCK_DECODE=auto``): the stock kernel for decode
#     batch >= 2 (on SDK 2.33 it is correct and much faster there) and
#     ``_forward_decode_torch_baseline`` (the plugin's own
#     ``_torch_attention_decode_impl``) at batch 1.
#
# X5 -- DEFERRED + COALESCED KV WRITE-BACK.
#     34 layers share only ~10 physical KV tensors (5 K + 5 V, ~7 layers apiece).
#     Each layer's ``index_put_`` is functionalized by lite's
#     ``InPlaceToOutOfPlacePass`` into ``index_put``, which PRODUCES a new
#     full-size tensor; ``AliasingOutputRewritePass`` can alias only the FINAL
#     result of a chain back to the input buffer, so the 6 intermediates become
#     REAL full-tensor HBM copies. Measured (device profile, Gemma3-4B, eager decode
#     ml=1024 ctx=768): 40 DMA packets of exactly 212,598,784 B = 8.504 GB, in a
#     fully serialized ping-pong chain spanning 13.686 -> 43.403 ms of a 47.2 ms
#     decode step (63%).
#
#     FIX: each layer DEFERS its KV scatter into a per-step list; the backbone
#     issues ONE ``index_put_`` per PHYSICAL cache tensor after the layer loop.
#     Measured TPOT 62.5 -> 20.0 ms (3.12x) at b1 TP=1 and 50.8 -> 8.0 ms (6.38x)
#     at b1 TP=4 (768-in/128-out, ml=1024, gpu_memory_utilization=0.9).
#     The ENTIRE speedup is from DEFERRAL. The data_ptr coalescing on top
#     takes 68 -> 10 scatters, which is a DMA-DESCRIPTOR-COUNT
#     reduction, not a byte reduction (deferral had already taken HBM write from
#     41.10 to 0.186 GB/step), so its own headroom is a few percent at most.
#
#     NOT APPLICABLE TO PREFILL, structurally -- see the comment in
#     ``_write_paged_kv_cache``.
#
# ENV GATES AND THEIR DEFAULTS (all read from ``os.environ`` at call time, never
# captured at import, so they can be flipped between engine launches):
#
#   GEMMA3_STOCK_DECODE   default "auto" -> stock NF.attention_decode for decode
#                                      batch >= 2, torch fallback at batch 1;
#                                      "1" = always stock, "0" = always torch
#   GEMMA3_DECODE_TORCH   default "0"  "1" -> force the torch baseline (redundant
#                                      at batch 1, kept for explicitness)
#   GEMMA3_KV_COALESCE    default "1"  "0" -> no deferral; per-layer interleaved
#                                      index_put_ (the pre-X5 behaviour)
#   GEMMA3_PREFILL_KV_COALESCE default "1" "0" -> inline per-layer prefill KV
#                                      write. Deferral applies only to single-
#                                      shot prefill; segmented always inline.
#   GEMMA3_KV_PTR_GROUP   default "1"  "0" -> deferral WITHOUT data_ptr grouping
#                                      (68 scatters; isolates deferral from
#                                      coalescing for A/B)
#
# NOTE ON WHERE THE ENV READS LIVE: every gate is read with a bare
# ``os.environ.get(...)`` INSIDE the traced forward, not captured at import time.
# That is deliberate and was validated on hardware --
# Dynamo treats the returned string as a constant and folds the branch, so the
# compiled graph contains only the selected arm. Do NOT hoist these into
# module-level constants (it removes the ability to A/B without a reinstall) and
# do NOT mutate any module-level state from inside the traced region: that is a
# Dynamo SIDE EFFECT which forces a recompile, and the runner sets
# ``torch.compiler.set_stance("fail_on_recompile")``, so it kills the engine
# outright (measured).
# =============================================================================



# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Gemma3's RMSNorm is (1 + gamma) * norm(x), NOT
#     gamma * norm(x). HF (modeling_gemma3.Gemma3RMSNorm.forward):
#
#         output = x.float() * rsqrt(x.float().pow(2).mean(-1) + eps)
#         output = output * (1.0 + self.weight.float())
#         return output.type_as(x)
#
#     Note also that HF initializes the gamma to ZEROS (so an untrained norm is
#     the identity) and multiplies in fp32 BEFORE the downcast -- unlike Llama,
#     which does ``x.to(dtype) * w``. See huggingface/transformers#29402.
#
#     We keep the standard ``w * x`` form in this module and FOLD THE "+1" INTO
#     THE LOADED GAMMA (``load_weights`` adds 1.0 to every RMSNorm tensor listed
#     in ``_RMSNORM_PARAM_SUFFIXES``). This is not cosmetic: the decode
#     mega-kernel's ``rmsnorm_QK_pre_rope_W_Q`` / ``_W_K`` arguments apply gamma
#     as ``w * x`` with no way to express ``(1 + w)``, so q_norm/k_norm MUST be
#     pre-incremented for decode to match prefill. Folding ALL norms the same
#     way keeps one convention instead of two.
#
#     Consequence: ``self.weight`` is initialized to ONES (== HF's ``1 + 0``),
#     so an unloaded norm is the identity, matching HF at init.
# =============================================================================


class Gemma3RMSNorm(nn.Module):
    """RMS Normalization. Gamma is stored PRE-INCREMENTED (HF gamma + 1).

    Computes ``(w * norm(x))`` in fp32 then casts, which reproduces HF's
    ``(norm(x) * (1 + gamma)).type_as(x)`` exactly once ``w == 1 + gamma``.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        # Ones, not zeros: this parameter holds (HF_gamma + 1).
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # fp32 multiply BEFORE the downcast -- matches Gemma3, not Llama.
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (DUAL)
# <-- MODEL-SPECIFIC: Gemma3 keeps TWO RoPE tables alive simultaneously and
#     selects per layer:
#       * sliding ("local") layers:  theta = rope_local_base_freq (1e4),
#                                    rope_type "default" -> NO scaling
#       * global  ("full")  layers:  theta = rope_theta (1e6) with
#                                    rope_type "linear", factor 8
#
#     HF applies linear scaling to the INVERSE FREQUENCIES
#     (``modeling_rope_utils._compute_linear_scaling_rope_parameters``:
#     ``inv_freq /= factor``), not to the position ids. The two are
#     algebraically equivalent (``emb = inv_freq @ positions``) but NOT
#     bit-identical in floating point, so we divide inv_freq to match HF.
#
#     Rotation style is rotate_half (non-interleaved halves), same as
#     Llama/Qwen -- ``modeling_gemma3.rotate_half`` + ``apply_rotary_pos_emb``.
#
# CRITICAL -- meta tensor avoidance: vLLM builds the model under
# ``torch.device("meta")``. Anything computed in ``__init__`` becomes a meta
# tensor and later fails with "Cannot copy out of meta tensor; no data!".
# ``__init__`` therefore stores ONLY python scalars; inv_freq is computed
# lazily in ``_compute_inv_freq(device)``, called from ``forward()``.
# =============================================================================


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: rotate_half style (contiguous halves)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    <-- MODEL-SPECIFIC: rotate_half style. Called AFTER QK-norm in Gemma3.

    Args:
        q: [Nh, T, Dh]  k: [Nkv, T, Dh]
        cos: [T, Dh]    sin: [T, Dh]
    """
    cos = cos.unsqueeze(0)  # [1, T, Dh]
    sin = sin.unsqueeze(0)  # [1, T, Dh]
    return (
        q * cos + _rotate_half(q) * sin,
        k * cos + _rotate_half(k) * sin,
    )


class Gemma3RotaryEmbedding(nn.Module):
    """Dual RoPE. Emits BOTH the sliding and global cos/sin per forward call.

    <-- MODEL-SPECIFIC: local theta 1e4 (no scaling) + global theta 1e6 with
    linear scaling factor 8. ``attention_scaling`` is 1.0 for both RoPE types
    in Gemma3 (``_compute_linear_scaling_rope_parameters`` and
    ``compute_default_rope_parameters`` both return ``attention_factor = 1.0``),
    so there is no post-hoc cos/sin scaling to apply.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.head_dim = config.head_dim
        # Python scalars only -- see the meta-tensor note above.
        self.local_base = float(config.rope_local_base_freq)
        self.global_base = float(config.rope_theta)
        # Linear scaling applies to the GLOBAL layers only.
        self.global_linear_factor = 1.0
        if config.rope_scaling and config.rope_scaling.get("rope_type") == "linear":
            self.global_linear_factor = float(config.rope_scaling["factor"])

    def _compute_inv_freq(
        self, base: float, linear_factor: float, device: torch.device
    ) -> torch.Tensor:
        """Lazily compute inv_freq on the tensor's own device (never in __init__)."""
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
                / self.head_dim
            )
        )
        if linear_factor != 1.0:
            # HF divides inv_freq (not the positions) -- see Section 2 header.
            inv_freq = inv_freq / linear_factor
        return inv_freq

    def _cos_sin(
        self,
        base: float,
        linear_factor: float,
        position_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._compute_inv_freq(base, linear_factor, device)  # [Hd/2]
        inv_freq_expanded = inv_freq[:, None].float()  # [Hd/2, 1]
        positions_expanded = position_ids[None, :].float()  # [1, T]
        freqs = (inv_freq_expanded @ positions_expanded).transpose(0, 1)  # [T, Hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Hd]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Return ``{"sliding": (cos, sin), "global": (cos, sin)}``.

        The backbone picks the table per layer; the decoder layer and attention
        module see a plain ``(cos, sin)`` tuple, exactly as in Qwen3/Llama.
        """
        return {
            "sliding": self._cos_sin(
                self.local_base, 1.0, position_ids, device, dtype
            ),
            "global": self._cos_sin(
                self.global_base,
                self.global_linear_factor,
                position_ids,
                device,
                dtype,
            ),
        }


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (GQA, per-head QK-norm before RoPE,
#        query_pre_attn_scalar softmax scale, per-layer sliding window,
#        per-layer RoPE table selection, head_dim=256)
#
# HEAD_DIM = 256 NOTES (these are kernel constraints, not model choices):
#   * The decode mega-kernel cannot fold the output projection for
#     ``head_dim > 128``, so this port passes ``W_out=None`` and runs
#     ``NF.o_proj`` separately. It likewise uses ``update_cache=False`` and
#     scatters the returned K/V itself.
#   * ``NF.flash_attention`` has ``MAX_HEAD_DIM = 128`` and SILENTLY falls back
#     to the torch implementation above it (no exception). Correct, slow.
#   * ``NF.segmented_attention`` has ``_MAX_HEAD_DIM = 128`` and RAISES. The
#     segmented/APC prefill path therefore cannot run until that cap is lifted
#     in ``vllm_neuron/functional/attention/attention_segmented_cte.py``. See
#     README "Known gaps".
# =============================================================================


class Gemma3Attention(nn.Module):
    """Multi-head attention with TP head sharding, per-head QK-norm and a
    per-layer sliding window.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size (GQA)
    - Prefill: all-gather -> QKV -> QK-norm -> RoPE -> attn -> O proj ->
      reduce-scatter
    - Decode: fused mega-kernel + TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA: 8 Q heads / 4 KV heads, head_dim 256
    - Per-head QK-norm (head_dim RMSNorm) applied BEFORE RoPE
    - Softmax scale = query_pre_attn_scalar ** -0.5
    - Sliding window 1024 on 5 of every 6 layers; every 6th layer is global
    - Per-layer RoPE table (local theta vs global theta)
    - No attention bias, no attention sinks, no logit soft-capping
    """

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # <-- MODEL-SPECIFIC: Gemma softmax scale. For Gemma3-4B
        # query_pre_attn_scalar == head_dim == 256 so the numeric value happens
        # to match 1/sqrt(head_dim), but the config fields are independent and
        # other Gemma3 sizes do differ. Always use query_pre_attn_scalar.
        self.scaling = float(config.query_pre_attn_scalar) ** -0.5

        # <-- MODEL-SPECIFIC: per-layer hybrid attention. `is_sliding` comes
        # from the config so the schedule (layer_types list, else the
        # `(i + 1) % pattern != 0` rule) lives in exactly one place. Do NOT use
        # gpt_oss's `layer_idx % 2 == 0` -- that is GPT-OSS's own schedule.
        self.is_sliding = config.is_sliding(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None
        # Which RoPE table this layer consumes.
        self.rope_key = "sliding" if self.is_sliding else "global"

        # head_dim > 128: the stock decode kernel can fold NEITHER the output
        # projection nor the KV cache update (both are kernel_asserts).
        # GEMMA3_FUSED_OUT_PROJ (default "auto"): in-kernel output projection for d>128 when the
        # installed NKI Library supports it (SUPPORTS_D256_OUT_PROJ); "0" forces the external o_proj.
        _fop = os.environ.get("GEMMA3_FUSED_OUT_PROJ", "auto")
        self.fused_out_proj = self.head_dim <= _MAX_FUSED_OUT_PROJ_HEAD_DIM or (
            self.head_dim % 128 == 0 and (_fop == "1" or (_fop == "auto" and _NKILIB_D256_OUT_PROJ))
        )

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Head sharding <<<
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes <<<
        # <-- MODEL-SPECIFIC: NON-SQUARE QKV. num_attention_heads * head_dim
        # (8 * 256 = 2048) != hidden_size (2560), so q_size must be derived from
        # the head counts and head_dim, never from hidden_size.
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = self.num_attention_heads * self.head_dim // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        # <-- MODEL-SPECIFIC: Per-head QK-norm (on head_dim, not hidden_size),
        # applied BEFORE RoPE. Gamma is stored pre-incremented (Section 1).
        self.q_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)
        self.k_norm = Gemma3RMSNorm(self.head_dim, config.rms_norm_eps, self.dtype)

        # KV caches bound externally via bind_kv_cache().
        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of QKV and O-proj weights. <<<
        <-- MODEL-SPECIFIC: separate Q, K, V in the checkpoint, NO bias
        (attention_bias=False). HF stores linear weights transposed.
        """
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_attention_heads * self.head_dim // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # ── Forward dispatch ──────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        kv_writeback: list | None = None,
    ):
        """>>> PARALLELISM: Dispatch to prefill or decode. <<<

        X5: ``kv_writeback`` is the per-step deferred-KV list threaded down from
        ``Gemma3Model.forward``. It is only consumed by the DECODE path -- prefill
        must write its KV cache inline (both prefill kernels read the slots back;
        see ``_write_paged_kv_cache``).
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_mask,
                attn_metadata,
                kv_writeback=kv_writeback,
            )
        else:
            # >>> PARALLELISM: All-gather from SP <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
                kv_writeback=kv_writeback,
            )

    # ── Prefill path ──────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        kv_writeback: list | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: QKV projection. RoPE is NOT fused into the kernel here
        # because QK-norm must run BETWEEN the projection and RoPE.
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
            bias=None,  # <-- MODEL-SPECIFIC: attention_bias=False
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        # Reshape to per-head layout: [T, Nh, Dh] -> [Nh, T, Dh]
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # Step 2a: <-- MODEL-SPECIFIC: per-head QK-norm BEFORE RoPE.
        # q/k are [Nh, T, Dh]; the norm reduces over the last dim (head_dim).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Step 2b: <-- MODEL-SPECIFIC: rotate_half RoPE with the table the
        # backbone selected for this layer (local theta on sliding layers,
        # scaled global theta on global layers).
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 3: Update KV cache.
        # >>> PARALLELISM: cache is per-rank (TP-sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]
        cached_seq_len = attn_metadata[layer_name].get("cached_seq_len")
        kv_segment_size = attn_metadata[layer_name].get("kv_segment_size")

        # PREFILL KV WRITE-BACK (GEMMA3_PREFILL_KV_COALESCE, default on; "0" restores inline):
        # the single-shot path (no kv_segment_size) attends with flash_attention
        # over the raw k/v locals below and never reads this layer's cache back,
        # so its write is a pure output and may be deferred + coalesced like
        # decode. The segmented path DOES read the active segment back from the
        # cache, so it keeps writing inline.
        if kv_writeback is not None and not kv_segment_size:
            self._defer_paged_kv_write(k, v, slot_mapping, block_size, kv_writeback)
        else:
            self._write_paged_kv_cache(k, v, slot_mapping, block_size)

        # Step 4: Attention.
        # <-- MODEL-SPECIFIC: per-layer sliding window; no sinks, no soft-cap.
        if kv_segment_size:
            # Segmented (chunked / prefix-caching) prefill. Reads prior K/V back
            # out of the paged cache over block_tables with an online softmax,
            # so it never materializes an O(ctx^2) score matrix.
            attn_output = NF.segmented_attention(
                q,  # [Nh, T, Dh] (tp_q=True layout)
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                block_tables=block_table,
                prior_tokens=cached_seq_len,
                block_size=block_size,
                kv_segment_size=kv_segment_size,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=True,
                tp_out=True,
            )  # [Nh, Dh, T]
        else:
            # Full (single-chunk) prefill: flash attention over the raw K/V.
            # GQA: NF.flash_attention can also handle GQA natively (no repeat), but
            # on trn2 it made no difference (prefill within
            # +-3%, E2E within +-1.6%), so the explicit repeat stays the default.
            # GEMMA3_PREFILL_GQA_REPEAT=0 selects the native path.
            if os.environ.get("GEMMA3_PREFILL_GQA_REPEAT", "1") == "1":
                k = k.repeat_interleave(self.num_key_value_groups, dim=0)
                v = v.repeat_interleave(self.num_key_value_groups, dim=0)

            q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
            k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
            v_flash = v  # [Nh, T, Dh]

            # Packed multi-request prefill (GEMMA3_PACKED_PREFILL=1 with a scheduler that packs several
            # prompts into one step): keep each token's attention inside its own request with per-query
            # KV bounds. A request starts where the position resets, so bound_min[i] = i - positions[i]
            # (the first token of i's request) and bound_max[i] = i + 1 (causal). For a single request
            # bound_min is 0 everywhere, so this is a no-op relative to plain causal attention.
            _bmin = _bmax = None
            if os.environ.get("GEMMA3_PACKED_PREFILL", "0") == "1" and positions is not None:
                _idx = torch.arange(tokens, device=positions.device, dtype=torch.int32)
                _pos = positions.reshape(-1).to(torch.int32)
                _bmin = (_idx - _pos).clamp(min=0).reshape(1, tokens, 1)
                _bmax = (_idx + 1).reshape(1, tokens, 1)
                _nh = q_flash.shape[0]
                _bmin = _bmin.expand(_nh, tokens, 1).contiguous()
                _bmax = _bmax.expand(_nh, tokens, 1).contiguous()
            attn_output = NF.flash_attention(
                q_flash,
                k_flash,
                v_flash,
                scale=self.scaling,
                sliding_window=self.sliding_window,
                tp_q=False,
                tp_out=True,
                bound_min=_bmin,
                bound_max=_bmax,
            )  # [Nh, Dh, T]

        # Step 5: Output projection.
        attn_output = attn_output.unsqueeze(0)  # [1, Nh, Dh, T]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight, None)  # [1, T, H]
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    def _defer_paged_kv_write(self, k, v, slot_mapping, block_size, kv_writeback):
        """Append this layer's prefill KV write to ``kv_writeback`` instead of
        issuing it. Same index math as ``_write_paged_kv_cache``; same tuple
        layout ``_coalesce_kv_writes`` consumes."""
        nkh = self.num_key_value_heads_per_rank
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        k_flat = _to_cache_dtype(k.reshape(-1, self.head_dim), self.k_cache.dtype)
        v_flat = _to_cache_dtype(v.reshape(-1, self.head_dim), self.v_cache.dtype)
        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=slot_mapping.device
        ).repeat_interleave(slot_mapping.shape[0])
        kv_writeback.append(
            (
                self.k_cache,
                self.v_cache,
                k_flat,
                v_flat,
                block_indices.repeat(nkh),
                head_indices_for_put,
                position_indices.repeat(nkh),
                self.layer_idx,
                slot_mapping,
            )
        )

    def _write_paged_kv_cache(self, k, v, slot_mapping, block_size):
        """Scatter post-RoPE K/V into the paged cache at ``slot_mapping``.

        >>> PARALLELISM: per-rank cache (TP-sharded KV heads) <<<
        The cache layout is the standard bf16
        ``[num_blocks, kv_heads_per_rank, block_size, head_dim]``. This port
        does not implement an FP8 KV cache (``bind_kv_cache`` rejects one), so
        there is no quantize-on-write branch.

        X5 DOES **NOT** APPLY HERE, and the reasons are STRUCTURAL, not untested:

        (1) DEFERRAL IS ILLEGAL. Both prefill kernels READ BACK the slots written
            here. ``attention_segmented_cte``'s SWA implementation loads the
            ACTIVE segment's K/V FROM THE CACHE, not from the ``k``/``v`` locals
            in this frame. Proven by controlled experiment: zeroing ONLY the
            active segment's cache slots changes 99.99967% of output elements and
            drives absmax to exactly 0.0. Deferring past the layer loop would
            make every layer attend to a cache missing its own active segment --
            silently wrong output, not a crash.

        (2) COALESCING IS UNREACHABLE EVEN IF DEFERRAL WERE LEGAL. Merging the
            writes of the ~7 layers sharing one storage requires their K/V to
            EXIST AT THE SAME TIME, but the data_ptr groups are CONTIGUOUS layer
            runs ([0..6], [7..13], [14..20], [21..27], [28..33]) and the residual
            stream makes those layers STRICTLY SEQUENTIAL -- layer 6's K/V cannot
            be computed until layer 5's attention has run, which by (1) needs
            layer 5's own cache write to have landed. The concatenation cannot
            even be FORMED.

        Decode escapes both because it writes ONE token per layer and the decode
        path consumes ``K_new``/``V_new`` DIRECTLY (never re-reading the slot it
        just wrote), so its writes are pure outputs with no in-step reader.
        """
        nkh = self.num_key_value_heads_per_rank
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        k_flat = _to_cache_dtype(k.reshape(-1, self.head_dim), self.k_cache.dtype)
        v_flat = _to_cache_dtype(v.reshape(-1, self.head_dim), self.v_cache.dtype)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=k.device
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)
        index = (block_indices_for_put, head_indices_for_put, position_indices_for_put)

        self.k_cache.index_put_(index, k_flat)
        self.v_cache.index_put_(index, v_flat)

    # ── Decode path ───────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None,
        attn_metadata: object,
        kv_writeback: list | None = None,
    ):
        """Decode dispatch for ``head_dim = 256``.

        X4 -- default ``GEMMA3_STOCK_DECODE=auto``: the fused
        ``NF.attention_decode`` mega-kernel for decode batch >= 2 and
        ``_forward_decode_torch_baseline`` (the plugin's own
        ``_torch_attention_decode_impl``) at batch 1. On SDK 2.32 / neuronx-cc
        2.27 the stock ``d_head > 128`` GQA decode path was numerically wrong
        inside the full model (see the X4 block at the top of this file).

        ``GEMMA3_STOCK_DECODE=1`` / ``0`` force the stock kernel / the fallback.
        ``GEMMA3_DECODE_TORCH=1`` forces the fallback regardless (it wins over
        ``GEMMA3_STOCK_DECODE``).
        """
        # Read inline: Dynamo folds
        # the string compare to a constant and only the taken arm is traced.
        # Default "auto": stock NKI kernel for B >= 2, torch fallback at B == 1
        # (stock is 1.50x at b16 / 5.23x at b64 with exact parity, but 0.86x at
        # b1). B is the static decode bucket, so this is a
        # trace-time choice, not a runtime branch.
        _sd = os.environ.get("GEMMA3_STOCK_DECODE", "auto")
        if _sd == "auto":
            _B = attn_metadata[f"layers.{self.layer_idx}.self_attn"]["block_table_tensor"].shape[0]
            _use_stock = _B >= 2
        else:
            _use_stock = _sd == "1"
        if os.environ.get("GEMMA3_DECODE_TORCH", "0") == "1" or not _use_stock:
            return self._forward_decode_torch_baseline(
                hidden_states,
                positions,
                position_embeddings,
                attn_mask,
                attn_metadata,
                kv_writeback=kv_writeback,
            )
        return self._forward_decode_stock(
            hidden_states,
            positions,
            position_embeddings,
            attn_mask,
            attn_metadata,
            kv_writeback=kv_writeback,
        )

    def _forward_decode_torch_baseline(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None,
        attn_metadata: object,
        kv_writeback: list | None = None,
    ):
        """X4: THE DEFAULT decode path -- ``_torch_attention_decode_impl``.

        This is the plugin's own PyTorch reference implementation of the fused
        TKG attention block, called with exactly the same Gemma3 arguments as the
        stock kernel would receive:

          * ``rmsnorm_QK_pre_rope_enabled=True`` with the PRE-INCREMENTED
            q_norm/k_norm gammas viewed as ``[1, head_dim]`` (Section 1 -- the
            fallback applies gamma as ``w * x`` with no ``(1 + w)`` form, exactly
            like the kernel, so the fold is what makes decode match prefill);
          * ``cos``/``sin`` sliced to ``head_dim // 2`` and permuted to
            ``(half_d, B, S_decode)`` with ``rope_contiguous_layout=True``
            (== rotate_half; the impl reconstructs the second half);
          * ``softmax_scale=self.scaling`` (query_pre_attn_scalar ** -0.5);
          * ``sink=None`` -- Gemma3 has no attention sinks;
          * ``update_cache=False`` and ``W_out=None`` -- head_dim=256, so both the
            KV write and the output projection happen externally here. NOTE these
            are NOT kernel_assert constraints in the torch impl (it can do both),
            but keeping them external makes this path a drop-in swap for the
            stock branch and lets X5 defer the cache write.

        CONSTRAINT (important, and the reason this is TP-degree sensitive): the
        torch fallback REJECTS a per-head 3D block table with ``kv_heads > 1``
        (``NotImplementedError`` -- its gather flattens the table as
        ``[B, num_blocks]``). So a 2D ``block_table`` is passed and the 4D cache is
        handed over as-is; ``_infer_kv_heads`` reads ``kv_heads`` off the cache's
        dim 1, and the impl's own 4D gather path handles GQA correctly. This is
        what was validated at
        BOTH TP=1 (nkh=4) and TP=4 (nkh=1).

        X5: when ``kv_writeback`` is a list this method does NOT perform its own
        ``index_put_``. It APPENDS the write to that list and the backbone issues
        one coalesced scatter per physical cache tensor after the layer loop. See
        ``_coalesce_kv_writes``.
        """
        from vllm_neuron.functional.attention.attention_decode import (
            _torch_attention_decode_impl,
        )

        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank
        nqh = self.num_attention_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # RoPE in kernel layout: [T, Dh] -> [Dh/2, B, S]. Only the first half is
        # passed; ``rope_contiguous_layout=True`` reconstructs the second half
        # (== rotate_half). Identical to the stock branch below.
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        # 2D table + 4D cache: the torch fallback cannot take a per-head 3D table
        # when kv_heads > 1 (see the docstring).
        active_blocks_table = block_table.to(torch.int32)

        output, K_new, V_new = _torch_attention_decode_impl(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,  # RMSNorm applied by the decoder layer
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,  # <-- MODEL-SPECIFIC: attention_bias=False
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            active_blocks_table=active_blocks_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attn_mask,
            softmax_scale=self.scaling,
            sink=None,  # <-- MODEL-SPECIFIC: Gemma3 has no attention sinks
            update_cache=False,  # head_dim=256 -> external scatter below
            W_out=None,  # head_dim=256 -> external NF.o_proj below
            bias_out=None,
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        # <-- MODEL-SPECIFIC: external output projection (head_dim=256).
        # output: [B, q_heads, d_head, S_decode] -> [1, B*S_decode, q_heads*d_head]
        attn_ctx = (
            output.permute(0, 3, 1, 2)  # [B, S_decode, q_heads, d_head]
            .reshape(B * S_decode, nqh * self.head_dim)
            .unsqueeze(0)
        )
        output = NF.o_proj(attn_ctx, self.o_proj_weight, None).squeeze(0)

        # >>> External KV-cache scatter (or X5 deferral) <<<
        # The torch impl returns K_new [d_head, B*nkh, S_decode] and
        # V_new [B, nkh, S_decode, d_head]. Both are reshaped to the flat
        # [nkh * B * S_decode, d_head] layout the (block, head, position)
        # index_put_ expects.
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]
        k_new = (
            K_new.permute(1, 2, 0)  # [B*nkh, S_decode, d_head]
            .reshape(B, nkh, S_decode, self.head_dim)
            .transpose(0, 1)  # [nkh, B, S_decode, d_head]
            .reshape(nkh, B * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)
        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        if kv_writeback is not None:
            # X5: DEFER the cache write. The backbone coalesces every layer that
            # shares this physical cache tensor into ONE index_put_.
            # ``layer_idx`` is carried so ``_coalesce_kv_writes`` can look this
            # layer up in the static data_ptr group table resolved in
            # ``bind_kv_cache``; ``slot_mapping`` is carried so it can enforce the
            # disjointness guard by OBJECT identity (free at trace time).
            kv_writeback.append(
                (
                    self.k_cache,
                    self.v_cache,
                    _to_cache_dtype(k_new_flat, self.k_cache.dtype),
                    _to_cache_dtype(v_new_flat, self.v_cache.dtype),
                    block_indices_for_put,
                    head_indices_for_put,
                    position_indices_for_put,
                    self.layer_idx,
                    slot_mapping,
                )
            )
        else:
            index = (
                block_indices_for_put,
                head_indices_for_put,
                position_indices_for_put,
            )
            self.k_cache.index_put_(index, _to_cache_dtype(k_new_flat, self.k_cache.dtype))
            self.v_cache.index_put_(index, _to_cache_dtype(v_new_flat, self.v_cache.dtype))

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output

    def _forward_decode_stock(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None,
        attn_metadata: object,
        kv_writeback: list | None = None,
    ):
        """>>> PARALLELISM: Fused mega-kernel with TP. <<<

        X4: used for decode batch >= 2 by default (``GEMMA3_STOCK_DECODE=auto``).
        On SDK 2.32 / neuronx-cc 2.27 this path was numerically wrong in the
        full model; on SDK 2.33 it passes the fp32 first-divergence gate.

        <-- MODEL-SPECIFIC:
        - QK-norm via ``rmsnorm_QK_pre_rope_enabled=True``, passing the
          pre-incremented gammas so decode matches prefill exactly.
        - Softmax scale ``query_pre_attn_scalar ** -0.5``.
        - The per-layer sliding-window / causal pattern arrives as the
          pre-computed ``attn_mask`` built once per step by
          ``_precompute_decode_attn_masks`` (gpt_oss pattern).
        - No sinks, no soft-cap, no attention bias.
        - ``head_dim = 256`` forces ``W_out=None`` + external ``NF.o_proj`` and
          ``update_cache=False`` + external scatter (see Section 3 header).
        - GQA with ``kv_heads_per_rank > 1`` (i.e. TP < 4) needs a PER-HEAD 3D
          block table to reach the kernel at all.
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B

        hidden_states = hidden_states.to(self.dtype)
        nkh = self.num_key_value_heads_per_rank
        nqh = self.num_attention_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        # RoPE in mega-kernel layout: [T, Dh] -> [Dh/2, B, S]. Only the first
        # half is passed; the kernel reconstructs the second half
        # (``rope_contiguous_layout=True`` == rotate_half).
        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        # >>> GQA table / cache selection <<<
        # nkh > 1 reaches the NKI kernel only via a per-head 3D block table with
        # the 4D cache kept as-is; nkh == 1 uses the 2D table + squeezed 3D
        # cache. Shared helper -- see the import note at the top of the file.
        # GEMMA3_KERNEL_KV_UPDATE: "auto" (default) = in-kernel KV write only when the neuronx-cc modular-flow
        # partitioner is disabled (VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD >= 1e12). With the plugin's default
        # threshold (10) a partition boundary lands inside the last KV buffer's alias chain and materialises a
        # 2.79 GB intermediate (+50 ms/step). Measured b64: 17.93 vs 18.35 ms, parity 12/12.
        _kv_in_kernel = _resolve_kernel_kv_update()
        _kv_layers = os.environ.get("GEMMA3_KV_UPD_LAYERS", "")
        if _kv_in_kernel and _kv_layers:
            _kv_in_kernel = self.layer_idx in {int(x) for x in _kv_layers.split(",") if x}
        _pp = False
        _chain = None
        if nkh > 1:
            active_blocks_table = build_per_head_block_table(
                block_table.to(torch.int32), nkh
            )
            # nkh > 1 always passes the bound 4D [num_blocks, nkh, block, d] cache (the kernel
            # folds it head-inner to num_blocks*nkh blocks). With the in-kernel write, thread the
            # same per-storage ping-pong chain as nkh == 1 so same-storage writers form one
            # in-place alias chain instead of each copying the full buffer.
            _pp = _kv_in_kernel and os.environ.get("GEMMA3_KV_PINGPONG", "1") == "1"
            _chain = attn_metadata.get("_g3_kv_chain") if _pp else None
            if _chain is not None and getattr(self, "kv_gid", None) in _chain:
                k_cache_arg, v_cache_arg = _chain[self.kv_gid]
            else:
                k_cache_arg = self.k_cache
                v_cache_arg = self.v_cache
        else:
            active_blocks_table = block_table.to(torch.int32)
            _pp = _kv_in_kernel and os.environ.get("GEMMA3_KV_PINGPONG", "1") == "1"
            _chain = attn_metadata.get("_g3_kv_chain") if _pp else None
            if _chain is not None and getattr(self, "kv_gid", None) in _chain:
                k_cache_arg, v_cache_arg = _chain[self.kv_gid]
            elif _kv_in_kernel and os.environ.get("GEMMA3_KV_UPD_4D", "1") == "1":
                # In-kernel cache write: pass the BOUND cache tensors themselves
                # (4D [num_blocks, 1, block, d]; the kernel folds the kv-head axis
                # internally). The in-place write is threaded back via FX aliasing of
                # the kernel's cache outputs onto these inputs; a squeeze(1) view
                # would break that aliasing (functional copy + stale reads).
                k_cache_arg = self.k_cache
                v_cache_arg = self.v_cache
            else:
                k_cache_arg = (
                    self.k_cache.squeeze(1) if self.k_cache.dim() == 4 else self.k_cache
                )
                v_cache_arg = (
                    self.v_cache.squeeze(1) if self.v_cache.dim() == 4 else self.v_cache
                )

        # <-- MODEL-SPECIFIC kernel call. Note bias_qkv / bias_out are None
        # (Gemma3 has no attention bias). If a variant ever added one it MUST be
        # passed as 2D `[1, size]` via `.unsqueeze(0)`; the mega-kernel rejects
        # 1D bias on hardware ("Bias shape must be [1, I]") while the CPU
        # simulator accepts it.
        # GEMMA3_KERNEL_KV_UPDATE=1: the kernel scatters the new K/V into the paged cache
        # itself (block KV; needs the patched nkilib with d>128 update_cache support), and the
        # external deferred/coalesced write below is skipped for this layer.
        _skip_null = os.environ.get("GEMMA3_KV_UPD_SKIPNULL", "1") == "1"
        _kv_upd_idx = None
        if _kv_in_kernel:
            # Padding rows: drop the write with an OOB index (the kernel scatter uses
            # oob_mode.skip) instead of letting every padding row hit null slot 0.
            _slot_oob = (
                (slot_mapping < 0)
                | (slot_mapping >= self.k_cache.shape[0] * self.k_cache.shape[-2])
                | (slot_mapping < (block_size if _skip_null else 0))  # block 0 = vLLM null block
            )
            if nkh == 1:
                # [B, S_tkg] base slots (unchanged nkh == 1 path).
                _kv_upd_idx = (
                    torch.where(_slot_oob, torch.full_like(slot_mapping, 2**31 - 1), slot_mapping)
                    .view(B, S_decode)
                    .to(torch.uint32)
                )
            else:
                # nkh > 1: the block-KV kernel wants [B, nkh, S_tkg] GLOBAL slots into the
                # head-inner num_blocks*nkh pool (global block = base*nkh + h), the same pool
                # build_per_head_block_table targets. Map OOB rows to slot 0 first so the
                # per-head arithmetic stays in range, then re-apply the skip sentinel per head.
                _per_head = build_per_head_slot_mapping(
                    torch.where(_slot_oob, torch.zeros_like(slot_mapping), slot_mapping),
                    block_size,
                    nkh,
                    B,
                )  # [B, nkh, S_decode]
                _kv_upd_idx = torch.where(
                    _slot_oob.view(B, 1, S_decode).expand(B, nkh, S_decode),
                    torch.full_like(_per_head, 2**31 - 1),
                    _per_head,
                ).to(torch.uint32)
        _res = NF.attention_decode(
            X=X,
            X_hidden_dim_actual=self.hidden_size,
            rmsnorm_X_enabled=False,  # RMSNorm applied by the decoder layer
            W_qkv=self.qkv_proj_weight,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_eps=self.q_norm.variance_epsilon,
            rmsnorm_QK_pre_rope_W_Q=self.q_norm.weight.view(1, -1),
            rmsnorm_QK_pre_rope_W_K=self.k_norm.weight.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache_arg,
            V_cache=v_cache_arg,
            attention_mask=attn_mask,
            softmax_scale=self.scaling,
            sink=None,  # <-- MODEL-SPECIFIC: Gemma3 has no attention sinks
            update_cache=_kv_in_kernel,
            **({"return_updated_kv": True} if _chain is not None else {}),
            kv_cache_update_idx=_kv_upd_idx,
            W_out=self.o_proj_weight if self.fused_out_proj else None,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
            **(
                {"k_scale": self.kv_scale_t, "v_scale": self.kv_scale_t}
                if getattr(self, "kv_scale_t", None) is not None
                else {}
            ),
            attention_dp=1,
            attention_dp_group=None,
            attention_dp_rank=0,
            kv_needs_a2a=False,
        )

        if _kv_in_kernel:
            if _chain is not None and isinstance(_res, (tuple, list)):
                output, _k_upd, _v_upd = _res
                _chain[self.kv_gid] = (_k_upd, _v_upd)
            else:
                output = _res[0] if isinstance(_res, (tuple, list)) else _res
            K_new = V_new = None
        else:
            output, K_new, V_new = _res

        if not self.fused_out_proj:
            # output: [B, q_heads, d_head, S_decode] ->
            #         [1, B*S_decode, q_heads*d_head]
            attn_ctx = (
                output.permute(0, 3, 1, 2)  # [B, S, q_heads, d_head]
                .reshape(B * S_decode, nqh * self.head_dim)
                .unsqueeze(0)
            )
            output = NF.o_proj(attn_ctx, self.o_proj_weight, None).squeeze(0)

        if _kv_in_kernel:
            # The kernel already wrote the cache; skip the external scatter but NOT the TP all-reduce
            # below (an early return here skipped it, so each rank kept only its partial o_proj sum).
            if self.world_size > 1:
                self.tp_group.all_reduce(output)
            return output

        # >>> External KV-cache scatter of the kernel's new K/V tokens <<<
        # With nkh == 1 the kernel drops K's head axis to 3D; re-insert the
        # singleton so the shared helper sees the uniform 4D contract.
        if nkh == 1:
            if K_new.dim() == 3:
                K_new = K_new.unsqueeze(2)  # [d_head, B, 1, S]
            if V_new.dim() == 3:
                V_new = V_new.unsqueeze(1)  # [B, 1, S, d_head]
        if kv_writeback is None:
            scatter_new_kv(
                self.k_cache,
                self.v_cache,
                K_new,
                V_new,
                slot_mapping,
                block_size,
                nkh,
                self.head_dim,
            )
        else:
            # X5 on the opt-in stock arm. ``scatter_new_kv`` does the index_put_
            # itself, so its permute/flatten mapping is reproduced here verbatim
            # (see ``qwen3_vl/utils/decode_kv.py``) and the write is deferred
            # instead, which keeps the stock/torch A/B a single-variable
            # comparison. With the in-kernel KV write (the default when the
            # modular-flow partitioner is off) this branch is not taken.
            num_tokens = slot_mapping.shape[0]
            block_indices = slot_mapping // block_size
            position_indices = slot_mapping % block_size
            k_new_flat = K_new.permute(2, 1, 3, 0).reshape(-1, self.head_dim)
            v_new_flat = V_new.permute(1, 0, 2, 3).reshape(-1, self.head_dim)
            head_indices_for_put = torch.arange(
                nkh, dtype=torch.long, device=slot_mapping.device
            ).repeat_interleave(num_tokens)
            kv_writeback.append(
                (
                    self.k_cache,
                    self.v_cache,
                    k_new_flat.to(self.k_cache.dtype),
                    v_new_flat.to(self.v_cache.dtype),
                    block_indices.repeat(nkh),
                    head_indices_for_put,
                    position_indices.repeat(nkh),
                    self.layer_idx,
                    slot_mapping,
                )
            )

        # >>> PARALLELISM: TP all-reduce <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: Dense MLP
# Mixed: PARALLELISM (TP intermediate sharding, SP collectives) +
#        MODEL-SPECIFIC (gelu_pytorch_tanh gate/up/down, no bias)
# =============================================================================


class Gemma3MLP(nn.Module):
    """Dense gated MLP with TP intermediate sharding.

    >>> PARALLELISM: TP <<<
    - gate_proj, up_proj: hidden -> intermediate/TP per rank
    - down_proj: intermediate/TP -> hidden per rank
    - Prefill: all-gather -> compute -> reduce-scatter (SP)
    - Decode: compute -> all-reduce

    <-- MODEL-SPECIFIC: ``gelu_pytorch_tanh`` gating
    (``down(gelu_tanh(gate(x)) * up(x))``), no bias. HF:
    ``Gemma3MLP.forward`` with ``act_fn = ACT2FN["gelu_pytorch_tanh"]``.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of MLP weights. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    @staticmethod
    def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
        """``gelu_pytorch_tanh`` written out EXPLICITLY.

        <-- MODEL-SPECIFIC and deliberately NOT
        ``torch.nn.functional.gelu(x, approximate="tanh")``: that is a C builtin
        that Dynamo marks as skipped, so it breaks a ``fullgraph=True`` trace
        ("Attempted to call function marked as skipped"). The closed form below
        lowers cleanly to HLO and the compiler fuses it with the surrounding
        matmuls.

            0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        Computed in fp32 then cast back, matching HF's fp32 activation math.
        """
        c = 0.7978845608028654  # sqrt(2 / pi)
        xf = x.to(torch.float32)
        out = 0.5 * xf * (1.0 + torch.tanh(c * (xf + 0.044715 * xf * xf * xf)))
        return out.to(x.dtype)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """>>> PARALLELISM: SP all-gather/reduce-scatter for prefill,
        all-reduce for decode. <<<
        """
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # <-- MODEL-SPECIFIC: gelu_pytorch_tanh gated MLP.
        # NF.mlp is NOT used here. Its nkilib kernel takes an ActFnType and does
        # support GELU_Tanh_Approx, but its PyTorch fallback (taken whenever the
        # kernel gate declines a shape, and always in CPU mode) routes to
        # `F.gelu(approximate="tanh")` -- the Dynamo-skipped builtin described
        # in _gelu_tanh. Expressing the MLP by hand keeps both the kernel-less
        # and the traced paths fullgraph-clean at the cost of losing the fused
        # kernel. See README "Known gaps".
        # DECODE: fused nkilib MLP kernel (gate+up matmul, GELU-tanh, multiply,
        # down matmul in one kernel) instead of three separate matmuls plus an
        # elementwise chain. Only on the decode path (T <= 128, the TKG kernel)
        # and only when the kernel can run; prefill and CPU mode keep the
        # hand-written fullgraph-clean path above. Default OFF: measured slower
        # (TPOT 22.6 -> 23.2 ms, E2E -2 to -4% on Gemma3-4B). =1 enables.
        if (
            not is_prefill
            and os.environ.get("GEMMA3_FUSED_MLP", "0") == "1"
            and hidden_states.shape[0] <= 128
            and can_run_kernel(hidden_states)
        ):
            output = NF.mlp(
                hidden_states,
                self.gate_proj_weight,
                self.up_proj_weight,
                self.down_proj_weight,
                act_fn=ActFnType.GELU_Tanh_Approx,
            )
        else:
            gate = hidden_states @ self.gate_proj_weight
            up = hidden_states @ self.up_proj_weight
            output = (self._gelu_tanh(gate) * up) @ self.down_proj_weight

        if self.world_size > 1:
            if is_prefill:
                output = self.tp_group.reduce_scatter(output, dim=0)
            else:
                self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: Decoder Layer
# <-- MODEL-SPECIFIC: Gemma3's 4-norm "sandwich". The post-norms are applied to
#     the SUBLAYER OUTPUT, BEFORE the residual add -- which is what makes this
#     different from Qwen3/Llama pre-norm, not just "two extra norms".
#
#     HF (modeling_gemma3.Gemma3DecoderLayer.forward):
#         residual = h
#         h = input_layernorm(h); h = self_attn(h)
#         h = post_attention_layernorm(h); h = residual + h
#         residual = h
#         h = pre_feedforward_layernorm(h); h = mlp(h)
#         h = post_feedforward_layernorm(h); h = residual + h
# =============================================================================


class Gemma3DecoderLayer(nn.Module):
    """Single Gemma3 transformer decoder layer (4-norm sandwich)."""

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Gemma3Attention(config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.layer_idx = layer_idx

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        kv_writeback: list | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_mask=attn_mask,
            attn_metadata=attn_metadata,
            # X5: the per-step deferred-KV list (None on prefill / when gated off).
            kv_writeback=kv_writeback,
        )
        # <-- MODEL-SPECIFIC: post-attention norm BEFORE the residual add.
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        # <-- MODEL-SPECIFIC: post-FFN norm BEFORE the residual add.
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Decode attention-mask precompute
# Adapted from gpt_oss's ``_precompute_decode_attn_masks``. Structure and the
# trimmed-frame / clamp reasoning are kept verbatim (they are SWA decode
# infrastructure, not GPT-OSS model specifics); only the layer-index selection
# is Gemma3's.
#
# <-- MODEL-SPECIFIC: Gemma3 has TWO distinct decode contexts -- the sliding
#     layers (window 1024, trimmed block table) and the global layers (full
#     causal over the untrimmed table) -- so exactly two masks are built per
#     step and shared by all 34 layers.
# =============================================================================


def _resolve_schedule_probe_layers(config: Gemma3Config) -> tuple[int, int]:
    """Return ``(first_sliding_layer_idx, first_global_layer_idx)``.

    Resolved ONCE at construction time (never inside the traced forward) so the
    decode-mask precompute reads its two ``attn_metadata`` entries by static
    index. With Gemma3-4B's ``sliding_window_pattern = 6`` this is ``(0, 5)``,
    but it is derived from ``config.is_sliding`` so an explicit ``layer_types``
    list (or a different pattern) stays consistent.

    A degenerate schedule (every layer sliding, or every layer global) falls
    back to the other class's index; the corresponding mask is then built over
    that layer's block table, which is harmless because no layer consumes it.
    """
    n = config.num_hidden_layers
    sliding = next((i for i in range(n) if config.is_sliding(i)), None)
    glob = next((i for i in range(n) if not config.is_sliding(i)), None)
    if sliding is None and glob is None:
        raise ValueError("Gemma3: model has no layers")
    if sliding is None:
        sliding = glob
    if glob is None:
        glob = sliding
    return sliding, glob


def _precompute_decode_attn_masks(
    config: Gemma3Config,
    attn_metadata: dict,
    positions: torch.Tensor,
    num_q_heads_per_rank: int,
    sliding_probe_layer: int,
    global_probe_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sliding_mask, global_mask)`` for one decode step.

    The model runner trims ``block_table`` to the window-relevant blocks on
    sliding layers and supplies a per-seq ``swa_kv_pos_offset``
    (= ``start_block * block_size``). The mask is generated over the TRIMMED
    cache, so BOTH the mask end (``pos_ids``) and the window start
    (``start_pos``) must live in the trimmed frame. Mixing frames (absolute
    end, trimmed start) degenerates the window test once the trim activates:
    every slot from the window start through the end of the trimmed cache --
    including stale, not-yet-overwritten slots -- would get attended.

    The ``clamp(min=0)`` on the shifted positions is not cosmetic. For a real
    decode row it is a no-op. It fires only on a no-token padding/freed row,
    whose position is zero-padded to 0 while ``swa_kv_pos_offset`` is positive:
    without the clamp ``pos - offset < 0`` drives the mask's wrap branch to mark
    the ENTIRE window valid over that row's all-(-1) block table, so the kernel
    reads stale K, produces NaN, and the distributed topk gathers -1.
    """
    # Layer 0 is always sliding in Gemma3 (``(0 + 1) % 6 != 0``) and the first
    # global layer is at ``sliding_window_pattern - 1``. Both indices were
    # resolved at construction time -- see _resolve_schedule_probe_layers.
    sliding_layer_name = f"layers.{sliding_probe_layer}.self_attn"
    global_layer_name = f"layers.{global_probe_layer}.self_attn"

    block_size = attn_metadata[sliding_layer_name]["block_size"]
    block_table = attn_metadata[sliding_layer_name]["block_table_tensor"]
    B_local = block_table.shape[0]
    S_decode = positions.shape[0] // B_local
    pos_ids_flat = positions.reshape(1, B_local * S_decode)

    # --- Sliding-window mask (the 5-of-6 local layers) ---
    swa_max_blocks = attn_metadata[sliding_layer_name]["max_blocks_per_seq"]
    S_ctx_swa = swa_max_blocks * block_size

    swa_pos_ids = pos_ids_flat
    swa_kv_pos_offset = attn_metadata[sliding_layer_name].get("swa_kv_pos_offset")
    if swa_kv_pos_offset is not None:
        swa_pos_ids = pos_ids_flat - swa_kv_pos_offset.reshape(B_local, 1).expand(
            B_local, S_decode
        ).reshape(1, B_local * S_decode)
        swa_pos_ids = torch.clamp(swa_pos_ids, min=0)
    swa_start_pos_ids = torch.clamp(swa_pos_ids - config.sliding_window + 1, min=0)

    sliding_mask = NF.gen_attention_decode_mask(
        pos_ids=swa_pos_ids.to(torch.float32),
        bs=B_local,
        q_head=num_q_heads_per_rank,
        s_active=S_decode,
        s_prior=S_ctx_swa,
        start_pos=swa_start_pos_ids.to(torch.float32),
        block_len=block_size,
    )

    # --- Full causal mask (the 1-of-6 global layers) ---
    full_max_blocks = attn_metadata[global_layer_name]["max_blocks_per_seq"]
    S_ctx_full = full_max_blocks * block_size

    global_mask = NF.gen_attention_decode_mask(
        pos_ids=pos_ids_flat.to(torch.float32),
        bs=B_local,
        q_head=num_q_heads_per_rank,
        s_active=S_decode,
        s_prior=S_ctx_full,
        start_pos=None,
        block_len=block_size,
    )

    return sliding_mask, global_mask


# =============================================================================
# X5: DEFERRED + COALESCED KV CACHE WRITE-BACK
#
# THE DEFECT (measured on hardware)
# ---------------------------------
# Gemma3-4B has 34 layers but the vllm_neuron plugin allocates only **10 physical
# KV cache tensors** (5 K + 5 V) of ``[num_blocks, 4, 32, 256]`` bf16 -- SEVEN
# layers share each one. The decode FX graph shows this directly: every cache
# placeholder has ``num_users=7``.
#
# Each layer then does ``k_cache.index_put_(idx, k_new)`` on its shared tensor.
# The ``libtorch_neuronx_lite`` FX pass ``InPlaceToOutOfPlacePass`` rewrites every
# in-place op to its FUNCTIONAL equivalent (the XLA/HLO backend cannot express
# in-place semantics), so ``index_put_`` becomes ``index_put``, which PRODUCES A
# NEW FULL-SIZE TENSOR. Chained seven deep on a tensor that is 202.75 MB per LNC2
# physical core, and with ``AliasingOutputRewritePass`` able to alias only the
# FINAL result of the chain back to the input buffer, the six intermediates become
# REAL full-tensor copies.
#
# The device profile is unambiguous (eager decode ml=1024
# ctx=768, ``DmaPacketAggregated``): **40 packets of exactly 212,598,784 B**
# (= half a KV tensor; LNC2 splits it across the 2 physical cores) totalling
# **8.504 GB**, all named ``intermediateNN <-> scatter.NN_sgNNNN``, in a fully
# SERIALIZED ping-pong chain spanning **13.686 -> 43.403 ms of a 47.2 ms decode
# step (63%)**.
#
# THE FIX
# -------
# Instead of, per step:
#     for each of ~7 layers sharing one storage:  cache.index_put_((b_i,h_i,p_i), v_i)
# do:
#     cache.index_put_((cat(b_*), cat(h_*), cat(p_*)), cat(v_*))
# which is numerically IDENTICAL (the layers write DISJOINT slots -- see the
# disjointness argument in ``_coalesce_kv_writes``) but produces ONE functional
# ``index_put`` instead of seven.
#
# WHAT ACTUALLY WON THE SPEEDUP (be precise about this)
# -----------------------------------------------------
# Measurement showed that the ORIGINAL coalescer grouped by tensor IDENTITY
# (``is``) and therefore NEVER GROUPED ANYTHING: the 34 ``k_cache`` attributes are
# 34 DISTINCT VIEW OBJECTS over only 5 storages (the runner rebuilds
# ``typed_tensor[0]`` per layer), so ``is`` yielded 34 groups of size 1 and 68
# scatters were issued -- exactly the baseline's 68. **The ENTIRE 3.12x (TP=1) /
# 6.38x (TP=4) TPOT win came from DEFERRAL alone** (moving the writes past the
# layer loop is enough to stop the aliasing pass materializing the intermediates).
# Grouping by ``data_ptr()`` fixed it to ``data_ptr()``, taking 68 -> 10 scatters,
# which is a DMA-DESCRIPTOR-COUNT reduction, not a byte reduction (deferral had
# already taken HBM write per step from 41.10 GB to 0.186 GB), so its own headroom
# is a few percent at most. Do not expect the coalescing half to show up big.
# =============================================================================


def _to_cache_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast K/V rows to the cache dtype. FP8 caches use unit scales; clamp to e4m3's finite range first."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return x.to(torch.float32).clamp(-240.0, 240.0).to(dtype)
    return x.to(dtype)


def _resolve_kernel_kv_update() -> bool:
    """Resolve GEMMA3_KERNEL_KV_UPDATE ("auto" default, "0", "1").

    "auto" enables the in-kernel d256 KV write only when the installed NKI Library supports it
    (SUPPORTS_D256_BLOCK_KV_UPDATE) and the neuronx-cc modular-flow partitioner is
    disabled (VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD >= 1e12). Under the plugin default threshold (10) a
    partition boundary falls inside the last shared KV buffer's alias chain and the compiler
    materialises a full-buffer intermediate (+50 ms/step at b64). Plain env reads only: safe to call
    inside the trace (Dynamo folds them to constants).
    """
    v = os.environ.get("GEMMA3_KERNEL_KV_UPDATE", "auto")
    if v != "auto":
        return v == "1"
    if not _NKILIB_D256_KV_UPDATE:
        return False
    try:
        return float(_modular_flow_mac_threshold()) >= 1e12
    except ValueError:
        return False


def _modular_flow_mac_threshold() -> str:
    return (
        os.environ.get("VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD")
        or os.environ.get("VLLM_NEURON_MODULAR_MAC_THRESHOLD")
        or "10"
    )


def _resolve_kv_ptr_groups(layers) -> list[int] | None:
    """Resolve ``layer_idx -> physical-cache group_id`` from ``data_ptr()``. ONCE.

    **THIS MUST NEVER BE CALLED FROM INSIDE THE TRACED REGION.** ``.data_ptr()``
    returns a Python int derived from a device pointer, and Dynamo cannot trace
    the int conversion::

        torch._dynamo.exc.Unsupported: Failed to trace builtin operator `int`
            ... builtin int [DataPtrVariable]

    which aborts the whole parallel trace. ``.untyped_storage()`` is not an
    alternative either -- it raises ``RuntimeError: Allocator for neuron is not a
    DeviceAllocator``.

    The safe pattern (generalisable to any Neuron model needing pointer identity
    at trace time):

      1. call this from a NON-TRACED lifecycle hook -- here ``bind_kv_cache``,
         which the runner invokes once, before any ``torch.compile`` tracing;
      2. cache the result as a **plain Python list of ints** on the module;
      3. have the traced code index that list with a static ``layer_idx``.

    This is sound because the grouping is a property of the KV **allocation**,
    which is fixed for the life of the engine: the runner allocates
    ``kv_cache_config.kv_cache_tensors`` once in ``initialize_kv_cache`` and binds
    the per-layer views immediately after. If the cache were ever reallocated,
    ``bind_kv_cache`` would run again and this would be re-resolved.

    Returns ``group_of[layer_idx]`` (contiguous group ids from 0), or ``None`` if
    the pointers could not be read -- in which case ``_coalesce_kv_writes``
    degrades to per-layer scatters (deferral only, which is the part that carries
    nearly all of the measured win anyway).
    """
    try:
        ptr_to_gid: dict[tuple[int, int], int] = {}
        group_of: list[int] = []
        for layer in layers:
            attn = layer.self_attn
            if attn.k_cache is None or attn.v_cache is None:
                return None
            # Key on the (k, v) PAIR: a group is only safe to coalesce if its
            # members share BOTH the K and the V storage. They always do here
            # (K and V are ``typed_tensor[0]`` / ``[1]`` of one buffer), but
            # keying on the pair makes that an assertion rather than an
            # assumption.
            key = (int(attn.k_cache.data_ptr()), int(attn.v_cache.data_ptr()))
            gid = ptr_to_gid.get(key)
            if gid is None:
                gid = len(ptr_to_gid)
                ptr_to_gid[key] = gid
            group_of.append(gid)
        return group_of
    except Exception:  # pragma: no cover - defensive; falls back to per-layer
        logger.warning(
            "Gemma3: could not resolve KV data_ptr groups; falling back to "
            "per-layer KV scatters (deferral still applies).",
            exc_info=True,
        )
        return None


def _coalesce_kv_writes(writeback: list, group_of: list | None = None) -> None:
    """Issue the deferred KV ``index_put_`` calls after the layer loop, grouped by
    PHYSICAL cache tensor so each storage is scattered exactly ONCE.

    ``writeback`` is a list of ``(k_cache, v_cache, k_flat, v_flat, block_idx,
    head_idx, pos_idx, layer_idx, slot_mapping)`` tuples appended by each decode
    layer in layer order. ``group_of`` is the static ``layer_idx -> group_id``
    table resolved OUTSIDE the trace by :func:`_resolve_kv_ptr_groups` (``None``
    -> per-layer fallback, i.e. deferral only).

    **WHY ``data_ptr()`` AND NOT ``is``**: this function IS inside the traced
    region, so it cannot call ``.data_ptr()`` itself. The first version therefore reached
    for tensor identity (``is``) -- and it never grouped anything, because the
    runner rebuilds ``typed_tensor[0]`` per layer, giving 34 distinct view objects
    over ~5 storages. Measured, on-instance, at BOTH TP degrees::

        34 layers  ->  group by `is`        : 34 groups, ALL of size 1  -> 68 scatters
                   ->  group by data_ptr()  :  5 groups                -> 10 scatters
                       [[0..6],[7..13],[14..20],[21..27],[28..33]]

    THE LAYER <-> STORAGE <-> SLOT_MAPPING MAPPING (measured)
    ---------------------------------------------------------
    Two facts look contradictory but are orthogonal axes, and their
    transversality is exactly what makes concatenation safe:

      * **5 physical KV buffers** (``kv_cache_config.kv_cache_tensors``, each
        1,422,393,344 B), ``shared_by`` CONTIGUOUS layer runs:
        ``[0-6] [7-13] [14-20] [21-27] [28-33]``.
      * **7 KV-cache GROUPS** (``kv_cache_config.kv_cache_groups``), each with its
        own ``attn_metadata`` dict and therefore its own ``slot_mapping``,
        striding layers ~7 apart: ``g0=[0,7,14,21,28] g1=[1,8,15,22,30]
        g2=[2,9,16,24,31] g3=[3,10,18,25,32] g4=[4,12,19,26,33] g5=[6,13,20,27]``
        (all SlidingWindowSpec, window 1024) and ``g6=[5,11,17,23,29]``
        (FullAttentionSpec -- the 5:1 global layers).

    So a physical buffer's 6-7 members are drawn from **7 different KV groups, one
    apiece** -- the storage axis (contiguous) is TRANSVERSAL to the slot_mapping
    axis (strided). Each member of a data_ptr group therefore holds a DIFFERENT
    ``slot_mapping`` object with a DIFFERENT slot value. Measured at both TP
    degrees, all 4 TP=4 ranks, every decode step::

        ptr group [0..6]   slots [63, 95, 127, 159, 191, 223, 255]  (step 1)
                           slots [256,288,320,352,384,416,448]      (step 2)
        ptr group [28..33] slots [63, 95, 127, 159, 191, 255]  (6 members)
        duplicate slots within a group: 0 / 5 groups / 3 steps / 5 ranks

    **Why disjointness is STRUCTURAL, not a lucky draw**: two layers in the same
    data_ptr group always belong to different KV groups (a KV group's members are
    ~7 apart, a storage's members are 6-7 CONSECUTIVE layers, so at most one of
    each KV group's layers falls in any one storage). Distinct KV groups get
    distinct ``BlockTable``s from the scheduler and hence distinct blocks, so their
    slots cannot collide. This MATTERS because ``index_put_(accumulate=False)``
    does not define which of two duplicate indices wins -- so if it ever failed,
    the coalesced write would be silently wrong rather than an error. It is
    therefore additionally ASSERTED at run time below via ``slot_mapping`` OBJECT
    identity (free: no device read, no ``.item()``, no graph break), with a
    per-layer fallback for any group that fails the check.
    """

    def _group(val_idx, cache_idx):
        """Group deferred writes by physical cache storage.

        Returns ``[[cache_tensor, [(vals, b, h, p), ...]], ...]``. Uses the static
        ``group_of`` table when available; falls back to one group per layer
        (== deferral only) otherwise.
        """
        if group_of is None:
            return [
                [item[cache_idx], [(item[val_idx], item[4], item[5], item[6])]]
                for item in writeback
            ]
        by_gid: dict = {}
        order: list = []
        for item in writeback:
            layer_idx = item[7]
            if layer_idx is None or layer_idx >= len(group_of):
                # Unknown layer -> its own group (safe: per-layer scatter).
                gid = ("solo", id(item))
            else:
                gid = group_of[layer_idx]
            if gid not in by_gid:
                by_gid[gid] = [item[cache_idx], [], []]
                order.append(gid)
            slot = item[8]
            g = by_gid[gid]
            # DISJOINTNESS GUARD (free, trace-safe): two layers may only be
            # concatenated if they carry DIFFERENT ``slot_mapping`` OBJECTS.
            # Layers sharing one slot_mapping object write the SAME slots, and
            # ``index_put_(accumulate=False)`` does not define which wins. This is
            # an object-identity check, so it costs nothing at trace time -- no
            # device read, no ``.item()``, no graph break. Measured: within every
            # data_ptr group all members carry distinct slot_mapping objects with
            # distinct values, at both TP degrees and on all 4 TP=4 ranks, so this
            # never trips today. It is here so a future scheduler / KV-group
            # change degrades to per-layer scatters instead of silently corrupting
            # the cache.
            entry = (item[val_idx], item[4], item[5], item[6])
            if slot is not None and any(slot is s for s in g[2]):
                solo = ("solo", id(item))
                by_gid[solo] = [item[cache_idx], [entry], [slot]]
                order.append(solo)
                continue
            g[1].append(entry)
            g[2].append(slot)
        return [[by_gid[gid][0], by_gid[gid][1]] for gid in order]

    for cache, items in _group(2, 0) + _group(3, 1):
        if len(items) == 1:
            vals, b, h, p = items[0]
            cache.index_put_((b, h, p), vals)
            continue
        # Disjoint slots (structural + asserted above) -> one scatter is
        # bit-identical to the sequential calls and collapses the 7-deep
        # functional-copy chain to depth 1.
        vals = torch.cat([it[0] for it in items], dim=0)
        b = torch.cat([it[1] for it in items], dim=0)
        h = torch.cat([it[2] for it in items], dim=0)
        p = torch.cat([it[3] for it in items], dim=0)
        cache.index_put_((b, h, p), vals)


# =============================================================================
# Section 6: Model Backbone
# Mixed: PARALLELISM (SP embedding/chunking) +
#        MODEL-SPECIFIC (sqrt(hidden) embedding scale, dual-RoPE dispatch,
#        per-layer mask selection, layer stack, final norm)
# =============================================================================


def _bf16_embed_scale(hidden_size: int) -> float:
    """``sqrt(hidden_size)`` ROUNDED TO BF16, returned as a python float.

    <-- MODEL-SPECIFIC and the rounding matters. HF's
    ``Gemma3TextScaledWordEmbedding.forward`` does
    ``embeddings * self.embed_scale.to(self.weight.dtype)``, i.e. the fp32
    ``sqrt(hidden_size)`` is DOWNCAST TO BF16 before the multiply. transformers
    calls this out explicitly for Gemma: "Gemma3 downcasts the below to
    bfloat16, causing sqrt(3072)=55.4256 to become 55.5"
    (huggingface/transformers#29402). Using the full-precision fp32 value
    instead produces a small but systematic error in every token embedding.

    ``device="cpu"`` is explicit so this is safe under the
    ``torch.device("meta")`` model-construction context: an explicit device
    kwarg overrides the ambient device, and returning a python ``float`` means
    the forward-path multiply carries no tensor and therefore no device.
    """
    return float(
        torch.tensor(hidden_size**0.5, dtype=torch.bfloat16, device="cpu").item()
    )


class Gemma3Model(nn.Module):
    """Gemma3 transformer backbone.

    >>> PARALLELISM: SP during prefill -- embed -> chunk -> layers ->
    all-gather. <<<
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # X5: ``layer_idx -> physical-KV-storage group_id``, filled in by
        # ``Gemma3ForCausalLM.bind_kv_cache`` (OUTSIDE the trace). ``None`` until
        # the cache is bound, and ``None`` forever if the pointers could not be
        # read -- in which case ``_coalesce_kv_writes`` degrades to per-layer
        # scatters (i.e. deferral only, which carries nearly all of the win).
        self.kv_ptr_group_of = None

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        # vocab_size 262208 is divisible by 1/2/4 so no padding is needed at any
        # valid TP degree; ``pad_shard=True`` below keeps it safe regardless.
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        # <-- MODEL-SPECIFIC: sqrt(hidden_size) embedding scale, bf16-rounded.
        # A python float, resolved at construction time -- see _bf16_embed_scale.
        self.embed_scale = _bf16_embed_scale(config.hidden_size)

        # <-- MODEL-SPECIFIC: 34 decoder layers
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm (also a (1 + gamma) norm)
        self.norm = Gemma3RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # <-- MODEL-SPECIFIC: DUAL RoPE
        self.rotary_emb = Gemma3RotaryEmbedding(config)

        # Static layer indices used by the decode-mask precompute, resolved
        # outside any traced region.
        self.sliding_probe_layer, self.global_probe_layer = (
            _resolve_schedule_probe_layers(config)
        )

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                is_storage_transposed=False,
                pad_shard=True,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # >>> PARALLELISM: SP prompt-embed path <<<
        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        # <-- MODEL-SPECIFIC: scale embeddings by bf16(sqrt(hidden_size)).
        # Applied AFTER merge_prompt_embeds, matching HF: the scale lives inside
        # Gemma3TextScaledWordEmbedding, and externally supplied prompt embeds
        # are expected in the same (scaled) space as the table output.
        hidden_states = hidden_states * self.embed_scale

        # <-- MODEL-SPECIFIC: compute BOTH RoPE tables once per step.
        rope_tables = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # <-- MODEL-SPECIFIC: pre-compute the two decode masks (sliding vs
        # global) once per step instead of once per layer.
        sliding_mask = None
        global_mask = None
        if not is_prefill:
            sliding_mask, global_mask = _precompute_decode_attn_masks(
                config=self.config,
                attn_metadata=attn_metadata,
                positions=positions,
                num_q_heads_per_rank=self.layers[
                    0
                ].self_attn.num_attention_heads_per_rank,
                sliding_probe_layer=self.sliding_probe_layer,
                global_probe_layer=self.global_probe_layer,
            )

        # <-- X5: DEFERRED + COALESCED KV WRITE-BACK (see _coalesce_kv_writes).
        # When enabled, each attention layer DEFERS its KV ``index_put_`` into
        # ``kv_writeback`` and the writes are issued after the layer loop instead
        # of interleaved with it. This removes the ~7.92 GB/step of full-tensor
        # copies that ``InPlaceToOutOfPlacePass`` materializes for 6 of every 7
        # chained scatters. Measured on Gemma3-4B: TPOT 62.5 -> 20.0 ms
        # (3.12x) at b1 TP=1 and 50.8 -> 8.0 ms (6.38x) at b1 TP=4
        # (768-in/128-out, ml=1024, gpu_memory_utilization=0.9).
        #
        # Prefill NEVER defers: both prefill kernels read back the slots they
        # write, so deferral there is not slow, it is WRONG. See
        # ``_write_paged_kv_cache``. Hence the ``not is_prefill`` guard, which is
        # load-bearing.
        #
        # GEMMA3_KV_COALESCE=0 restores the interleaved per-layer write.
        _coalesce = (not is_prefill) and (
            os.environ.get("GEMMA3_KV_COALESCE", "1") == "1"
        )
        # Prefill write-back: legal only on the single-shot path (enforced per
        # layer in forward_prefill, which falls back to an inline write whenever
        # kv_segment_size is set). Default ON (token-identical to inline in 3 checks).
        if is_prefill and os.environ.get("GEMMA3_PREFILL_KV_COALESCE", "1") == "1":
            _coalesce = True
        kv_writeback = [] if _coalesce else None
        # GEMMA3_KV_PINGPONG: per-step {storage_gid: (K, V)} so each in-kernel KV writer takes the
        # previous same-storage writer's UPDATED cache (the kernel's alias output) as its input.
        # That makes one linear in-place chain per physical buffer instead of N writers all reading
        # the original input (which forces a full-buffer copy per buffer).
        if not is_prefill:
            attn_metadata["_g3_kv_chain"] = {}

        for decoder_layer in self.layers:
            # <-- MODEL-SPECIFIC: per-layer RoPE table + mask selection driven
            # by the 5:1 hybrid schedule.
            is_sliding = decoder_layer.self_attn.is_sliding
            position_embeddings = rope_tables[
                "sliding" if is_sliding else "global"
            ]
            attn_mask = sliding_mask if is_sliding else global_mask
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_mask=attn_mask,
                attn_metadata=attn_metadata,
                kv_writeback=kv_writeback,
            )

        if not is_prefill:
            attn_metadata.pop("_g3_kv_chain", None)

        if kv_writeback:
            # X5: pass the STATIC ``layer_idx -> storage group_id`` table resolved
            # outside the trace in ``bind_kv_cache``. This is what makes the
            # coalescing actually coalesce (68 -> 10 index_put_ calls).
            # GEMMA3_KV_PTR_GROUP=0 restores the per-layer scatters (deferral
            # only) for A/B measurement of deferral vs coalescing.
            _pg = self.kv_ptr_group_of
            if os.environ.get("GEMMA3_KV_PTR_GROUP", "1") != "1":
                _pg = None
            _coalesce_kv_writes(kv_writeback, _pg)

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP -- all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states, []


# =============================================================================
# Section 7: LM Head + Weight Loading
# Mixed: PARALLELISM (column-parallel LM head) +
#        MODEL-SPECIFIC (tied embeddings, text-prefix stripping, vision drop,
#        RMSNorm (1 + gamma) fold)
# =============================================================================


def _detect_text_prefix(checkpoint_keys) -> str:
    """Return the HF text-decoder key prefix present in the checkpoint.

    <-- MODEL-SPECIFIC. The multimodal ``Gemma3ForConditionalGeneration``
    checkpoint stores the decoder under
    ``language_model.model.`` (e.g.
    ``language_model.model.layers.0.self_attn.q_proj.weight``), while a
    text-only ``gemma3_text`` re-export would use a bare ``model.``. Probing
    the checkpoint keeps both working and fails loudly (via ``strict=True``
    loading) rather than silently loading nothing if a third layout appears.
    """
    keys = set(checkpoint_keys)
    for cand in _TEXT_PREFIX_CANDIDATES:
        if any(k.startswith(cand + "embed_tokens") for k in keys) or any(
            k.startswith(cand + "layers.0.") for k in keys
        ):
            logger.info("Gemma3: detected text-decoder checkpoint prefix %r", cand)
            return cand
    raise RuntimeError(
        "Gemma3: could not locate the text decoder in the checkpoint. Tried "
        f"prefixes {_TEXT_PREFIX_CANDIDATES}. This port loads the Gemma3 TEXT "
        "stack only."
    )


class Gemma3ForCausalLM(nn.Module):
    """Gemma3 text decoder with LM head.

    >>> PARALLELISM: Column-parallel LM head. <<<
    <-- MODEL-SPECIFIC: TIED embeddings (``tie_word_embeddings=True``); the
    checkpoint has no ``lm_head.weight``.
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # >>> PARALLELISM: Column-parallel LM head <<<
        # <-- MODEL-SPECIFIC: tied to the input embedding. The head is NOT
        # aliased to `embed_tokens.weight` in python -- see load_weights for
        # why, and for the mapping that gives both parameters real data.
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,  # <-- MODEL-SPECIFIC: no lm_head bias
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=False,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]

        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, _ = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )

        # <-- MODEL-SPECIFIC: no final_logit_softcapping in Gemma3 (dropped
        # from Gemma2, and the factory rejects a config that sets it).
        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Gemma3Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV Cache ──────────────────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<
    # <-- MODEL-SPECIFIC: sliding_window_size varies by layer (5 local : 1
    #     global). Emitting it per layer is what lets the model runner allocate
    #     a smaller cache and trim the block table for the sliding layers.

    def get_kv_spec(self):
        """Returns the KV cache specification for vLLM integration."""
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    # None on the global layers, 1024 on the sliding ones.
                    sliding_window_size=layer.self_attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise KeyError(f"KV cache for layer {layer_name} not initialized")
            k_cache = kv_caches[layer_name][0]
            v_cache = kv_caches[layer_name][1]
            # This port implements a bf16 KV cache only. Fail loudly instead of
            # silently writing unscaled values into an FP8 cache (which would
            # run and produce garbage).
            fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
            if (k_cache.dtype in fp8_dtypes or v_cache.dtype in fp8_dtypes) and os.environ.get("GEMMA3_FP8_KV", "0") != "1":
                raise ValueError(
                    "Gemma3 on Neuron does not implement an FP8 KV cache "
                    f"(got K={k_cache.dtype}, V={v_cache.dtype}). Run with "
                    "kv_cache_dtype='auto'."
                )
            layer.self_attn.k_cache = k_cache
            layer.self_attn.v_cache = v_cache
            # FP8 KV (GEMMA3_FP8_KV=1): per-tensor scale 1.0 (no calibrated scales for this checkpoint).
            # Post-norm K and V stay well inside e4m3's +-240 range. Scales are (128,1) fp32 buffers, the
            # layout the kernel's _quantize_to_fp8 accepts.
            if k_cache.dtype in fp8_dtypes:
                layer.self_attn.kv_scale_t = torch.ones((128, 1), dtype=torch.float32, device=k_cache.device)
            else:
                layer.self_attn.kv_scale_t = None

        # X5: resolve the ``layer_idx -> physical-storage group_id`` table HERE,
        # i.e. OUTSIDE the traced region. This is the ONLY place it can be done:
        # ``.data_ptr()`` inside the compiled graph aborts Dynamo with "Failed to
        # trace builtin operator `int` ... [DataPtrVariable]", and
        # ``.untyped_storage()`` raises "Allocator for neuron is not a
        # DeviceAllocator". ``bind_kv_cache`` runs once, before any tracing, and is
        # re-run if the cache is ever reallocated -- so the cached table cannot go
        # stale. ``_coalesce_kv_writes`` then consults only this plain list of ints
        # and indexes it with a static ``layer_idx``. See ``_resolve_kv_ptr_groups``.
        group_of = _resolve_kv_ptr_groups(self.model.layers)
        self.model.kv_ptr_group_of = group_of
        logger.info(
            "Gemma3 decode KV write: GEMMA3_KERNEL_KV_UPDATE=%s modular-flow MAC threshold=%s -> %s",
            os.environ.get("GEMMA3_KERNEL_KV_UPDATE", "auto"),
            _modular_flow_mac_threshold(),
            "in-kernel" if _resolve_kernel_kv_update() else "deferred external scatter",
        )
        logger.info(
            "Gemma3 NKI Library d256 support: out_proj=%s block_kv_update=%s",
            _NKILIB_D256_OUT_PROJ,
            _NKILIB_D256_KV_UPDATE,
        )
        for i, layer in enumerate(self.model.layers):
            # Static physical-storage id per layer (ping-pong threading of the in-kernel KV write).
            layer.self_attn.kv_gid = group_of[i] if group_of is not None else i
        if group_of is not None:
            n_groups = len(set(group_of))
            logger.info(
                "Gemma3 KV coalescing: %d layers -> %d physical KV storages "
                "(data_ptr grouping, resolved outside the trace) -> %d "
                "index_put_ calls/step instead of %d. groups=%s",
                len(group_of),
                n_groups,
                2 * n_groups,
                2 * len(group_of),
                [
                    [i for i, g in enumerate(group_of) if g == gid]
                    for gid in sorted(set(group_of))
                ],
            )

    # ── Weight Loading ────────────────────────────────────────────────────

    def _build_weight_mappings(self, prefix: str) -> dict:
        """Build ``{our_param_name: hf_checkpoint_key(s)}``.

        <-- MODEL-SPECIFIC. Three Gemma3-specific things happen here:

        1. **Text-prefix stripping.** Every key is built from the detected
           ``prefix`` (``language_model.model.`` for the multimodal checkpoint),
           so our flat ``model.*`` parameter names map onto the nested layout.

        2. **Vision drop.** ``vision_tower.*`` and ``multi_modal_projector.*``
           are simply never referenced. ``load_sharded_pipelined`` only reads
           keys named in this mapping, so those ~439 tensors are never loaded
           and never allocated -- they show up in ``unexpected_keys`` and are
           filtered from the audit log below.

        3. **Tied LM head.** Gemma3 ties, so the checkpoint has NO
           ``lm_head.weight``. Per the bringup guide, both parameters are mapped
           to the SAME checkpoint key and each loads through its own weight
           loader. Do NOT alias ``lm_head.weight = embed_tokens.weight`` in
           python before loading: vLLM builds the model on ``meta`` and
           ``load_state_dict(assign=True)`` replaces the tensor in one module's
           ``_parameters`` dict only, leaving the other pointing at a stale meta
           tensor -> "Cannot copy out of meta tensor".

        Q/K/V are SEPARATE tensors in the checkpoint and are fused by
        ``fused_qkv_weight_loader`` (a 3-element list value). There are no
        biases anywhere in the Gemma3 text stack.
        """
        mappings: dict = {}

        # Embedding, final norm, LM head.
        mappings["model.embed_tokens.weight"] = f"{prefix}embed_tokens.weight"
        mappings["model.norm.weight"] = f"{prefix}norm.weight"
        if self.config.tie_word_embeddings:
            mappings["lm_head.weight"] = f"{prefix}embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"

        for layer_id in range(len(self.model.layers)):
            hf = f"{prefix}layers.{layer_id}"
            our = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: separate Q, K, V -> fused QKV, no bias.
            mappings[f"{our}.self_attn.qkv_proj_weight"] = [
                f"{hf}.self_attn.q_proj.weight",
                f"{hf}.self_attn.k_proj.weight",
                f"{hf}.self_attn.v_proj.weight",
            ]
            mappings[f"{our}.self_attn.o_proj_weight"] = (
                f"{hf}.self_attn.o_proj.weight"
            )

            # <-- MODEL-SPECIFIC: per-head QK-norm gammas. NOT TP-sharded --
            # they are per-head (head_dim) vectors, identical on every rank.
            mappings[f"{our}.self_attn.q_norm.weight"] = f"{hf}.self_attn.q_norm.weight"
            mappings[f"{our}.self_attn.k_norm.weight"] = f"{hf}.self_attn.k_norm.weight"

            # <-- MODEL-SPECIFIC: the 4-norm sandwich.
            mappings[f"{our}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
            mappings[f"{our}.post_attention_layernorm.weight"] = (
                f"{hf}.post_attention_layernorm.weight"
            )
            mappings[f"{our}.pre_feedforward_layernorm.weight"] = (
                f"{hf}.pre_feedforward_layernorm.weight"
            )
            mappings[f"{our}.post_feedforward_layernorm.weight"] = (
                f"{hf}.post_feedforward_layernorm.weight"
            )

            # <-- MODEL-SPECIFIC: gated MLP, no bias.
            mappings[f"{our}.mlp.gate_proj_weight"] = f"{hf}.mlp.gate_proj.weight"
            mappings[f"{our}.mlp.up_proj_weight"] = f"{hf}.mlp.up_proj.weight"
            mappings[f"{our}.mlp.down_proj_weight"] = f"{hf}.mlp.down_proj.weight"

        return mappings

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from the Gemma3 HF checkpoint.

        >>> PARALLELISM: Weight loaders handle TP sharding. <<<
        <-- MODEL-SPECIFIC: nested text prefix, vision drop, tied LM head, and
        the ``(1 + gamma)`` RMSNorm fold.
        """
        tp_rank = self.rank
        tp_size = self.world_size

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        prefix = _detect_text_prefix(checkpoint.get_tensor_names())
        mappings = self._build_weight_mappings(prefix)

        # strict=True: every key we reference must exist. The vision tensors are
        # not referenced at all, so they are dropped without tripping this.
        result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        )
        rank_sharded = result.state_dict

        # Audit: separate the intentionally dropped vision tensors from a
        # genuinely unexpected key, so a clean text-only load reads 0.
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        vision_dropped = [
            k for k in unexpected if any(p in k for p in _VISION_PREFIXES)
        ]
        unexpected_non_vision = [k for k in unexpected if k not in vision_dropped]
        logger.info(
            "Gemma3 weight load: prefix=%r vision tensors dropped=%d "
            "unexpected(non-vision)=%d tied_lm_head=%s",
            prefix,
            len(vision_dropped),
            len(unexpected_non_vision),
            self.config.tie_word_embeddings,
        )
        if unexpected_non_vision:
            logger.warning(
                "Gemma3 unexpected checkpoint keys (not loaded): %s",
                unexpected_non_vision[:20],
            )

        target_dtype = self.config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        # <-- MODEL-SPECIFIC: Gemma3's RMSNorm is (1 + gamma) * norm(x) and HF
        # stores gamma centered on 0. Add 1.0 here so both this module's plain
        # ``w * x`` and the decode mega-kernel's ``rmsnorm_QK_pre_rope_W_*``
        # (which has no (1 + w) form) reproduce HF exactly. Every norm must be
        # covered -- see _RMSNORM_PARAM_SUFFIXES.
        folded = 0
        for name in list(rank_sharded.keys()):
            if name.endswith(_RMSNORM_PARAM_SUFFIXES):
                rank_sharded[name] = rank_sharded[name] + 1.0
                folded += 1
        # 4 sandwich norms + 2 QK norms per layer, plus the final norm.
        expected_norms = 6 * len(self.model.layers) + 1
        if folded != expected_norms:
            # Not fatal (a user could load a partial checkpoint), but a mismatch
            # means some norm silently kept HF's 0-centered gamma, which is a
            # correctness bug that no smoke test would catch.
            logger.warning(
                "Gemma3: folded (1 + gamma) into %d RMSNorm tensors, expected "
                "%d. Check _RMSNORM_PARAM_SUFFIXES against the parameter names.",
                folded,
                expected_norms,
            )

        self.load_state_dict(rank_sharded, strict=False, assign=True)
