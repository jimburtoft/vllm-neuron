# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy4 (``HYV4ForCausalLM``) text decoder for Neuron.

78 layers of ``iHC -> gated MLA -> iHC -> MoE``, one hidden state carried as
``hc_mult`` = 4 streams. See ``config.py`` for the shape summary and ``moe.py``
for the experts. This file is everything else, and four things in it give wrong
answers without failing:

* **The residual is four streams, not one.** Each sub-block reads a sigmoid-
  weighted sum of the streams (``HYV4HCPre``) and writes its output back to all
  four with a second set of gates (``HYV4HCPost``); there is no plain
  ``x + f(x)`` anywhere. The gate projections, their RMS factor and the
  combination are fp32, as in every reference implementation.
* **MLA is computed in latent space.** The cache holds, per token, the normed
  512-dim KV latent, the shared 64-dim rope key and the 128-dim DSA indexer key
  (stored first) -- 704 values, stored as the K and V halves of a 352-wide "head" so that one
  page is exactly the bytes the planner reserved. ``kv_b_proj`` is absorbed:
  the no-rope query is multiplied into latent space, attention runs against
  the latent directly, and the value up-projection is applied to the attention
  output. Identical in exact arithmetic to expanding K/V per head. Note what
  it does and does not save: the latent is 704 values against 64 x 512 for
  all heads expanded, but every rank holds a full copy, and at one head per
  rank an expanded cache would be 512 + 128 -- so at world 64 the latent
  cache is ~10% *larger* per rank. Sharding the latent's tokens across ranks
  (context parallelism) is what would turn it into a saving.
* **Every head has a learned sink logit**, a column with no value that only
  takes probability mass, and the attention output is gated elementwise by
  ``sigmoid(gate_proj(x))`` before ``o_proj``.
* **The rope pairing is interleaved** -- see ``ROPE_STYLE`` in ``config.py``.

**DSA.** Each query attends to the top ``index_topk`` = 2048 keys a lightning
indexer picks (``HYV4Indexer``); the 20 ``"full"`` layers run one, the other 58
reuse the selection of the last full layer before them. Below 2048 visible keys
the selection is every key. The sparse attention is computed as dense attention
masked to the selection: exact, but its cost still grows with the context.

**Prefill is chunked** (the plugin's segmented prefill): each step is one chunk
of one request, attending to the ``cached_seq_len`` tokens already in the cache
plus causally to itself, so a prompt longer than the prefill bucket is handled
across steps and per-step cost does not grow with the bucket.

Parallelism is plain TP over the 64 query heads (one per rank at world 64) and
over every MLP's intermediate dim. The MLA down-projections (``q_a_proj``,
``kv_a_proj_with_mqa``) are replicated: their outputs feed per-token norms over
the full width.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import ROPE_STYLE, HYV4Config

# HY4_DEBUG_CAPTURE=1 traces ``capture_tensor`` calls for the layers listed in
# HY4_DEBUG_LAYERS (default "0") into the graph (used with
# neuron_config.tensor_capture). Off, nothing is traced.
DEBUG_CAPTURE = os.environ.get("HY4_DEBUG_CAPTURE") == "1"
DEBUG_LAYERS = {int(x) for x in os.environ.get("HY4_DEBUG_LAYERS", "0").split(",")}
# HY4_PERF_SKIP=collectives,attn,moe -- timing experiments only, outputs are
# wrong: "collectives" makes the per-layer all-reduces the identity and the
# DSA mask all-gather a local repeat; "attn"/"moe" replace that sub-block's
# output with zeros (attention then writes no KV and selects nothing).
PERF_SKIP = {x for x in os.environ.get("HY4_PERF_SKIP", "").split(",") if x}
def _platform_target() -> str:
    """``trn2`` / ``trn3`` / ... for the default attention paths; ``""`` off device.

    Read once, at model construction (``resolve_platform_defaults``), never
    inside a traced forward.
    """
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "")
    if target:
        return target
    try:
        from libtorch_neuronx_lite.compile.platform import get_platform_target

        return str(get_platform_target())
    except Exception:  # no runtime (CPU tests)
        return ""


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value == "1"


# Defaults per platform, each overridable with its environment variable:
#   ``HY4_DEC_CP`` (decode attention sharded by KEY across the TP ranks,
#   ``HYV4Attention.forward_decode_cp``) and ``HY4_DEC_SPARSE_NKI``: on, on
#   Trn2 and Trn3; ``HY4_PREFILL_SPARSE_NKI``: on for Trn2 only -- on Trn3 its
#   kernels currently run slower than the dense blocked attention, so Trn3
#   keeps the dense prefill. Off device (CPU tests) everything defaults off.
# The values below are the environment-only defaults; the model sets the
# platform defaults at construction (``resolve_platform_defaults``).
DEC_CP = _flag("HY4_DEC_CP", False)
# ``HY4_DEC_SPARSE_NKI=1`` (with HY4_DEC_CP=1): each rank reads only the indexer
# keys of its shard plus the latent rows DSA selected, not whole entries for the
# shard (``HYV4Attention._cp_sparse_local``). On Neuron the read and the local
# attention are NKI kernels (``nki_sparse_decode.py``); on CPU the torch
# references below implement the same contracts.
DEC_SPARSE_NKI = _flag("HY4_DEC_SPARSE_NKI", False)
# ``HY4_PREFILL_SPARSE_NKI=1``: prefill attention split by QUERY instead of by
# head (``HYV4Attention.forward_prefill_sparse``): each rank attends its ``C / world``
# queries for every head over only the rows DSA selected, with the decode
# kernel ``sparse_decode_attn`` on Neuron (torch reference on CPU).
PREFILL_SPARSE_NKI = _flag("HY4_PREFILL_SPARSE_NKI", False)
# ``HY4_PREFILL_SPARSE_BACKEND``: the sparse prefill attention kernel. "v1" =
# ``sparse_decode_attn`` in position mode (per-128-key-tile indirect DMAs
# through position tables); "cte" = ``nki_sparse_prefill.sparse_prefill_attn``
# (per-slot flat rows built in torch, one batched indirect DMA per query and
# source, after nkilib's ``mla_sparse_attention_cte``). Platform default: "cte"
# on Trn3, "v1" elsewhere. CPU runs the matching torch reference.
PREFILL_SPARSE_BACKEND = os.environ.get("HY4_PREFILL_SPARSE_BACKEND", "v1")
# ``HY4_PREFILL_CTE_DMA_T`` (cte backend): how many of the 5 feature-major key
# chunks (4 latent + the rope window) an indirect ``dma_transpose`` lands
# directly; the rest are PE transposes of the gathered rows. Each DMA-transposed
# chunk moves PE / eviction work onto the DMA queues, whose traffic it
# multiplies (one partition write per element); nkilib's MLA decode kernel
# keeps one chunk on the DMA for that reason. To be swept on device.
PREFILL_CTE_DMA_T = int(os.environ.get("HY4_PREFILL_CTE_DMA_T", "1"))


def resolve_platform_defaults() -> str:
    """Set the attention-path defaults for this host's platform; -> the target.

    Called once from ``HYV4ForCausalLM.__init__``, before any layer is built
    (layers read these flags at construction).
    """
    global DEC_CP, DEC_SPARSE_NKI, PREFILL_SPARSE_NKI, PREFILL_SPARSE_BACKEND
    target = _platform_target()
    on_neuron, on_trn2 = target.startswith("trn"), target.startswith("trn2")
    DEC_CP = _flag("HY4_DEC_CP", on_neuron)
    DEC_SPARSE_NKI = _flag("HY4_DEC_SPARSE_NKI", on_neuron)
    PREFILL_SPARSE_NKI = _flag("HY4_PREFILL_SPARSE_NKI", on_trn2)
    PREFILL_SPARSE_BACKEND = os.environ.get(
        "HY4_PREFILL_SPARSE_BACKEND", "cte" if target.startswith("trn3") else "v1"
    )
    return target


# The sparse prefill runs for graphs whose prefix span (static per graph, from
# the kv_segment bound) is at least this many keys; shorter spans keep the
# dense blocked attention. Shared-indexer layers follow their full layer.
PREFILL_SPARSE_MIN_SPAN = int(os.environ.get("HY4_PREFILL_SPARSE_MIN_SPAN", "4096"))
# Selected-row slots per query for the sparse prefill (rounded up to 128). It
# must hold the top-k plus keys tied with the k-th score; a query with more
# selected keys than slots drops the excess (inexact).
PREFILL_KSEL = int(os.environ.get("HY4_PREFILL_KSEL", "2176") or 2176)
# Cap on the compacted per-rank key list. 0 (default) = the whole shard, which
# is always exact. A smaller cap (rounded up to 128) is exact only while no
# rank holds more selected keys than the cap: the global top-k holds 2048
# keys, but keys tied with the k-th score can add a few more.
SPARSE_KLOC = int(os.environ.get("HY4_SPARSE_KLOC", "0") or 0)
# Row index of a key the sparse path must not read: "oob" (default) = one past
# the last cache row, which the kernels' indirect DMA skips (``oob_mode.skip``,
# destination pre-zeroed); "null" = the null block's first row, a valid finite
# entry that is read and masked (a fallback should a skipped DMA misbehave).
SPARSE_PAD_ROW = os.environ.get("HY4_SPARSE_PAD_ROW", "oob")
if SPARSE_PAD_ROW not in ("oob", "null"):
    raise ValueError(f"HY4_SPARSE_PAD_ROW must be oob or null, got {SPARSE_PAD_ROW!r}")

# Prefill prefix bound from ``kv_segment_size`` (see ``prefill_span_bound``).
# Inert unless neuron_config.kv_segment_size_buckets has more than one entry;
# ``HY4_PREFILL_SPAN_KVSEG=0`` turns it off.
PREFILL_SPAN_KVSEG = os.environ.get("HY4_PREFILL_SPAN_KVSEG", "1") == "1"
# Dtype of the prefill DSA-selection all-gather: "bf16" (the PR's) or "uint8"
# (half the bytes; 0/1 values, so exact either way). Opt-in until it has run on
# device: a uint8 collective is not exercised elsewhere in this model.
SEL_GATHER_DTYPE = os.environ.get("HY4_SEL_GATHER_DTYPE", "bf16")
if SEL_GATHER_DTYPE not in ("bf16", "uint8"):
    raise ValueError(f"HY4_SEL_GATHER_DTYPE must be bf16 or uint8, got {SEL_GATHER_DTYPE!r}")


def prefill_span_bound(kv_segment_size: int, kv_segment_size_buckets) -> int | None:
    """Upper bound on ``cached_seq_len`` implied by a prefill graph's ``kv_segment_size``.

    The runner picks ``kv_segment_size`` with ``pick_kv_segment_size``, which
    returns the smallest configured bucket that holds the whole prefix in one
    segment, and the largest bucket for any prefix longer than that. So a
    segment size *below* the largest bucket means ``cached_seq_len <=
    kv_segment_size``; the largest bucket bounds nothing. ``None`` = no bound.
    ``check_prefill_span_bound`` verifies this against the plugin's picker.
    """
    buckets = sorted(kv_segment_size_buckets or [])
    if not PREFILL_SPAN_KVSEG or len(buckets) < 2 or not kv_segment_size:
        return None
    if kv_segment_size >= buckets[-1]:
        return None
    # Rounded up to whole ``ATTN_KEY_BLOCK``s, so the online softmax sees the
    # same key blocks as the full span (only fully masked ones are dropped)
    # and the output is bitwise identical, not just equal up to rounding.
    return -(-int(kv_segment_size) // ATTN_KEY_BLOCK) * ATTN_KEY_BLOCK


def check_prefill_span_bound(kv_segment_size_buckets, limit: int = 1 << 17) -> bool:
    """True if the plugin's ``pick_kv_segment_size`` honours ``prefill_span_bound``."""
    buckets = sorted(kv_segment_size_buckets or [])
    if len(buckets) < 2:
        return True
    try:
        from vllm_neuron.utils.bucket_utils import pick_kv_segment_size
    except ImportError:
        return False
    for cached in range(0, limit + 1):
        bound = prefill_span_bound(pick_kv_segment_size(cached, buckets), buckets)
        if bound is not None and cached > bound:
            return False
    return True


def _all_reduce(group, t: torch.Tensor) -> torch.Tensor:
    return t if "collectives" in PERF_SKIP else group.all_reduce(t)


def _all_gather(group, t: torch.Tensor, dim: int, world: int) -> torch.Tensor:
    if "collectives" in PERF_SKIP:
        return torch.cat([t] * world, dim=dim)
    return group.all_gather(t, dim=dim)


def _capture(layer_idx: int, name: str, tensor: torch.Tensor) -> None:
    if DEBUG_CAPTURE and layer_idx in DEBUG_LAYERS:
        from vllm_neuron.accuracy import capture_tensor

        capture_tensor(f"hy4.l{layer_idx}.{name}", tensor)


from .moe import (
    FORMAT_FP8_ROW,
    FORMAT_MXFP8,
    HYV4MLP,
    MX_BACKEND_ENV,
    MX_BACKEND_KERNEL,
    MX_BACKEND_TORCH,
    HYV4MoE,
    mx_backend,
    require_mx_platform,
)

logger = logging.getLogger(__name__)

HF_PREFIX = "model"


def _replicated_transposed() -> SafetensorsWeightLoader:
    """Whole ``[out, in]`` checkpoint matrix, stored ``[in, out]`` for ``x @ W``."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


class HYV4RMSNorm(nn.Module):
    """``weight * normalise(x)``, normalised in fp32 and cast back *before* the scale.

    That order is ``HYV4RMSNorm``'s (``self.weight * hidden.to(input_dtype)``),
    and it is not the zero-centred norm Qwen3.5 uses.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(dtype)


# ── iHC ──────────────────────────────────────────────────────────────────


class HYV4HCPre(nn.Module):
    """Reduce ``[T, hc, H]`` streams to one ``[T, H]`` input, and emit the post gates.

    ``fn`` projects the RMS-normalised flattened streams to ``2 * hc`` logits:
    the first ``hc`` weigh the streams into the sub-block input, the second ``hc``
    are how strongly the sub-block's output is written back to each stream.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        hc, h = config.hc_mult, config.hidden_size
        self.hc_mult = hc
        self.eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.magnitude = config.hc_magnitude
        self.fn = nn.Parameter(torch.empty(2 * hc, hc * h, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(2 * hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2, dtype=torch.float32))

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = streams.shape[0]
        hc = self.hc_mult
        x = streams.float()
        flat = x.reshape(tokens, -1)
        rsqrt = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = (flat @ self.fn.t()) * rsqrt  # [T, 2 hc]
        pre = (
            torch.sigmoid(mixes[:, :hc] * self.scale[0] + self.base[:hc]) + self.hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(mixes[:, hc:] * self.scale[1] + self.base[hc:])
            + self.hc_eps
        )
        out = (pre.unsqueeze(-1) * x).sum(dim=1)
        return out.to(streams.dtype), post


def hc_post(
    out: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
) -> torch.Tensor:
    """``residual[t, i] + post[t, i] * out[t]``, in fp32, back in the streams' dtype."""
    y = post.unsqueeze(-1) * out.float().unsqueeze(1) + residual.float()
    return y.to(residual.dtype)


class HYV4HCHead(nn.Module):
    """Final collapse of the streams before ``model.norm`` (no post gates)."""

    def __init__(self, config: HYV4Config):
        super().__init__()
        hc, h = config.hc_mult, config.hidden_size
        self.eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = nn.Parameter(torch.empty(hc, hc * h, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(hc, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        tokens = streams.shape[0]
        x = streams.float()
        flat = x.reshape(tokens, -1)
        rsqrt = torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mixes = (flat @ self.fn.t()) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre.unsqueeze(-1) * x).sum(dim=1).to(streams.dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────


class HYV4RotaryEmbedding(nn.Module):
    """``(cos, sin)`` of ``[T, rope_dim / 2]`` for the 64 rope dims of MLA."""

    inv_freq: torch.Tensor

    def __init__(self, config: HYV4Config):
        super().__init__()
        dim = config.qk_rope_head_dim
        # device="cpu": the runner builds the model under the meta device, and a
        # buffer that no checkpoint tensor overwrites would stay meta.
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [T, ..., 64]`` by ``cos/sin [T, 32]``, broadcast over middle dims.

    Interleaved pairs ``(x[2i], x[2i+1])`` by default, ``(x[i], x[i+32])`` under
    ``HY4_ROPE_STYLE=neox``. Rotation is done in fp32.
    """
    dtype = x.dtype
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x = x.float()
    if ROPE_STYLE == "interleaved":
        pairs = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = pairs[..., 0], pairs[..., 1]
        out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        return out.flatten(-2).to(dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype)


def apply_rope_nocat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rope`` without ``torch.stack`` / ``torch.cat`` (bit-identical).

    The rotation's partner of each element comes from a ``flip`` of its pair
    (interleaved) or half (neox) axis, signed ``[-1, +1]``: ``a - b`` is
    computed as ``a + (-b)``, which is exact. For tensors that feed an NKI
    kernel, where a concatenation failed neuronx-cc (NCC_INIC902).
    """
    dtype = x.dtype
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x = x.float()
    half = x.shape[-1] // 2
    # [-1, +1] built from arange, not torch.tensor: a literal tensor created in
    # the traced forward is not a fake tensor and breaks dynamo's tracing.
    sign = torch.arange(2, dtype=torch.float32, device=x.device) * 2.0 - 1.0
    if ROPE_STYLE == "interleaved":
        pairs = x.view(*x.shape[:-1], half, 2)
        partner = pairs.flip(-1) * sign
        out = pairs * cos.unsqueeze(-1) + partner * sin.unsqueeze(-1)
    else:
        halves = x.view(*x.shape[:-1], 2, half)
        partner = halves.flip(-2) * sign.view(2, 1)
        out = halves * cos.unsqueeze(-2) + partner * sin.unsqueeze(-2)
    return out.flatten(-2).to(dtype)


# ── Attention ────────────────────────────────────────────────────────────

# Bisection steps for the top-k threshold; see ``select_topk``. 24 halvings of
# the per-row score range leave ties within ~6e-8 of it unresolved, which only
# ever widens the selection by keys tied with the k-th.
TOPK_BISECTION_STEPS = 24
# Keys per block of the prefill attention's online softmax (see
# ``HYV4Attention._attend_blocked``).
ATTN_KEY_BLOCK = 2048
# Stand-in for -inf/+inf in the selection, where real infinities would turn the
# bisection midpoint into NaN.
_BIG = 3.0e38


def select_topk(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of each row's top-``k`` visible keys (all of them if fewer).

    ``scores``/``visible`` are ``[rows, keys]``; every row must have at least one
    visible key (each query always sees itself).

    By bisection on a per-row threshold rather than ``torch.topk``: it is a few
    elementwise passes over ``[rows, keys]``, with no k=2048 sort and no scatter
    to turn indices back into a mask. The invariant is ``count(s >= lo) >= k``
    (or ``lo`` is the row minimum, which selects everything visible), so the
    result is exactly the top-k up to keys tied with the k-th score.
    """
    s = scores.float()
    return visible & (s >= topk_threshold(s, visible, k))


def topk_threshold(
    s: torch.Tensor,
    visible: torch.Tensor,
    k: int,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """``select_topk``'s per-row threshold ``lo`` ``[rows, 1]`` over fp32 ``s``.

    It depends only on the multiset of visible scores per row (max, min and
    counts), not on their order, so it can run over keys gathered from ranks
    in any order (``forward_decode_cp``). ``extra`` ``[rows, 1]`` is one more
    always-visible score, counted as if it were a column of ``s`` (so callers
    need not concatenate it).
    """
    hi = s.masked_fill(~visible, -_BIG).amax(dim=-1, keepdim=True)
    lo = s.masked_fill(~visible, _BIG).amin(dim=-1, keepdim=True)
    if extra is not None:
        hi = torch.maximum(hi, extra)
        lo = torch.minimum(lo, extra)
    for _ in range(TOPK_BISECTION_STEPS):
        mid = (lo + hi) * 0.5
        count = ((s >= mid) & visible).float().sum(dim=-1, keepdim=True)
        if extra is not None:
            count = count + (extra >= mid).float()
        up = count >= k
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid)
    return lo


def topk_threshold2(
    s1: torch.Tensor, v1: torch.Tensor, s2: torch.Tensor, v2: torch.Tensor, k: int
) -> torch.Tensor:
    """``topk_threshold`` over the columns of ``[s1 | s2]`` without concatenating:
    max, min and the (integer, exact in fp32) counts combine per piece, so the
    result is bit-identical to the one over the concatenation."""
    hi = torch.maximum(
        s1.masked_fill(~v1, -_BIG).amax(dim=-1, keepdim=True),
        s2.masked_fill(~v2, -_BIG).amax(dim=-1, keepdim=True),
    )
    lo = torch.minimum(
        s1.masked_fill(~v1, _BIG).amin(dim=-1, keepdim=True),
        s2.masked_fill(~v2, _BIG).amin(dim=-1, keepdim=True),
    )
    for _ in range(TOPK_BISECTION_STEPS):
        mid = (lo + hi) * 0.5
        count = ((s1 >= mid) & v1).float().sum(dim=-1, keepdim=True) + (
            (s2 >= mid) & v2
        ).float().sum(dim=-1, keepdim=True)
        up = count >= k
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid)
    return lo


_SPARSE_KERNELS: list = []


def _sparse_kernels():
    """``(attn, gather_cols, select_to_rows, prefill_attn, topk_rows)`` NKI kernels
    through the plugin's ``wrap_nki`` (lazy)."""
    if not _SPARSE_KERNELS:
        try:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        except ImportError:  # source-tree layout
            from vllm_neuron.nki.nki_hop import wrap_nki
        from .nki_sparse_decode import gather_rows_cols, sparse_decode_attn
        from .nki_sparse_prefill import select_to_rows, sparse_prefill_attn, topk_rows

        _SPARSE_KERNELS.extend(
            (
                wrap_nki(sparse_decode_attn),
                wrap_nki(gather_rows_cols),
                wrap_nki(select_to_rows),
                wrap_nki(sparse_prefill_attn),
                wrap_nki(topk_rows),
            )
        )
    return _SPARSE_KERNELS


def _sparse_kernels_or_none() -> tuple:
    """The kernels where nki and the plugin's ``wrap_nki`` import; else ``None``s
    (CPU unit tests), in which case the torch references run."""
    try:
        return tuple(_sparse_kernels())
    except ImportError:
        return (None, None, None, None, None)


def _sparse_grid() -> int:
    """Launch grid of the sparse kernels: the logical NC config the graph compiles at
    (neuronx-cc rejects a kernel launched at another degree, NCC_EVRF066)."""
    try:
        from vllm_neuron.utils.hardware_config import get_logical_nc_config
    except ImportError:  # CPU unit tests without vLLM
        return 1
    from vllm_neuron import envs

    return 1 if envs.VLLM_NEURON_CPU_MODE else get_logical_nc_config()


def _use_sparse_kernels(t: torch.Tensor) -> bool:
    """The plugin's rule (Neuron device, or CPU mode with ``NKI_SIMULATOR=1``)."""
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ImportError:  # CPU unit tests without vLLM
        return t.device.type != "cpu"
    return can_run_kernel(t)


def _flat_rows_ref(kv_pages: torch.Tensor) -> torch.Tensor:
    return kv_pages.reshape(-1, kv_pages.shape[-1])


def gather_rows_cols_ref(
    kv_pages: torch.Tensor, rows: torch.Tensor, col_off: int, width: int
) -> torch.Tensor:
    """``nki_sparse_decode.gather_rows_cols``: ``out[r, k] = cache_row[rows[r, k]][col_off:+width]``;
    out-of-range rows give zeros."""
    flat = _flat_rows_ref(kv_pages)
    n_rows = flat.shape[0]
    ok = (rows >= 0) & (rows < n_rows)
    safe = torch.where(ok, rows, torch.zeros_like(rows)).to(torch.long)
    out = flat[safe][..., col_off : col_off + width]
    return out * ok.unsqueeze(-1).to(out.dtype)


def sparse_decode_ref(
    q_all: torch.Tensor,
    kv_pages: torch.Tensor,
    rows: torch.Tensor,
    bound: torch.Tensor,
    scale: float,
    col_off: int,
    lat_dim: int,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``nki_sparse_decode.sparse_decode_attn`` (``lat_off = 0``, rope after the latent).

    ``s = min(scale * q . row[col_off:col_off+D], bound)``; -> unnormalised
    ``m [B, H]``, ``den [B, H]``, ``num [B, H, lat_dim]``, all fp32.
    """
    width = q_all.shape[-1]
    keys = gather_rows_cols_ref(kv_pages, rows, col_off, width)  # [B, n, D]
    s = (q_all.float() @ keys.float().transpose(1, 2)) * scale  # [B, H, n]
    s = torch.minimum(s, bound.unsqueeze(1))
    m = s.amax(dim=-1)
    p = torch.exp(s - m.unsqueeze(-1))
    den = p.sum(dim=-1)
    num = (p.to(value_dtype) @ keys[..., :lat_dim].to(value_dtype)).float()
    return m, den, num


def sparse_prefill_ref(
    q_lat: torch.Tensor,
    q_rope: torch.Tensor,
    kv_pages: torch.Tensor,
    pos_map: torch.Tensor,
    own_lat: torch.Tensor,
    own_rope: torch.Tensor,
    rows: torch.Tensor,
    bound: torch.Tensor,
    scale: float,
    col_off: int,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sparse_decode_attn`` in position mode (CPU): ``rows`` are key positions,
    ``p < P`` -> cache row ``pos_map[p]``, ``P <= p < P + C`` -> own row ``p - P``,
    else zeros. -> ``m, den, num`` as ``sparse_decode_ref``."""
    n_pos, n_own = pos_map.shape[0], own_lat.shape[0]
    lat_dim, rope_dim = own_lat.shape[1], own_rope.shape[1]
    r = rows.to(torch.long)
    is_ctx = r < n_pos
    is_own = (r >= n_pos) & (r < n_pos + n_own)
    flat = pos_map.view(-1).to(torch.long)[torch.where(is_ctx, r, torch.zeros_like(r))]
    flat = torch.where(is_ctx, flat, torch.full_like(flat, -1))
    ctx = gather_rows_cols_ref(kv_pages, flat, col_off, lat_dim + rope_dim).float()
    oi = torch.where(is_own, r - n_pos, torch.zeros_like(r))
    m_own = is_own.unsqueeze(-1).float()
    lat = ctx[..., :lat_dim] + own_lat.float()[oi] * m_own  # [B, n, L]
    rope = ctx[..., lat_dim:] + own_rope.float()[oi] * m_own
    s = (q_lat.float() @ lat.transpose(1, 2) + q_rope.float() @ rope.transpose(1, 2)) * scale
    s = torch.minimum(s, bound.unsqueeze(1))
    m = s.amax(dim=-1)
    p = torch.exp(s - m.unsqueeze(-1))
    den = p.sum(dim=-1)
    num = (p.to(value_dtype) @ lat.to(value_dtype)).float()
    return m, den, num


def sparse_prefill_cte_ref(
    q_lat: torch.Tensor,
    q_rope: torch.Tensor,
    kv_pages: torch.Tensor,
    rows_cache: torch.Tensor,
    own_lat: torch.Tensor,
    own_rope: torch.Tensor,
    rows_own: torch.Tensor,
    bound: torch.Tensor,
    scale: float,
    col_off: int,
    value_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``nki_sparse_prefill.sparse_prefill_attn`` (CPU): slot ``k`` reads cache row
    ``rows_cache[q, k]`` and own row ``rows_own[q, k]`` (out of range = zeros),
    summed. -> ``m, den, num`` as ``sparse_decode_ref``."""
    n_own = own_lat.shape[0]
    lat_dim, rope_dim = own_lat.shape[1], own_rope.shape[1]
    ctx = gather_rows_cols_ref(kv_pages, rows_cache, col_off, lat_dim + rope_dim).float()
    ro = rows_own.to(torch.long)
    ok = (ro >= 0) & (ro < n_own)
    oi = torch.where(ok, ro, torch.zeros_like(ro))
    m_own = ok.unsqueeze(-1).float()
    own = torch.cat((own_lat.float()[oi], own_rope.float()[oi]), dim=-1) * m_own
    keys = ctx + own
    lat, rope = keys[..., :lat_dim], keys[..., lat_dim:]
    s = (q_lat.float() @ lat.transpose(1, 2) + q_rope.float() @ rope.transpose(1, 2)) * scale
    s = torch.minimum(s, bound.unsqueeze(1))
    m = s.amax(dim=-1)
    p = torch.exp(s - m.unsqueeze(-1))
    den = p.sum(dim=-1)
    num = (p.to(value_dtype) @ lat.to(value_dtype)).float()
    return m, den, num


def _zero_beyond(ctx: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Zero the gathered cache entries at positions ``>= limit`` (per row).

    Not needed for correctness -- those keys are masked out of the scores, and
    this model's single KV group only ever holds finite entries -- but in the
    one build without it the 32K prefill was ~2.7x slower (150 s vs 55 s per
    chunk) and its graph twice the size (596K vs 284K instructions per
    subgraph). Why is not understood; the masked build is the measured one.

    The predicate here is computed *separately* from ``in_prefix`` (float
    subtraction, not the integer compare) on purpose. Applying ``in_prefix``
    itself to the ``[rows, 32K, 704]`` context made neuronx-cc 2.27 compute
    ``in_prefix`` wrongly wherever it was reused -- 128-key blocks set past the
    prefix, even on a dead row at position 0 -- so every 32K decode attended
    to the wrong keys while the CPU, and the same ops compiled separately,
    were right. A second, independent predicate compiles correctly (checked
    on device in a standalone graph of the decode attention chain).
    """
    rows = ctx.shape[0] if ctx.dim() == 3 else 1
    ctx_len = ctx.shape[-2]
    positions = torch.arange(ctx_len, device=ctx.device, dtype=torch.float32)
    # A tensor threshold: comparing against a Python float lowers to an f64
    # constant, which neuronx-cc rejects (NCC_ESPP004).
    half = torch.full((1, 1), 0.5, dtype=torch.float32, device=ctx.device)
    keep = (limit.float().view(rows, 1) - positions.view(1, ctx_len)) > half
    keep = keep.to(ctx.dtype)
    return ctx * (keep.unsqueeze(-1) if ctx.dim() == 3 else keep.view(ctx_len, 1))


def _zero_beyond_at(
    ctx: torch.Tensor, limit: torch.Tensor, key_pos: torch.Tensor
) -> torch.Tensor:
    """``_zero_beyond`` for ``ctx [rows, n, D]`` whose keys sit at ``key_pos [n]``.

    Same separately computed float predicate (see ``_zero_beyond``), for a
    rank's key shard whose positions are not ``arange(n)``.
    """
    rows, n = ctx.shape[0], ctx.shape[1]
    half = torch.full((1, 1), 0.5, dtype=torch.float32, device=ctx.device)
    keep = (limit.float().view(rows, 1) - key_pos.float().view(1, n)) > half
    return ctx * keep.to(ctx.dtype).unsqueeze(-1)


class HYV4Indexer(nn.Module):
    """DSA lightning indexer: scores every visible key, the top ``index_topk`` attend.

    ``score[t, s] = sum_h w[t, h] * relu(q[t, h] . k[s])`` with 32 query heads of
    128 against one 128-wide key per token (LayerNorm'd, with the rope applied to
    its *last* 64 dims, as is the query's). Replicated on every rank: it is small
    (~9.4M parameters) and its output, a selection over keys, is needed by each
    rank's attention head in full. The reference's Hadamard rotation and fp8
    scoring are precision/speed devices that leave the selection's definition
    unchanged, and are skipped, as transformers does.
    """

    def __init__(self, config: HYV4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        hidden = config.hidden_size
        self.wq_b_weight = nn.Parameter(
            torch.empty(
                config.q_lora_rank, self.n_heads * self.head_dim, dtype=self.dtype
            )
        )
        self.wk_weight = nn.Parameter(
            torch.empty(hidden, self.head_dim, dtype=self.dtype)
        )
        self.k_norm_weight = nn.Parameter(
            torch.ones(self.head_dim, dtype=torch.float32)
        )
        self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
        self.weights_proj_weight = nn.Parameter(
            torch.empty(hidden, self.n_heads, dtype=torch.float32)
        )
        # vLLM's HYV4 indexer uses 1e-6 here, not rms_norm_eps.
        self.k_norm_eps = 1e-6
        self.weight_scale = self.head_dim**-0.5 * self.n_heads**-0.5
        self.rope_first = config.indexer_rope_first
        for param in (self.wq_b_weight, self.wk_weight, self.weights_proj_weight):
            set_weight_loader(param, _replicated_transposed())

    def _rope_tail(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if self.rope_first:  # GLM-5.x: the rotated dims lead
            r = self.rope_dim
            return torch.cat((apply_rope(x[..., :r], cos, sin), x[..., r:]), dim=-1)
        keep = self.head_dim - self.rope_dim
        return torch.cat((x[..., :keep], apply_rope(x[..., keep:], cos, sin)), dim=-1)

    def keys(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """``x [T, H]`` -> ``[T, 128]``, what the cache stores for this layer."""
        k = F.layer_norm(
            (x @ self.wk_weight).float(),
            (self.head_dim,),
            self.k_norm_weight,
            self.k_norm_bias,
            self.k_norm_eps,
        )
        return self._rope_tail(k, cos, sin).to(self.dtype)

    def queries(
        self,
        q_resid: torch.Tensor,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (``[T, heads, 128]`` queries, ``[T, heads]`` fp32 head weights)."""
        q = (q_resid @ self.wq_b_weight).view(-1, self.n_heads, self.head_dim)
        weights = (x.float() @ self.weights_proj_weight) * self.weight_scale
        return self._rope_tail(q, cos, sin), weights


class HYV4Attention(nn.Module):
    """Gated MLA with learnable sinks over the keys DSA selects.

    The cache entry per token is 704 wide: the 128-wide indexer key, then the
    576-wide MLA latent (normed kv latent + rope key), stored as the K and V
    halves of one 352-wide head. Layers whose indexer is ``"shared"``
    leave the indexer slot zero and reuse the selection of the last ``"full"``
    layer before them, which the decoder threads through.
    """

    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.neuron_config = config.neuron_config
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        heads = config.num_attention_heads
        if heads % self.world_size:
            raise ValueError(
                f"num_attention_heads={heads} must be divisible by "
                f"tp_size={self.world_size}"
            )
        self.num_heads = heads // self.world_size
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = config.qk_head_dim
        self.v_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.latent_dim = config.kv_latent_dim
        self.index_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.entry_dim = self.latent_dim + self.index_dim
        # Entry layout [index key 128 | kv latent 512 | rope key 64]: the
        # indexer key first, so both slices start on a 128-column boundary.
        # With it last (column 576), the prefill indexer's matmul against
        # ``keys[:, 576:]`` failed neuronx-cc ISA validation at -O1 ("Copy;
        # inst failed assertion check: 'start_addr_active_channels'").
        self.index_slice = slice(0, self.index_dim)
        self.latent_slice = slice(self.index_dim, self.entry_dim)
        if self.entry_dim % 2:
            raise ValueError(f"cache entry width {self.entry_dim} must be even")
        # The K and V halves of the cache page, each this wide.
        self.cache_head_dim = self.entry_dim // 2
        self.scaling = self.qk_dim**-0.5
        hidden = config.hidden_size
        nh = self.num_heads

        self.indexer = (
            HYV4Indexer(config) if config.indexer_types[layer_idx] == "full" else None
        )

        self.q_a_proj_weight = nn.Parameter(
            torch.empty(hidden, config.q_lora_rank, dtype=self.dtype)
        )
        self.q_a_layernorm = HYV4RMSNorm(
            config.q_lora_rank, config.rms_norm_eps, self.dtype
        )
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(config.q_lora_rank, nh * self.qk_dim, dtype=self.dtype)
        )
        self.kv_a_proj_weight = nn.Parameter(
            torch.empty(hidden, self.latent_dim, dtype=self.dtype)
        )
        self.kv_a_layernorm = HYV4RMSNorm(
            self.kv_lora_rank, config.rms_norm_eps, self.dtype
        )
        # kv_b_proj split per head into its key and value halves. The key half keeps the
        # checkpoint's [out, latent] orientation (it contracts over out); the value half is
        # stored [latent, out], the way it contracts. In [out, latent] the value up-projection
        # transposed the weight every step, and in GLM-5.3's decode graph that transpose's
        # spill/reload gets an out-of-range address (NCC_INLA001 / ISA validation failure).
        self.w_uk = nn.Parameter(
            torch.empty(nh, self.nope_dim, self.kv_lora_rank, dtype=self.dtype)
        )
        self.w_uv = nn.Parameter(
            torch.empty(nh, self.kv_lora_rank, self.v_dim, dtype=self.dtype)
        )
        self.gated = config.gated_mla
        if self.gated:
            self.gate_proj_weight = nn.Parameter(
                torch.empty(hidden, nh * self.v_dim, dtype=self.dtype)
            )
        self.o_proj_weight = nn.Parameter(
            torch.empty(nh * self.v_dim, hidden, dtype=self.dtype)
        )
        self.learnable_sink = config.learnable_sink
        if self.learnable_sink:
            self.sinks = nn.Parameter(torch.zeros(nh, dtype=torch.float32))
        else:
            # No sinks (GLM-5.x): a -1e30 logit takes exactly zero weight in every softmax
            # here, dense and key-blocked, so no path changes.
            # device="cpu": the model is built under the meta device, and a buffer made
            # there has no data to move (as for HYV4RotaryEmbedding.inv_freq).
            self.register_buffer(
                "sinks", torch.full((nh,), -1.0e30, device="cpu"), persistent=False
            )
        self.kv_pages: torch.Tensor | None = None
        # Read once here, not in the traced forward.
        use_kernels = DEC_SPARSE_NKI or PREFILL_SPARSE_NKI
        self.sparse_grid = _sparse_grid() if use_kernels else 1
        # Built here, never inside the traced forward: a lazy build there puts a
        # guard on the module-level cache that flips after the first trace and
        # forces a recompile at serving time.
        (
            self._sparse_attn_k,
            self._sparse_gather_k,
            self._select_rows_k,
            self._prefill_attn_k,
            self._topk_rows_k,
        ) = _sparse_kernels_or_none() if use_kernels else (None, None, None, None, None)
        self.prefill_backend = PREFILL_SPARSE_BACKEND
        self.prefill_cte_dma_t = PREFILL_CTE_DMA_T

        set_weight_loader(self.q_a_proj_weight, _replicated_transposed())
        set_weight_loader(self.kv_a_proj_weight, _replicated_transposed())
        set_weight_loader(
            self.q_b_proj_weight,
            sharding_weight_loader(
                shard_dim=1,
                shard_size=nh * self.qk_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.gated:
            set_weight_loader(
                self.gate_proj_weight,
                sharding_weight_loader(
                    shard_dim=1,
                    shard_size=nh * self.v_dim,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                ),
            )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=nh * self.v_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.learnable_sink:
            set_weight_loader(
                self.sinks,
                sharding_weight_loader(
                    shard_dim=0, shard_size=nh, num_shards=self.world_size
                ),
            )
        per_head = self.nope_dim + self.v_dim

        def kv_b(part: str) -> SafetensorsWeightLoader:
            def transform(slices: list, rank: int) -> torch.Tensor:
                (source,) = slices  # [heads * (nope + v), kv_lora_rank]
                rows = source[rank * nh * per_head : (rank + 1) * nh * per_head, :]
                rows = rows.reshape(nh, per_head, self.kv_lora_rank)
                if part == "k":
                    return rows[:, : self.nope_dim]
                return rows[:, self.nope_dim :].transpose(1, 2).contiguous()

            return SafetensorsWeightLoader(transform=transform)

        set_weight_loader(self.w_uk, kv_b("k"))
        set_weight_loader(self.w_uv, kv_b("v"))

    # ── shared pieces ────────────────────────────────────────────────────

    def _project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x [T, H]`` -> (query ``[T, nh, 576]``, entry ``[T, 704]``, gate, q_resid).

        The query's no-rope part is absorbed through ``w_uk`` so that it scores
        directly against the cached latent; its rope part scores against the
        cached rope key. Concatenating both gives one 576-wide dot product.
        ``entry`` is what this token writes to the cache.
        """
        q_lat, q_rope, index_key, c, k_rope, gate, q_resid = self._project_parts(x, cos, sin)
        query = torch.cat((q_lat, q_rope), dim=-1)
        entry = torch.cat((index_key.to(c.dtype), c, k_rope.to(c.dtype)), dim=-1)
        return query, entry, gate, q_resid

    def _project_parts(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, nocat: bool = False
    ) -> tuple:
        """``_project`` before its two concatenations: (q_lat ``[T, nh, 512]``,
        q_rope ``[T, nh, 64]``, index_key ``[T, 128]``, latent ``[T, 512]``,
        k_rope ``[T, 64]``, gate, q_resid). ``nocat``: the ropes by
        ``apply_rope_nocat`` (same values; they feed NKI kernels)."""
        rope = apply_rope_nocat if nocat else apply_rope
        tokens = x.shape[0]
        q_resid = self.q_a_layernorm(x @ self.q_a_proj_weight)
        q = (q_resid @ self.q_b_proj_weight).view(tokens, self.num_heads, self.qk_dim)
        q_nope, q_rope = q[..., : self.nope_dim], q[..., self.nope_dim :]
        q_rope = rope(q_rope, cos, sin)
        q_lat = torch.einsum("tnd,ndc->tnc", q_nope, self.w_uk)

        kv = x @ self.kv_a_proj_weight
        c = self.kv_a_layernorm(kv[:, : self.kv_lora_rank])
        k_rope = rope(kv[:, self.kv_lora_rank :], cos, sin)
        if self.indexer is not None:
            index_key = self.indexer.keys(x, cos, sin)
        else:
            index_key = torch.zeros(
                tokens, self.index_dim, dtype=self.dtype, device=x.device
            )
        gate = x @ self.gate_proj_weight if self.gated else None
        return q_lat, q_rope, index_key, c, k_rope, gate, q_resid

    def _finish(self, o_lat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``o_lat [T, nh, 512]`` -> value up-projection, gate, ``o_proj`` partial."""
        tokens = o_lat.shape[0]
        out = torch.einsum("tnc,ncv->tnv", o_lat.to(self.dtype), self.w_uv)
        out = out.reshape(tokens, self.num_heads * self.v_dim)
        if gate is not None:
            out = out * torch.sigmoid(gate.float()).to(out.dtype)
        out = out @ self.o_proj_weight
        if self.world_size > 1:
            out = _all_reduce(self.tp_group, out)
        return out

    # vLLM's null block: the scheduler never hands it to a request, so padded writes go
    # there and dead rows / table padding read it. It is not zeros (padded writes land in
    # it), but everything written is a finite entry, so masked keys still get exactly 0.
    _NULL_BLOCK = 0

    def _num_pages(self) -> int:
        return self.kv_pages.shape[0]

    def _block_size(self) -> int:
        return self.kv_pages.shape[2]

    def _rows(self) -> torch.Tensor:
        """The paged cache as ``[pages, block * 704]``: one contiguous entry row per slot.

        The cache is a key-only one (``LayerSpec(key_only=True)``): vLLM allocates it as
        ``[pages, 1, block, 704]``, one 704-value entry per slot, so a page is one
        contiguous 45 KB read.
        """
        return self.kv_pages.view(self._num_pages(), -1)

    def _write_kv(self, entry: torch.Tensor, metadata: dict) -> None:
        """Scatter ``[T, 704]`` entries into their slots (see ``_rows`` for the layout).

        Padded tokens (``PAD_SLOT_ID`` or out of range) go to the null block, via the
        plugin's ``paged_slot_indices``.
        """
        blocks, offsets = paged_slot_indices(
            metadata["slot_mapping"], self._block_size(), self._num_pages()
        )
        # Index the [pages, 1, block, 704] cache as the plugin's own models do: block,
        # head, position. The head index is an arange, not a literal 0 -- a constant index
        # is folded to rank 0 and fails the graph (NCC_IRRW901) -- and any reshaped view of
        # the cache fails to lower (NCC_ILTO901 "Incorrect reshape basis").
        heads = torch.zeros_like(blocks) + torch.arange(1, device=blocks.device)
        self.kv_pages.index_put_(
            (blocks.to(torch.long), heads.to(torch.long), offsets.to(torch.long)),
            entry.to(self.kv_pages.dtype),
        )

    def _gather(
        self, block_table: torch.Tensor, live_rows: torch.Tensor
    ) -> torch.Tensor:
        """``[R, W]`` block ids -> ``[R, W * block_size, 704]`` cache entries.

        One page gather over ``_rows``. Dead rows and table padding (-1) read the null
        block.
        """
        num_rows = block_table.shape[0]
        bt = block_table.to(torch.long)
        num_pages, block_size = self._num_pages(), self._block_size()
        blocks = torch.where(
            live_rows & (bt > 0) & (bt < num_pages), bt, torch.full_like(bt, self._NULL_BLOCK)
        )
        ctx_len = blocks.shape[1] * block_size
        # Gathered as [pages, 2 * block, 352] (the bytes of _rows' [pages, block * 704]):
        # over the 2-D view, neuronx-cc split the gather into millions of DMA descriptors
        # and one decode layer took 15+ min to compile; this is the shape it handles.
        half = self.kv_pages.view(num_pages, 2 * block_size, self.cache_head_dim)
        pages = torch.index_select(half, 0, blocks.reshape(-1))
        return pages.reshape(num_rows, ctx_len, self.entry_dim)

    def _with_sinks(self, scores: torch.Tensor) -> torch.Tensor:
        """Softmax over ``[..., keys]`` plus one sink column per head, sink dropped.

        ``scores`` is ``[nh, rows, keys]``-shaped with heads first.
        """
        sink = self.sinks.view(-1, 1, 1).expand(scores.shape[0], scores.shape[1], 1)
        probs = torch.softmax(torch.cat((scores, sink.float()), dim=-1), dim=-1)
        return probs[..., :-1]

    def _attend_blocked(
        self, query: torch.Tensor, keys: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        """Prefill attention with sinks, ``[nh, C, 512]``, in key blocks.

        ``query`` ``[nh, C, 576]``, ``keys`` ``[K, 704]``, ``selected`` ``[C, K]``.

        At 32K the ``[C, K]`` score matrix is ~285 MB of fp32 per head per
        layer against ~24 MB of SBUF per core, and a softmax over its full rows
        made neuronx-cc spill it: profiled on one layer, 2.6 GB spilled out and
        7.3 GB reloaded per step, as tens of millions of ~150-byte DMA packets,
        with the Tensor engine 6% busy. So keys are processed ``ATTN_KEY_BLOCK``
        at a time with an online softmax (running max ``m``, running sum ``l``,
        and the output rescaled by ``exp(m_old - m_new)``), and the sink -- a
        logit with no value -- joins at the end, in the max and the sum.
        """
        nh, chunk, _ = query.shape
        q = query.float() * self.scaling
        # A finite floor, not -inf: with -inf a row that has nothing selected
        # in the blocks so far computes exp(-inf - -inf) = NaN. With the floor,
        # its masked keys briefly count with weight 1, and the first selected
        # key (every row has one) rescales them by exp(floor - m) = 0.
        floor = -1.0e30
        m = torch.full((nh, chunk, 1), floor, device=q.device)
        l = torch.zeros((nh, chunk, 1), device=q.device)
        acc = torch.zeros((nh, chunk, self.kv_lora_rank), device=q.device)
        total = keys.shape[0]
        for start in range(0, total, ATTN_KEY_BLOCK):
            stop = min(start + ATTN_KEY_BLOCK, total)
            lat = keys[start:stop, self.latent_slice]
            s = q @ lat.float().t().unsqueeze(0)  # [nh, C, B]
            s = s.masked_fill(~selected[:, start:stop].unsqueeze(0), floor)
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            corr = torch.exp(m - m_new)
            e = torch.exp(s - m_new)
            l = l * corr + e.sum(dim=-1, keepdim=True)
            acc = (
                acc * corr
                + (e.to(self.dtype) @ lat[:, : self.kv_lora_rank].unsqueeze(0)).float()
            )
            m = m_new
        sink = self.sinks.view(-1, 1, 1).float()
        m_f = torch.maximum(m, sink)
        scale = torch.exp(m - m_f)
        return acc * scale / (l * scale + torch.exp(sink - m_f))

    # ── phases ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        selected: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (output, this layer's key selection, for the next shared layers)."""
        metadata = attn_metadata[self.layer_name]
        if self.indexer is None and selected is None:
            raise ValueError(
                f"{self.layer_name} shares its DSA selection but no full-indexer "
                f"layer ran before it"
            )
        if metadata["max_query_len"] <= metadata["decode_token_threshold"]:
            if DEC_CP and self.cp_supported(metadata["block_table_tensor"].shape[1]):
                return self.forward_decode_cp(
                    x, positions, position_embeddings, metadata, selected, rank
                )
            rows = metadata["block_table_tensor"].shape[0]
            # See forward_decode_split for why the batch picks the form.
            split = rows == 1 or rows >= 16
            decode = self.forward_decode_split if split else self.forward_decode
            return decode(x, positions, position_embeddings, metadata, selected)
        return self.forward_prefill(x, position_embeddings, metadata, selected, rank)

    def _select_prefill(
        self,
        iq: torch.Tensor,
        iw: torch.Tensor,
        ik: torch.Tensor,
        visible: torch.Tensor,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        """DSA selection ``[C, K]`` for a prefill chunk, split by query across ranks.

        Every rank needs the whole selection (its head attends to every query's
        keys), but each query's selection depends only on that query. So rank
        ``r`` scores and selects for its ``C / world`` queries against all keys,
        and one all-gather assembles the mask. Done replicated instead, the
        indexer and the top-k bisection were ~90% of the prefill graph at 32K
        context -- 15.7M instructions at -O3, over neuronx-cc's 10M limit.

        ``rank`` is the runner's tensor, not ``rank_in_group``: a Python rank
        would bake into the trace and make 64 different graphs.
        """
        chunk, world = iq.shape[0], self.world_size
        if world == 1 or chunk % world:
            rows = slice(None)
            local_q, local_w, local_vis = iq, iw, visible
        else:
            per_rank = chunk // world
            if rank is None:
                rank = torch.tensor(self.tp_group.rank_in_group, device=iq.device)
            rows = rank.reshape(1).to(torch.long) * per_rank + torch.arange(
                per_rank, device=iq.device
            )
            local_q = torch.index_select(iq, 0, rows)
            local_w = torch.index_select(iw, 0, rows)
            local_vis = torch.index_select(visible, 0, rows)
        # A flat matmul, not ``einsum("qhd,kd->qhk")``: with the key-blocked
        # attention in the same graph, neuronx-cc 2.27 failed ISA validation on
        # the einsum's lowering ("Copy; inst failed assertion check:
        # 'start_addr_active_channels'") at -O1.
        rows_q, heads, dim = local_q.shape
        s = (local_q.reshape(rows_q * heads, dim) @ ik.t()).view(rows_q, heads, -1)
        s = torch.relu(s.float())
        scores = (s * local_w.unsqueeze(-1)).sum(dim=1)  # [C / world, K]
        local = select_topk(scores, local_vis, self.index_topk)
        _capture(self.layer_idx, "index_scores_local", scores)
        _capture(self.layer_idx, "selected_local", local.to(torch.bfloat16))
        if isinstance(rows, slice):
            return local
        # The all-gather's block order is the collective's, which on Trn2 is
        # *not* tp-rank order (the rank -> device mapping is a mesh; see
        # ``_TRN2_MESH``). Assuming it was gave every query some other query's
        # selection -- coherent-looking garbage from the first token. So the
        # ranks are gathered too, and a 0/1 permutation matmul puts block ``j``
        # (from rank ``j``) at rows ``[j * per_rank, (j + 1) * per_rank)``.
        # Exact in bf16: every output element is one 0/1 product.
        #
        # bf16 for the collective, which bool may not be on every backend; back
        # by a cast, not ``> 0.5``: a bf16-vs-Python-float compare lowers to an
        # f64 constant, which neuronx-cc rejects (NCC_ESPP004).
        # ``HY4_SEL_GATHER_DTYPE=uint8`` halves the bytes; the 0/1 values are
        # cast to bf16 after the collective, so ``ordered`` is identical.
        wire = torch.uint8 if SEL_GATHER_DTYPE == "uint8" else torch.bfloat16
        gathered = _all_gather(self.tp_group, local.to(wire), 0, world)
        gathered = gathered.to(torch.bfloat16)
        if "collectives" in PERF_SKIP:
            sources = torch.arange(world, dtype=torch.float32, device=iq.device)
        else:
            sources = self.tp_group.all_gather(rank.reshape(1).to(torch.float32), dim=0)
        targets = torch.arange(world, dtype=torch.float32, device=iq.device)
        perm = (targets.view(world, 1) == sources.view(1, world)).to(torch.bfloat16)
        ordered = perm @ gathered.reshape(world, -1)
        _capture(self.layer_idx, "gather_sources", sources)
        return ordered.reshape(chunk, -1).to(torch.bool)

    def forward_prefill(
        self,
        x: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of one sequence: the cached prefix, then causal over the chunk.

        The scheduler runs one request per prefill step, and ``cached_seq_len``
        tokens of it are already in the cache -- an earlier chunk, or a prefix
        hit. The prefix is read from the cache; the chunk's own keys come from
        this step's projections, because the ``index_put_`` writing them is not
        visible to a gather in the same graph. Keys are laid out
        ``[cache span (W * block_size) | chunk (C)]``; a cache key is visible iff
        its position is below ``cached_seq_len``.
        """
        cos, sin = position_embeddings
        chunk = x.shape[0]

        prior = metadata["cached_seq_len"].reshape(1, 1).to(torch.long)
        block_table = metadata["block_table_tensor"][:1]
        # Static prefix bound for this graph (``prefill_span_bound``): keys past
        # it are past ``cached_seq_len``, i.e. masked, so dropping them is exact.
        # The plugin's ``prefill_context_length_buckets`` instead trims the
        # table itself, which ``_gather`` already follows.
        nc = self.neuron_config
        bound = prefill_span_bound(
            metadata.get("kv_segment_size", 0),
            getattr(nc, "kv_segment_size_buckets", None) if nc is not None else None,
        )
        if bound is not None:
            blocks = -(-bound // self._block_size())
            if blocks < block_table.shape[1]:
                block_table = block_table[:, :blocks]
        span = block_table.shape[1] * self._block_size()
        if isinstance(selected, tuple) or (
            self.indexer is not None and self.prefill_sparse_supported(chunk, span)
        ):
            return self.forward_prefill_sparse(
                x, position_embeddings, metadata, selected, rank, block_table
            )
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)
        live = torch.ones(1, 1, dtype=torch.bool, device=x.device)
        ctx = self._gather(block_table, live)[0]  # [ctx_len, 704]
        ctx_len = ctx.shape[0]
        in_prefix = torch.arange(ctx_len, device=x.device).view(1, ctx_len) < prior
        ctx = _zero_beyond(ctx, prior.reshape(1, 1))

        keys = torch.cat((ctx, entry.to(ctx.dtype)), dim=0)  # [K, 704]
        causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
        visible = torch.cat((in_prefix.expand(chunk, ctx_len), causal), dim=1)

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)
            selected = self._select_prefill(
                iq, iw, keys[:, self.index_slice], visible, rank
            )

        o_lat = self._attend_blocked(query.transpose(0, 1), keys, selected)
        out = self._finish(o_lat.transpose(0, 1), gate)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    # ── sparse prefill, split by query (HY4_PREFILL_SPARSE_NKI=1) ──────────
    #
    # No torch.cat / torch.stack / torch.gather / topk / sort on any tensor
    # that feeds a kernel here (a concatenation consumed by an NKI kernel
    # failed neuronx-cc with NCC_INIC902 in "concatenate.*_pftranspose"). The
    # prefix keys are read straight from the paged cache, the chunk's own
    # keys from this step's projections, as separate inputs.

    def prefill_sparse_supported(self, chunk: int, span: int) -> bool:
        """The query split needs ``C % world == 0``; the kernels, the span and the
        chunk in whole 128-key tiles."""
        return (
            PREFILL_SPARSE_NKI
            and span >= PREFILL_SPARSE_MIN_SPAN
            and chunk % self.world_size == 0
            and span % 128 == 0
            and chunk % 128 == 0
        )

    def _prefill_ksel(self, n_keys: int) -> int:
        return min(n_keys, -(-max(PREFILL_KSEL, 1) // 128) * 128)

    def _prefill_pos_map(self, block_table: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """Flat cache row of each prefix position ``[span]`` int32; positions at or
        past ``cached_seq_len`` (and table padding) get one past the last row,
        which the kernels' DMA skips (zeros, no read)."""
        bs, num_pages = self._block_size(), self._num_pages()
        bt = block_table[0].to(torch.long)
        blocks = torch.where(
            (bt > 0) & (bt < num_pages), bt, torch.full_like(bt, self._NULL_BLOCK)
        )
        span = bt.shape[0] * bs
        j = torch.arange(span, device=bt.device)
        flat = torch.index_select(blocks, 0, j // bs) * bs + j % bs
        pad = torch.full_like(flat, self._n_cache_rows())
        return torch.where(j < prior.reshape(1), flat, pad).to(torch.int32)

    def _prefill_rows(
        self, sel_ctx: torch.Tensor, sel_own: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selection over ``[prefix (P) | chunk (C)]`` as two bool pieces -> the first
        ``ksel`` selected positions per row, ascending, ``[Q, ksel]`` int32 (pad
        ``P + C``), and their bound (``+BIG`` / ``-BIG``).

        On Neuron, the NKI kernel ``select_to_rows`` (GpSimd nonzero compaction)
        takes the two pieces as separate 0/1 inputs; on CPU the torch reference
        ``_prefill_rows_ref`` runs (its ``[Q, ksel, K / 128]`` intermediates
        must not reach the device graph).
        """
        n_keys = sel_ctx.shape[1] + sel_own.shape[1]
        dev = sel_ctx.device
        ksel = self._prefill_ksel(n_keys)
        if self._select_rows_k is not None and _use_sparse_kernels(sel_ctx):
            rows = self._select_rows_k[self.sparse_grid](
                sel=sel_ctx.to(torch.float32).contiguous(),
                sel2=sel_own.to(torch.float32).contiguous(),
                ksel=ksel,
            )[:, :ksel]
            big = 1.0e30
            # A tensor limit, not a Python int compare (see _zero_beyond).
            filled = rows < torch.full((1, 1), n_keys, dtype=torch.int32, device=dev)
            bound = torch.where(
                filled,
                torch.full(filled.shape, big, dtype=torch.float32, device=dev),
                torch.full(filled.shape, -big, dtype=torch.float32, device=dev),
            )
            return rows, bound
        # CPU only: the concatenation never reaches a device graph.
        return self._prefill_rows_ref(torch.cat((sel_ctx, sel_own), dim=1))

    _TOPK_BIG = 1.0e30  # invisible-key score / bound magnitude of the cte top-k

    def _prefill_topk(
        self,
        sc_ctx: torch.Tensor,
        vis_ctx: torch.Tensor,
        sc_own: torch.Tensor,
        vis_own: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cte backend: the exact top-``k`` positions over ``[prefix | chunk]`` and
        their bound, ``[Q, ksel]`` (``k = min(index_topk, keys)``, ``ksel`` = ``k``
        rounded up to 128; slots past ``k`` get the pad ``P + C`` / ``-BIG``).

        Invisible keys score ``-BIG``; a selected invisible key (a query with
        fewer than ``k`` visible) gets bound ``-BIG``, so the attended set is
        the visible top-k. Unlike the threshold path, keys tied with the k-th
        score are not all kept: exactly ``k`` are. On Neuron the NKI kernel
        ``topk_rows`` (``nisa.topk``) takes the two pieces separately; on CPU
        ``torch.topk`` over their concatenation (never in a device graph).
        """
        big = self._TOPK_BIG
        dev = sc_ctx.device
        n_keys = sc_ctx.shape[1] + sc_own.shape[1]
        k = min(self.index_topk, n_keys)
        width = min(n_keys, -(-k // 128) * 128)
        neg = torch.full((1, 1), -big, dtype=torch.float32, device=dev)
        s_ctx = torch.where(vis_ctx, sc_ctx.float(), neg)
        s_own = torch.where(vis_own, sc_own.float(), neg)
        if self._topk_rows_k is not None and _use_sparse_kernels(sc_ctx):
            return self._topk_rows_k[self.sparse_grid](
                scores_ctx=s_ctx.contiguous(), scores_own=s_own.contiguous(),
                k=k, big=big, width=width,
            )
        # CPU only: the concatenation never reaches a device graph.
        top = torch.topk(torch.cat((s_ctx, s_own), dim=1), k, dim=1)
        rows = top.indices.to(torch.int32)
        bound = torch.where(
            top.values > -0.5 * big,
            torch.full_like(top.values, big),
            torch.full_like(top.values, -big),
        )
        if width > k:
            q = rows.shape[0]
            rows = torch.cat((rows, torch.full((q, width - k), n_keys, dtype=torch.int32)), dim=1)
            bound = torch.cat((bound, torch.full((q, width - k), -big)), dim=1)
        return rows, bound

    def _prefill_rows_ref(self, local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Torch reference of ``_prefill_rows`` (CPU).

        Unfilled slots get ``K`` (out of range: the kernel's DMA skips it). The
        ``j``-th selected key is the number of keys whose inclusive selected
        count is ``<= j``. Counted per 128-key block: whole blocks before the one
        holding it count 128 each, and inside that block (picked by a one-hot
        matmul over the blocks' within-block prefix counts) the keys whose count
        is ``<= j`` minus the block's offset. All counts are integers in fp32
        (exact), all prefix sums are triangular matmuls.
        """
        q, n_keys = local.shape
        dev = local.device
        nb = n_keys // 128
        ksel = self._prefill_ksel(n_keys)
        sel = local.to(torch.float32).view(q, nb, 128)
        tri = torch.ones(128, 128, device=dev).triu()  # [i, j] = i <= j
        within = sel @ tri  # [Q, nb, 128] inclusive count inside the block
        cnt = within[..., 127]  # [Q, nb]
        strict = torch.ones(nb, nb, device=dev).triu(diagonal=1)  # [i, j] = i < j
        excl = cnt @ strict  # [Q, nb] selected keys before the block
        incl = excl + cnt
        slots = torch.arange(ksel, dtype=torch.float32, device=dev).view(1, ksel, 1)
        before = (incl.unsqueeze(1) <= slots).to(torch.float32)  # [Q, ksel, nb]
        holds = (excl.unsqueeze(1) <= slots).to(torch.float32) - before  # boundary block
        off = (holds * excl.unsqueeze(1)).sum(dim=-1)  # [Q, ksel]
        w_b = holds @ within  # [Q, ksel, 128]
        target = slots.view(1, ksel) - off
        partial = (w_b <= target.unsqueeze(-1)).to(torch.float32).sum(dim=-1)
        idx = before.sum(dim=-1) * 128.0 + partial  # [Q, ksel]
        total = cnt.sum(dim=-1, keepdim=True)  # [Q, 1]
        filled = slots.view(1, ksel) < total
        pad = torch.full_like(idx, float(n_keys))
        rows = torch.where(filled, idx, pad).to(torch.int32)
        big = 1.0e30
        bound = torch.where(
            filled,
            torch.full(filled.shape, big, dtype=torch.float32, device=dev),
            torch.full(filled.shape, -big, dtype=torch.float32, device=dev),
        )
        return rows, bound

    def forward_prefill_sparse(
        self,
        x: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: tuple | None,
        rank: torch.Tensor | None,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        """``forward_prefill`` split by query; same keys, same selection.

        Positions ``[0, P)`` are the cache span (``P = W * block_size``) and
        ``[P, P + C)`` the chunk. Rank ``r`` owns queries
        ``[r * Q, (r + 1) * Q)``, ``Q = C / world``:

        1. full layers: indexer keys of the span read from the cache
           (``gather_rows_cols``, prefix rows only); scores against the span and
           against the chunk's own keys kept apart; the top-k threshold over
           both (``topk_threshold2``, bit-identical to ``select_topk``); the
           selection as row lists (``_prefill_rows``). Shared layers reuse
           ``(rows, bound, sources)``;
        2. all-gather the absorbed query parts (latent, rope) over heads;
        3. ``sparse_decode_attn`` in position mode: prefix rows straight from
           the paged cache, chunk rows from this step's latent / rope keys;
           the sink is folded in once per head;
        4. all-gather the bf16 ``[Q, heads, 512]``, take this rank's heads.
        """
        cos, sin = position_embeddings
        chunk = x.shape[0]
        world, nh = self.world_size, self.num_heads
        per_rank = chunk // world
        dev = x.device
        x = x.to(self.dtype)
        q_lat, q_rope, index_key, lat_own, k_rope, gate, q_resid = self._project_parts(
            x, cos, sin, nocat=True
        )
        # The cache write alone takes the concatenated entry (index_put_, no kernel).
        entry = torch.cat((index_key.to(lat_own.dtype), lat_own, k_rope.to(lat_own.dtype)), dim=-1)
        self._write_kv(entry, metadata)

        if rank is None:
            rank = torch.tensor(self.tp_group.rank_in_group, device=dev)
        r = rank.reshape(1).to(torch.long)
        mine = r * per_rank + torch.arange(per_rank, device=dev)  # this rank's queries
        prior = metadata["cached_seq_len"].reshape(1, 1).to(torch.long)
        pos_map = self._prefill_pos_map(block_table, prior)  # [P]
        span = pos_map.shape[0]

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)
            local_q = torch.index_select(iq, 0, mine)
            local_w = torch.index_select(iw, 0, mine)
            ik_ctx = self._gather_cols(pos_map.view(1, span), self.index_slice.start, self.index_dim)[0]
            rows_q, heads, dim = local_q.shape
            flat_q = local_q.reshape(rows_q * heads, dim)
            s_ctx = torch.relu((flat_q @ ik_ctx.t()).float()).view(rows_q, heads, span)
            s_own = torch.relu((flat_q @ index_key.to(ik_ctx.dtype).t()).float()).view(rows_q, heads, chunk)
            sc_ctx = (s_ctx * local_w.unsqueeze(-1)).sum(dim=1)  # [Q, P]
            sc_own = (s_own * local_w.unsqueeze(-1)).sum(dim=1)  # [Q, C]
            vis_ctx = (torch.arange(span, device=dev).view(1, span) < prior).expand(rows_q, span)
            vis_own = torch.arange(chunk, device=dev).view(1, chunk) <= mine.view(rows_q, 1)
            if "collectives" in PERF_SKIP:
                sources = torch.arange(world, dtype=torch.float32, device=dev)
            else:
                sources = self.tp_group.all_gather(r.to(torch.float32), dim=0)
            # nisa.topk ranks fewer than 65536 keys (the threshold path otherwise).
            if self.prefill_backend == "cte" and span + chunk < 65536:
                rows, bound = self._prefill_topk(sc_ctx, vis_ctx, sc_own, vis_own)
            else:
                lo = topk_threshold2(sc_ctx, vis_ctx, sc_own, vis_own, self.index_topk)
                sel_ctx = vis_ctx & (sc_ctx >= lo)
                sel_own = vis_own & (sc_own >= lo)
                _capture(self.layer_idx, "selected_local", sel_own.to(torch.bfloat16))
                rows, bound = self._prefill_rows(sel_ctx, sel_own)
            selected = (rows, bound, sources)
        rows, bound, sources = selected

        ql_all = _all_gather(self.tp_group, q_lat, 1, world)  # [C, heads, 512]
        qr_all = _all_gather(self.tp_group, q_rope, 1, world)  # [C, heads, 64]
        ql_loc = torch.index_select(ql_all, 0, mine).float()
        qr_loc = torch.index_select(qr_all, 0, mine).float()
        sinks = _all_gather(self.tp_group, self.sinks.float(), 0, world)  # [heads]
        own_lat = lat_own.to(self.kv_pages.dtype)
        own_rope = k_rope.to(self.kv_pages.dtype)
        col_off = self.latent_slice.start
        if self.prefill_backend == "cte":
            m, den, num = self._prefill_attn_cte(
                ql_loc, qr_loc, pos_map, own_lat, own_rope, rows, bound, col_off
            )
        elif self._sparse_attn_k is not None and _use_sparse_kernels(ql_loc):
            # Tokens per group bounded so the gathered rows (bf16, 576 wide)
            # stay a fraction of a partition's SBUF.
            # Position tables over [prefix (P) | chunk (C)] for the kernel, built
            # without concatenation: cache row for p < P (else out of range), own
            # row p - P for p >= P (else out of range). No integer mod on device.
            n_own = own_lat.shape[0]
            pos = torch.arange(span + n_own, device=pos_map.device)
            in_ctx = pos < span
            pos_full = torch.where(
                in_ctx,
                torch.index_select(pos_map, 0, torch.clamp(pos, max=span - 1)),
                torch.full_like(pos, self._n_cache_rows(), dtype=pos_map.dtype),
            )
            own_map = torch.where(in_ctx, torch.full_like(pos, n_own), pos - span).to(torch.int32)
            kt = -(-rows.shape[1] // 128)
            tg = max(1, min(8, 49152 // (kt * self.latent_dim * 2)))
            m, den, num = self._sparse_attn_k[self.sparse_grid](
                q_all=ql_loc.contiguous(), kv_cache=self.kv_pages,
                rows=rows.contiguous(), bound=bound.contiguous(),
                scale=float(self.scaling), col_off=col_off, lat_off=0,
                lat_dim=self.kv_lora_rank, rope_off=self.kv_lora_rank,
                rope_dim=self.rope_dim, tg=tg,
                pos_map=pos_full.view(span + n_own, 1).contiguous(),
                own_lat=own_lat.contiguous(), own_rope=own_rope.contiguous(),
                q_rope=qr_loc.contiguous(),
                own_map=own_map.view(span + n_own, 1).contiguous(),
            )
        else:
            m, den, num = sparse_prefill_ref(
                ql_loc, qr_loc, self.kv_pages, pos_map, own_lat, own_rope, rows, bound,
                self.scaling, col_off, self.dtype,
            )
        sink = sinks.view(1, -1)
        m_f = torch.maximum(m, sink)
        scale = torch.exp(m - m_f)
        o_loc = num * scale.unsqueeze(-1) / (
            den * scale + torch.exp(sink - m_f)
        ).unsqueeze(-1)  # [Q, heads, 512]

        rank_v = self.kv_lora_rank
        o_all = _all_gather(self.tp_group, o_loc.to(self.dtype), 0, world)  # [C, heads, 512]
        pick = self._pick_from(sources, r, dev).to(self.dtype)  # [nh, heads]
        o_mine = torch.einsum("jh,qhc->qjc", pick, o_all)  # blocks in gather order
        targets = torch.arange(world, dtype=torch.float32, device=dev)
        perm = (targets.view(world, 1) == sources.view(1, world)).to(self.dtype)
        ordered = perm @ o_mine.reshape(world, per_rank * nh * rank_v)
        out = self._finish(ordered.reshape(chunk, nh, rank_v), gate)
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    def _prefill_attn_cte(
        self,
        ql_loc: torch.Tensor,
        qr_loc: torch.Tensor,
        pos_map: torch.Tensor,
        own_lat: torch.Tensor,
        own_rope: torch.Tensor,
        rows: torch.Tensor,
        bound: torch.Tensor,
        col_off: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse prefill attention, ``cte`` backend: per-slot flat rows built here.

        Lookup tables over positions ``[0, P + C]`` (the pad ``P + C`` included),
        built with ``arange`` / ``where`` (no concatenation, no integer ``mod``),
        then one ``index_select`` per table by the slot positions ``rows``:
        ``rows_cache`` (prefix -> flat cache row), ``rows_own`` (chunk -> own row);
        a slot's other source (and both, for the pad) is out of range, which
        the kernel's DMA skips.
        """
        span, n_own = pos_map.shape[0], own_lat.shape[0]
        n_rows = self._n_cache_rows()
        q, ksel = rows.shape
        dev = pos_map.device
        pos = torch.arange(span + n_own + 1, device=dev)
        in_ctx = pos < span
        in_own = (pos >= span) & (pos < span + n_own)
        cache_row = torch.where(
            in_ctx,
            torch.index_select(pos_map, 0, torch.clamp(pos, max=span - 1)),
            torch.full_like(pos, n_rows, dtype=pos_map.dtype),
        )
        own_t = torch.where(in_own, pos - span, torch.full_like(pos, n_own))
        flat = rows.reshape(q * ksel).to(torch.long)
        rows_cache = torch.index_select(cache_row.to(torch.int32), 0, flat).view(q, ksel)
        rows_own = torch.index_select(own_t.to(torch.int32), 0, flat).view(q, ksel)
        if self._prefill_attn_k is not None and _use_sparse_kernels(ql_loc):
            # bf16 query (the cache dtype): the score matmuls run at the bf16
            # rate, accumulating in fp32 PSUM; the softmax stays fp32.
            qdt = self.kv_pages.dtype
            return self._prefill_attn_k[self.sparse_grid](
                q_lat=ql_loc.to(qdt).contiguous(), q_rope=qr_loc.to(qdt).contiguous(),
                kv_cache=self.kv_pages,
                rows_cache=rows_cache.contiguous(), own_lat=own_lat.contiguous(),
                own_rope=own_rope.contiguous(), rows_own=rows_own.contiguous(),
                bound=bound.contiguous(), scale=float(self.scaling), col_off=col_off,
                dma_t=self.prefill_cte_dma_t,
            )
        qdt = self.kv_pages.dtype
        return sparse_prefill_cte_ref(
            ql_loc.to(qdt), qr_loc.to(qdt), self.kv_pages, rows_cache, own_lat, own_rope, rows_own, bound,
            self.scaling, col_off, self.dtype,
        )

    def forward_decode(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One new token per request against its paged history.

        Keys are ``[cache span | the token itself]``: the current token's entry is
        appended, since the ``index_put_`` above is not visible to the gather.
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        ctx = self._gather(block_table, live_rows)  # [reqs, ctx_len, 704]
        ctx_len = ctx.shape[1]
        key_positions = torch.arange(ctx_len, device=x.device).view(1, ctx_len)
        in_prefix = key_positions < positions.view(num_reqs, 1)  # [reqs, ctx_len]
        ctx = _zero_beyond(ctx, positions.view(num_reqs, 1))

        keys = torch.cat(
            (ctx, entry.to(ctx.dtype).unsqueeze(1)), dim=1
        )  # [reqs, K, 704]
        visible = torch.cat(
            (in_prefix, torch.ones(num_reqs, 1, dtype=torch.bool, device=x.device)),
            dim=1,
        )

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)  # [reqs, 32, 128]
            ik = keys[..., self.index_slice]  # [reqs, K, 128]
            s = torch.relu((iq @ ik.transpose(1, 2)).float())  # [reqs, 32, K]
            index_scores = (s * iw.unsqueeze(-1)).sum(dim=1)
            selected = select_topk(index_scores, visible, self.index_topk)

        lat = keys[..., self.latent_slice]  # [reqs, K, 576]
        q = query.float()  # [reqs, nh, 576]
        scores = (q @ lat.float().transpose(1, 2)) * self.scaling  # [reqs, nh, K]
        scores = scores.masked_fill(~selected.unsqueeze(1), float("-inf"))
        # Heads first for the sink broadcast, then back.
        probs = self._with_sinks(scores.transpose(0, 1)).transpose(0, 1)
        o_lat = probs.to(self.dtype) @ lat[..., : self.kv_lora_rank]  # [reqs, nh, 512]
        out = self._finish(o_lat, gate)
        _capture(self.layer_idx, "attn_in", x)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    def forward_decode_split(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward_decode`` with the token's own scores computed apart.

        Same result; it never concatenates the ``[reqs, span, 704]`` keys and does
        the indexer's head-weighted sum as a matmul. Which form neuronx-cc 2.27
        lowers better depends on the batch, measured per full-indexer layer at a
        4K span (attention only, one core): 1 row 1.7 vs 22.5 ms, 16 rows 9.3 vs
        16.8 ms -- but 2/4/8 rows 3.9/6.1/10.5 vs 3.0/2.7/3.8 ms. At 1 row the
        original made the batch-1 graph 3x slower per token than the 4-row one.
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs = block_table.shape[0]
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        ctx = self._gather(block_table, live_rows)  # [reqs, ctx_len, 704]
        ctx_len = ctx.shape[1]
        key_positions = torch.arange(ctx_len, device=x.device).view(1, ctx_len)
        in_prefix = key_positions < positions.view(num_reqs, 1)  # [reqs, ctx_len]
        ctx = _zero_beyond(ctx, positions.view(num_reqs, 1))
        own = entry.to(ctx.dtype).unsqueeze(1)  # [reqs, 1, 704]
        visible = torch.cat(
            (in_prefix, torch.ones(num_reqs, 1, dtype=torch.bool, device=x.device)),
            dim=1,
        )
        # Scores against the cache and against the token itself are computed
        # separately and concatenated, rather than concatenating the
        # [reqs, 32K, 704] keys (a full copy of the context per layer).

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)  # [reqs, 32, 128]
            s = torch.cat(
                (
                    iq @ ctx[..., self.index_slice].transpose(1, 2),
                    iq @ own[..., self.index_slice].transpose(1, 2),
                ),
                dim=-1,
            )
            s = torch.relu(s.float())  # [reqs, 32, K]
            index_scores = (iw.unsqueeze(1) @ s).squeeze(1)  # [reqs, K]
            selected = select_topk(index_scores, visible, self.index_topk)

        lat = ctx[..., self.latent_slice]  # [reqs, ctx_len, 576]
        own_lat = own[..., self.latent_slice]  # [reqs, 1, 576]
        q = query.float()  # [reqs, nh, 576]
        scores = torch.cat(
            (q @ lat.float().transpose(1, 2), q @ own_lat.float().transpose(1, 2)),
            dim=-1,
        ) * self.scaling  # [reqs, nh, K]
        scores = scores.masked_fill(~selected.unsqueeze(1), float("-inf"))
        # Heads first for the sink broadcast, then back.
        probs = self._with_sinks(scores.transpose(0, 1)).transpose(0, 1).to(self.dtype)
        rank_v = self.kv_lora_rank
        o_lat = (
            probs[..., :ctx_len] @ lat[..., :rank_v]
            + probs[..., ctx_len:] @ own_lat[..., :rank_v]
        )  # [reqs, nh, 512]
        out = self._finish(o_lat, gate)
        _capture(self.layer_idx, "attn_in", x)
        _capture(self.layer_idx, "selected", selected.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected

    # ── decode, sharded by key (HY4_DEC_CP=1) ────────────────────────────
    #
    # No torch.cat / torch.stack anywhere in this path (an earlier version
    # failed neuronx-cc with NCC_IFML902 "FlattenMacroLoop ... Cannot remove an
    # edge" on an aten cat inside it): the current token's key is handled as
    # separate terms, and each collective carries one unpadded tensor.

    def cp_supported(self, width: int) -> bool:
        """Key-sharded decode needs the block table to split evenly over ranks."""
        return self.world_size > 1 and width % self.world_size == 0

    def _cp_pick(self, r: torch.Tensor, dev) -> torch.Tensor:
        """``[nh, world * nh]`` 0/1 matrix taking this rank's heads out of a
        head axis in all-gather block order (not tp-rank order on Trn2; the
        ranks are gathered too, as in ``_select_prefill``)."""
        world = self.world_size
        if "collectives" in PERF_SKIP:
            sources = torch.arange(world, dtype=torch.float32, device=dev)
        else:
            sources = self.tp_group.all_gather(r.reshape(1).to(torch.float32), dim=0)
        return self._pick_from(sources, r, dev)

    def _pick_from(self, sources: torch.Tensor, r: torch.Tensor, dev) -> torch.Tensor:
        """``_cp_pick`` from already gathered ranks ``sources [world]`` (fp32)."""
        world, nh = self.world_size, self.num_heads
        at_me = (sources.view(world) == r.reshape(1).to(torch.float32)).to(torch.float32)
        eye = torch.eye(nh, dtype=torch.float32, device=dev)
        return (eye.view(nh, 1, nh) * at_me.view(1, world, 1)).reshape(nh, world * nh)

    # ── sparse local attention (HY4_DEC_SPARSE_NKI=1) ──────────────────────

    def _n_cache_rows(self) -> int:
        return self._num_pages() * self._block_size()

    def _pad_row(self) -> int:
        """Row index for keys the sparse path does not attend (``HY4_SPARSE_PAD_ROW``)."""
        if SPARSE_PAD_ROW == "null":
            return self._NULL_BLOCK * self._block_size()
        return self._n_cache_rows()

    def _cp_shard_rows(
        self, local_bt: torch.Tensor, live_rows: torch.Tensor, in_prefix: torch.Tensor
    ) -> torch.Tensor:
        """Flat cache row (``page * block + slot``) of each shard key, ``[reqs, shard]`` int32.

        Dead rows and table padding map to the null block, as in ``_gather``; keys
        past the prefix get ``_pad_row()`` (by default one past the last row, which
        the kernels' indirect DMA skips: zeros, no read). Their indexer scores are
        masked by ``in_prefix`` either way.
        """
        bs, num_pages = self._block_size(), self._num_pages()
        bt = local_bt.to(torch.long)
        blocks = torch.where(
            live_rows & (bt > 0) & (bt < num_pages), bt, torch.full_like(bt, self._NULL_BLOCK)
        )
        shard = bt.shape[1] * bs
        j = torch.arange(shard, device=bt.device)
        page = torch.index_select(blocks, 1, j // bs)  # [reqs, shard]
        flat = page * bs + (j % bs).view(1, shard)
        pad = torch.full_like(flat, self._pad_row())
        return torch.where(in_prefix, flat, pad).to(torch.int32)

    def _gather_cols(self, rows: torch.Tensor, col_off: int, width: int) -> torch.Tensor:
        """``[R, n]`` flat rows -> ``[R, n, width]`` columns ``col_off:`` of each entry."""
        if self._sparse_gather_k is not None and _use_sparse_kernels(rows):
            gather = self._sparse_gather_k
            return gather[self.sparse_grid](kv_cache=self.kv_pages, rows=rows, col_off=col_off, width=width)
        return gather_rows_cols_ref(self.kv_pages, rows, col_off, width)

    def _cp_sparse_rows(
        self, shard_rows: torch.Tensor, sel_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Selected rows and their keep/drop bound, ``[reqs, K]`` each.

        Unselected keys get ``_pad_row()`` and ``-BIG``. With
        ``HY4_SPARSE_KLOC`` below the shard, the selected keys are compacted to
        the front first: rank among the selected by a prefix-sum matmul, then a
        one-hot placement (no sort, top-k, gather or scatter). Exact while no
        rank holds more than K selected keys.
        """
        reqs, shard = sel_ctx.shape
        dev = sel_ctx.device
        n_rows = self._n_cache_rows()
        pad_row = self._pad_row()
        big = 1.0e30
        k = shard
        if 0 < SPARSE_KLOC < shard:
            k = min(shard, -(-SPARSE_KLOC // 128) * 128)
        if k == shard:
            rows = torch.where(sel_ctx, shard_rows, torch.full_like(shard_rows, pad_row))
        else:
            if n_rows >= 1 << 24:
                raise ValueError(f"{n_rows} cache rows: fp32 one-hot compaction is inexact")
            sel = sel_ctx.to(torch.float32)
            upper = torch.ones(shard, shard, device=dev).triu()  # [i, j] = i <= j
            pos = sel @ upper - 1.0  # [reqs, shard], rank among the selected
            slots = torch.arange(k, dtype=torch.float32, device=dev)
            onehot = (pos.unsqueeze(-1) == slots.view(1, 1, k)).to(torch.float32) * sel.unsqueeze(-1)
            filled = onehot.sum(dim=1) > 0.5  # [reqs, k]
            placed = (onehot * shard_rows.to(torch.float32).unsqueeze(-1)).sum(dim=1)
            pad = torch.full_like(placed, pad_row).to(torch.int32)
            rows = torch.where(filled, placed.to(torch.int32), pad)
            sel_ctx = filled
        bound = torch.where(
            sel_ctx,
            torch.full(sel_ctx.shape, big, dtype=torch.float32, device=dev),
            torch.full(sel_ctx.shape, -big, dtype=torch.float32, device=dev),
        )
        return rows.to(torch.int32), bound

    def _cp_sparse_local(
        self, q_all: torch.Tensor, shard_rows: torch.Tensor, sel_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unnormalised local attention partials over the selected rows only.

        -> ``m_loc [reqs, heads]`` (max over the selected keys, ``-BIG`` if none),
        ``den [reqs, heads]`` and ``num [reqs, heads, 512]`` relative to ``m_loc``.
        Reads the 576-wide ``[latent | rope]`` slice of the selected entries.
        """
        rows, bound = self._cp_sparse_rows(shard_rows, sel_ctx)
        col_off = self.latent_slice.start
        if self._sparse_attn_k is not None and _use_sparse_kernels(q_all):
            attn = self._sparse_attn_k
            return attn[self.sparse_grid](
                q_all=q_all.contiguous(), kv_cache=self.kv_pages, rows=rows.contiguous(),
                bound=bound.contiguous(), scale=float(self.scaling), col_off=col_off,
                lat_off=0, lat_dim=self.kv_lora_rank, rope_off=self.kv_lora_rank,
                rope_dim=self.rope_dim,
            )
        return sparse_decode_ref(
            q_all, self.kv_pages, rows, bound, self.scaling, col_off, self.kv_lora_rank, self.dtype
        )

    def forward_decode_cp(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple,
        metadata: dict,
        selected: tuple | None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Decode attention sharded by KEY across the TP ranks; same result as
        ``forward_decode_split``.

        The cache is still replicated, but rank ``r`` gathers only block-table
        columns ``[r * nb, (r + 1) * nb)`` (``nb = width / world``), i.e. key
        positions ``[r * n, (r + 1) * n)``, and handles them for ALL heads.
        The current token's own key is not in the gather; its query, entry
        and therefore its scores are identical on every rank, so it is added
        as a separate term (once, after the merge), never concatenated.

        1. all-gather the absorbed queries ``[reqs, nh, 576]`` -> all heads;
        2. full-indexer layers: all-gather the local indexer scores
           ``[reqs, n]`` (invisible at ``-_BIG``); the global top-k threshold
           is ``select_topk``'s bisection over them plus the own-token score
           as an always-visible extra term. It depends only on the multiset of
           scores, so the gather order does not matter, and it is bit-identical
           on all ranks. -> local mask ``[reqs, n]`` and own-token bit
           ``[reqs, 1]``. Also gathers the ranks for the head pick (step 5).
           Shared layers reuse all three (``selected`` is a tuple here);
        3. local scores ``[reqs, heads, n]``; all-gather the per-head local
           maxima; ``M`` = max over ranks and the selected own-token score;
        4. all-reduce the unnormalised numerators ``[reqs, heads, 512]`` and
           denominators ``[reqs, heads]``, then add the own-token term;
        5. take this rank's heads (``_cp_pick``), fold the sink in once
           (rescaling to ``max(M, sink)``), then ``w_uv`` / gate / ``o_proj``.
        """
        cos, sin = position_embeddings
        block_table = metadata["block_table_tensor"]
        num_reqs, width = block_table.shape
        world, nh = self.world_size, self.num_heads
        dev = x.device
        x = x.to(self.dtype)
        query, entry, gate, q_resid = self._project(x, cos, sin)
        self._write_kv(entry, metadata)

        if rank is None:
            rank = torch.tensor(self.tp_group.rank_in_group, device=dev)
        r = rank.reshape(1).to(torch.long)
        nb = width // world
        shard = nb * self._block_size()
        cols = r * nb + torch.arange(nb, device=dev)
        local_bt = torch.index_select(block_table, 1, cols)
        live_rows = (metadata["slot_mapping"].view(num_reqs, -1)[:, 0] > 0).view(-1, 1)
        key_pos = r * shard + torch.arange(shard, device=dev)  # [shard]
        limit = positions.view(num_reqs, 1)
        in_prefix = key_pos.view(1, shard) < limit  # [reqs, shard]
        # The kernels take <= 128 keys or a multiple of 128.
        sparse = DEC_SPARSE_NKI and (shard <= 128 or shard % 128 == 0)
        if sparse:
            # Only the indexer keys of the shard; latent rows are read after the
            # selection (_cp_sparse_local). Keys past the prefix read nothing.
            shard_rows = self._cp_shard_rows(local_bt, live_rows, in_prefix)
            ik_ctx = self._gather_cols(shard_rows, self.index_slice.start, self.index_dim)
            ctx = None
        else:
            ctx = self._gather(local_bt, live_rows)  # [reqs, shard, 704]
            ctx = _zero_beyond_at(ctx, limit, key_pos)
            ik_ctx = ctx[..., self.index_slice]
        own = entry.to(self.dtype)  # [reqs, 704], the same on every rank

        if self.indexer is not None:
            iq, iw = self.indexer.queries(q_resid, x, cos, sin)  # [reqs, 32, 128]
            s_ctx = torch.relu((iq @ ik_ctx.transpose(1, 2)).float())
            idx_ctx = (iw.unsqueeze(1) @ s_ctx).squeeze(1)  # [reqs, shard]
            s_own = torch.relu(
                torch.einsum("rhd,rd->rh", iq, own[:, self.index_slice]).float()
            )
            idx_own = (iw * s_own).sum(dim=-1, keepdim=True)  # [reqs, 1]
            masked = idx_ctx.masked_fill(~in_prefix, -_BIG)
            every = _all_gather(self.tp_group, masked, 1, world)  # [reqs, world*shard]
            half = torch.full((1, 1), -0.5 * _BIG, dtype=torch.float32, device=dev)
            lo = topk_threshold(every, every > half, self.index_topk, idx_own)
            selected = (in_prefix & (idx_ctx >= lo), idx_own >= lo, self._cp_pick(r, dev))
        sel_ctx, sel_own, pick = selected

        # 1. every head's absorbed query (order: all-gather blocks).
        q_all = _all_gather(self.tp_group, query.float(), 1, world)  # [reqs, heads, 576]

        # 3. local scores, global max.
        own_lat = own[:, self.latent_slice]  # [reqs, 576]
        # A finite floor (see _attend_blocked): a rank may have no selected key.
        floor = -1.0e30
        rank_v = self.kv_lora_rank
        if sparse:
            # Local partials relative to the local max, over the selected rows only.
            m_loc, den_l, num_l = self._cp_sparse_local(q_all, shard_rows, sel_ctx)
        else:
            lat = ctx[..., self.latent_slice]  # [reqs, shard, 576]
            scores = (q_all @ lat.float().transpose(1, 2)) * self.scaling  # [reqs, heads, shard]
            scores = scores.masked_fill(~sel_ctx.unsqueeze(1), floor)
            m_loc = scores.amax(dim=-1)  # [reqs, heads]
        a = torch.einsum("rhc,rc->rh", q_all, own_lat.float()) * self.scaling
        a = a.masked_fill(~sel_own, floor)  # [reqs, heads]
        m_all = _all_gather(self.tp_group, m_loc.unsqueeze(0), 0, world).amax(dim=0)
        m = torch.maximum(m_all, a)

        # 4. unnormalised local partials, merged across ranks.
        if sparse:
            corr = torch.exp(m_loc - m)  # 0 on a rank with no selected key
            den = den_l * corr
            num = num_l * corr.unsqueeze(-1)
        else:
            p = torch.exp(scores - m.unsqueeze(-1))
            den = p.sum(dim=-1)  # [reqs, heads]
            # Contract the whole 576-wide row and slice the small output, not the
            # gathered latent (a strided view of it spills).
            num = (p.to(self.dtype) @ lat).float()[..., :rank_v]  # [reqs, heads, 512]
        pa = torch.exp(a - m)  # [reqs, heads]; 0 when the own key is unselected
        own_v = own_lat[:, :rank_v].float()  # [reqs, 512]
        m_mine = torch.einsum("jh,rh->rj", pick, m)  # [reqs, nh]
        pa_mine = torch.einsum("jh,rh->rj", pick, pa)
        num_all = _all_reduce(self.tp_group, num)
        den_all = _all_reduce(self.tp_group, den)
        num_mine = torch.einsum("jh,rhc->rjc", pick, num_all)  # [reqs, nh, 512]
        den_mine = torch.einsum("jh,rh->rj", pick, den_all)  # [reqs, nh]
        num_mine = num_mine + pa_mine.unsqueeze(-1) * own_v.unsqueeze(1)
        den_mine = den_mine + pa_mine

        # 5. the sink joins once, as in _attend_blocked.
        sink = self.sinks.float().view(1, nh)
        m_f = torch.maximum(m_mine, sink)
        scale = torch.exp(m_mine - m_f)
        o_lat = num_mine * scale.unsqueeze(-1) / (
            den_mine * scale + torch.exp(sink - m_f)
        ).unsqueeze(-1)  # [reqs, nh, 512]
        out = self._finish(o_lat, gate)
        _capture(self.layer_idx, "attn_in", x)
        _capture(self.layer_idx, "selected", sel_ctx.to(torch.bfloat16))
        _capture(self.layer_idx, "attn_out", out)
        return out, selected


# ── Decoder ──────────────────────────────────────────────────────────────


class HYV4DecoderLayer(nn.Module):
    def __init__(self, config: HYV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        dtype = config.torch_dtype
        self.input_layernorm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.post_attention_layernorm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype
        )
        self.attn_hc = HYV4HCPre(config)
        self.ffn_hc = HYV4HCPre(config)
        self.self_attn = HYV4Attention(config, layer_idx)
        self.mixer_name = self.self_attn.layer_name
        self.is_moe = config.is_moe_layer(layer_idx)
        self.mlp = (
            HYV4MoE(config)
            if self.is_moe
            else HYV4MLP(
                config.hidden_size, config.intermediate_size, self.world_size, dtype
            )
        )

    def forward(
        self,
        streams: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        selected: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (streams, DSA key selection for the layers that share it)."""
        residual = streams
        x, post = self.attn_hc(streams)
        if "attn" in PERF_SKIP:
            x = torch.zeros_like(x)
        else:
            x, selected = self.self_attn(
                self.input_layernorm(x),
                positions,
                position_embeddings,
                attn_metadata,
                selected,
                rank,
            )
        streams = hc_post(x, residual, post)

        residual = streams
        x, post = self.ffn_hc(streams)
        x = self.post_attention_layernorm(x)
        _capture(self.layer_idx, "mlp_in", x)
        x = torch.zeros_like(x) if "moe" in PERF_SKIP else self.mlp(x)
        if self.world_size > 1:
            x = _all_reduce(self.tp_group, x)
        _capture(self.layer_idx, "mlp_out", x)
        return hc_post(x, residual, post), selected


class HYV4TextModel(nn.Module):
    def __init__(self, config: HYV4Config):
        super().__init__()
        target = resolve_platform_defaults()
        logger.info(
            "hy4_preview attention paths (platform %r): HY4_DEC_CP=%d "
            "HY4_DEC_SPARSE_NKI=%d HY4_PREFILL_SPARSE_NKI=%d HY4_PREFILL_SPARSE_BACKEND=%s",
            target, DEC_CP, DEC_SPARSE_NKI, PREFILL_SPARSE_NKI, PREFILL_SPARSE_BACKEND,
        )
        self.config = config
        self.tp_group = get_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
            ),
        )
        self.layers = nn.ModuleList(
            HYV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.hc_head = HYV4HCHead(config)
        self.norm = HYV4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = HYV4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        streams = x.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()
        position_embeddings = self.rotary_emb(positions)
        selected = None
        for layer in self.layers:
            streams, selected = layer(
                streams, positions, position_embeddings, attn_metadata, selected, rank
            )
        return self.norm(self.hc_head(streams))


def checkpoint_mappings(config: HYV4Config) -> dict[str, str]:
    """Port parameter name -> converted-checkpoint key, for every loaded parameter."""
    m: dict[str, str] = {}
    mx_experts = (config.neuron_quantization or {}).get("format") == FORMAT_MXFP8
    for i in range(config.num_hidden_layers):
        hf, ours = f"{HF_PREFIX}.layers.{i}", f"model.layers.{i}"
        m[f"{ours}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
        m[f"{ours}.post_attention_layernorm.weight"] = (
            f"{hf}.post_attention_layernorm.weight"
        )
        for mine, theirs in (("attn_hc", "hc_attn_layer"), ("ffn_hc", "hc_mlp_layer")):
            for p in ("fn", "base", "scale"):
                m[f"{ours}.{mine}.{p}"] = f"{hf}.{theirs}.hc_pre.hc_{p}"

        a, ha = f"{ours}.self_attn", f"{hf}.self_attn"
        m[f"{a}.q_a_proj_weight"] = f"{ha}.q_a_proj.weight"
        m[f"{a}.q_a_layernorm.weight"] = f"{ha}.q_a_layernorm.weight"
        m[f"{a}.q_b_proj_weight"] = f"{ha}.q_b_proj.weight"
        m[f"{a}.kv_a_proj_weight"] = f"{ha}.kv_a_proj_with_mqa.weight"
        m[f"{a}.kv_a_layernorm.weight"] = f"{ha}.kv_a_layernorm.weight"
        m[f"{a}.w_uk"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.w_uv"] = f"{ha}.kv_b_proj.weight"
        m[f"{a}.gate_proj_weight"] = f"{ha}.linear_gate.weight"
        m[f"{a}.o_proj_weight"] = f"{ha}.o_proj.weight"
        m[f"{a}.sinks"] = f"{ha}.learnable_sink_param"
        if config.indexer_types[i] == "full":
            ia, hi = f"{a}.indexer", f"{ha}.indexer"
            m[f"{ia}.wq_b_weight"] = f"{hi}.wq_b.weight"
            m[f"{ia}.wk_weight"] = f"{hi}.wk.weight"
            m[f"{ia}.k_norm_weight"] = f"{hi}.k_norm.weight"
            m[f"{ia}.k_norm_bias"] = f"{hi}.k_norm.bias"
            m[f"{ia}.weights_proj_weight"] = f"{hi}.weights_proj.weight"

        mlp, hm = f"{ours}.mlp", f"{hf}.mlp"
        if config.is_moe_layer(i):
            m[f"{mlp}.router_weight"] = f"{hm}.gate.weight"
            m[f"{mlp}.e_score_correction_bias"] = f"{hm}.gate.e_score_correction_bias"
            m[f"{mlp}.experts_gate_up_weight"] = f"{hm}.experts.gate_up_proj"
            m[f"{mlp}.experts_gate_up_scale"] = f"{hm}.experts.gate_up_proj_scale"
            # fp8-row checkpoints store the down weight transposed; the MXFP8
            # expert checkpoint keeps the published [E, H, I] ``down_proj``.
            m[f"{mlp}.experts_down_weight"] = (
                f"{hm}.experts.down_proj" if mx_experts else f"{hm}.experts.down_proj_t"
            )
            m[f"{mlp}.experts_down_scale"] = f"{hm}.experts.down_proj_scale"
            shared, hs = f"{mlp}.shared_experts", f"{hm}.shared_experts"
        else:
            shared, hs = mlp, hm
        for p in ("gate_proj", "up_proj", "down_proj"):
            m[f"{shared}.{p}_weight"] = f"{hs}.{p}.weight"

    m["model.embed_tokens.weight"] = f"{HF_PREFIX}.embed_tokens.weight"
    m["model.norm.weight"] = f"{HF_PREFIX}.norm.weight"
    for p in ("fn", "base", "scale"):
        m[f"model.hc_head.{p}"] = f"{HF_PREFIX}.hc_head.hc_head_{p}"
    m["lm_head.weight"] = (
        f"{HF_PREFIX}.embed_tokens.weight"
        if config.tie_word_embeddings
        else "lm_head.weight"
    )
    return m


class HYV4ForCausalLM(nn.Module):
    """Hy4-preview text model, on the ``convert_mxfp8_to_neuron_fp8.py`` checkpoint.

    The MTP layer (``model.mtp_layers.0``) is not loaded.
    """

    # Overridden by models built on this port (glm_dsa).
    text_model_cls = HYV4TextModel
    checkpoint_mappings = staticmethod(checkpoint_mappings)

    def __init__(
        self, config: HYV4Config | None = None, *, vllm_config=None, prefix: str = ""
    ):
        # ``vllm_config``/``prefix`` exist only so vLLM's engine process, which
        # has no class of its own for this architecture and so inspects this
        # one, recognises it as a text-generation model (``is_vllm_model``
        # checks the keyword). The runner builds it through ``from_configs``.
        if config is None:
            raise TypeError("build HYV4ForCausalLM with from_configs()")
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.neuron_config
        seg_buckets = getattr(nc, "kv_segment_size_buckets", None) if nc else None
        if PREFILL_SPAN_KVSEG and seg_buckets and len(seg_buckets) > 1:
            # The bound trusts pick_kv_segment_size's rule; refuse to run if the
            # installed plugin picks differently (set HY4_PREFILL_SPAN_KVSEG=0).
            if not check_prefill_span_bound(seg_buckets):
                raise RuntimeError(
                    f"kv_segment_size_buckets={seg_buckets}: the plugin's "
                    "pick_kv_segment_size does not bound cached_seq_len as "
                    "prefill_span_bound assumes; set HY4_PREFILL_SPAN_KVSEG=0"
                )

        self.model = self.text_model_cls(config)

        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (
            nc.max_logprobs != 0 or nc.debug_logits_dir is not None
        )
        self.lm_dtype = torch.float32 if config.lm_head_fp32 else config.torch_dtype
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=self.lm_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config, process_group=self.tp_group.device_group
            )

    # ── KV cache ─────────────────────────────────────────────────────────

    def compiler_args_for(self, graph_type, example_inputs, base):
        """neuronx-cc arguments for each graph: the runner's, minus the mac threshold.

        The runner passes ``--modular-flow-mac-threshold=10`` inside
        ``--internal-hlo2tensorizer-options`` because NKI kernels do not report MAC
        counts yet; on graphs built mostly from torch ops it makes neuronx-cc fail
        codegen, so it is dropped here (the fp8 cast flag next to it stays).
        ``HY4_TENSORIZER_OPTIONS`` adds ``--tensorizer-options`` when set.
        (The optimization level is not set here: vllm-neuron's native compile route
        appends its own ``-O2``, so only an explicit ``--optimization-level`` changes it.)
        """
        prefix = "--internal-hlo2tensorizer-options="
        args = []
        for arg in base or []:
            if arg.startswith(prefix):
                opts = arg[len(prefix):].replace("--modular-flow-mac-threshold=10", "").split()
                if not opts:
                    continue
                arg = prefix + " ".join(opts)
            args.append(arg)
        extra = os.environ.get("HY4_TENSORIZER_OPTIONS")
        if extra:
            args.insert(len(args) - 1, f"--tensorizer-options={extra}")
        return args

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(
            layers=[
                LayerSpec(
                    name=layer.self_attn.layer_name,
                    num_kv_heads=1,
                    head_size=layer.self_attn.entry_dim,
                    dtype=self.config.torch_dtype,
                    # One [index | latent | rope] entry per token, not a K/V pair.
                    key_only=True,
                )
                for layer in self.model.layers
            ]
        )

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]) -> None:
        for layer in self.model.layers:
            attn = layer.self_attn
            tensors = kv_caches.get(attn.layer_name)
            if tensors is None:
                raise Exception(f"cache for layer {attn.layer_name} not initialized")
            if len(tensors) != 1:
                raise Exception(
                    f"layer {attn.layer_name} expected one key-only cache tensor, "
                    f"got {len(tensors)}"
                )
            attn.kv_pages = tensors[0]  # [pages, 1, block, 704]

    # ── Forward ──────────────────────────────────────────────────────────

    # The other half of vLLM's text-generation protocol (see ``__init__``).
    # The runner calls ``forward`` only; these are not on its path.
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids, scatter_tokens=False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.to(self.lm_dtype))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        rotary_position_ids: torch.Tensor | None = None,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ):
        positions = positions.to(torch.int32)
        hidden = self.model(input_ids, positions, attn_metadata, rank=rank)
        hidden = torch.index_select(hidden, 0, sampling_positions)
        logits = self.lm_head(hidden.to(self.lm_dtype))

        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)
        if self.on_device_sampling_config is None:
            return logits
        sampled = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled)
        return sampled, gathered_logits

    # ── Construction and weights ─────────────────────────────────────────

    @classmethod
    def from_configs(
        cls, hf_config, text_neuron_config=None, vision_neuron_config=None
    ):
        if vision_neuron_config is not None:
            raise NotImplementedError("Hy4 is a text-only model")
        config = HYV4Config.from_configs(hf_config, neuron_config=text_neuron_config)
        quant = config.neuron_quantization or {}
        fmt = quant.get("format")
        if fmt == FORMAT_MXFP8:
            if mx_backend() == MX_BACKEND_KERNEL:
                # nkilib MX moe_tkg is Trn3-only. Checked once here, outside
                # any traced region.
                require_mx_platform()
            else:
                logger.info(
                    "neuron_quantization.format=%s: routed experts dequantised in "
                    "the graph (%s=%s); this format is intended for Trn3 (set "
                    "%s=%s for nkilib MX moe_tkg)",
                    FORMAT_MXFP8, MX_BACKEND_ENV, MX_BACKEND_TORCH,
                    MX_BACKEND_ENV, MX_BACKEND_KERNEL,
                )
        elif fmt != FORMAT_FP8_ROW:
            raise ValueError(
                "HYV4ForCausalLM on Neuron loads the Trn2 fp8 checkpoint written by "
                "contrib/models/hy4_preview/scripts/convert_mxfp8_to_neuron_fp8.py "
                "(config.json has no neuron_quantization.format=fp8_row_experts). "
                "The bf16 checkpoint does not fit one trn2.48xlarge, and the "
                "published MXFP8 one has no Trn2 kernel. On Trn3, convert with "
                "--keep-mx-experts (format mxfp8_experts)."
            )
        return cls(config)

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            self.rank,
            self.world_size,
            self,
            self.checkpoint_mappings(self.config),
            device,
            strict=False,
        ).state_dict

        # strict=False (buffers are not in the checkpoint), so a wrong mapping key
        # would leave a parameter at torch.empty and generate fluent garbage.
        params = dict(self.named_parameters())
        unfilled = sorted(set(params) - set(rank_sharded))
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} parameter(s) got no checkpoint tensor: {unfilled[:8]}"
                + (" ..." if len(unfilled) > 8 else "")
            )
        for name, tensor in rank_sharded.items():
            want = params[name].dtype
            if tensor.dtype != want:
                rank_sharded[name] = tensor.to(want)
        self.load_state_dict(rank_sharded, strict=False, assign=True)


__all__ = ["HYV4ForCausalLM", "HYV4TextModel"]
