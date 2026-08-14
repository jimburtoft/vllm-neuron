# SPDX-License-Identifier: Apache-2.0
"""
Medusa speculative-decoding HEADS + head LOADER for native-backend Whisper.

Task 020 Milestone 1. This module provides:

  1. ``ResBlock``     -- the single Medusa residual block ``x + SiLU(Linear(x))``.
  2. ``MedusaHeads``  -- an ``nn.Module`` holding N independent heads that read
     the Whisper decoder's LAST hidden state and predict K future tokens.
  3. ``load_medusa_heads`` -- a loader that either synthesizes placeholder heads
     (``zero`` / ``random``) so the framework runs UNTRAINED, or loads
     customer-supplied head weights (voxtral-medusa external formats +
     a generic ``medusa_heads.{i}.*`` scheme).

Design (M0 spec Task 020 §2, §3)
--------------------------------
* **N = K heads, no tree.** Head ``j`` predicts the token at position ``t+j+1``
  from the decoder's last hidden state at position ``t``. Default N=K=5.
* **Head architecture (Medusa paper / NxDI-verbatim):**
    ``ResBlock(d) -> output_projection``
  where ``ResBlock(d) = x + SiLU(Linear(d, d))(x)`` (``medusa_num_layers=1`` is
  the common case; L>1 stacks L ResBlocks). d_model=1280, vocab=51866.
* **Medusa-1 weight tie (default):** the output projection of every head is the
  model's EXISTING sharded LM head (``WhisperForConditionalGeneration.lm_head``,
  the vocab-parallel ``ColumnParallelLinear`` from Task 019). Heads therefore
  inherit the TP vocab-sharding for free and add only the per-head ResBlock
  params (``[d, d]`` each). No per-head ``[vocab, d]`` output Linear is stored.
  The loader IGNORES any per-head output-Linear weights found in a checkpoint
  (they are the target's lm_head under the tie).
* **dtype bf16** to match the model; the tied projection runs the model's fp32
  LM-head path (``_logits_local`` casts to fp32) so head logits match
  ``compute_logits`` precision (M3 tie-stability footgun, spec §2c).
* **Compile-friendly:** ``MedusaHeads.forward`` is pure tensor ops (SiLU written
  out to avoid a fullgraph break, mirroring the model's manual ``gelu``). The
  heads read the last hidden state IN-GRAPH -- no CPU bounce, no separate NEFF.

What M2 consumes
----------------
``MedusaHeads.propose_from_hidden(last_hidden, project_fn)`` returns the K draft
token ids ``[N]`` (int32) argmaxed from each head's (tied) logits at a single
anchor position. The M2 proposer/verify NEFF calls this on the LAST candidate
position's hidden state to produce the next K drafts. ``MedusaHeads.forward``
returns the raw head hidden-states (pre-projection) for callers that want to run
the projection + argmax themselves (e.g. the sharded distributed-argmax path).

Loader key layouts accepted
----------------------------
* ``zero``   -- synth: ResBlock Linear weight+bias = 0 (ResBlock == identity, so
  every head proposes ``argmax(lm_head(h))`` -> the same token; rejection sampler
  rejects at the first divergence -> framework == greedy). NO checkpoint needed.
* ``random`` -- synth: small random ResBlock init (garbage drafts, mostly
  rejected). NO checkpoint needed.
* ``<path>`` -- load customer weights. Supported external schemes (delegated to
  ``head_formats`` normalization; see that module's docstring):
    - upstream FasterDecoding : ``{i}.{j}.linear.weight`` (+ ``{i}.{L}.weight``)
    - vLLM speculators        : ``blocks.{i}.layers.{j}.weight`` (+ ``lm_heads.{i}.weight``)
    - internal / generic      : ``medusa_head_{i}.{j}.linear.weight`` and/or
                                ``medusa_heads.{i}.{j}.linear.weight``
  For the Medusa-1 tie, only the ResBlock Linear weights (``.{j}.linear.*``) are
  loaded; any output-Linear / ``lm_heads.*`` / ``shared_lm_head_weight`` tensors
  are dropped (the tie uses the model's own lm_head).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Head modules
# --------------------------------------------------------------------------- #
def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU written out (x * sigmoid(x)). nn.SiLU / F.silu are graph-safe on
    this stack, but we match the model's manual-activation convention (the erf
    ``gelu`` is written out to avoid a fullgraph break) for consistency and to
    guarantee Dynamo traces it in-graph."""
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    """Single Medusa ResBlock (upstream Medusa & NxDI-verbatim):

        ResBlock(x) = x + SiLU(Linear(x))

    The internal Linear is named ``.linear`` to match the upstream/internal
    checkpoint key scheme (``medusa_head_{i}.{j}.linear.weight``).
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + _silu(self.linear(x))


class MedusaHeads(nn.Module):
    """N Medusa-1 heads reading the Whisper decoder's LAST hidden state.

    Each head is ``L`` stacked ResBlocks (default L=1). The output projection is
    NOT owned here -- under the Medusa-1 tie it is the model's sharded lm_head,
    supplied to ``propose_from_hidden`` / ``head_logits`` as a callable.

    State-dict layout (owned params, internal scheme):
        medusa_head_{i}.{j}.linear.weight   # head i, ResBlock j
        medusa_head_{i}.{j}.linear.bias
    """

    def __init__(
        self,
        num_heads: int = 5,
        hidden_size: int = 1280,
        medusa_num_layers: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.medusa_num_layers = medusa_num_layers
        self.dtype = dtype
        for i in range(num_heads):
            blocks = nn.ModuleList(
                [ResBlock(hidden_size, dtype=dtype) for _ in range(medusa_num_layers)]
            )
            setattr(self, f"medusa_head_{i}", blocks)

    def _head_blocks(self, i: int) -> nn.ModuleList:
        return getattr(self, f"medusa_head_{i}")

    def _apply_head(self, i: int, h: torch.Tensor) -> torch.Tensor:
        """Run head i's ResBlock stack on hidden state ``h`` -> head hidden [.., d]."""
        x = h
        for block in self._head_blocks(i):
            x = block(x)
        return x

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        """hidden: [.., d_model]. Returns a list of N head hidden-states [.., d]
        (PRE-projection). Callers project each with the tied lm_head."""
        return [self._apply_head(i, hidden) for i in range(self.num_heads)]

    def head_logits(
        self, hidden: torch.Tensor, project_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> list[torch.Tensor]:
        """Run all heads then the tied projection. ``project_fn`` is the model's
        lm-head projection (e.g. ``model._logits_local`` for the sharded slice,
        or ``model.compute_logits`` for full vocab). Returns N logits tensors."""
        return [project_fn(self._apply_head(i, hidden)) for i in range(self.num_heads)]

    def propose_from_hidden(
        self,
        last_hidden: torch.Tensor,
        project_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Greedy draft proposal for a SINGLE anchor position.

        Args:
            last_hidden: [d_model] or [1, d_model] the decoder's last hidden
                state at the anchor position.
            project_fn:  the tied lm-head projection producing FULL-vocab logits
                (so the argmax is over the real vocab id space). Use
                ``model.compute_logits`` (all-gathers the sharded slices) for a
                real token id. Head j -> token at anchor+1+j.

        Returns:
            drafts: [N] int32 -- the K greedy draft token ids, one per head.
        """
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)  # [1, d]
        ids = []
        for i in range(self.num_heads):
            head_h = self._apply_head(i, last_hidden)  # [1, d]
            logits = project_fn(head_h)  # [1, vocab]
            ids.append(torch.argmax(logits, dim=-1).to(torch.int32))  # [1]
        return torch.cat(ids, dim=0)  # [N]

    def propose_from_hidden_sharded(
        self,
        last_hidden: torch.Tensor,
        argmax_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Greedy draft proposal via the SHARDED distributed argmax (Task 023).

        Unlike ``propose_from_hidden`` (which calls ``model.compute_logits`` ->
        a full-vocab all-gather per head), this stacks all N head hidden-states
        into one [N, d] tensor and runs a SINGLE sharded distributed argmax over
        the local vocab slices -- avoiding N full-vocab all-gathers per verify
        step. ``argmax_fn`` maps [N, d] head hidden-states -> [N] int32 token
        ids (the model's sharded ``_logits_local`` + pad-mask + Sampler path).

        Args:
            last_hidden: [d_model] or [1, d_model] decoder last hidden state.
            argmax_fn:   [N, d] -> [N] int32 sharded distributed argmax.

        Returns:
            drafts: [N] int32 -- the K greedy draft token ids, one per head.
        """
        if last_hidden.dim() == 1:
            last_hidden = last_hidden.unsqueeze(0)  # [1, d]
        head_hs = [
            self._apply_head(i, last_hidden) for i in range(self.num_heads)
        ]  # N x [1, d]
        stacked = torch.cat(head_hs, dim=0)  # [N, d]
        return argmax_fn(stacked).to(torch.int32)  # [N]



# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def _synthesize_state(
    heads: MedusaHeads, mode: str, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Build a state dict for placeholder heads (no checkpoint required).

    zero   -- ResBlock Linear weight & bias = 0 -> ResBlock == identity -> every
              head emits argmax(lm_head(h)) (the same token). Rejected at the
              first divergence -> framework == greedy.
    random -- small normal init (std 0.02) -> garbage drafts, mostly rejected.
    """
    if mode not in ("zero", "random"):
        raise ValueError(f"synth mode must be 'zero' or 'random', got {mode!r}")
    g = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for i in range(heads.num_heads):
        for j in range(heads.medusa_num_layers):
            wkey = f"medusa_head_{i}.{j}.linear.weight"
            bkey = f"medusa_head_{i}.{j}.linear.bias"
            if mode == "zero":
                w = torch.zeros(heads.hidden_size, heads.hidden_size)
                b = torch.zeros(heads.hidden_size)
            else:
                w = torch.empty(heads.hidden_size, heads.hidden_size)
                w.normal_(mean=0.0, std=0.02, generator=g)
                b = torch.zeros(heads.hidden_size)
            state[wkey] = w.to(heads.dtype)
            state[bkey] = b.to(heads.dtype)
    return state


def _load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    """Load a raw head state dict from .pt/.bin/.safetensors, unwrapping a
    common ``{'heads': state}`` outer dict."""
    if path.endswith(".safetensors"):
        import safetensors.torch

        state = safetensors.torch.load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "heads" in state and isinstance(
        state["heads"], dict
    ):
        state = state["heads"]
    return state


def _normalize_external_state(
    raw: dict[str, torch.Tensor], heads: MedusaHeads
) -> dict[str, torch.Tensor]:
    """Convert a customer/external head state dict to this module's internal
    ResBlock layout ``medusa_head_{i}.{j}.linear.{weight,bias}``, DROPPING any
    per-head output-projection tensors (Medusa-1 tie uses the model's lm_head).

    Accepts (in priority order):
      internal / generic : medusa_head_{i}.{j}.linear.*  OR  medusa_heads.{i}.{j}.linear.*
      upstream           : {i}.{j}.linear.*
      vLLM speculators    : blocks.{i}.layers.{j}.*
    Uses head_formats detection/refusal when available; falls back to a direct
    regex remap so the loader also works without that reference module present.
    """
    # Strip a generic ``medusa_heads.`` outer prefix (our public config scheme).
    if all(k.startswith("medusa_heads.") for k in raw):
        raw = {k[len("medusa_heads."):]: v for k, v in raw.items()}

    # Try the reference head_formats normalizer first (does refusal of EAGLE /
    # Hydra / Medusa-2 and format detection). Optional dependency.
    normalized: dict[str, torch.Tensor] | None = None
    try:
        from .head_formats import convert_to_internal, detect_medusa_format

        info = detect_medusa_format(raw)
        normalized = convert_to_internal(raw, info)
        logger.info(
            "Medusa head loader: detected format=%s variant=%s n_heads=%s L=%s",
            info.get("source_format"),
            info.get("variant"),
            info.get("n_heads"),
            info.get("medusa_num_layers"),
        )
    except ImportError:
        normalized = None
    except Exception as e:  # detection failed -> fall back to direct remap
        logger.warning("head_formats detection failed (%s); using direct remap", e)
        normalized = None

    if normalized is None:
        normalized = {}
        for k, v in raw.items():
            # generic public scheme: medusa_heads.{i}.{j}.linear.*
            m = re.match(r"^medusa_heads\.(\d+)\.(\d+)\.linear\.(weight|bias)$", k)
            if m:
                normalized[
                    f"medusa_head_{m.group(1)}.{m.group(2)}.linear.{m.group(3)}"
                ] = v
                continue
            # internal: medusa_head_{i}.{j}.linear.*
            m = re.match(r"^medusa_head_(\d+)\.(\d+)\.linear\.(weight|bias)$", k)
            if m:
                normalized[k] = v
                continue
            # upstream: {i}.{j}.linear.*
            m = re.match(r"^(\d+)\.(\d+)\.linear\.(weight|bias)$", k)
            if m:
                normalized[
                    f"medusa_head_{m.group(1)}.{m.group(2)}.linear.{m.group(3)}"
                ] = v
                continue
            # vLLM speculators: blocks.{i}.layers.{j}.{weight,bias}
            m = re.match(r"^blocks\.(\d+)\.layers\.(\d+)\.(weight|bias)$", k)
            if m:
                normalized[
                    f"medusa_head_{m.group(1)}.{m.group(2)}.linear.{m.group(3)}"
                ] = v
                continue
            # everything else (output projections, lm_heads, token_map, shared) dropped

    # Keep ONLY ResBlock Linear tensors (drop output projections under the tie).
    kept: dict[str, torch.Tensor] = {}
    for k, v in normalized.items():
        if re.match(r"^medusa_head_(\d+)\.(\d+)\.linear\.(weight|bias)$", k):
            kept[k] = v.to(heads.dtype)
    if not kept:
        raise ValueError(
            "No ResBlock Linear tensors found after normalization. Expected keys "
            "like 'medusa_head_{i}.{j}.linear.weight'. Got (first 8): "
            f"{list(raw.keys())[:8]}"
        )
    return kept


def load_medusa_heads(
    num_heads: int = 5,
    hidden_size: int = 1280,
    medusa_num_layers: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    init: str = "random",
    heads_path: str | None = None,
    seed: int = 0,
) -> tuple[MedusaHeads, dict[str, int]]:
    """Construct MedusaHeads and load its ResBlock params.

    Args:
        num_heads:  N = K lookahead heads (default 5).
        hidden_size: d_model (1280 for whisper-large-v3).
        medusa_num_layers: ResBlocks per head (default 1).
        dtype:      head param dtype (bf16 to match the model).
        init:       "zero" | "random" | "load". If "load", ``heads_path`` must
                    be set. ``zero``/``random`` need NO checkpoint (framework
                    runs untrained).
        heads_path: checkpoint path for init="load" (.pt/.bin/.safetensors).
        seed:       RNG seed for init="random".

    Returns:
        (heads, report) where report = {"missing": n, "unexpected": n} for the
        head params (both must be 0 for a clean load).
    """
    heads = MedusaHeads(
        num_heads=num_heads,
        hidden_size=hidden_size,
        medusa_num_layers=medusa_num_layers,
        dtype=dtype,
    )
    # Heads are built on meta or real device by the caller; materialize storage
    # on CPU so load_state_dict(assign=True) attaches concrete tensors.
    heads = heads.to_empty(device="cpu") if _is_meta(heads) else heads

    if init in ("zero", "random"):
        state = _synthesize_state(heads, init, seed=seed)
    elif init == "load":
        if not heads_path:
            raise ValueError("init='load' requires heads_path")
        raw = _load_checkpoint_state(heads_path)
        state = _normalize_external_state(raw, heads)
    else:
        raise ValueError(
            f"init must be 'zero' | 'random' | 'load', got {init!r}"
        )

    missing, unexpected = heads.load_state_dict(state, strict=False, assign=True)
    report = {"missing": len(missing), "unexpected": len(unexpected)}
    if missing or unexpected:
        logger.warning(
            "MedusaHeads load: missing=%s unexpected=%s", missing, unexpected
        )
    logger.info(
        "MedusaHeads: N=%d L=%d d=%d dtype=%s init=%s missing=%d unexpected=%d",
        num_heads,
        medusa_num_layers,
        hidden_size,
        dtype,
        init,
        report["missing"],
        report["unexpected"],
    )
    return heads, report


def _is_meta(module: nn.Module) -> bool:
    for p in module.parameters():
        return p.is_meta
    return False
