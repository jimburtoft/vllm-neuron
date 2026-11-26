# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec


class KeyOnlyAttentionSpec(MLAAttentionSpec):
    """Key-only KV cache spec that will not merge caches of different widths.

    ``MLAAttentionSpec`` budgets one vector per token rather than a K/V pair,
    which is what a key-only cache needs. Its ``merge`` asserts on the
    quantization fields but takes ``head_size``/``num_kv_heads``/``dtype`` from
    the first spec, and vLLM decides two layers share a KV cache group by
    calling ``merge`` and checking whether it raises
    (``kv_cache_utils.is_kv_cache_spec_uniform``). A 576-wide MLA cache and a
    128-wide indexer cache therefore merge silently into one 576-wide group,
    and every layer in that group is budgeted AND shaped at 576 -- the indexer
    both wastes 448 columns and rejects its own 128-wide write.

    Asserting on the width keeps differing caches out of one group, so vLLM
    falls through to ``UniformTypeKVCacheSpecs``, which budgets each layer at
    its own page size and exposes per-layer specs to the worker.
    """

    @classmethod
    def merge(cls, specs):
        widths = {(s.num_kv_heads, s.head_size, s.dtype) for s in specs}
        assert len(widths) == 1, (
            "KV caches of different widths cannot share a cache group: "
            f"{sorted(str(w) for w in widths)}"
        )
        return super().merge(specs)


@dataclass
class LayerSpec:
    """
    Defines the KV cache specification for a single transformer layer.

    Used to specify the memory requirements and configuration for storing
    key-value pairs in the attention mechanism of a transformer layer.
    """

    name: str
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    sliding_window_size: int | None = None
    chunk_size: int | None = None
    # One vector per token instead of a K/V pair, budgeted and allocated as a
    # single cache. Set by MLA layers, which store one combined
    # [rope | compressed_kv] entry, and by key-only caches such as the DeepSeek
    # V3.2 sparse-attention indexer. bind_kv_cache receives a 1-element list.
    key_only: bool = False


@dataclass
class KVSpec:
    """
    Defines the KV cache needs of a model by specifying all layer configurations.

    Contains a list of LayerSpec objects that collectively define the complete
    KV cache requirements for an entire transformer model.
    """

    layers: list[LayerSpec]


# ---------------------------------------------------------------------------
# Paged-KV slot arithmetic
# ---------------------------------------------------------------------------
# Write-side padding target: the code redirects an invalid slot to block 0. The
# scheduler reserves block 0 as the null block for this purpose. This mirrors
# ``NULL_BLOCK_ID`` in ``neuron_model_runner``, whose context-parallel path
# already does ``np.where(oob_mask, NULL_BLOCK_ID, slot_mapping)``.
_NULL_BLOCK_ID = 0


def paged_slot_indices(
    slot_mapping: torch.Tensor,
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive bounds-checked ``(block_indices, position_indices)`` for a KV write.

    The scheduler marks an inactive write slot with ``PAD_SLOT_ID`` (-1). A
    negative entry is not safe to use as a cache coordinate. ``-1 // block_size``
    floor-divides to ``-1``, not to ``0``. Eager torch wraps that ``-1`` to the
    LAST block and corrupts the KV of a live sequence. On Neuron the same
    ``index_put_`` becomes an HLO ``scatter`` that runs as an indirect vector-DGE
    copy, so ``-1`` becomes an out-of-range HBM address. The runtime then raises
    "scatter/gather (indirect memory copy via vector DGE) out-of-bound access",
    which wedges the NEFF execution queue and stops the engine core.

    This function redirects an entry outside ``[0, num_blocks * block_size)`` to
    the null block before it splits the slot into a block and a position. The
    write for that token still happens, and the null block discards it. A
    redirect keeps the returned shapes static, so the enclosing ``index_put_``
    stays one traceable operation.

    Call this function instead of a direct ``//`` and ``%`` on a slot mapping.
    It clamps before it derives both coordinates. A clamp after the ``//`` and
    the ``%`` leaves ``position_indices`` at ``block_size - 1`` for a pad entry.

    Args:
        slot_mapping: ``[num_tokens]`` flat slot ids, possibly padded with -1.
        block_size: Tokens per KV-cache block.
        num_blocks: Block count of the target cache (its dim-0 extent).

    Returns:
        ``(block_indices, position_indices)``. Both have the shape and the dtype
        of ``slot_mapping``. Every value is in range for ``num_blocks`` blocks.

    Example:
        >>> import torch
        >>> slots = torch.tensor([0, 5, -1, 9])  # -1 is PAD_SLOT_ID; 9 overflows
        >>> blocks, positions = paged_slot_indices(slots, 4, 2)
        >>> blocks
        tensor([0, 1, 0, 0])
        >>> positions
        tensor([0, 1, 0, 0])
    """
    max_slot = num_blocks * block_size
    safe_slots = torch.where(
        (slot_mapping < 0) | (slot_mapping >= max_slot),
        torch.full_like(slot_mapping, _NULL_BLOCK_ID),
        slot_mapping,
    )
    return safe_slots // block_size, safe_slots % block_size


def build_per_head_slot_mapping(
    slot_mapping: torch.Tensor,
    block_size: int,
    kv_heads: int,
    batch_size: int,
) -> torch.Tensor:
    """Build the per-head ``kv_cache_update_idx`` of the in-kernel KV cache update.

    ``update_cache=True`` makes the fused decode kernel write the new K/V into the
    paged cache itself, so the caller does no scatter. For block KV the kernel wants
    physical slots, and the shape depends on the head count
    (``attention_block_tkg`` docstring):

    * ``kv_heads == 1``: ``[B, S_tkg]``
    * ``kv_heads >  1``: ``[B, kv_heads, S_tkg]``

    A slot must index the SAME flat pool that ``build_per_head_block_table`` targets:
    the kernel folds the 4D cache to ``num_blocks * kv_heads`` blocks, so the global
    block of ``(base_block, h)`` is ``base_block * kv_heads + h``. A slot is then
    ``global_block * block_size + position_in_block``. This function and
    ``build_per_head_block_table`` (in ``qwen3_vl.utils.decode_kv``) must agree.

    Args:
        slot_mapping: ``[B * S_tkg]`` flat slots, ``base_block * block_size + pos``.
        block_size: Tokens in one cache block.
        kv_heads: KV heads of this rank.
        batch_size: Requests of this decode step.

    Returns:
        ``[B, kv_heads, S_tkg]`` slots, or ``[B, S_tkg]`` when ``kv_heads == 1``.

    Example:
        >>> import torch
        >>> # 1 request, 1 token at base block 2 position 3, block_size 4
        >>> slots = torch.tensor([2 * 4 + 3])
        >>> build_per_head_slot_mapping(slots, block_size=4, kv_heads=2, batch_size=1)
        tensor([[[19],
                 [23]]])
    """
    if kv_heads == 1:
        return slot_mapping.view(batch_size, -1)

    base_block = torch.div(slot_mapping, block_size, rounding_mode="floor")
    position = slot_mapping - base_block * block_size

    kv_h = torch.arange(
        kv_heads, dtype=slot_mapping.dtype, device=slot_mapping.device
    ).view(1, kv_heads)
    # [B*S, kv_heads] -> global slot of each (token, head)
    global_block = base_block.unsqueeze(1) * kv_heads + kv_h
    global_slot = global_block * block_size + position.unsqueeze(1)
    # [B*S, kv_heads] -> [B, kv_heads, S_tkg]
    return global_slot.view(batch_size, -1, kv_heads).transpose(1, 2).contiguous()

# ---------------------------------------------------------------------------
# Full-prefill KV write in whole cache blocks (port of private-vllm-neuron
# PR #5335, ba997f7576 + d15674ce35).
# ---------------------------------------------------------------------------
from torch._subclasses.fake_tensor import FakeTensor

from vllm_neuron import envs as _envs

_PREFILL_KV_BLOCK_WRITE = _envs.VLLM_NEURON_PREFILL_KV_BLOCK_WRITE
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _eager_values_visible(t: torch.Tensor) -> bool:
    """Return True when ``t`` holds real values that a Python check can read.

    Dynamo folds a tensor-valued Python assertion before the graph reaches a
    backend, and a FakeTensor holds no values. Only a real eager tensor can
    enforce a value contract.
    """
    return not torch.compiler.is_dynamo_compiling() and not isinstance(t, FakeTensor)


def prefill_kv_block_write_ok(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_tokens: int,
    block_size: int,
    num_kv_heads: int,
    *,
    full_prefill: bool,
    dcp_size: int = 1,
    fp8_packed: bool = False,
) -> bool:
    """Return True when ``write_prefill_kv`` can write whole cache blocks.

    Every input is a trace-time constant (Python ints, flags, and tensor
    shapes), so the result selects one write path for the whole graph. No
    branch depends on tensor values.

    The block write needs all of these conditions:

    * ``VLLM_NEURON_PREFILL_KV_BLOCK_WRITE`` allows it. ``"packed"`` (the
      default) allows it for a packed FP8 K cache only, ``"all"`` for every
      cache, and ``"off"`` for none. For a packed K cache, the per-token path
      unswizzles and reswizzles the whole cache, and the block write does not.
      For a bf16 or a non-packed FP8 cache, the per-token scatter is about as
      fast on trn2, so the default keeps it.
    * The step is a full prefill (no ``kv_segment_size``), so the tokens of the
      step start at in-block position 0 and fill each block in order.
    * ``dcp_size == 1``. A DCP rank owns only some tokens of each block.
    * ``num_tokens`` is a whole number of blocks.
    * The block dim of the cache is ``block_size`` and the head dim of the
      cache is ``num_kv_heads``. A packed FP8 K cache holds ``block_size // 2``
      token pairs in each block.

    Example:
        >>> import torch
        >>> k = torch.zeros(4, 2, 8, 8, 2, dtype=torch.float8_e4m3fn)
        >>> v = torch.zeros(4, 2, 16, 8, dtype=torch.float8_e4m3fn)
        >>> ok = prefill_kv_block_write_ok
        >>> ok(k, v, 32, 16, 2, full_prefill=True, fp8_packed=True)
        True
        >>> ok(k, v, 24, 16, 2, full_prefill=True, fp8_packed=True)
        False
    """
    if _PREFILL_KV_BLOCK_WRITE == "off" or not full_prefill or dcp_size > 1:
        return False
    if _PREFILL_KV_BLOCK_WRITE == "packed" and not fp8_packed:
        return False
    if block_size <= 0 or num_tokens <= 0 or num_tokens % block_size != 0:
        return False
    if v_cache.dim() != 4 or tuple(v_cache.shape[1:3]) != (num_kv_heads, block_size):
        return False
    if fp8_packed:
        return (
            k_cache.dim() == 5
            and k_cache.shape[-1] == 2
            and block_size % 2 == 0
            and tuple(k_cache.shape[1:3]) == (num_kv_heads, block_size // 2)
        )
    return k_cache.dim() == 4 and tuple(k_cache.shape[1:3]) == (
        num_kv_heads,
        block_size,
    )


def _check_block_aligned(
    block_indices: torch.Tensor,
    position_indices: torch.Tensor,
    block_size: int,
) -> None:
    """Check in eager mode that each ``block_size`` chunk fills one cache block.

    Chunk ``c`` holds tokens ``[c * bs, (c + 1) * bs)``. A valid token (a block
    other than the null block) must sit in the block of the first token of its
    chunk, at in-block position ``token - c * bs``. A chunk whose first token is
    padding must hold padding only.
    """
    blocks = block_indices.view(-1, block_size)
    positions = position_indices.view(-1, block_size)
    valid = blocks != _NULL_BLOCK_ID
    offset = torch.arange(block_size, device=blocks.device).view(1, -1)
    aligned = (blocks == blocks[:, :1]) & (positions == offset)
    assert bool(torch.all(~valid | aligned)), (
        "write_prefill_kv: the slots of a full prefill are not block-aligned. "
        "Each chunk of block_size tokens must start at in-block position 0 of "
        "one block, or hold padding only. Pass full_prefill=False to use the "
        "per-token write."
    )


def write_prefill_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    *,
    full_prefill: bool,
    dcp_size: int = 1,
    fp8_packed: bool = False,
    packed_k_fallback: str = "pairs",
) -> None:
    """Write the K/V of a prefill step into the paged KV cache, in place.

    The caller quantizes or casts K/V first. This function only moves bytes, so
    the cache contents of each real token are identical on both paths.

    **Block path.** When ``prefill_kv_block_write_ok`` is True, each chunk of
    ``block_size`` tokens is one cache block, and the function writes one
    ``index_put_`` row for each (block, head). By default only a packed FP8 K
    cache takes this path (see ``prefill_kv_block_write_ok``). The block of
    chunk ``c`` is the block of its first token. The runner points a pad slot
    at the null block (block 0), and ``paged_slot_indices`` also redirects
    ``-1``, so a chunk of pad tokens writes the null block. The pad tail of the
    last real chunk goes to positions past the prompt in the last block of that
    request. No reader uses these positions before decode writes them: the
    attention mask stops at the sequence length, a sliding window ends at the
    sequence length too, and the prefix cache shares full blocks only. A
    reused block already holds stale bytes at those positions on the per-token
    path. A packed FP8 K block gets both lanes of each token pair from the new
    tokens, so the write does not read the cache.

    **Per-token path.** Otherwise, write one row for each (token, head) with
    ``index_put_`` for K and V. A packed FP8 K cache uses
    ``scatter_packed_k`` (``packed_k_fallback="pairs"``), or a whole-cache
    unswizzle, scatter, and reswizzle (``packed_k_fallback="reswizzle"``).
    ``scatter_packed_k`` needs an even token count.

    In eager mode the block path checks that the slots are block-aligned.

    Args:
        k_cache: ``[nb, nkh, bs, d]``, or packed FP8 ``[nb, nkh, bs // 2, d, 2]``.
        v_cache: ``[nb, nkh, bs, d]``.
        k_new: ``[nkh, T, d]`` or ``[nkh * T, d]`` (head-major), in the cache
            dtype (FP8 caches: already scaled, clamped, and cast).
        v_new: Same layout as ``k_new``.
        slot_mapping: ``[T]`` flat slots. Pad slots are 0 (null block) or -1.
        block_size: Tokens in one cache block of this layer.
        full_prefill: True when the step has no ``kv_segment_size``.
        dcp_size: Decode-context-parallel size. A value above 1 selects the
            per-token path.
        fp8_packed: True when ``k_cache`` uses the packed FP8 layout.
        packed_k_fallback: ``"pairs"`` or ``"reswizzle"``. The packed K write
            of the per-token path.

    Example:
        >>> import torch
        >>> k_cache = torch.zeros(3, 1, 2, 1)
        >>> v_cache = torch.zeros(3, 1, 2, 1)
        >>> k = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # [nkh=1, T=4, d=1]
        >>> slots = torch.tensor([4, 5, 2, 3])  # block 2, then block 1
        >>> write_prefill_kv(k_cache, v_cache, k, k, slots, 2, full_prefill=True)
        >>> k_cache[:, 0, :, 0]
        tensor([[0., 0.],
                [3., 4.],
                [1., 2.]])
    """
    num_tokens = slot_mapping.shape[0]
    d_head = k_new.shape[-1]
    nkh = k_new.numel() // (num_tokens * d_head)
    block_indices, position_indices = paged_slot_indices(
        slot_mapping, block_size, k_cache.shape[0]
    )

    if prefill_kv_block_write_ok(
        k_cache,
        v_cache,
        num_tokens,
        block_size,
        nkh,
        full_prefill=full_prefill,
        dcp_size=dcp_size,
        fp8_packed=fp8_packed,
    ):
        if _eager_values_visible(slot_mapping):
            _check_block_aligned(block_indices, position_indices, block_size)
        num_blocks = num_tokens // block_size
        # One row for each (head, block), in head-major order. Row h * nblk + c
        # is block c of head h. It is a contiguous [bs, d] slice of the
        # head-major K/V, so the payload needs no transpose.
        blk = block_indices.view(num_blocks, block_size)[:, 0]
        index = (
            blk.repeat(nkh),
            torch.arange(nkh, dtype=blk.dtype, device=blk.device).repeat_interleave(
                num_blocks
            ),
        )
        rows = (nkh * num_blocks, block_size, d_head)
        v_cache.index_put_(index, v_new.reshape(rows).to(v_cache.dtype))
        if fp8_packed:
            # Lane p % 2 of packed row p // 2 holds in-block position p. Swap
            # the lane and the head dims in bf16: the fp8 -> bf16 -> fp8 round
            # trip is exact, so the stored bytes do not change.
            k_rows = k_new.reshape(nkh * num_blocks, block_size // 2, 2, d_head)
            if k_rows.dtype in _FP8_DTYPES:
                k_rows = k_rows.to(torch.bfloat16)
            k_rows = k_rows.transpose(-1, -2).contiguous()
        else:
            k_rows = k_new.reshape(rows)
        k_cache.index_put_(index, k_rows.to(k_cache.dtype))
        return

    head_indices = torch.arange(
        nkh, dtype=torch.long, device=k_new.device
    ).repeat_interleave(num_tokens)
    index = (block_indices.repeat(nkh), head_indices, position_indices.repeat(nkh))
    k_flat = k_new.reshape(-1, d_head).to(k_cache.dtype)
    v_flat = v_new.reshape(-1, d_head).to(v_cache.dtype)
    v_cache.index_put_(index, v_flat)
    if not fp8_packed:
        k_cache.index_put_(index, k_flat)
        return

    # Import here: attention_decode imports the NKI kernels, and kv_cache is
    # imported by the model runner before they are needed.
    from vllm_neuron.functional.attention.attention_decode import (
        _swizzle_packed_k,
        _unswizzle_packed_k,
        scatter_packed_k,
    )

    if packed_k_fallback == "reswizzle":
        k_unpacked = _unswizzle_packed_k(k_cache)
        k_unpacked.index_put_(index, k_flat)
        k_cache.copy_(_swizzle_packed_k(k_unpacked))
    elif packed_k_fallback == "pairs":
        scatter_packed_k(k_cache, k_flat, *index)
    else:
        raise ValueError(
            f"packed_k_fallback {packed_k_fallback!r} is not known. "
            "Use 'pairs' or 'reswizzle'."
        )




def scatter_packed_k_pairs_nogather(
    k_cache: torch.Tensor,
    k_new: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write new K token pairs into a packed FP8 K cache without reading it.

    For a prefill chunk that starts at an even position and has its pad slots
    (0 or -1) only after its real tokens -- a segmented-prefill chunk, or a
    single prompt from position 0. Then each token pair (2k, 2k+1) is either
    two real tokens, a real token followed by a pad, or two pads. A real + pad
    pair writes the pad into the position after the prompt's last token, in the
    request's own block; no reader uses that position before decode writes it
    (the same argument as the block path of ``write_prefill_kv``). Two pads go
    to the null block.

    ``scatter_packed_k`` instead gathers the stored pair to merge one lane; that
    gather went out of bounds on trn2 with two KV heads per rank (gpt-oss-trn2
    Task 086, ministral3-8b Task 030).

    Args:
        k_cache: packed FP8 K cache ``[nb, nkh, bs // 2, d, 2]``.
        k_new: ``[nkh, T, d]`` or ``[nkh * T, d]`` (head-major), cache dtype.
        slot_mapping: ``[T]`` flat slots, pads 0 or -1 after the real tokens.
        block_size: tokens in one cache block.

    Example:
        >>> import torch
        >>> kc = torch.zeros(3, 1, 2, 1, 2)
        >>> k = torch.tensor([[[1.0], [2.0], [3.0], [0.0]]])  # 3 real + 1 pad
        >>> scatter_packed_k_pairs_nogather(kc, k, torch.tensor([8, 9, 10, 0]), 4)
        >>> kc[2, 0, :, 0].tolist()
        [[1.0, 2.0], [3.0, 0.0]]
    """
    num_tokens = slot_mapping.shape[0]
    d_head = k_new.shape[-1]
    nkh = k_new.numel() // (num_tokens * d_head)
    if _eager_values_visible(slot_mapping):
        assert num_tokens % 2 == 0, "scatter_packed_k_pairs_nogather needs an even token count"
    blocks, positions = paged_slot_indices(slot_mapping, block_size, k_cache.shape[0])
    lane0_blocks = blocks.view(-1, 2)[:, 0]
    lane1_blocks = blocks.view(-1, 2)[:, 1]
    lane0_outer = positions.view(-1, 2)[:, 0] // 2
    lane1_outer = positions.view(-1, 2)[:, 1] // 2
    # Lane 0 is real unless the whole pair is padding; a pad-then-real pair
    # cannot occur for this layout. Pick lane 1's coordinate only for that
    # impossible case to keep the write in range.
    use_lane0 = lane0_blocks > 0
    pair_blocks = torch.where(use_lane0, lane0_blocks, lane1_blocks)
    pair_outer = torch.where(use_lane0, lane0_outer, lane1_outer)
    num_pairs = num_tokens // 2
    index = (
        pair_blocks.repeat(nkh),
        torch.arange(nkh, dtype=pair_blocks.dtype, device=pair_blocks.device).repeat_interleave(num_pairs),
        pair_outer.repeat(nkh),
    )
    k_rows = k_new.reshape(nkh * num_pairs, 2, d_head)
    if k_rows.dtype in _FP8_DTYPES:
        k_rows = k_rows.to(torch.bfloat16)
    k_rows = k_rows.transpose(-1, -2).contiguous()  # [pairs, d, 2]
    k_cache.index_put_(index, k_rows.to(k_cache.dtype))


def scatter_packed_k_by_position(
    k_cache: torch.Tensor,
    k_new: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Write new K into a packed FP8 K cache by cache position, with no cache read.

    For a step whose prompts each start at a block boundary (fresh prompts at
    position 0, or segmented chunks), packed back to back in any lengths: a
    packed-prefill step with odd-length prompts breaks the step-level token
    pairing that ``scatter_packed_k_pairs_nogather`` assumes, but every cache
    pair (2k, 2k + 1) still holds two consecutive tokens of one prompt.

    Each token at an even cache position writes its pair: lane 0 is the token,
    lane 1 the next token of the step if that is the next position of the same
    block, else zero (the position after a prompt's last token, unread until
    decode writes it). Tokens at odd positions and padding (slot 0 or -1) write
    to the null block. The partner is gathered from ``k_new``, never from the
    cache.

    Args:
        k_cache: packed FP8 K cache ``[nb, nkh, bs // 2, d, 2]``.
        k_new: ``[nkh, T, d]`` or ``[nkh * T, d]`` (head-major), cache dtype.
        slot_mapping: ``[T]`` flat slots; padding 0 or -1.
        block_size: tokens in one cache block.

    Example:
        >>> import torch
        >>> kc = torch.zeros(4, 1, 2, 1, 2)
        >>> # prompt A (3 tokens, block 2), prompt B (2 tokens, block 3), 1 pad
        >>> k = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [0.0]]])
        >>> scatter_packed_k_by_position(kc, k, torch.tensor([8, 9, 10, 12, 13, 0]), 4)
        >>> kc[2, 0, :, 0].tolist(), kc[3, 0, 0, 0].tolist()
        ([[1.0, 2.0], [3.0, 0.0]], [4.0, 5.0])
    """
    num_tokens = slot_mapping.shape[0]
    d_head = k_new.shape[-1]
    nkh = k_new.numel() // (num_tokens * d_head)
    blocks, positions = paged_slot_indices(slot_mapping, block_size, k_cache.shape[0])
    dev = blocks.device
    t = torch.arange(num_tokens, dtype=blocks.dtype, device=dev)
    nxt = torch.clamp(t + 1, max=num_tokens - 1)
    has_partner = (
        (blocks[nxt] == blocks) & (positions[nxt] == positions + 1) & (t + 1 < num_tokens)
    )
    lead = (torch.remainder(positions, 2) == 0) & (blocks > 0)
    zero = torch.zeros_like(blocks)
    pair_block = torch.where(lead, blocks, zero)
    pair_outer = torch.where(lead, torch.div(positions, 2, rounding_mode="floor"), zero)
    k3 = k_new.reshape(nkh, num_tokens, d_head)
    if k3.dtype in _FP8_DTYPES:
        k3 = k3.to(torch.bfloat16)
    lane1 = torch.where(has_partner.view(1, num_tokens, 1), k3[:, nxt, :], torch.zeros_like(k3))
    rows = torch.stack([k3, lane1], dim=-1).reshape(nkh * num_tokens, d_head, 2)
    index = (
        pair_block.repeat(nkh),
        torch.arange(nkh, dtype=blocks.dtype, device=dev).repeat_interleave(num_tokens),
        pair_outer.repeat(nkh),
    )
    k_cache.index_put_(index, rows.to(k_cache.dtype))
