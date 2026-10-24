# SPDX-License-Identifier: Apache-2.0
"""Whisper large-v3 (encoder-decoder) in BF16 for the native vLLM Neuron backend.

Design (the same as the 0.21 port):

* The runner sees a decoder-only model with a multimodal (audio) encoder. The
  runner calls ``embed_multimodal`` one time for each request, at prefill.
* ``audio_tower`` is a ``WhisperCrossKVEncoder``. It runs the encoder, projects
  the cross-attention K/V for each decoder layer, and writes them into
  model-owned ``register_buffer`` tensors. All of this occurs in one compiled
  graph. The aliasing FX pass keeps the in-place writes in HBM.
* The decoder self-attention uses the block-managed paged KV cache of the
  runner (``get_kv_spec`` / ``bind_kv_cache``).
* The decoder cross-attention reads the cross-KV buffers. They are not in
  ``get_kv_spec``, so the block manager does not see them.
* The LM head is tied to ``decoder.embed_tokens``. It is vocab-parallel and
  in fp32. The on-device sampler does a distributed greedy argmax.

The cross-KV buffers have one slot for each request (``max_num_seqs`` slots
plus one scratch slot). The runner gives each request a stable slot
(``RequestSlotTable``) and sends the slot of each batch row in
``attn_metadata[layer]["request_slots"]``.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ...     TP-sharded.
  # <-- MODEL-SPECIFIC: ...  Whisper-specific.

Whisper properties that are different from llama3:
  * No RoPE. The model adds learned positions to the token embeddings.
  * 20 attention heads (MHA). 20 % 8 != 0, so TP is 1, 2 or 4.
  * head_dim = 64.
  * q_proj and v_proj have a bias. k_proj has no bias.
  * out_proj and mlp.fc2 keep an fp32 bias. RowParallelLinear accepts only an
    fp32 bias at tp > 1.
  * The activation is erf GELU, written out. ``F.gelu`` breaks the full graph.
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm_neuron.model.kv_cache import (
    KVSpec,
    LayerSpec,
    build_per_head_slot_mapping,
    paged_slot_indices,
)
from vllm_neuron.model.qwen3_vl.utils.decode_kv import build_per_head_block_table
from vllm_neuron.model.neuron_config import EncoderNeuronConfig, NeuronConfig
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
from vllm_neuron.nn.sampler import Sampler

from .config import WhisperConfig
from .weight_loaders import build_state, load_hf_state, resolve_checkpoint_dir

logger = logging.getLogger(__name__)


def encoder_nf_enabled() -> bool:
    """Return True when ``WHISPER_ENCODER_NF=1`` fuses the encoder LayerNorm + QKV and the out-proj.

    The encoder self-attention then runs ``NF.qkv_proj`` (LayerNorm, the fused
    q/k/v weight and bias), the NKI flash attention, and ``NF.o_proj``, the same
    path as the Qwen3-ASR audio encoder. It needs ``WHISPER_ENCODER_FLASH=1``.
    The values are not byte-identical to the separate ops: gate it on WER.
    """
    return os.environ.get("WHISPER_ENCODER_NF", "0") == "1"


def encoder_flash_enabled() -> bool:
    """Return True when ``WHISPER_ENCODER_FLASH=1`` selects NKI flash attention in the encoder."""
    return os.environ.get("WHISPER_ENCODER_FLASH", "0") == "1"


def fused_decode_enabled() -> bool:
    """Return True when ``WHISPER_FUSED_DECODE=1`` selects the NKI decode attention.

    Whisper is MHA, so the kernel folds several KV heads per rank into the
    batch. The decode mask must have the folded layout. Without that fold the
    kernel output is wrong, and nothing fails, so check that it is present.

    Raises:
        RuntimeError: If the fused decode is selected and the GQA mask fold is
            not in ``attention_decode``.
    """
    if os.environ.get("WHISPER_FUSED_DECODE", "0") != "1":
        return False
    from vllm_neuron.functional.attention import attention_decode as _ad

    if not hasattr(_ad, "_fold_gqa_block_kv_mask"):
        raise RuntimeError(
            "WHISPER_FUSED_DECODE=1 needs the GQA decode-mask fold in "
            "attention_decode (_fold_gqa_block_kv_mask). This tree does not have "
            "it. Unset WHISPER_FUSED_DECODE, or use a tree with the fold."
        )
    return True

# Whisper always pads or cuts the audio to a 30 s window: 3000 mel frames, then
# 1500 encoder positions after the stride-2 convolution.
def fused_decode_max_rows() -> int:
    """Return the largest decode batch that uses the fused decode kernel.

    ``WHISPER_FUSED_DECODE_MAX_B`` (default: no limit). The kernel folds the
    5 MHA heads of a rank into the batch, so its cost grows with the rows.
    The torch path is faster from about 64 rows at TP4 (Task 013).
    """
    return int(os.environ.get("WHISPER_FUSED_DECODE_MAX_B", "1000000"))


def in_out_parts() -> set[str]:
    """Return the parts whose linear weights are stored as ``[in, out]`` (``WHISPER_IN_OUT``).

    ``WHISPER_IN_OUT`` is a comma list of ``decoder`` (decoder MLP, self-attention
    q/k/v/out and cross-attention q/out), ``lm_head`` and ``encoder``. The default is
    ``decoder,lm_head``. ``none`` stores all weights as ``[out, in]``.

    With ``[out, in]`` weights, the compiler adds one PE transpose for each 128 x 128
    weight tile on every decode step (Task 022: 30.4 k transposes, 2.6 ms at B=16).
    ``[in, out]`` weights give the same values: the matmul reads the same products
    in the same order.
    """
    raw = os.environ.get("WHISPER_IN_OUT", "decoder,lm_head")
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    unknown = parts - {"decoder", "lm_head", "encoder", "none"}
    if unknown:
        raise ValueError(
            f"WHISPER_IN_OUT has unknown parts {sorted(unknown)}. "
            "Use a comma list of decoder, lm_head and encoder, or none."
        )
    return parts - {"none"}


def self_kv_mode() -> str:
    """Return ``WHISPER_SELF_KV``: where the decoder self-attention keeps its K/V.

    * ``paged`` (default): the block-managed paged KV cache of the runner.
    * ``slot``: model-owned contiguous K/V, one ``[heads, 448, d]`` region for each
      request slot (the same slots as the cross-KV). Decode reads the first ``ctx``
      positions of the slot of each row, where ``ctx`` is the decode-context bucket
      (the block-table width). Each read is a large contiguous DMA, not a paged gather.

    Raises:
        ValueError: If the value is not ``paged`` or ``slot``.
    """
    mode = os.environ.get("WHISPER_SELF_KV", "paged")
    if mode not in ("paged", "slot"):
        raise ValueError(f"WHISPER_SELF_KV={mode!r}: use paged or slot.")
    return mode


def slot_ctx(max_target_positions: int) -> int:
    """Return ``WHISPER_SLOT_CTX``: the positions of each slot of the slot self-KV (default 448).

    A shorter slot saves HBM (``2 x layers x heads x d x 2`` bytes per position and
    slot), which leaves room for more slots. Every position that decode writes must
    fit, so the server ``max_model_len`` must not be larger.

    Raises:
        ValueError: If the value is not a multiple of 128 up to ``max_target_positions``,
            or if the vLLM ``max_model_len`` is larger.
    """
    from .config import current_max_model_len

    ctx = int(os.environ.get("WHISPER_SLOT_CTX", str(max_target_positions)))
    if ctx != max_target_positions and (ctx % 128 != 0 or ctx > max_target_positions):
        raise ValueError(
            f"WHISPER_SLOT_CTX={ctx}: use a multiple of 128 up to {max_target_positions}."
        )
    mml = current_max_model_len()
    if mml is not None and mml > ctx:
        raise ValueError(
            f"WHISPER_SLOT_CTX={ctx} is below max_model_len={mml}: a decode position "
            "would not fit in its slot. Set --max-model-len to at most WHISPER_SLOT_CTX."
        )
    return ctx


def self_kv_write() -> str:
    """Return ``WHISPER_SELF_WRITE``: how slot mode writes the new K/V (``index`` or ``where``).

    * ``index`` (default): ``index_put_`` of one row for each token and head.
    * ``where``: a one-hot ``torch.where`` over the whole buffer (intern-whisper style).
      It reads and writes the full buffer on each step.
    """
    mode = os.environ.get("WHISPER_SELF_WRITE", "index")
    if mode not in ("index", "where"):
        raise ValueError(f"WHISPER_SELF_WRITE={mode!r}: use index or where.")
    return mode


def to_in_out(linear: nn.Module) -> None:
    """Store the weight of a ``ColumnParallelLinear`` as ``[in, out]``.

    ``ColumnParallelLinear`` then runs ``x @ W + b`` instead of ``F.linear``.
    Call this method after the weights load.
    """
    if getattr(linear, "weight_layout", "out_in") == "in_out":
        return
    with torch.no_grad():
        linear.weight = nn.Parameter(linear.weight.t().contiguous(), requires_grad=False)
    linear.weight_layout = "in_out"


WHISPER_ENCODER_TOKENS = 1500
# The kernel cross-attention mode keeps the cross-KV of a slot in
# CROSS_BLOCKS_PER_SLOT blocks of CROSS_BLOCK tokens: 1536 positions, the 1500
# encoder positions and then pad. WHISPER_CROSS_BLOCK sets the block length.
# The decode kernel needs a multiple of 128 blocks for each row, and shrinks a
# larger block until it gets one (128-token blocks become 4-token blocks), so
# 12-token blocks (128 for each slot) keep each DMA at 12 rows.
CROSS_CTX = 1536
CROSS_BLOCK = int(os.environ.get("WHISPER_CROSS_BLOCK", "128"))
CROSS_BLOCKS_PER_SLOT = CROSS_CTX // CROSS_BLOCK


def cross_k_transposed() -> bool:
    """Return True when ``WHISPER_CROSS_KT=1`` stores the cross-K as ``[.., d, src]``.

    With d on the partition axis, the score matmul reads K as it is, with no
    transpose of each slot and head on every decode step.
    """
    return os.environ.get("WHISPER_CROSS_KT", "0") == "1"


def mqa_group(heads: int) -> int:
    """Return the heads in one MQA cross-attention call (``WHISPER_CROSS_MQA_GROUP``, default 5).

    One call spans ``group x 1536`` positions, padded by ``mqa_span``. Without the
    pad, the decode kernel's flash tiling fails above 8192 positions.
    """
    g = min(int(os.environ.get("WHISPER_CROSS_MQA_GROUP", "5")), heads)
    if heads % g != 0:
        raise ValueError(f"WHISPER_CROSS_MQA_GROUP={g} must divide the {heads} heads of a rank.")
    return g


def mqa_span(group: int) -> int:
    """Return the positions of one MQA call: ``group x 1536``, padded to whole flash tiles.

    ``WHISPER_CROSS_MQA_PAD=1`` (default) pads a span above 8192 positions to a
    multiple of 8192. The decode kernel's flash tiling fails on a partial last
    tile (10 heads: 15360 positions), and runs a padded span (16384) correctly
    (Task 025). The pad positions are masked. ``WHISPER_CROSS_MQA_PAD=0`` keeps the
    unpadded span.
    """
    span = group * CROSS_CTX
    if os.environ.get("WHISPER_CROSS_MQA_PAD", "1") == "1" and span > 8192:
        span = -(-span // 8192) * 8192
    return span


def cross_mode() -> str:
    """Return ``WHISPER_CROSS_MODE``: how a batched decode step reads the cross-KV.

    * ``mask`` (default): attention over all slots, -inf for the other slots.
    * ``perm``: a one-hot matmul moves the query of each row to the row of its
      slot, attention runs once for each slot over its own cross-KV, and a
      second one-hot matmul moves the result back. Both moves are exact.
    * ``kernel``: the NKI ``attention_decode`` kernel over the cross-KV in
      blocks, with a per-head block table for each row (MHA, heads folded).
    * ``mqa``: the same kernel with one KV head: the heads of a slot are laid
      end to end (head-major, 1536 positions each), and a per-head mask keeps
      each q head on its own span. The kernel then runs B rows, not B * heads.
    * ``xkernel``: the Whisper cross-attention decode kernel
      (``vllm_neuron/_kernels/whisper``). Each slot keeps K^T ``[h, d, 1536]`` and
      V ``[h, 1536, d]``; the kernel reads the K and V of each row in large DMAs
      and runs the 1500-position softmax in fp32 (Task 034).
    * ``onehot`` / ``gather``: older modes, kept for comparison.
    """
    return os.environ.get("WHISPER_CROSS_MODE", "mask")


def cross_kv_fused() -> bool:
    """Return True when ``WHISPER_CROSS_KV_FUSED=1``: one GEMM projects the cross K|V of all layers.

    The encoder graph then runs one ``[T, H] x [H, layers * 2 * H_local]`` matmul
    instead of ``2 x layers`` matmuls of ``[H, H_local]`` (Task 033). The weights are
    a concatenated copy, ``[H, layers * 2 * H_local]`` (about 210 MB at TP1).
    """
    return os.environ.get("WHISPER_CROSS_KV_FUSED", "0") == "1"


def xkernel_dma() -> tuple[int, int, int, int]:
    """Return ``WHISPER_XKERNEL_DMA=KP,VH[,VB[,HW]]`` (default ``5,10,5,0``).

    KP head pairs per K DMA, VH heads per V DMA, VB units per blocked-PV matmul
    (0: one-column PV), HW 1 for hardware DGE on the K and V loads.
    """
    parts = [int(x) for x in os.environ.get("WHISPER_XKERNEL_DMA", "5,10,5,0").split(",")]
    parts += [5, 0][len(parts) - 2:] if len(parts) < 4 else []
    return parts[0], parts[1], parts[2], parts[3]
WHISPER_MEL_FRAMES = 3000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def gelu(x):
    """Return the exact (erf) GELU, written out so that Dynamo can trace it."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    """Return the tanh-form GELU, written as ``x * sigmoid(2z)``.

    It lowers to one ScalarE sigmoid instead of the multi-op erf. It is the
    GELU approximation that HF calls ``gelu_pytorch_tanh``, so it is not
    byte-identical to the erf GELU: gate it on WER.
    """
    return x * torch.sigmoid(1.5957691216057308 * (x + 0.044715 * x * x * x))


def gelu_for(part: str):
    """Return the GELU of ``part`` ("encoder" or "decoder").

    ``WHISPER_GELU`` is a comma list of the parts that use the tanh form, for
    example ``encoder`` or ``encoder,decoder``. The default is the erf GELU.
    """
    parts = os.environ.get("WHISPER_GELU", "").split(",")
    return gelu_tanh if part in parts else gelu


def sinusoids(length: int, channels: int, max_timescale: float = 10000.0):
    """Return the OpenAI Whisper sinusoidal position embeddings (encoder)."""
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def tp_info() -> tuple[object | None, int, int]:
    """Return ``(device_group, world_size, rank)`` of the text TP group.

    Return ``(None, 1, 0)`` when the vLLM parallel state is not initialized,
    for example in a CPU unit test.
    """
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
    except (AssertionError, RuntimeError, ImportError):
        return None, 1, 0
    return group.device_group, group.world_size, group.rank_in_group


class RowParallelLinearFP32Bias(nn.Module):
    """A RowParallelLinear without bias, then an fp32 bias add.

    The add occurs after the all-reduce. Cast the bias to the activation dtype
    and add in that dtype. This order matches the byte-identical reference. An
    fp32 add gives a different bf16 rounding, which can change a near-tie at the
    first token.
    """

    def __init__(self, in_features, out_features, dtype, tp_group=None):
        super().__init__()
        self.rpl = RowParallelLinear(
            in_features, out_features, bias=False, dtype=dtype, tp_group=tp_group
        )
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self.in_out = False

    @torch.no_grad()
    def set_in_out(self) -> None:
        """Store the weight as ``[in_local, out]`` and run ``x @ W``, then the all-reduce."""
        if self.in_out:
            return
        self.rpl.weight = nn.Parameter(self.rpl.weight.t().contiguous(), requires_grad=False)
        self.in_out = True

    def forward(self, x):
        if self.in_out:
            y = torch.matmul(x, self.rpl.weight)
            if self.rpl.tp_size > 1:
                from torch.distributed._functional_collectives import all_reduce

                y = all_reduce(y, reduceOp="sum", group=self.rpl.tp_group)
        else:
            y = self.rpl(x)
        return y + self.bias.to(y.dtype)


def _cpl(in_f, out_f, bias, dtype, tp_group):
    return ColumnParallelLinear(
        in_f, out_f, bias=bias, gather_output=False, dtype=dtype, tp_group=tp_group
    )


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class EncoderSelfAttention(nn.Module):
    """Non-causal encoder self-attention. The softmax is in fp32."""

    def __init__(self, embed_dim, num_heads, dtype, tp_group, tp):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        if num_heads % tp != 0:
            raise ValueError(f"Encoder heads {num_heads} are not divisible by TP {tp}.")
        self.num_heads = num_heads // tp
        self.q_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.k_proj = _cpl(embed_dim, embed_dim, False, dtype, tp_group)
        self.v_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype, tp_group)
        # Read the flags one time: a read in forward adds an os.environ guard.
        self.flash = encoder_flash_enabled()
        self.in_out = "encoder" in in_out_parts()
        self.nf = encoder_nf_enabled()
        if self.nf and not self.flash:
            raise ValueError("WHISPER_ENCODER_NF=1 needs WHISPER_ENCODER_FLASH=1.")
        if self.nf:
            d_local = self.num_heads * self.head_dim
            self.register_buffer(
                "qkv_weight_nf", torch.zeros(embed_dim, 3 * d_local, dtype=dtype), persistent=False
            )
            self.register_buffer(
                "qkv_bias_nf", torch.zeros(1, 3 * d_local, dtype=dtype), persistent=False
            )
            self.register_buffer(
                "out_weight_nf", torch.zeros(d_local, embed_dim, dtype=dtype), persistent=False
            )

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Store q/k/v/out as ``[in, out]`` when ``encoder`` is in ``WHISPER_IN_OUT``.

        With ``WHISPER_ENCODER_NF=1``, build the fused ``[H, 3 * d_local]`` q/k/v
        weight (q scale folded in, zero k bias) and the ``[d_local, H]`` out weight.
        """
        if self.nf:
            wq = self.q_proj.weight * self.scaling
            self.qkv_weight_nf.copy_(
                torch.cat([wq, self.k_proj.weight, self.v_proj.weight], dim=0).t().contiguous()
            )
            bq = self.q_proj.bias * self.scaling
            self.qkv_bias_nf.copy_(
                torch.cat([bq, torch.zeros_like(bq), self.v_proj.bias]).view(1, -1)
            )
            self.out_weight_nf.copy_(self.out_proj.rpl.weight.t().contiguous())
        if not self.in_out:
            return
        for lin in (self.q_proj, self.k_proj, self.v_proj):
            to_in_out(lin)
        self.out_proj.set_in_out()

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states):
        bsz, seqlen, _ = hidden_states.shape
        if self.flash:
            return self._forward_flash(hidden_states)
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.num_heads * self.head_dim)
        return self.out_proj(out)


    def forward_nf(self, x, norm):
        """LayerNorm + fused QKV (``NF.qkv_proj``), flash attention, and ``NF.o_proj``.

        Args:
            x: The residual stream ``[b, s, H]`` (before the LayerNorm).
            norm: The ``self_attn_layer_norm`` of the layer.
        """
        from vllm_neuron import functional as NF
        from nkilib.core.utils.common_types import NormType

        bsz, seqlen, _ = x.shape
        h, d = self.num_heads, self.head_dim
        qkv = NF.qkv_proj(
            hidden=x,
            qkv_weights=self.qkv_weight_nf,
            norm_weights=norm.weight.unsqueeze(0),
            eps=norm.eps,
            norm_type=NormType.LAYER_NORM,
            bias=self.qkv_bias_nf,
            norm_b=norm.bias.unsqueeze(0),
        )
        hd = h * d
        q, k, v = (
            qkv[:, :, i * hd : (i + 1) * hd]
            .reshape(bsz, seqlen, h, d)
            .permute(0, 2, 1, 3)
            .reshape(bsz * h, seqlen, d)
            for i in range(3)
        )
        out = NF.flash_attention(
            q, k, v, scale=1.0, causal_mask=False, tp_q=True, tp_k=True, tp_out=True,
        )  # [b * h, d, s]
        y = NF.o_proj(out.view(bsz, h, d, seqlen), self.out_weight_nf)  # [b, s, H]
        if self.out_proj.rpl.tp_size > 1:
            from torch.distributed._functional_collectives import all_reduce

            y = all_reduce(y, reduceOp="sum", group=self.out_proj.rpl.tp_group)
        return y + self.out_proj.bias.to(y.dtype)

    def _forward_flash(self, hidden_states):
        """Run the encoder attention with the NKI flash attention kernel (non-causal)."""
        from vllm_neuron import functional as NF

        bsz, seqlen, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, seqlen, bsz)
        k = self._shape(self.k_proj(hidden_states), seqlen, bsz)
        v = self._shape(self.v_proj(hidden_states), seqlen, bsz)
        bh = bsz * self.num_heads
        out = NF.flash_attention(
            q.reshape(bh, seqlen, self.head_dim),
            k.reshape(bh, seqlen, self.head_dim),
            v.reshape(bh, seqlen, self.head_dim),
            scale=1.0,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            tp_out=False,
        )
        out = out.view(bsz, self.num_heads, seqlen, self.head_dim).transpose(1, 2)
        return self.out_proj(out.reshape(bsz, seqlen, self.num_heads * self.head_dim))


FP8_MAX = 240.0  # trn2 nkilib float8_e4m3; a value <= 240 has the same bits in float8_e4m3fn


def fp8_parts() -> set[str]:
    """Return ``WHISPER_FP8``: the parts with fp8 weights (``decoder_mlp`` or empty).

    ``decoder_mlp``: the decoder fc1 / fc2 weights in float8 e4m3 with one fp32 scale
    per output channel (``absmax / 240``), quantized at load. Decode runs the nkilib
    MLP kernel (``QuantizationType.ROW``, weight-only: the activations stay bf16).
    Prefill dequantizes in torch. This is a precision change: gate it on WER.
    """
    raw = os.environ.get("WHISPER_FP8", "")
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    unknown = parts - {"decoder_mlp"}
    if unknown:
        raise ValueError(f"WHISPER_FP8 has unknown parts {sorted(unknown)}. Use decoder_mlp.")
    return parts


def quantize_per_out_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fp8 [K, N], fp32 scale [1, N])`` for a ``[K, N]`` (contraction-major) weight."""
    wf = w.to(torch.float32)
    scale = (wf.abs().amax(dim=0, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


class MLP(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dtype, tp_group, part="encoder"):
        super().__init__()
        self.fc1 = _cpl(embed_dim, ffn_dim, True, dtype, tp_group)
        self.fc2 = RowParallelLinearFP32Bias(ffn_dim, embed_dim, dtype, tp_group)
        self.act = gelu_for(part)
        # Read the flag one time: a read in forward adds an os.environ guard.
        self.in_out = part in in_out_parts()
        self.fp8 = part == "decoder" and "decoder_mlp" in fp8_parts()
        self.dtype = dtype
        if self.fp8:
            i_local = self.fc1.weight.shape[0]
            # The fp8 weights replace fc1.weight / fc2.rpl.weight after loading.
            self.register_buffer("w1_q", torch.zeros(embed_dim, i_local, dtype=torch.float8_e4m3fn), persistent=False)
            self.register_buffer("w1_s", torch.zeros(128, i_local, dtype=torch.float32), persistent=False)
            self.register_buffer("w2_q", torch.zeros(i_local, embed_dim, dtype=torch.float8_e4m3fn), persistent=False)
            self.register_buffer("w2_s", torch.zeros(128, embed_dim, dtype=torch.float32), persistent=False)
            self.register_buffer("b1", torch.zeros(1, i_local, dtype=dtype), persistent=False)
            self.register_buffer("b0", torch.zeros(1, i_local, dtype=dtype), persistent=False)

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Store the fc1 / fc2 weights as ``[in, out]`` (``WHISPER_IN_OUT``), or quantize them (``WHISPER_FP8``)."""
        if self.fp8:
            w1 = self.fc1.weight if getattr(self.fc1, "weight_layout", "out_in") == "in_out" else self.fc1.weight.t()
            w2 = self.fc2.rpl.weight if self.fc2.in_out else self.fc2.rpl.weight.t()
            q1, s1 = quantize_per_out_channel(w1)
            q2, s2 = quantize_per_out_channel(w2)
            self.w1_q.copy_(q1)
            self.w1_s.copy_(s1.expand(128, -1))
            self.w2_q.copy_(q2)
            self.w2_s.copy_(s2.expand(128, -1))
            self.b1.copy_(self.fc1.bias.view(1, -1).to(self.b1.dtype))
            # Free the bf16 copies: an empty weight keeps the module tree loadable.
            self.fc1.weight = nn.Parameter(torch.empty(0, dtype=self.dtype), requires_grad=False)
            self.fc2.rpl.weight = nn.Parameter(torch.empty(0, dtype=self.dtype), requires_grad=False)
            return
        if not self.in_out:
            return
        to_in_out(self.fc1)
        self.fc2.set_in_out()

    def _dequant(self, q, s):
        return (q.to(torch.float32) * s[0:1]).to(self.dtype)

    def forward(self, x, is_prefill=True):
        if not self.fp8:
            return self.fc2(self.act(self.fc1(x)))
        from vllm_neuron.utils.neuron_utils import can_run_kernel

        if is_prefill or not can_run_kernel(x):
            h = self.act(torch.matmul(x, self._dequant(self.w1_q, self.w1_s)) + self.b1)
            y = torch.matmul(h, self._dequant(self.w2_q, self.w2_s))
        else:
            from nkilib.core.utils.common_types import ActFnType, QuantizationType

            from vllm_neuron import functional as NF

            # skip_gate: GELU(x W1 + b1) W2. b0 is a zero gate bias: nkilib takes the
            # bias tile dtype from the gate bias (it fails with an up bias only).
            y = NF.mlp(
                x, self.w1_q, self.w1_q, self.w2_q, skip_gate=True, act_fn=ActFnType.GELU,
                gate_b=self.b0, up_b=self.b1, quantization_type=QuantizationType.ROW,
                gate_w_scale=self.w1_s, up_w_scale=self.w1_s, down_w_scale=self.w2_s,
                output_dtype="bfloat16",
            )
        if self.fc2.rpl.tp_size > 1:
            from torch.distributed._functional_collectives import all_reduce

            y = all_reduce(y, reduceOp="sum", group=self.fc2.rpl.tp_group)
        return y + self.fc2.bias.to(y.dtype)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.self_attn = EncoderSelfAttention(
            d, cfg.encoder_attention_heads, dtype, tp_group, tp
        )
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.encoder_ffn_dim, dtype, tp_group)
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, x):
        if self.self_attn.nf:
            x = x + self.self_attn.forward_nf(x, self.self_attn_layer_norm)
        else:
            x = x + self.self_attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


class WhisperEncoder(nn.Module):
    """Audio encoder: mel ``[b, 128, 3000]`` to hidden ``[b, 1500, d]``."""

    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, kernel_size=3, padding=1, dtype=dtype)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, dtype, tp_group, tp) for _ in range(cfg.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)

    def forward(self, input_features):
        act = gelu_for("encoder")
        x = act(self.conv1(input_features))
        x = act(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1], :]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class WhisperCrossKVEncoder(nn.Module):
    """The ``audio_tower`` graph: the encoder plus the cross-KV projection.

    The runner compiles this module as a separate graph. ``forward`` runs the
    encoder, projects K/V for each decoder layer, and writes them into the
    ``cross_k_{i}`` / ``cross_v_{i}`` buffers in place. The buffers belong to
    this module, so the aliasing FX pass adds an input-output alias for each
    write. The decode graph reads the same tensors.

    Each buffer is ``[num_slots, heads_local, 1500, head_dim]``. Clip ``j``
    goes to slot ``write_block_ids[j]``. The warmup of the worker sends
    ``write_block_ids`` as zeros, so warmup writes slot 0. Whisper does not use
    ``encoder_cache_buffer``.
    """

    def __init__(
        self,
        encoder: WhisperEncoder,
        cross_attns: nn.ModuleList,
        n_layers: int,
        n_heads_local: int,
        head_dim: int,
        n_ctx: int,
        dtype,
        num_slots: int = 2,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.encoder = encoder
        # The same module objects as the decoder cross-attention layers. The
        # weights load under the decoder names. There is no copy.
        self.cross_attns = cross_attns
        self.n_layers = n_layers
        # The kernel mode stores each slot as CROSS_BLOCKS_PER_SLOT blocks of
        # [heads, CROSS_BLOCK, d], the same 4D layout as the paged self-KV.
        self.mqa = cross_mode() == "mqa"
        self.blocked = cross_mode() in ("kernel", "mqa")
        # xkernel: K^T [slots, h, d, 1536] and V [slots, h, 1536, d], zero pad.
        self.xkernel = cross_mode() == "xkernel"
        self.kt = (cross_k_transposed() and not self.blocked) or self.xkernel
        self.n_heads_local = n_heads_local
        if self.mqa:
            # One KV head: each slot holds `groups` spans of g x 1536 positions,
            # each span in 128 blocks of g x 12 positions.
            self.mqa_g = mqa_group(n_heads_local)
            self.mqa_groups = n_heads_local // self.mqa_g
            self.mqa_span = mqa_span(self.mqa_g)
            self.mqa_block = self.mqa_span // 128
            shape = (num_slots * self.mqa_groups * 128, 1, self.mqa_block, head_dim)
        elif self.blocked:
            shape = (num_slots * CROSS_BLOCKS_PER_SLOT, n_heads_local, CROSS_BLOCK, head_dim)
        elif self.xkernel:
            from vllm_neuron._kernels.whisper.cross_attention_decode import xkernel_heads

            # The kernel pairs heads: an odd head count (TP4: 5) gets one zero head.
            n_ctx = CROSS_CTX
            self.xk_heads = xkernel_heads(n_heads_local)
            shape = (num_slots, self.xk_heads, n_ctx, head_dim)
        else:
            shape = (num_slots, n_heads_local, n_ctx, head_dim)
        k_shape = (num_slots, shape[1], head_dim, n_ctx) if self.kt else shape
        for i in range(n_layers):
            for kind, shp in (("k", k_shape), ("v", shape)):
                self.register_buffer(
                    f"cross_{kind}_{i}", torch.zeros(shp, dtype=dtype), persistent=False
                )
        self.kv_fused = cross_kv_fused()
        if self.kv_fused:
            hidden = encoder.conv2.out_channels
            local = n_heads_local * head_dim
            self.register_buffer(
                "kv_weight_fused", torch.zeros(hidden, n_layers * 2 * local, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "kv_bias_fused", torch.zeros(n_layers * 2 * local, dtype=dtype), persistent=False
            )

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Build the fused cross K|V weight (``WHISPER_CROSS_KV_FUSED=1``) from the loaded weights.

        Columns ``[2 i L, (2 i + 1) L)`` are the k_proj of layer i and the next L
        columns its v_proj (``L = H_local``). k_proj has no bias.
        """
        if not self.kv_fused:
            return
        local = self.n_heads_local * self.cross_attns[0].head_dim
        for i, ca in enumerate(self.cross_attns):
            for j, lin in enumerate((ca.k_proj, ca.v_proj)):
                w = lin.weight if getattr(lin, "weight_layout", "out_in") == "in_out" else lin.weight.t()
                cols = slice((2 * i + j) * local, (2 * i + j + 1) * local)
                self.kv_weight_fused[:, cols].copy_(w)
                if lin.bias is not None:
                    self.kv_bias_fused[cols].copy_(lin.bias)
                else:
                    self.kv_bias_fused[cols].zero_()

    def forward(self, input_features, encoder_cache_buffer=None, write_block_ids=None):
        encoder_hidden = self.encoder(input_features)
        if write_block_ids is None:
            write_block_ids = torch.zeros(
                input_features.shape[0], dtype=torch.int64, device=input_features.device
            )
        slots = write_block_ids.to(torch.int64)
        if self.blocked:
            nb = 128 * self.mqa_groups if self.mqa else CROSS_BLOCKS_PER_SLOT
            rows = (
                slots.view(-1, 1) * nb + torch.arange(nb, device=slots.device, dtype=slots.dtype)
            ).view(-1)
        if self.kv_fused:
            n, t, _ = encoder_hidden.shape
            hl, d = self.n_heads_local, self.cross_attns[0].head_dim
            kv_all = torch.matmul(encoder_hidden, self.kv_weight_fused) + self.kv_bias_fused
            # [n, t, layers, 2, h, d] -> [layers, 2, n, h, t, d]
            kv_all = kv_all.view(n, t, self.n_layers, 2, hl, d).permute(2, 3, 0, 4, 1, 5)
        for i in range(self.n_layers):
            if self.kv_fused:
                k, v = kv_all[i, 0], kv_all[i, 1]
            else:
                k, v = self.cross_attns[i].project_kv(encoder_hidden)
            if self.mqa:
                k, v = self._to_mqa_blocks(k), self._to_mqa_blocks(v)
                getattr(self, f"cross_k_{i}").index_put_((rows,), k)
                getattr(self, f"cross_v_{i}").index_put_((rows,), v)
            elif self.blocked:
                k, v = self._to_blocks(k), self._to_blocks(v)
                getattr(self, f"cross_k_{i}").index_put_((rows,), k)
                getattr(self, f"cross_v_{i}").index_put_((rows,), v)
            else:
                if self.xkernel:
                    pad = CROSS_CTX - k.shape[2]
                    hpad = self.xk_heads - k.shape[1]
                    k, v = F.pad(k, (0, 0, 0, pad, 0, hpad)), F.pad(v, (0, 0, 0, pad, 0, hpad))
                if self.kt:
                    k = k.transpose(-1, -2).contiguous()
                getattr(self, f"cross_k_{i}").index_put_((slots,), k)
                getattr(self, f"cross_v_{i}").index_put_((slots,), v)
        # Return a tensor, so that the warmup trace has an output.
        return encoder_hidden

    def _to_mqa_blocks(self, x):
        """``[n, h, 1500, d]`` to ``[n * groups * 128, 1, span / 128, d]``, head-major.

        Each group holds its g heads end to end (1536 positions each), then zero
        pad to the span.
        """
        n, h, t, d = x.shape
        x = F.pad(x, (0, 0, 0, CROSS_CTX - t))  # [n, h, 1536, d]
        x = x.reshape(n, self.mqa_groups, self.mqa_g * CROSS_CTX, d)
        x = F.pad(x, (0, 0, 0, self.mqa_span - self.mqa_g * CROSS_CTX))
        return x.reshape(n * self.mqa_groups * 128, 1, self.mqa_block, d)

    @staticmethod
    def _to_blocks(x):
        """``[n, h, 1500, d]`` to ``[n * 12, h, 128, d]``, zero-padded to 1536 positions."""
        n, h, t, d = x.shape
        pad = CROSS_BLOCK * CROSS_BLOCKS_PER_SLOT - t
        x = F.pad(x, (0, 0, 0, pad))
        x = x.view(n, h, CROSS_BLOCKS_PER_SLOT, CROSS_BLOCK, d).permute(0, 2, 1, 3, 4)
        return x.reshape(n * CROSS_BLOCKS_PER_SLOT, h, CROSS_BLOCK, d)


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class CrossAttention(nn.Module):
    """Decoder cross-attention. Q comes from the decoder. K/V come from the buffers."""

    def __init__(self, embed_dim, num_heads, dtype, tp_group, tp):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_heads = num_heads // tp
        self.q_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.k_proj = _cpl(embed_dim, embed_dim, False, dtype, tp_group)
        self.v_proj = _cpl(embed_dim, embed_dim, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(embed_dim, embed_dim, dtype, tp_group)
        self.tp_group = tp_group
        self.tp = tp
        self.xkernel = cross_mode() == "xkernel"
        # xkernel keeps K^T, so its prefill (attn_cached) reads K^T too.
        self.kt = (cross_k_transposed() and cross_mode() != "kernel") or self.xkernel
        self.xkernel_dma = xkernel_dma() if self.xkernel else None
        # Kernel mode: the kernel projects Q, K and V of the decoder token. Only
        # Q is used, so the K and V columns are zero. The active key is masked.
        self.kernel = cross_mode() in ("kernel", "mqa")
        if self.kernel:
            d_local = self.num_heads * self.head_dim
            # MHA: q, k, v columns for every head. MQA: q for every head and one
            # k and one v head (zero: the active key is masked).
            self.mqa = cross_mode() == "mqa"
            self.mqa_g = mqa_group(self.num_heads) if self.mqa else self.num_heads
            self.mqa_groups = self.num_heads // self.mqa_g
            gd = self.mqa_g * self.head_dim
            kv_cols = 2 * self.head_dim if self.mqa else 2 * d_local
            self.register_buffer(
                "qkv_weight_fused",
                torch.zeros(self.mqa_groups, embed_dim, gd + kv_cols, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "qkv_bias_fused", torch.zeros(self.mqa_groups, 1, gd + kv_cols, dtype=dtype),
                persistent=False,
            )
            self.register_buffer(
                "out_weight_fused", torch.zeros(self.mqa_groups, gd, embed_dim, dtype=dtype),
                persistent=False,
            )

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Build the kernel-mode weights: the scaled q weight and bias, zero K/V columns.

        With MQA head groups, group ``i`` gets the q columns and the out_proj
        rows of its heads.
        """
        self._build_kernel_weights()
        parts = in_out_parts()
        if "decoder" in parts:
            to_in_out(self.q_proj)
            self.out_proj.set_in_out()
        if "encoder" in parts:
            # k_proj / v_proj run in the encoder graph (the cross-KV projection).
            to_in_out(self.k_proj)
            to_in_out(self.v_proj)

    def _build_kernel_weights(self) -> None:
        if not self.kernel:
            return
        gd = self.mqa_g * self.head_dim
        wq = (self.q_proj.weight * self.scaling).t()  # [H, d_local]
        bq = self.q_proj.bias * self.scaling
        wo = self.out_proj.rpl.weight.t()  # [d_local, H]
        self.qkv_weight_fused.zero_()
        self.qkv_bias_fused.zero_()
        for i in range(self.mqa_groups):
            cols = slice(i * gd, (i + 1) * gd)
            self.qkv_weight_fused[i, :, :gd].copy_(wq[:, cols])
            self.qkv_bias_fused[i, 0, :gd].copy_(bq[cols])
            self.out_weight_fused[i].copy_(wo[cols].contiguous())

    def attn_kernel(self, hidden_states, cross_k, cross_v, table, mask):
        """Cross-attention with the NKI decode kernel over the blocked cross-KV.

        Args:
            hidden_states: ``[B, 1, H]``, after the LayerNorm.
            cross_k, cross_v: ``[slots * 12, h, 128, d]``.
            table: ``[B, h, 12]`` int32 per-head block table.
            mask: ``[1536, B, h, 1]`` block-KV mask: the 1500 encoder positions
                are valid; the pad and the active key are not.
        """
        from vllm_neuron import functional as NF

        B, S, hidden = hidden_states.shape
        out = None
        for i in range(self.mqa_groups):
            # MQA groups: group i reads blocks [i * 128, (i + 1) * 128) of each slot.
            tbl = table if i == 0 else table + i * 128
            o = NF.attention_decode(
                X=hidden_states,
                X_hidden_dim_actual=hidden,
                rmsnorm_X_enabled=False,
                W_qkv=self.qkv_weight_fused[i],
                bias_qkv=self.qkv_bias_fused[i],
                K_cache_transposed=False,
                active_blocks_table=tbl,
                K_cache=cross_k,
                V_cache=cross_v,
                attention_mask=mask,
                softmax_scale=1.0,
                update_cache=False,
                W_out=self.out_weight_fused[i],
                bias_out=None,
                transposed_out=False,
            )
            if isinstance(o, (tuple, list)):
                o = o[0]
            out = o if out is None else out + o
        out = out.reshape(B * S, hidden)
        if self.tp > 1:
            from torch.distributed._functional_collectives import all_reduce

            out = all_reduce(out, reduceOp="sum", group=self.tp_group)
        return (out + self.out_proj.bias.to(out.dtype)).view(B, S, hidden)

    def attn_xkernel(self, hc, cross_k, cross_v, slots, qmask, group):
        """Cross-attention of B decode rows with the Whisper cross-attention kernel.

        Args:
            hc: ``[B, H]`` after the LayerNorm.
            cross_k: ``[S, h, d, 1536]`` K^T of all slots.
            cross_v: ``[S, h, 1536, d]`` V of all slots.
            slots: ``[B]`` int32, the slot of each row.
            qmask: the kernel head-pair mask (see ``xkernel_qmask``).
            group: rows per kernel group.
        """
        from vllm_neuron.functional.whisper_cross_attention import (
            whisper_cross_attention_decode,
        )

        B = hc.shape[0]
        h, d = self.num_heads, self.head_dim
        q = (self.q_proj(hc) * self.scaling).view(B, h, d)
        hk = cross_k.shape[1]
        if hk != h:
            # Odd heads per rank: the zero pad head of the buffers gets a zero q.
            q = F.pad(q, (0, 0, 0, hk - h))
        kp, vh, vb, hw = self.xkernel_dma
        ctx = whisper_cross_attention_decode(
            q, cross_k, cross_v, slots, qmask, group, k_pairs=kp, v_heads=vh, pv_block=vb,
            hw_dge=hw,
        )
        return self.out_proj(ctx[:, :h].reshape(B, h * d))

    def _shape(self, x, seqlen, bsz):
        return x.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, encoder_hidden_states):
        """Project the encoder output to cross K and V, ``[b, h, src, d]``."""
        bsz, src_len, _ = encoder_hidden_states.shape
        k = self._shape(self.k_proj(encoder_hidden_states), src_len, bsz)
        v = self._shape(self.v_proj(encoder_hidden_states), src_len, bsz)
        return k, v

    def attn_all_slots(self, hidden_states, cross_k, cross_v, slots):
        """Cross-attention of each row over its own slot, without a gather.

        ``cross_k`` and ``cross_v`` hold all slots, ``[S, h, src, d]``. The
        scores of row ``b`` for every slot other than ``slots[b]`` get -inf,
        so their softmax weights are exactly zero. A gather of the cross-KV
        lowers to software-DGE indirect DMA, which is slow for large rows.

        Args:
            hidden_states: ``[B, tgt, H]``.
            cross_k: ``[S, h, src, d]``.
            cross_v: ``[S, h, src, d]``.
            slots: ``[B]`` int32, the slot of each row.
        """
        bsz, tgt_len, _ = hidden_states.shape
        num_slots, src_len = cross_k.shape[0], cross_v.shape[2]
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        if self.kt:
            scores = torch.einsum("bhqd,shdt->bhqst", q, cross_k)
        else:
            scores = torch.einsum("bhqd,shtd->bhqst", q, cross_k)
        own = torch.arange(num_slots, device=slots.device, dtype=slots.dtype).view(
            1, num_slots
        ) == slots.view(bsz, 1)
        neg = torch.full((), float("-inf"), dtype=torch.float32, device=scores.device)
        zero = torch.zeros((), dtype=torch.float32, device=scores.device)
        bias = torch.where(own, zero, neg).view(bsz, 1, 1, num_slots, 1)
        scores = scores.to(torch.float32) + bias
        probs = F.softmax(scores.view(bsz, self.num_heads, tgt_len, num_slots * src_len), dim=-1)
        probs = probs.view(bsz, self.num_heads, tgt_len, num_slots, src_len).to(cross_v.dtype)
        ctx = torch.einsum("bhqst,shtd->bhqd", probs, cross_v)
        out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        return self.out_proj(out)

    def attn_cached(self, hidden_states, cross_k, cross_v):
        bsz, tgt_len, _ = hidden_states.shape
        q = self._shape(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        attn = torch.matmul(q, cross_k if self.kt else cross_k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(cross_v.dtype)
        ctx = torch.matmul(attn, cross_v)
        out = ctx.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        return self.out_proj(out)

    def attn_hybrid(self, hc, cross_k, cross_v, perm, active):
        """``attn_perm`` over only the first ``active`` slots of the cross-KV.

        The runner gives the low slot numbers first, so the live slots of a
        batch of B rows are mostly below B. ``active`` is a static bucket (the
        decode-batch bucket); ``perm`` must be ``[B, active]``.
        """
        return self.attn_perm(hc, cross_k[:active], cross_v[:active], perm)

    def attn_perm(self, hc, cross_k, cross_v, perm):
        """Cross-attention of one query row for each slot, then back to the batch rows.

        ``perm`` is the ``[B, S]`` one-hot of the slot of each row. The q
        projection runs on the B rows; one exact one-hot matmul moves q to the
        slot rows. Attention runs once for each slot. A second one-hot matmul
        moves the context back to the B rows, and out_proj runs on B rows.
        A slot with no row gets q = 0 and its context is dropped.

        Args:
            hc: ``[B, H]`` after the LayerNorm.
            cross_k, cross_v: ``[S, h, src, d]`` (``cross_k`` is ``[S, h, d, src]``
                with ``WHISPER_CROSS_KT=1``).
            perm: ``[B, S]`` one-hot, activation dtype.
        """
        num_slots = perm.shape[1]
        h, d = self.num_heads, self.head_dim
        q = self.q_proj(hc) * self.scaling  # [B, h * d]
        qs = torch.matmul(perm.t(), q).view(num_slots, 1, h, d).transpose(1, 2)
        attn = torch.matmul(qs, cross_k if self.kt else cross_k.transpose(-1, -2))
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(cross_v.dtype)
        ctx = torch.matmul(attn, cross_v)  # [S, h, 1, d]
        ctx = torch.matmul(perm, ctx.transpose(1, 2).reshape(num_slots, h * d))
        return self.out_proj(ctx)


class DecoderSelfAttention(nn.Module):
    """Causal self-attention on the block-managed paged KV cache.

    The runner allocates the K/V caches from ``get_kv_spec`` and binds them
    with ``bind_kv_cache``. The cache shape is
    ``[num_blocks, num_kv_heads, block_size, head_dim]``.
    """

    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype, tp_group, tp):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"decoder.layers.{layer_idx}.self_attn"
        d = cfg.d_model
        num_heads = cfg.decoder_attention_heads
        self.head_dim = d // num_heads
        self.scaling = self.head_dim**-0.5
        if num_heads % tp != 0:
            raise ValueError(
                f"Decoder heads {num_heads} are not divisible by TP {tp}. "
                "Use TP 1, 2 or 4 for Whisper large-v3."
            )
        self.num_heads = num_heads // tp
        # Whisper is MHA: the KV heads are the Q heads.
        self.num_key_value_heads_per_rank = self.num_heads

        self.q_proj = _cpl(d, d, True, dtype, tp_group)
        self.k_proj = _cpl(d, d, False, dtype, tp_group)
        self.v_proj = _cpl(d, d, True, dtype, tp_group)
        self.out_proj = RowParallelLinearFP32Bias(d, d, dtype, tp_group)
        self.tp_group = tp_group
        self.tp = tp

        # Fused decode (NKI attention_decode): one [H, 3 * d_local] QKV weight
        # and bias, and the [d_local, H] out_proj weight. finalize_weights
        # fills them after weight loading. The k slice of the bias is zero.
        self.fused = fused_decode_enabled()
        if self.fused:
            d_local = self.num_heads * self.head_dim
            self.register_buffer(
                "qkv_weight_fused", torch.zeros(d, 3 * d_local, dtype=dtype), persistent=False
            )
            self.register_buffer(
                "qkv_bias_fused", torch.zeros(1, 3 * d_local, dtype=dtype), persistent=False
            )
            self.register_buffer(
                "out_weight_fused", torch.zeros(d_local, d, dtype=dtype), persistent=False
            )

        self.k_cache = None
        self.v_cache = None
        # WHISPER_SELF_KV=slot: model-owned contiguous self-KV, one [448, d] row of
        # each (request slot, head), as a 3D [slots * heads, 448, d] buffer. The
        # writes and reads index only dim 0 (and the position for a decode write):
        # neuronx-cc fails on index_put_ into a reshaped view of a flat buffer
        # (NCC_ILTO901, Task 024). The last slot is the scratch slot (pad rows and
        # pad tokens). Read the flags one time (see DecoderLayer.__init__).
        self.slot_kv = self_kv_mode() == "slot"
        self.slot_ctx = slot_ctx(cfg.max_target_positions) if self.slot_kv else cfg.max_target_positions
        self.num_slots = cfg.max_num_seqs + 1
        self.slot_write_where = self.slot_kv and self_kv_write() == "where"
        if self.slot_kv:
            shp = (self.num_slots * self.num_heads, self.slot_ctx, self.head_dim)
            self.register_buffer("slot_k", torch.zeros(shp, dtype=dtype), persistent=False)
            self.register_buffer("slot_v", torch.zeros(shp, dtype=dtype), persistent=False)

    def _row_slots(self, meta, rows, device):
        """Return the request slot of each batch row, ``[rows]`` int64 (slot 0 if the runner sends none)."""
        slots = meta.get("request_slots")
        if slots is None:
            return torch.zeros(rows, dtype=torch.int64, device=device)
        return slots.to(torch.int64).view(rows)

    def _slot_head_rows(self, slots):
        """Return the buffer row of each (slot, head), ``[len(slots) * heads]``, slot-major."""
        h = self.num_heads
        hh = torch.arange(h, device=slots.device, dtype=torch.int64)
        return (slots.view(-1, 1) * h + hh.view(1, h)).view(-1)

    def _slot_write_prefill(self, k, v, meta):
        """Prefill write: the whole ``[448, d]`` row of each (slot, head) of a batch row.

        Whisper prefill always starts at position 0 (no prefix cache, no chunked
        prefill for encoder-decoder models), so token ``j`` of a row is at position
        ``j``. The pad tokens after the prompt and the zero tail go to positions that
        decode masks until it writes them. A pad row writes the scratch slot.
        """
        h, T, d = k.shape
        C = self.slot_ctx
        rows = meta["block_table_tensor"].shape[0]
        s = T // rows
        idx = self._slot_head_rows(self._row_slots(meta, rows, k.device))
        for buf, x in ((self.slot_k, k), (self.slot_v, v)):
            val = x.view(h, rows, s, d).transpose(0, 1)[:, :, : min(s, C)]
            val = F.pad(val, (0, 0, 0, C - val.shape[2])).to(buf.dtype)
            buf.index_put_((idx,), val.reshape(rows * h, C, d))

    def _slot_write_decode(self, k, v, meta, positions):
        """Decode write: K/V ``[h, B, d]`` of each row to ``(slot, head, position)``.

        A pad row (slot_mapping -1) writes the scratch slot.
        """
        h, B, d = k.shape
        slots = self._row_slots(meta, B, k.device)
        scratch = torch.full_like(slots, self.num_slots - 1)
        slots = torch.where(meta["slot_mapping"].view(B) < 0, scratch, slots)
        idx = self._slot_head_rows(slots)
        pos = positions.to(torch.int64).view(B, 1).expand(B, h).reshape(B * h)
        self.slot_k.index_put_((idx, pos), k.transpose(0, 1).reshape(B * h, d).to(self.slot_k.dtype))
        self.slot_v.index_put_((idx, pos), v.transpose(0, 1).reshape(B * h, d).to(self.slot_v.dtype))

    def _slot_write_where(self, k, v, meta, positions):
        """Decode write with a one-hot ``torch.where`` (``WHISPER_SELF_WRITE=where``).

        One exact one-hot matmul moves the K/V and the position of each row to its
        slot. A slot with no row (and the scratch slot) is left as it is. The where
        then reads and writes the whole buffer.
        """
        h, B, d = k.shape
        S, C = self.num_slots, self.slot_ctx
        slots = self._row_slots(meta, B, k.device)
        live = (meta["slot_mapping"].view(B) >= 0) & (slots < S - 1)
        sel = (
            (torch.arange(S, device=k.device, dtype=torch.int64).view(1, S) == slots.view(B, 1))
            & live.view(B, 1)
        ).to(k.dtype)  # [B, S]
        ks = torch.matmul(sel.t(), k.transpose(0, 1).reshape(B, h * d)).view(S * h, 1, d)
        vs = torch.matmul(sel.t(), v.transpose(0, 1).reshape(B, h * d)).view(S * h, 1, d)
        pos = positions.to(torch.int64).view(B)
        hit = (sel.t().to(torch.int64) * pos.view(1, B)).sum(dim=1)  # [S]
        has = sel.sum(dim=0) > 0
        ar = torch.arange(C, device=k.device, dtype=torch.int64)
        mask = ((ar.view(1, C) == hit.view(S, 1)) & has.view(S, 1))
        mask = mask.view(S, 1, C).expand(S, h, C).reshape(S * h, C, 1)
        self.slot_k.copy_(torch.where(mask, ks.to(self.slot_k.dtype), self.slot_k))
        self.slot_v.copy_(torch.where(mask, vs.to(self.slot_v.dtype), self.slot_v))

    def forward_decode_slot(self, hidden_states, meta, positions):
        """Decode over the slot self-KV: read ``ctx`` positions of each row's slot and attend.

        ``ctx`` is the decode-context bucket: the block-table width times the block
        size, capped at 448. The runner picks the bucket from the largest position
        in the batch (``decode_context_length_buckets``), so every position of the
        batch is below ``ctx``. Positions after the query are masked.
        """
        B = meta["block_table_tensor"].shape[0]
        h, d, C = self.num_heads, self.head_dim, self.slot_ctx
        c = min(meta["block_table_tensor"].shape[1] * meta["block_size"], C)
        q, k, v = self._project(hidden_states)  # [h, B, d]
        if self.slot_write_where:
            self._slot_write_where(k, v, meta, positions)
        else:
            self._slot_write_decode(k, v, meta, positions)
        idx = self._slot_head_rows(self._row_slots(meta, B, hidden_states.device))
        kc = self.slot_k[:, :c].index_select(0, idx).view(B, h, c, d)
        vc = self.slot_v[:, :c].index_select(0, idx).view(B, h, c, d)
        pos = positions.to(torch.int64).view(B)
        qd = q.transpose(0, 1).unsqueeze(2)  # [B, h, 1, d]
        attn = torch.matmul(qd, kc.transpose(-1, -2))  # [B, h, 1, c]
        ar = torch.arange(c, device=attn.device, dtype=torch.int64)
        valid = ar.view(1, c) <= pos.view(B, 1)
        neg = torch.full((), float("-inf"), dtype=attn.dtype, device=attn.device)
        zero = torch.zeros((), dtype=attn.dtype, device=attn.device)
        attn = attn + torch.where(valid, zero, neg).view(B, 1, 1, c)
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(vc.dtype)
        ctx = torch.matmul(attn, vc)  # [B, h, 1, d]
        return self.out_proj(ctx.reshape(B, h * d))

    @torch.no_grad()
    def finalize_weights(self) -> None:
        """Build the fused decode weights from the loaded q/k/v/out weights.

        The q scale is folded into the q weight and the q bias, so the kernel
        runs with ``softmax_scale=1``. This is the same math as ``_project``,
        which scales q after the bias.
        """
        self._build_fused_weights()
        if "decoder" in in_out_parts():
            for lin in (self.q_proj, self.k_proj, self.v_proj):
                to_in_out(lin)
            self.out_proj.set_in_out()

    def _build_fused_weights(self) -> None:
        if not self.fused:
            return
        wq = self.q_proj.weight * self.scaling  # [d_local, H]
        wk = self.k_proj.weight
        wv = self.v_proj.weight
        self.qkv_weight_fused.copy_(torch.cat([wq, wk, wv], dim=0).t().contiguous())
        bq = self.q_proj.bias * self.scaling
        bk = torch.zeros_like(bq)
        bv = self.v_proj.bias
        self.qkv_bias_fused.copy_(torch.cat([bq, bk, bv]).view(1, -1))
        self.out_weight_fused.copy_(self.out_proj.rpl.weight.t().contiguous())

    def _project(self, hidden_states):
        """Project ``[T, H]`` to q, k, v, each ``[heads_local, T, head_dim]``."""
        tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _write_kv_cache(self, k, v, slot_mapping, block_size):
        """Write K/V into the paged cache.

        ``paged_slot_indices`` sends a pad slot (-1) to the null block. Thus the
        scatter never gets an out-of-range address.
        """
        num_blocks = self.k_cache.shape[0]
        blk_idx, pos_idx = paged_slot_indices(slot_mapping, block_size, num_blocks)
        num_slots = slot_mapping.shape[0]
        nkh = self.num_key_value_heads_per_rank
        k_f = k.reshape(-1, self.head_dim).to(self.k_cache.dtype)
        v_f = v.reshape(-1, self.head_dim).to(self.v_cache.dtype)
        h_idx = torch.arange(nkh, dtype=blk_idx.dtype, device=k.device).repeat_interleave(
            num_slots
        )
        index = (blk_idx.repeat(nkh), h_idx, pos_idx.repeat(nkh))
        self.k_cache.index_put_(index, k_f)
        self.v_cache.index_put_(index, v_f)

    def forward_prefill(self, hidden_states, meta, positions=None):
        """Causal attention over the prompt, with the reference SDPA math.

        The decoder prompt has about 4 tokens, so a flash kernel gives no gain.
        The plain math keeps the first token byte-identical to the reference.
        The pad tokens are after the real tokens, so the causal mask hides them.
        """
        q, k, v = self._project(hidden_states)
        if self.slot_kv:
            self._slot_write_prefill(k, v, meta)
        else:
            self._write_kv_cache(k, v, meta["slot_mapping"], meta["block_size"])

        tokens = hidden_states.shape[0]
        rows = meta["block_table_tensor"].shape[0]
        if rows > 1:
            return self._prefill_rows(q, k, v, rows, tokens)
        attn = torch.matmul(q, k.transpose(-1, -2))
        mask = torch.triu(
            torch.full((tokens, tokens), float("-inf"), dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = F.softmax((attn + mask).to(torch.float32), dim=-1).to(v.dtype)
        ctx = torch.matmul(attn, v)
        out = ctx.transpose(0, 1).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(out)

    def _prefill_rows(self, q, k, v, rows, tokens):
        """Causal attention for ``rows`` prompts of ``tokens // rows`` tokens each.

        Each row attends only to its own prompt. This is the per-row form of the
        one-row math above, so each row gets the same values as a one-row call.
        """
        s = tokens // rows
        h, d = self.num_heads, self.head_dim
        qr = q.view(h, rows, s, d).transpose(0, 1)
        kr = k.view(h, rows, s, d).transpose(0, 1)
        vr = v.view(h, rows, s, d).transpose(0, 1)
        attn = torch.matmul(qr, kr.transpose(-1, -2))
        mask = torch.triu(
            torch.full((s, s), float("-inf"), dtype=attn.dtype, device=attn.device),
            diagonal=1,
        )
        attn = F.softmax((attn + mask).to(torch.float32), dim=-1).to(v.dtype)
        ctx = torch.matmul(attn, vr)  # [rows, h, s, d]
        out = ctx.permute(0, 2, 1, 3).reshape(tokens, h * d)
        return self.out_proj(out)

    def forward_decode_fused(self, hidden_states, meta, positions):
        """Decode with the NKI fused attention block (QKV, attention, KV write, out_proj).

        The pre-attention LayerNorm stays outside the kernel (the kernel has
        RMSNorm only). The kernel writes the new K/V into the paged cache. The
        out_proj bias is added after the TP all-reduce, as in the torch path.
        """
        from vllm_neuron import functional as NF

        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        nkh = self.num_key_value_heads_per_rank
        S_ctx = block_table.shape[1] * block_size

        # The mask, the slots and the table are the same for every layer.
        # Build them one time for each step and keep them in the shared
        # metadata dict (the llama3 decode-mask cache does the same).
        cached = meta.get("_whisper_fused_decode_inputs")
        if cached is not None:
            attention_mask, update_idx, table = cached
            return self._fused_decode_call(
                hidden_states, attention_mask, update_idx, table, B, S, tokens, hidden
            )
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=positions.view(1, B * S).to(torch.float32),
            bs=B,
            q_head=self.num_heads,
            s_active=S,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )
        # The NKI kernel needs per-head slots and a per-head [B, kv_heads, blocks]
        # table for MHA. The CPU torch fallback takes the base [B, S] slots and
        # the 2D table with the 4D cache.
        if hidden_states.device.type == "cpu":
            update_idx = meta["slot_mapping"].view(B, S)
            table = block_table.to(torch.int32)
        else:
            update_idx = build_per_head_slot_mapping(
                meta["slot_mapping"], block_size, nkh, B
            ).to(torch.uint32)
            table = build_per_head_block_table(block_table.to(torch.int32), nkh).to(
                torch.int32
            )
        meta["_whisper_fused_decode_inputs"] = (attention_mask, update_idx, table)
        return self._fused_decode_call(
            hidden_states, attention_mask, update_idx, table, B, S, tokens, hidden
        )

    def _fused_decode_call(
        self, hidden_states, attention_mask, update_idx, table, B, S, tokens, hidden
    ):
        """Run the fused decode kernel, the TP all-reduce, and the out_proj bias."""
        from vllm_neuron import functional as NF

        out = NF.attention_decode(
            X=hidden_states.view(B, S, hidden),
            X_hidden_dim_actual=hidden,
            rmsnorm_X_enabled=False,
            W_qkv=self.qkv_weight_fused,
            bias_qkv=self.qkv_bias_fused,
            K_cache_transposed=False,
            active_blocks_table=table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=1.0,
            update_cache=True,
            kv_cache_update_idx=update_idx,
            W_out=self.out_weight_fused,
            bias_out=None,
            transposed_out=False,
        )
        out = out.reshape(tokens, hidden)
        if self.tp > 1:
            from torch.distributed._functional_collectives import all_reduce

            out = all_reduce(out, reduceOp="sum", group=self.tp_group)
        return out + self.out_proj.bias.to(out.dtype)

    def forward_decode(self, hidden_states, meta, positions):
        """Decode: write the new K/V, gather the sequence cache, and attend.

        The runner sets an unused block-table entry to -1. Change it to the
        null block (0) before the gather. The position mask hides each slot
        after the position of the query, which includes the null-block slots.
        """
        block_size = meta["block_size"]
        block_table = meta["block_table_tensor"]
        block_table = torch.where(
            block_table < 0, torch.zeros_like(block_table), block_table
        )

        B_local = block_table.shape[0]
        tokens = hidden_states.shape[0]
        S_decode = tokens // B_local

        q, k, v = self._project(hidden_states)
        self._write_kv_cache(k, v, meta["slot_mapping"], block_size)

        nkh = self.num_key_value_heads_per_rank
        max_blocks = block_table.shape[1]
        S_ctx = max_blocks * block_size
        kc = self.k_cache[block_table]  # [B, max_blocks, nkh, block_size, d]
        vc = self.v_cache[block_table]
        kc = kc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)
        vc = vc.permute(0, 2, 1, 3, 4).reshape(B_local, nkh, S_ctx, self.head_dim)

        qd = q.transpose(0, 1).reshape(B_local, S_decode, nkh, self.head_dim).transpose(1, 2)
        attn = torch.matmul(qd, kc.transpose(-1, -2))
        # Slot s is valid for query j only if s <= position(j). The gathered
        # slot index is the logical sequence position.
        pos = positions.to(torch.int32).view(B_local, S_decode)
        ar = torch.arange(S_ctx, device=attn.device, dtype=torch.int32)
        valid = ar.view(1, 1, S_ctx) <= pos.view(B_local, S_decode, 1)
        neg = torch.full((), float("-inf"), dtype=attn.dtype, device=attn.device)
        zero = torch.zeros((), dtype=attn.dtype, device=attn.device)
        attn = attn + torch.where(valid, zero, neg).unsqueeze(1)
        attn = F.softmax(attn.to(torch.float32), dim=-1).to(vc.dtype)
        ctx = torch.matmul(attn, vc)
        ctx = ctx.transpose(1, 2).reshape(tokens, self.num_heads * self.head_dim)
        return self.out_proj(ctx)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: WhisperConfig, layer_idx: int, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.self_attn = DecoderSelfAttention(cfg, layer_idx, dtype, tp_group, tp)
        self.self_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.encoder_attn = CrossAttention(
            d, cfg.decoder_attention_heads, dtype, tp_group, tp
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(d, dtype=dtype)
        self.mlp = MLP(d, cfg.decoder_ffn_dim, dtype, tp_group, part="decoder")
        self.final_layer_norm = nn.LayerNorm(d, dtype=dtype)
        # Read the env flags one time. A read in forward puts an os.environ guard
        # in the compiled graph, and the warm restore of the compiled package
        # then fails when the variable is not set (KeyError on the guard).
        self.fused_max_rows = fused_decode_max_rows()
        self.skip_cross = cross_mode() == "none"

    def forward(
        self, x, cross_k, cross_v, meta, is_prefill, positions, slots=None, cross_meta=None,
        perm=None,
    ):
        # x is token-flat, [T, H]. cross_k and cross_v are [B, h, 1500, d], one
        # row for each request in the batch. When slots is set, they hold all
        # slots, [S, h, 1500, d], and slots gives the slot of each row.
        h = self.self_attn_layer_norm(x)
        if is_prefill:
            sa = self.self_attn.forward_prefill(h, meta, positions)
        elif self.self_attn.slot_kv:
            sa = self.self_attn.forward_decode_slot(h, meta, positions)
        elif self.self_attn.fused and meta["block_table_tensor"].shape[0] <= self.fused_max_rows:
            sa = self.self_attn.forward_decode_fused(h, meta, positions)
        else:
            sa = self.self_attn.forward_decode(h, meta, positions)
        x = x + sa

        tokens, hidden = x.shape
        if self.skip_cross and not is_prefill:
            # Timing only: skip the cross-attention (the output is wrong).
            x = x + self.mlp(self.final_layer_norm(x), is_prefill)
            return x
        if cross_meta is not None and isinstance(cross_meta[0], str):
            _, xslots, qmask, group = cross_meta
            hc = self.encoder_attn_layer_norm(x)
            ca = self.encoder_attn.attn_xkernel(hc, cross_k, cross_v, xslots, qmask, group)
        elif cross_meta is not None:
            table, mask = cross_meta
            num_rows = table.shape[0]
            hc = self.encoder_attn_layer_norm(x).view(num_rows, tokens // num_rows, hidden)
            ca = self.encoder_attn.attn_kernel(hc, cross_k, cross_v, table, mask)
        elif perm is not None:
            hc = self.encoder_attn_layer_norm(x)
            ca = self.encoder_attn.attn_perm(hc, cross_k, cross_v, perm)
        elif slots is not None:
            num_rows = slots.shape[0]
            hc = self.encoder_attn_layer_norm(x).view(num_rows, tokens // num_rows, hidden)
            ca = self.encoder_attn.attn_all_slots(hc, cross_k, cross_v, slots)
        else:
            num_rows = cross_k.shape[0]
            hc = self.encoder_attn_layer_norm(x).view(num_rows, tokens // num_rows, hidden)
            ca = self.encoder_attn.attn_cached(hc, cross_k, cross_v)
        x = x + ca.reshape(tokens, hidden)

        x = x + self.mlp(self.final_layer_norm(x), is_prefill)
        return x


class WhisperDecoder(nn.Module):
    def __init__(self, cfg: WhisperConfig, dtype, tp_group, tp):
        super().__init__()
        d = cfg.d_model
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d, cfg.pad_token_id, dtype=dtype)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d, dtype=dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i, dtype, tp_group, tp) for i in range(cfg.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(d, dtype=dtype)
        self._cross_kv_owner: WhisperCrossKVEncoder | None = None
        # Read the env flags one time (see DecoderLayer.__init__).
        self.cross_mode = cross_mode()
        self.cross_active = os.environ.get("WHISPER_CROSS_ACTIVE", "0") == "1"

    def bind_cross_kv(self, owner: WhisperCrossKVEncoder) -> None:
        """Point the cross-KV reads at the module that owns the buffers.

        ``forward`` reads the buffers from the owner at each call. A ``.to()``
        or ``to_empty`` call replaces the buffer tensors, so a stored tensor
        reference becomes stale. ``object.__setattr__`` keeps the owner out of
        the decoder module tree, which prevents a module cycle.
        """
        object.__setattr__(self, "_cross_kv_owner", owner)

    def forward(self, input_ids, positions, attn_metadata, is_prefill):
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        owner = self._cross_kv_owner
        # All decoder layers are in one KV-cache group, so they share one
        # metadata dict. Read it one time: a lookup for each layer adds a
        # Dynamo guard for each layer key on every step.
        meta = attn_metadata[self.layers[0].self_attn.layer_name]
        # With one request slot, read slot 0 directly. A gather of the cross-KV
        # becomes an indirect DMA that copies all of the cross-KV on each step.
        slots = None if owner.num_slots <= 2 else self._request_slots(attn_metadata, x.device)
        # How a decode step with request slots reads the cross-KV of its rows
        # (WHISPER_CROSS_MODE):
        #   mask (default): attention over all slots with -inf for the other
        #     slots. Measured fastest (TP4, 16 rows).
        #   onehot: a one-hot matmul selects the slot of each row. It
        #     is exact (1 * v + 0 * others) and runs as one dense matmul.
        #   gather: index_select (software-DGE indirect DMA, slow).
        # Prefill has one row, so it always uses the gather.
        mode = self.cross_mode
        blocked = owner.blocked
        if owner.xkernel:
            # Prefill gathers the rows' slots and uses the plain attention, as "gather".
            mode = "xgather" if is_prefill else "xkernel"
        elif owner.mqa:
            mode = "mqagather" if is_prefill else "mqa"
        elif blocked:
            # Decode uses the kernel with a block table. Prefill has up to 128
            # tokens for each row, more than the kernel takes, so it gathers
            # the 12 blocks of each row and uses the plain attention.
            mode = "blockgather" if is_prefill else "kernel"
        elif slots is None or is_prefill:
            mode = "gather"
        all_slots = mode == "mask"
        cross_meta = None
        if mode == "kernel":
            cross_meta = self._cross_kernel_inputs(x, slots, meta, owner)
        elif mode == "mqa":
            cross_meta = self._cross_mqa_inputs(x, slots, meta, owner)
        elif mode == "xkernel":
            rows = meta["block_table_tensor"].shape[0]
            xslots = (
                torch.zeros(rows, dtype=torch.int32, device=x.device) if slots is None else slots
            )
            from vllm_neuron._kernels.whisper.cross_attention_decode import (
                xkernel_group,
                xkernel_qmask,
            )

            from vllm_neuron.utils.hardware_config import get_logical_nc_config

            group = xkernel_group(owner.xk_heads, rows, lnc=get_logical_nc_config())
            qmask = xkernel_qmask(
                owner.xk_heads, group, device=x.device, dtype=self.embed_tokens.weight.dtype
            )
            cross_meta = ("xkernel", xslots.to(torch.int32), qmask, group)
        if mode == "xgather":
            rows = meta["block_table_tensor"].shape[0]
            xslots = (
                torch.zeros(rows, dtype=torch.int64, device=x.device) if slots is None else slots
            ).to(torch.int64)
        if mode == "mqagather":
            rows = meta["block_table_tensor"].shape[0]
            row_slots = (
                torch.zeros(rows, dtype=torch.int32, device=x.device) if slots is None else slots
            )
            nbm = 128 * owner.mqa_groups
        if mode == "blockgather":
            rows = meta["block_table_tensor"].shape[0]
            row_slots = (
                torch.zeros(rows, dtype=torch.int32, device=x.device) if slots is None else slots
            )
            nb = CROSS_BLOCKS_PER_SLOT
            block_ids = (
                row_slots.view(-1, 1).to(torch.int64) * nb
                + torch.arange(nb, device=x.device, dtype=torch.int64)
            ).view(-1)
        onehot = None
        active = None
        if mode == "perm" and not is_prefill:
            # WHISPER_CROSS_ACTIVE=1: attend only the first B slots for a batch
            # of B rows. The scheduler takes the lowest free slot, so this holds
            # when the live slots are 0..B-1; the runner must fall back
            # otherwise (the CPU test and the harness hold it by construction).
            if self.cross_active:
                active = min(slots.shape[0], owner.num_slots)
        if mode in ("onehot", "perm"):
            num_slots = owner.num_slots if active is None else active
            onehot = (
                torch.arange(num_slots, device=slots.device, dtype=slots.dtype).view(1, -1)
                == slots.view(-1, 1)
            ).to(self.embed_tokens.weight.dtype)
        for i, layer in enumerate(self.layers):
            cross_k = getattr(owner, f"cross_k_{i}")
            cross_v = getattr(owner, f"cross_v_{i}")
            if mode in ("kernel", "mqa", "xkernel"):
                pass
            elif mode == "xgather":
                # [rows, h, d, 1536] -> [rows, h, d, 1500]; the K^T layout (kt) of attn_cached.
                hl = owner.n_heads_local
                cross_k = cross_k.index_select(0, xslots)[:, :hl, :, :WHISPER_ENCODER_TOKENS]
                cross_v = cross_v.index_select(0, xslots)[:, :hl, :WHISPER_ENCODER_TOKENS]
            elif mode == "mqagather":
                # The blocks of a slot are contiguous, so gather one slot per row
                # instead of nbm blocks per row. A 512-block gather does not finish
                # compiling with neuronx-cc 2.28 (SDK 2.33).
                hl, d = owner.n_heads_local, cross_k.shape[-1]
                g, groups, span = owner.mqa_g, owner.mqa_groups, owner.mqa_span
                cross_k, cross_v = (
                    c.view(-1, nbm * c.shape[1] * c.shape[2], d)
                    .index_select(0, row_slots.to(torch.int64))
                    .reshape(rows, groups, span, d)[:, :, : g * CROSS_CTX]
                    .reshape(rows, hl, CROSS_CTX, d)[:, :, :WHISPER_ENCODER_TOKENS]
                    for c in (cross_k, cross_v)
                )
            elif mode == "blockgather":
                h, d = cross_k.shape[1], cross_k.shape[3]
                cross_k, cross_v = (
                    c.index_select(0, block_ids)
                    .view(rows, nb, h, CROSS_BLOCK, d)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(rows, h, nb * CROSS_BLOCK, d)[:, :, :WHISPER_ENCODER_TOKENS]
                    for c in (cross_k, cross_v)
                )
            elif slots is None:
                cross_k, cross_v = cross_k[:1], cross_v[:1]
            elif mode in ("perm", "none") and not is_prefill:
                if active is not None:
                    cross_k, cross_v = cross_k[:active], cross_v[:active]
            elif onehot is not None:
                n = onehot.shape[0]
                cross_k = torch.matmul(onehot, cross_k.view(cross_k.shape[0], -1)).view(
                    n, *cross_k.shape[1:]
                )
                cross_v = torch.matmul(onehot, cross_v.view(cross_v.shape[0], -1)).view(
                    n, *cross_v.shape[1:]
                )
            elif not all_slots:
                # Gather whole flat rows, so that each DMA descriptor moves one
                # slot of one layer (about 1 MB at TP4), not one 128-byte head row.
                n = slots.shape[0]
                cross_k = cross_k.view(cross_k.shape[0], -1).index_select(0, slots).view(
                    n, *cross_k.shape[1:]
                )
                cross_v = cross_v.view(cross_v.shape[0], -1).index_select(0, slots).view(
                    n, *cross_v.shape[1:]
                )
            x = layer(
                x, cross_k, cross_v, meta, is_prefill, positions,
                slots=slots if all_slots else None, cross_meta=cross_meta,
                perm=onehot if mode == "perm" else None,
            )
        return self.layer_norm(x)

    def _cross_kernel_inputs(self, x, slots, meta, owner):
        """Build the per-head block table and the mask of the kernel cross-attention.

        The mask marks the 1500 encoder positions of each row as valid, and the
        pad positions and the active key (the last position) as not valid. It
        has the kernel's block-KV layout for ``(B, heads)``; ``attention_decode``
        moves it to the folded MHA layout.
        """
        from vllm_neuron.functional.attention.attention_decode_mask import (
            _torch_gen_attention_decode_mask_impl,
        )

        rows = meta["block_table_tensor"].shape[0]
        if slots is None:
            slots = torch.zeros(rows, dtype=torch.int32, device=x.device)
        nb = CROSS_BLOCKS_PER_SLOT
        heads = owner.cross_attns[0].num_heads
        base = slots.view(-1, 1).to(torch.int32) * nb + torch.arange(
            nb, device=x.device, dtype=torch.int32
        )
        # The NKI kernel needs the per-head table for MHA. The CPU torch
        # fallback takes the 2D table with the 4D cache.
        if x.device.type == "cpu":
            table = base
        else:
            table = build_per_head_block_table(base, heads).to(torch.int32)
        s_ctx = CROSS_CTX
        pos = torch.full((1, rows), float(WHISPER_ENCODER_TOKENS), device=x.device)
        active = torch.zeros(1, rows, heads, 1, dtype=torch.float32, device=x.device)
        mask = _torch_gen_attention_decode_mask_impl(
            pos, bs=rows, q_head=heads, s_active=1, s_prior=s_ctx,
            block_len=CROSS_BLOCK, active_mask=active,
        )
        return table, mask

    def _cross_mqa_inputs(self, x, slots, meta, owner):
        """Build the 2D block table and the per-head mask of the MQA cross-attention.

        Each slot holds ``heads x 1536`` positions in 128 blocks (head-major).
        Q head ``j`` is valid only on positions ``[j * 1536, j * 1536 + 1500)``.
        The mask is built in sequence order and then moved to the kernel's
        block-KV layout (sequence order is the block-length-1 layout).
        """
        from vllm_neuron.functional.attention.attention_decode_mask import (
            _resize_block_len,
            relayout_block_kv_mask,
        )

        rows = meta["block_table_tensor"].shape[0]
        if slots is None:
            slots = torch.zeros(rows, dtype=torch.int32, device=x.device)
        # One head group: g heads, g x 1536 positions in 128 blocks. Group i of
        # slot s starts at block (s * groups + i) * 128; attn_kernel adds i * 128.
        heads = owner.mqa_g
        s_ctx = owner.mqa_span
        table = slots.view(-1, 1).to(torch.int32) * (128 * owner.mqa_groups) + torch.arange(
            128, device=x.device, dtype=torch.int32
        )
        t = torch.arange(s_ctx, device=x.device, dtype=torch.int32)
        h = torch.arange(heads, device=x.device, dtype=torch.int32)
        valid = ((t // CROSS_CTX).view(-1, 1) == h.view(1, -1)) & (
            (t % CROSS_CTX) < WHISPER_ENCODER_TOKENS
        ).view(-1, 1)
        mask = valid.to(torch.float32).view(s_ctx, 1, heads, 1).expand(s_ctx, rows, heads, 1)
        to_bl = _resize_block_len(owner.mqa_block, rows, heads, 1, s_ctx)
        mask = relayout_block_kv_mask(mask.contiguous(), 1, to_bl)
        return table, mask

    def _request_slots(self, attn_metadata, device) -> torch.Tensor:
        """Return the cross-KV slot of each batch row, ``[num_rows]``.

        The runner sends the slots in ``attn_metadata[layer]["request_slots"]``.
        A runner that does not send them runs one request at a time, so use
        slot 0.
        """
        meta = attn_metadata[self.layers[0].self_attn.layer_name]
        slots = meta.get("request_slots")
        if slots is None:
            num_rows = meta["block_table_tensor"].shape[0]
            slots = torch.zeros(num_rows, dtype=torch.int32, device=device)
        return slots


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class WhisperForConditionalGeneration(nn.Module):
    """Native Whisper large-v3 (encoder-decoder) for the vLLM Neuron runner.

    The runner reads ``supports_transcription`` and
    ``supports_transcription_only`` on this instance. It then reports only the
    "transcription" task. The serving class methods are on the registered
    factory class.
    """

    is_text_generation_model = True
    supports_transcription = True
    supports_transcription_only = True
    supports_multimodal = True
    # The runner gives each request a stable cross-KV slot and sends it in
    # attn_metadata["request_slots"] and embed_multimodal(request_slots=...).
    uses_request_slots = True

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        tp_group, world, rank = tp_info()
        self.tp_group = tp_group
        self.world_size = world
        self.rank = rank

        enc_cfg = config.encoder_neuron_config
        if enc_cfg is not None and (enc_cfg.tp_size != world or enc_cfg.dp_size != 1):
            raise ValueError(
                f"Whisper encoder_neuron_config has tp_size={enc_cfg.tp_size}, "
                f"dp_size={enc_cfg.dp_size}. Set tp_size={world} and dp_size=1. "
                "The encoder shards the cross-KV heads in the same way as the decoder."
            )

        self.encoder = WhisperEncoder(config, self.dtype, tp_group, world)
        self.decoder = WhisperDecoder(config, self.dtype, tp_group, world)

        # >>> PARALLELISM: the cross-KV heads have the decoder sharding.
        n_heads_local = config.decoder_attention_heads // world
        # One cross-KV slot for each request, plus the scratch slot.
        self.num_request_slots = config.max_num_seqs + 1
        self.audio_tower = WhisperCrossKVEncoder(
            self.encoder,
            nn.ModuleList([layer.encoder_attn for layer in self.decoder.layers]),
            n_layers=config.decoder_layers,
            n_heads_local=n_heads_local,
            head_dim=config.head_dim,
            n_ctx=config.max_source_positions,
            dtype=self.dtype,
            num_slots=self.num_request_slots,
        )
        self.decoder.bind_cross_kv(self.audio_tower)

        # >>> PARALLELISM: vocab-parallel fp32 LM head, tied to embed_tokens.
        # Pad the vocab to a multiple of TP. The pad columns get -inf before
        # the argmax, so they never win.
        vocab = config.vocab_size
        pad = (-vocab) % world
        self.padded_vocab = vocab + pad
        self.vocab_pad = pad
        self.per_rank = self.padded_vocab // world
        self.lm_head = ColumnParallelLinear(
            config.d_model,
            self.padded_vocab,
            bias=False,
            gather_output=False,
            dtype=torch.float32,
            tp_group=tp_group,
        )

        self.lm_in_out = "lm_head" in in_out_parts()
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self.sampler = None
        if self.on_device_sampling_config is not None:
            # The sampler does the cross-rank greedy argmax over the same group
            # that shards the vocab. On a tie, the lowest global index wins,
            # the same as an argmax over the full vocab.
            self.sampler = Sampler(self.on_device_sampling_config, process_group=tp_group)

    def compiler_args_for(self, graph_type, example_inputs, base):
        """Give each graph its own neuronx-cc arguments.

        ``WHISPER_DECODE_OPT_LEVEL`` sets the ``-O`` level of the plain decode
        graph. ``WHISPER_ENCODER_OPT_LEVEL`` sets it for the encoder graph. When
        a variable is not set, the graph keeps the base arguments.

        Example:
            >>> model.compiler_args_for("prefill", None, ["-O2"])  # doctest: +SKIP
            ['-O2']
        """
        import os

        from vllm_neuron.utils.compiler_args_ctx import with_optimization_level

        base = list(base or [])
        level = {
            "decode": os.environ.get("WHISPER_DECODE_OPT_LEVEL"),
            "encoder": os.environ.get("WHISPER_ENCODER_OPT_LEVEL"),
        }.get(graph_type)
        return with_optimization_level(base, int(level)) if level else base

    @property
    def dtype(self) -> torch.dtype:
        """Return the activation dtype.

        A property, not an attribute: Dynamo saves an unguarded module
        attribute as ``_Missing`` by object identity. A ``torch.dtype`` is a
        singleton, so an attribute that holds it also changes the dtype of each
        saved tensor to ``_Missing``, and the warm restore of the compiled
        package fails.
        """
        return self.config.torch_dtype

    # ── construction ──────────────────────────────────────────────────────
    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        encoder_neuron_config: EncoderNeuronConfig | None = None,
        **overrides,
    ) -> "WhisperForConditionalGeneration":
        """Build the model from the HF config and the neuron configs."""
        config = WhisperConfig.from_configs(
            hf_config, text_neuron_config, encoder_neuron_config, **overrides
        )
        return cls(config)

    # ── vLLM generative-model detection ───────────────────────────────────
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed_tokens(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the full-vocab fp32 logits (the host-sampling path)."""
        local = self._logits_local(hidden_states)
        if self.world_size == 1:
            full = local
        else:
            from torch.distributed._functional_collectives import all_gather_tensor

            full = all_gather_tensor(local, local.dim() - 1, self.tp_group)
        return full[..., : self.config.vocab_size]

    def _logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.to(torch.float32))

    def _mask_pad_local(self, local: torch.Tensor, rank) -> torch.Tensor:
        """Set the local logits of the pad columns to -inf.

        SPMD: derive the global vocab index from the runtime ``rank`` tensor,
        so that all ranks trace the same graph.
        """
        if self.vocab_pad == 0:
            return local
        if rank is None:
            rank = torch.tensor(self.rank, dtype=torch.int32)
        offset = rank.to(device=local.device, dtype=torch.int32).reshape(()) * self.per_rank
        idx = torch.arange(self.per_rank, device=local.device, dtype=torch.int32) + offset
        neg = torch.full((), float("-inf"), dtype=local.dtype, device=local.device)
        return torch.where(idx < self.config.vocab_size, local, neg)

    def _pad_logit_mask(self, logit_mask):
        """Pad a full-vocab logit mask to the padded vocab (pad columns False)."""
        if logit_mask is None or logit_mask.shape[-1] == self.padded_vocab:
            return logit_mask
        extra = self.padded_vocab - logit_mask.shape[-1]
        pad = torch.zeros(
            (*logit_mask.shape[:-1], extra), dtype=logit_mask.dtype, device=logit_mask.device
        )
        return torch.cat([logit_mask, pad], dim=-1)

    # ── encoder (SupportsEncoderWarmup) ───────────────────────────────────
    def get_encoder(self) -> nn.Module:
        return self.audio_tower

    def build_encoder_synthetic_inputs(
        self,
        bucket: int,
        encoder_neuron_config: EncoderNeuronConfig,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Return a zero mel batch for one encoder bucket.

        A bucket counts encoder tokens. One clip has 1500 tokens, so a bucket
        of ``n * 1500`` holds ``n`` clips.
        """
        num_clips = max(1, bucket // WHISPER_ENCODER_TOKENS)
        return {
            "input_features": torch.zeros(
                num_clips,
                self.config.num_mel_bins,
                WHISPER_MEL_FRAMES,
                dtype=self.dtype,
                device=device,
            )
        }

    def embed_multimodal(
        self,
        input_features: torch.Tensor | None = None,
        encoder_cache=None,
        mm_hashes: list[str] | None = None,
        request_slots: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Run the audio encoder and write the cross-KV slot of each clip.

        Whisper does not write the encoder cache buffer. The runner still reads
        the cache bookkeeping at prefill, so reserve one block (1500 tokens) for
        each clip. Without the reservation, prefill raises
        ``EncoderCacheReadError``.

        Args:
            input_features: Mel features ``[b, 128, 3000]``.
            encoder_cache: The ``EncoderCacheBlocks`` of the runner.
            mm_hashes: One identifier for each clip.
            request_slots: The cross-KV slot of each clip. When the runner
                does not send slots, it runs one request at a time, so use
                slot 0.
        """
        if input_features is None:
            raise ValueError(
                "Whisper embed_multimodal received no input_features. "
                "Send the audio through the Whisper multimodal processor."
            )
        if isinstance(input_features, (list, tuple)):
            input_features = torch.stack(list(input_features))
        mel = input_features
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        num_clips = mel.shape[0]
        if request_slots is None:
            if num_clips != 1 or self.config.max_num_seqs != 1:
                raise ValueError(
                    f"Whisper received {num_clips} clips without request slots "
                    f"(max_num_seqs={self.config.max_num_seqs}). Use a runner that "
                    "sends request_slots, or set max_num_seqs=1."
                )
            request_slots = [0]
        if len(request_slots) != num_clips:
            raise ValueError(
                f"Whisper received {num_clips} clips and {len(request_slots)} "
                "request slots. Send one slot for each clip."
            )
        if mm_hashes is None or len(mm_hashes) != num_clips:
            raise ValueError(
                f"Whisper embed_multimodal expects one mm_hash for each clip: "
                f"{num_clips} clips, {0 if mm_hashes is None else len(mm_hashes)} hashes."
            )
        if encoder_cache is not None:
            for mm_hash in mm_hashes:
                encoder_cache.allocate(mm_hash, [WHISPER_ENCODER_TOKENS])

        device = next(self.encoder.parameters()).device
        # Cast on the source device first. A combined device and dtype .to() on
        # Neuron raises "Expected self.dtype() == dst.dtype()".
        mel = mel.to(dtype=self.dtype).to(device=device)
        slots = torch.tensor(request_slots, dtype=torch.int64).to(device=device)
        self.audio_tower(mel, write_block_ids=slots)

    # ── KV cache ──────────────────────────────────────────────────────────
    def get_kv_spec(self) -> KVSpec:
        # Slot self-KV mode keeps K/V in model-owned buffers. The runner still
        # needs a paged spec to schedule, so give it one KV head: the paged cache
        # is never read or written, and takes 1/heads of the HBM.
        layers = [
            LayerSpec(
                name=layer.self_attn.layer_name,
                num_kv_heads=(
                    1 if layer.self_attn.slot_kv else layer.self_attn.num_key_value_heads_per_rank
                ),
                head_size=layer.self_attn.head_dim,
                dtype=self.dtype,
            )
            for layer in self.decoder.layers
        ]
        # The cross-KV buffers are model-owned and not in the spec.
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for layer in self.decoder.layers:
            name = layer.self_attn.layer_name
            if name not in kv_caches:
                raise KeyError(f"The KV cache for layer {name} is not initialized.")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    # ── weights ───────────────────────────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None = None
    ) -> None:
        """Load the HF safetensors into the encoder and the decoder.

        The runner builds the model on the meta device. Materialize all storage
        on CPU, load the weights, and tie the LM head. The runner then moves the
        model to the device.
        """
        self.to_empty(device="cpu")
        with torch.no_grad():
            for name, buf in self.audio_tower.named_buffers(recurse=False):
                buf.zero_()
            # to_empty leaves the slot self-KV uninitialized. A NaN in a masked
            # position still gives NaN (NaN + -inf, 0 * NaN), so zero it.
            for layer in self.decoder.layers:
                if layer.self_attn.slot_kv:
                    layer.self_attn.slot_k.zero_()
                    layer.self_attn.slot_v.zero_()
        self.decoder.bind_cross_kv(self.audio_tower)

        hf_sd = load_hf_state(resolve_checkpoint_dir(checkpoint_path, cache_dir))
        enc_state = build_state(hf_sd, "model.encoder.", self.dtype)
        dec_state = build_state(hf_sd, "model.decoder.", self.dtype)

        enc_missing, enc_unexpected = self.encoder.load_state_dict(
            enc_state, strict=False, assign=True
        )
        dec_missing, dec_unexpected = self.decoder.load_state_dict(
            dec_state, strict=False, assign=True
        )
        if "embed_positions.weight" in enc_missing:
            enc_missing.remove("embed_positions.weight")
            with torch.no_grad():
                self.encoder.embed_positions.weight.copy_(
                    sinusoids(self.config.max_source_positions, self.config.d_model).to(
                        self.dtype
                    )
                )
        self._tie_lm_head()
        if self.lm_in_out:
            # The fp32 LM head as [d, vocab_local]: the decode GEMV reads the weight
            # tiles without a PE transpose.
            to_in_out(self.lm_head)
        for layer in self.encoder.layers:
            layer.self_attn.finalize_weights()
            layer.mlp.finalize_weights()
        for layer in self.decoder.layers:
            layer.self_attn.finalize_weights()
            layer.encoder_attn.finalize_weights()
            layer.mlp.finalize_weights()
        self.audio_tower.finalize_weights()

        problems = enc_missing + enc_unexpected + dec_missing + dec_unexpected
        logger.info(
            "Whisper load_weights: encoder missing=%d unexpected=%d, "
            "decoder missing=%d unexpected=%d",
            len(enc_missing),
            len(enc_unexpected),
            len(dec_missing),
            len(dec_unexpected),
        )
        if problems:
            raise RuntimeError(
                "Whisper weight names do not match the checkpoint: "
                f"encoder missing={enc_missing} unexpected={enc_unexpected}, "
                f"decoder missing={dec_missing} unexpected={dec_unexpected}."
            )

    def _tie_lm_head(self) -> None:
        """Copy the vocab slice of this rank from embed_tokens into the LM head.

        The rows after the real vocab stay zero. The argmax mask hides them.
        Call this method on CPU, after the decoder weights load.
        """
        vocab = self.config.vocab_size
        start = self.rank * self.per_rank
        end = min(start + self.per_rank, vocab)
        w = self.decoder.embed_tokens.weight.detach().to("cpu")
        shard = torch.zeros(self.per_rank, w.shape[1], dtype=self.lm_head.weight.dtype)
        if end > start:
            shard[: end - start] = w[start:end].to(shard.dtype)
        with torch.no_grad():
            self.lm_head.weight.copy_(shard.to(self.lm_head.weight.device))

    # ── forward ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: dict | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        # The runner adds these prefill inputs for every multimodal model.
        # Whisper reads the cross-KV buffers instead, so it ignores them.
        encoder_embedding_buffer: torch.Tensor | None = None,
        encoder_block_ids: torch.Tensor | None = None,
        encoder_positions: torch.Tensor | None = None,
        **kwargs,
    ):
        positions = positions.to(torch.int32)
        meta = attn_metadata[self.decoder.layers[0].self_attn.layer_name]
        is_prefill = meta["max_query_len"] > meta["decode_token_threshold"]

        hidden_states = self.decoder(input_ids, positions, attn_metadata, is_prefill)
        if sampling_positions is not None:
            hidden_states = torch.index_select(hidden_states, 0, sampling_positions)

        if self.sampler is None:
            return self.compute_logits(hidden_states)

        local = self._mask_pad_local(self._logits_local(hidden_states), rank)
        sampled = self.sampler(
            local,
            sampling_params,
            logit_mask=self._pad_logit_mask(logit_mask),
            tp_rank=rank,
        )
        return sampled, None
