# SPDX-License-Identifier: Apache-2.0
"""CPU tests: Qwen3-ASR as a speculative-decoding (DFlash / Eagle3) target.

- The text decoder captures the INPUT of each configured layer (pre-layer convention; DFlash
  requests HF output-hidden-state id + 1) and returns ``(hidden, aux)``; with no aux layers the
  hidden output is unchanged.
- ``forward`` returns the concatenated aux states, runs the greedy rejection sampler on a verify
  step, and the ``@async_speculative_decoding`` wrapper appends the last accepted token.
- The interface the runner and the DFlash drafter use: ``SupportsEagle3``,
  ``set_aux_hidden_state_layers`` (range-checked), ``model.embed_tokens`` alias, and the
  text-decoder fields on the composed config.

    VLLM_NEURON_CPU_MODE=1 pytest \
    test/vllm_neuron/model/qwen3_asr/bf16/modules/test_spec_decode_hooks.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.qwen3_asr import model_bf16 as mb
from vllm_neuron.model.qwen3_asr.config import Qwen3ASRConfig, Qwen3ASRTextConfig
from vllm_neuron.model.qwen3_asr.model_bf16 import Qwen3ASRForConditionalGeneration
from vllm_neuron.utils.executor import MPExecutor

pytestmark = [pytest.mark.fast]

BLOCK = 32
HEAD_DIM = 128
HIDDEN = 256
VOCAB = 512
NUM_BLOCKS = 12
NUM_LAYERS = 4


def _text_config():
    return Qwen3ASRTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        num_hidden_layers=NUM_LAYERS,
        vocab_size=VOCAB,
        torch_dtype=torch.bfloat16,
        neuron_config=NeuronConfig(),
    )


def _make_load(aux_layers):
    def _model_load():
        cfg = _text_config()
        lm = mb.Qwen3ASRTextModel(cfg)
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for name, p in lm.named_parameters():
                if "norm" in name:
                    p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
                else:
                    p.copy_((torch.randn(p.shape, generator=g) * 0.06).to(p.dtype))
        caches = {
            f"layers.{i}.self_attn": [
                torch.zeros(NUM_BLOCKS, 2, BLOCK, HEAD_DIM, dtype=torch.bfloat16),
                torch.zeros(NUM_BLOCKS, 2, BLOCK, HEAD_DIM, dtype=torch.bfloat16),
            ]
            for i in range(NUM_LAYERS)
        }
        holder = SimpleNamespace(language_model=lm)
        Qwen3ASRForConditionalGeneration.bind_kv_cache(holder, caches)
        lm.aux_hidden_state_layers = list(aux_layers)
        seen = {}
        for i, layer in enumerate(lm.layers):
            layer.register_forward_pre_hook(
                lambda mod, args, i=i: seen.__setitem__(i, args[0].detach().clone())
            )

        class Runner(torch.nn.Module):
            def forward(self, ids, positions, md):
                with torch.no_grad():
                    for k, v in caches.values():
                        k.zero_()
                        v.zero_()
                    seen.clear()
                    hidden, aux = lm(ids, positions, attn_metadata=md)
                    layer_inputs = torch.stack([seen[i] for i in range(NUM_LAYERS)])
                return (
                    hidden.float(),
                    torch.stack(aux).float() if aux else torch.zeros(0),
                    layer_inputs.float(),
                )

        return Runner()

    return _model_load


def _prefill_inputs(T):
    ids = torch.randint(0, VOCAB, (T,), generator=torch.Generator().manual_seed(7))
    pos = torch.arange(T)
    table = torch.arange(1, 1 + (T + BLOCK - 1) // BLOCK)
    slots = table[pos // BLOCK] * BLOCK + pos % BLOCK
    meta = {
        "block_table_tensor": table[None].to(torch.int32),
        "slot_mapping": slots,
        "max_query_len": T,
        "block_size": BLOCK,
        "max_blocks_per_seq": table.shape[0],
        "decode_token_threshold": 1,
    }
    md = {f"layers.{i}.self_attn": meta for i in range(NUM_LAYERS)}
    return ids, pos, md


def _run(aux_layers, args):
    ex = MPExecutor(world_size=1, model_load=_make_load(aux_layers), device="cpu")
    try:
        ex.dispatch(*args)
        return ex.collect(timeout=600)[0]
    finally:
        ex.shutdown()


def test_aux_capture_is_layer_input_and_does_not_change_output():
    args = _prefill_inputs(64)
    h0, aux0, _ = _run([], args)
    h1, aux1, layer_inputs = _run([0, 2, 3], args)
    assert aux0.numel() == 0
    torch.testing.assert_close(h1, h0, atol=0, rtol=0)
    assert aux1.shape == (3, 64, HIDDEN)
    for j, layer in enumerate([0, 2, 3]):
        torch.testing.assert_close(aux1[j], layer_inputs[layer], atol=0, rtol=0)
    # Layer 0's input is the embedding (+ merged audio rows); later ones differ.
    assert not torch.equal(aux1[1], aux1[0])


# ---------------------------------------------------------------------------
# forward tail on a stand-in (no device graph)
# ---------------------------------------------------------------------------


class _StandIn:
    """Just enough of Qwen3ASRForConditionalGeneration for its forward."""

    def __init__(self, hidden, aux, w):
        self.world_size = 1
        self._gather_logits = False
        self._fused_lmhead = False
        self.config = SimpleNamespace(vocab_size=w.shape[1])
        self.on_device_sampling_config = SimpleNamespace(all_greedy=True)
        self.language_model = lambda *a, **k: (hidden, aux)
        self.lm_head = lambda x: torch.matmul(x, w)
        self.sampler = lambda logits, *a, **k: torch.argmax(logits, -1).to(torch.int32)


def _make_meta(drafts, num_draft):
    cu = torch.cumsum(torch.tensor(num_draft, dtype=torch.int32), 0)
    return SimpleNamespace(
        draft_token_ids=drafts.to(torch.int32),
        num_draft_tokens=num_draft,
        max_spec_len=max(num_draft),
        cu_num_draft_tokens=cu,
    )


def _call(stand_in, T, decode, spec=None):
    md = {"layers.0.self_attn": {"max_query_len": 1 if decode else T, "decode_token_threshold": 1}}
    if spec is not None:
        md["layers.0.self_attn"]["max_query_len"] = T // spec.cu_num_draft_tokens.shape[0]
        md["layers.0.self_attn"]["decode_token_threshold"] = md["layers.0.self_attn"]["max_query_len"]
    return Qwen3ASRForConditionalGeneration.forward(
        stand_in,
        torch.zeros(T, dtype=torch.long),
        positions=torch.arange(T),
        attn_metadata=md,
        sampling_positions=torch.arange(T) if (decode or spec is not None) else torch.tensor([T - 1]),
        spec_decode_metadata=spec,
        rank=torch.tensor(0),
    )


def test_forward_returns_aux_concat():
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(4, 64, generator=g)
    aux = [torch.randn(4, 64, generator=g) for _ in range(3)]
    w = torch.randn(64, 100, generator=g)
    tokens, aux_out, logits = _call(_StandIn(hidden, aux, w), 4, decode=True)
    assert logits is None
    assert torch.equal(tokens, torch.argmax(hidden @ w, -1).to(torch.int32))
    assert torch.equal(aux_out, torch.cat(aux, -1)) and aux_out.shape == (4, 192)
    # no draft configured: the original 2-tuple
    tokens2, logits2 = _call(_StandIn(hidden, [], w), 4, decode=True)
    assert torch.equal(tokens2, tokens) and logits2 is None


def test_verify_step_rejection_and_last_accepted():
    """Two requests, K=3: request 0 accepts 2 drafts, request 1 rejects the first."""
    K, V = 3, 50
    # the target's greedy token per verify row = row index pattern below
    target = torch.tensor([11, 12, 13, 14, 21, 22, 23, 24])
    hidden = torch.nn.functional.one_hot(target, V).float()
    w = torch.eye(V)
    drafts = torch.tensor([11, 12, 99, 98, 22, 23])
    meta = _make_meta(drafts, [K, K])
    out = _call(_StandIn(hidden, [torch.ones(8, 4)], w), 8, decode=True, spec=meta)
    rej, aux, logits, last = out
    assert rej.tolist() == [[11, 12, 13, -1], [21, -1, -1, -1]]
    assert last.tolist() == [13, 21]
    assert aux.shape == (8, 4) and logits is None


def test_interface():
    assert supports_eagle3(Qwen3ASRForConditionalGeneration)
    assert getattr(Qwen3ASRForConditionalGeneration, "_async_spec_decoded", False)
    lm = SimpleNamespace(layers=[None] * 28, aux_hidden_state_layers=[], embed_tokens="E")
    stub = SimpleNamespace(language_model=lm)
    Qwen3ASRForConditionalGeneration.set_aux_hidden_state_layers(stub, (2, 8, 14, 20, 26))
    assert lm.aux_hidden_state_layers == [2, 8, 14, 20, 26]
    assert Qwen3ASRForConditionalGeneration.get_eagle3_aux_hidden_state_layers(stub) == (2, 8, 14, 20, 26)
    with pytest.raises(ValueError, match="out of range"):
        Qwen3ASRForConditionalGeneration.set_aux_hidden_state_layers(stub, (2, 28))
    assert Qwen3ASRForConditionalGeneration.model.fget(stub) is lm


def test_composed_config_exposes_text_fields():
    tc = _text_config()
    c = Qwen3ASRConfig(text_config=tc)
    assert (c.hidden_size, c.vocab_size, c.num_hidden_layers) == (HIDDEN, VOCAB, NUM_LAYERS)
