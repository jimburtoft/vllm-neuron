# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Whisper draft-model proposer graph (``WhisperDraftPropose``).

A small random Whisper is the draft. The test compares the K draft tokens of one
proposer call with K greedy decode steps of the same model (no sampler, fp32 argmax):

* after the prefill (the bonus token goes after the prompt),
* after a verify step with partial acceptance (the block has the verified tokens,
  the bonus token, and one extra column).

Run with:
    pytest test/vllm_neuron/vllm/spec_decode/test_draft_model.py -v
"""

import pytest
import torch

pytestmark = [pytest.mark.fast]

transformers = pytest.importorskip("transformers")

D_MODEL = 64
HEADS = 4
LAYERS = 2
VOCAB = 51866
BLOCK_SIZE = 8
NUM_BLOCKS = 20
MAX_BLOCKS = 8
PROMPT = [50258, 50259, 50360, 50364]
K = 4


@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.WhisperConfig(
        vocab_size=VOCAB, num_mel_bins=128, d_model=D_MODEL, encoder_layers=LAYERS,
        decoder_layers=LAYERS, encoder_attention_heads=HEADS, decoder_attention_heads=HEADS,
        encoder_ffn_dim=4 * D_MODEL, decoder_ffn_dim=4 * D_MODEL, max_source_positions=1500,
        max_target_positions=448, pad_token_id=50256,
    )
    hf = transformers.WhisperForConditionalGeneration(cfg).eval()
    path = tmp_path_factory.mktemp("whisper_draft")
    hf.save_pretrained(path, safe_serialization=True)

    from vllm_neuron.model.whisper.model_bf16 import WhisperForConditionalGeneration

    with torch.device("meta"):
        m = WhisperForConditionalGeneration.from_configs(
            transformers.WhisperConfig.from_pretrained(path), None, None, check_dims=False,
            torch_dtype=torch.float32, max_num_seqs=1, layer_name_prefix="draft",
        )
    m.load_weights(str(path), torch.device("cpu"), None)
    m.eval()
    assert m.decoder.layers[0].self_attn.layer_name == "draft.layers.0.self_attn"
    torch.manual_seed(1)
    m.embed_multimodal(input_features=torch.randn(1, 128, 3000), mm_hashes=["c"])
    return m


def _fresh_caches(m):
    caches = {}
    for layer in m.decoder.layers:
        sa = layer.self_attn
        shape = (NUM_BLOCKS, sa.num_key_value_heads_per_rank, BLOCK_SIZE, sa.head_dim)
        caches[sa.layer_name] = [torch.randn(shape), torch.randn(shape)]
    m.bind_kv_cache(caches)


BLOCKS = [3, 5, 7]


def _table():
    t = torch.full((1, MAX_BLOCKS), -1, dtype=torch.int32)
    t[0, : len(BLOCKS)] = torch.tensor(BLOCKS, dtype=torch.int32)
    return t


def _slot(p):
    return BLOCKS[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE


def _meta(m, slots, S, thr):
    meta = {"slot_mapping": torch.tensor(slots, dtype=torch.int64), "block_size": BLOCK_SIZE,
            "block_table_tensor": _table(), "max_query_len": S, "decode_token_threshold": thr}
    return {layer.self_attn.layer_name: meta for layer in m.decoder.layers}


def _greedy(m, forced, n_new):
    """Prefill of the prompt, then the forced tokens, then n_new greedy tokens (one step each)."""
    _fresh_caches(m)
    n = len(PROMPT)
    m(input_ids=torch.tensor(PROMPT, dtype=torch.int32), positions=torch.arange(n, dtype=torch.int32),
      attn_metadata=_meta(m, [_slot(p) for p in range(n)], n, 1), sampling_positions=torch.tensor([n - 1]))
    seq = list(PROMPT) + [forced[0]]
    for i in range(1, len(forced) + n_new):
        p = len(seq) - 1
        out = m(input_ids=torch.tensor([seq[-1]], dtype=torch.int32), positions=torch.tensor([p], dtype=torch.int32),
                attn_metadata=_meta(m, [_slot(p)], 1, 1), sampling_positions=torch.tensor([0]))
        seq.append(forced[i] if i < len(forced) else int(out[0].argmax()))
    return seq


def _propose(m, **kw):
    from vllm_neuron.vllm.spec_decode.draft_model import WhisperDraftPropose

    g = WhisperDraftPropose(m, K, VOCAB)

    # CPU: the model has no on-device sampler and returns full logits. Take the argmax.
    def sample(ids, positions, meta, sampling_positions, rank):
        logits = m(input_ids=ids, positions=positions.to(torch.int32), attn_metadata=meta,
                   sampling_positions=sampling_positions)
        return logits.argmax(dim=-1).to(torch.int32)

    g._sample = sample
    return g(block_table=_table(), block_size=BLOCK_SIZE, rank=torch.tensor(0), **kw)


def test_propose_after_prefill(draft):
    m = draft
    n = len(PROMPT)
    bonus = 220
    ref = _greedy(m, [bonus], K)  # prompt, bonus, then K greedy draft tokens
    _fresh_caches(m)
    S = 8  # the prefill bucket: the prompt and pad columns
    ids = torch.tensor(PROMPT + [0] * (S - n), dtype=torch.int32)
    stacked, drafts = _propose(
        m, input_ids=ids, positions=torch.tensor(list(range(n)) + [n - 1] * (S - n)),
        last_token_indices=torch.tensor([n - 1]), raw_sampled_token_ids=torch.tensor([[bonus]], dtype=torch.int32),
    )
    assert stacked[0, 0].item() == bonus
    assert drafts[0].tolist() == ref[n + 1: n + 1 + K]


@pytest.mark.parametrize("accepted", [0, 2, K])
def test_propose_after_verify(draft, accepted):
    m = draft
    n = len(PROMPT)
    # The verified sequence: prompt, then t0 (the last token before the step), then
    # K draft tokens of which `accepted` match, then the bonus.
    t0, dr = 220, [1000 + j for j in range(K)]
    bonus = 777
    hist = [t0] + dr[:accepted] + [bonus]
    ref = _greedy(m, hist, K)
    # Draft KV of the prompt (the proposer wrote it at the prefill step).
    _fresh_caches(m)
    m(input_ids=torch.tensor(PROMPT, dtype=torch.int32), positions=torch.arange(n, dtype=torch.int32),
      attn_metadata=_meta(m, [_slot(p) for p in range(n)], n, 1), sampling_positions=torch.tensor([n - 1]))
    raw = torch.full((1, K + 1), -1, dtype=torch.int32)
    raw[0, :accepted] = torch.tensor(dr[:accepted], dtype=torch.int32)
    raw[0, accepted] = bonus
    stacked, drafts = _propose(
        m, input_ids=torch.tensor([t0] + dr, dtype=torch.int32), positions=torch.arange(n, n + K + 1),
        last_token_indices=torch.tensor([K]), raw_sampled_token_ids=raw,
    )
    assert stacked[0, 0].item() == bonus
    start = n + len(hist)
    assert drafts[0].tolist() == ref[start: start + K]
