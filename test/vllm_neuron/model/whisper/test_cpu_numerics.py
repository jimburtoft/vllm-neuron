# SPDX-License-Identifier: Apache-2.0
"""CPU numerics test for the native Whisper model against HF transformers.

The test builds a small random Whisper with HF transformers and saves it as
safetensors. It loads the checkpoint into the native model, and then does a
greedy decode through the paged KV path, as the runner does:

* prefill of the 4-token decoder prompt, padded with ``PAD_SLOT_ID`` (-1) slots,
* decode steps with a block table that holds -1 for unused blocks,
* the encoder-cache bookkeeping in ``embed_multimodal``.

The test compares the tokens and the logits with the HF model on each step.

Run with:
    pytest test/vllm_neuron/model/whisper/test_cpu_numerics.py -v
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
MAX_BLOCKS_PER_SEQ = 6
PROMPT = [50258, 50259, 50360, 50364]  # <|sot|><|en|><|transcribe|><|notimestamps|>


class _FakeEncoderCache:
    """Record the allocate calls of embed_multimodal."""

    def __init__(self):
        self.allocated = {}

    def allocate(self, mm_hash, tokens_per_block):
        self.allocated[mm_hash] = list(tokens_per_block)
        return [0]


@pytest.fixture(scope="module")
def hf_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.WhisperConfig(
        vocab_size=VOCAB,
        num_mel_bins=128,
        d_model=D_MODEL,
        encoder_layers=LAYERS,
        decoder_layers=LAYERS,
        encoder_attention_heads=HEADS,
        decoder_attention_heads=HEADS,
        encoder_ffn_dim=4 * D_MODEL,
        decoder_ffn_dim=4 * D_MODEL,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
    )
    hf = transformers.WhisperForConditionalGeneration(cfg).eval().to(torch.float32)
    # Make the biases non-zero, so the test checks the bias order.
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if name.endswith("bias"):
                p.normal_(0, 0.02)
    path = tmp_path_factory.mktemp("whisper_tiny")
    hf.save_pretrained(path, safe_serialization=True)
    return path, hf


def _build_native(path, max_num_seqs=1):
    from vllm_neuron.model.whisper.model_bf16 import WhisperForConditionalGeneration

    hf_cfg = transformers.WhisperConfig.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_configs(
        hf_cfg,
        None,
        None,
        check_dims=False,
        torch_dtype=torch.float32,
        max_num_seqs=max_num_seqs,
    )
    model.load_weights(str(path), torch.device("cpu"), None)
    model.eval()
    caches = {}
    for layer in model.decoder.layers:
        sa = layer.self_attn
        shape = (NUM_BLOCKS, sa.num_key_value_heads_per_rank, BLOCK_SIZE, sa.head_dim)
        # Fill with noise. A read of a slot that is not written must show.
        caches[sa.layer_name] = [torch.randn(shape), torch.randn(shape)]
    model.bind_kv_cache(caches)
    return model


def _metadata(model, slot_mapping, block_table, max_query_len, request_slots=None):
    meta = {
        "slot_mapping": slot_mapping,
        "block_size": BLOCK_SIZE,
        "block_table_tensor": block_table,
        "max_query_len": max_query_len,
        "decode_token_threshold": 1,
    }
    if request_slots is not None:
        meta["request_slots"] = torch.tensor(request_slots, dtype=torch.int32)
    return {layer.self_attn.layer_name: meta for layer in model.decoder.layers}


def test_greedy_matches_hf(hf_checkpoint):
    path, hf = hf_checkpoint
    model = _build_native(path)

    torch.manual_seed(1)
    mel = torch.randn(1, 128, 3000)
    cache = _FakeEncoderCache()
    model.embed_multimodal(input_features=mel, encoder_cache=cache, mm_hashes=["clip0"])
    assert cache.allocated == {"clip0": [1500]}
    assert getattr(model.audio_tower, "cross_k_0").abs().sum() > 0

    # The sequence uses blocks 3 and 5. Block 0 is the null block.
    seq_blocks = [3, 5]
    block_table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
    block_table[0, : len(seq_blocks)] = torch.tensor(seq_blocks, dtype=torch.int32)

    def slot(pos):
        return seq_blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE

    # Prefill: 4 real tokens, padded to 8 with PAD_SLOT_ID, as the runner does.
    n = len(PROMPT)
    pad = 4
    input_ids = torch.tensor(PROMPT + [0] * pad, dtype=torch.int32)
    positions = torch.tensor(list(range(n)) + [n - 1] * pad, dtype=torch.int32)
    slots = torch.tensor([slot(p) for p in range(n)] + [-1] * pad, dtype=torch.int64)
    out = model(
        input_ids=input_ids,
        positions=positions,
        attn_metadata=_metadata(model, slots, block_table, n + pad),
        sampling_positions=torch.tensor([n - 1]),
        encoder_embedding_buffer=torch.zeros(2, 1500, D_MODEL),
        encoder_block_ids=torch.zeros(1, dtype=torch.int32),
        encoder_positions=torch.zeros(1, 1500, dtype=torch.int32),
    )
    tokens = list(PROMPT)
    native_logits = [out[0]]
    tokens.append(int(out[0].argmax()))

    for _ in range(9):
        pos = len(tokens) - 1
        out = model(
            input_ids=torch.tensor([tokens[-1]], dtype=torch.int32),
            positions=torch.tensor([pos], dtype=torch.int32),
            attn_metadata=_metadata(
                model, torch.tensor([slot(pos)], dtype=torch.int64), block_table, 1
            ),
            sampling_positions=torch.tensor([0]),
        )
        native_logits.append(out[0])
        tokens.append(int(out[0].argmax()))

    # HF reference: a full forward on each prefix (no cache).
    with torch.no_grad():
        enc = hf.model.encoder(mel).last_hidden_state
        ref_tokens = list(PROMPT)
        for step in range(len(native_logits)):
            dec = torch.tensor([ref_tokens])
            logits = hf(encoder_outputs=(enc,), decoder_input_ids=dec).logits[0, -1]
            torch.testing.assert_close(native_logits[step], logits, atol=2e-4, rtol=2e-4)
            ref_tokens.append(int(logits.argmax()))
    assert tokens == ref_tokens


def test_pad_mask_uses_runtime_rank(hf_checkpoint):
    """Only the columns at or after the real vocab get -inf, for each rank."""
    path, _ = hf_checkpoint
    model = _build_native(path)
    # Use a vocab layout of 4 ranks: 51866 pads to 51868, 12967 columns per rank.
    model.vocab_pad, model.padded_vocab, model.per_rank = 2, 51868, 12967
    local = torch.zeros(1, model.per_rank)
    for r in range(4):
        out = model._mask_pad_local(local, torch.tensor(r, dtype=torch.int32))
        n_neg = int(torch.isinf(out).sum())
        assert n_neg == (2 if r == 3 else 0)
    mask = model._pad_logit_mask(torch.ones(1, VOCAB, dtype=torch.bool))
    assert mask.shape[-1] == 51868 and not mask[0, VOCAB:].any()


def _hf_greedy(hf, mel, steps):
    with torch.no_grad():
        enc = hf.model.encoder(mel).last_hidden_state
        tokens = list(PROMPT)
        for _ in range(steps):
            logits = hf(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([tokens])).logits
            tokens.append(int(logits[0, -1].argmax()))
    return tokens


def test_batched_requests_use_own_cross_kv(hf_checkpoint):
    """Two requests in their own slots: prefill one at a time, then decode as a batch.

    The batch rows are in the reverse order of the slots, and a padded third
    row reads the scratch slot. Each request must match its own HF greedy
    decode.
    """
    path, hf = hf_checkpoint
    model = _build_native(path, max_num_seqs=3)
    assert getattr(model.audio_tower, "cross_k_0").shape[0] == 4

    torch.manual_seed(2)
    mels = [torch.randn(1, 128, 3000), torch.randn(1, 128, 3000)]
    req_slots = [2, 0]
    blocks = [[4], [7]]
    cache = _FakeEncoderCache()
    for r in range(2):
        model.embed_multimodal(
            input_features=mels[r], encoder_cache=cache, mm_hashes=[f"c{r}"],
            request_slots=[req_slots[r]],
        )

    def slot(r, pos):
        return blocks[r][pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE

    tokens = [list(PROMPT), list(PROMPT)]
    n = len(PROMPT)
    for r in range(2):
        table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
        table[0, 0] = blocks[r][0]
        out = model(
            input_ids=torch.tensor(PROMPT, dtype=torch.int32),
            positions=torch.arange(n, dtype=torch.int32),
            attn_metadata=_metadata(
                model, torch.tensor([slot(r, p) for p in range(n)]), table, n,
                request_slots=[req_slots[r]],
            ),
            sampling_positions=torch.tensor([n - 1]),
        )
        tokens[r].append(int(out[0].argmax()))

    # Batch rows: row 0 = request 1, row 1 = request 0, row 2 = padding.
    order = [1, 0]
    table = torch.full((3, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
    for row, r in enumerate(order):
        table[row, 0] = blocks[r][0]
    for _ in range(4):
        pos = [len(tokens[r]) - 1 for r in order]
        out = model(
            input_ids=torch.tensor([tokens[r][-1] for r in order] + [0], dtype=torch.int32),
            positions=torch.tensor(pos + [0], dtype=torch.int32),
            attn_metadata=_metadata(
                model,
                torch.tensor([slot(r, p) for r, p in zip(order, pos)] + [-1]),
                table,
                1,
                request_slots=[req_slots[r] for r in order] + [3],
            ),
            sampling_positions=torch.arange(3),
        )
        for row, r in enumerate(order):
            tokens[r].append(int(out[row].argmax()))

    for r in range(2):
        assert tokens[r] == _hf_greedy(hf, mels[r], len(tokens[r]) - n), r


def test_batched_prefill_matches_single(hf_checkpoint):
    """One batched encoder call + one prefill of several rows equals the one-row path."""
    path, hf = hf_checkpoint
    model = _build_native(path, max_num_seqs=4)
    torch.manual_seed(3)
    mels = torch.randn(3, 128, 3000)
    model.embed_multimodal(
        input_features=mels, encoder_cache=_FakeEncoderCache(),
        mm_hashes=["a", "b", "c"], request_slots=[2, 0, 1],
    )
    blocks = {2: 4, 0: 7, 1: 9}
    n, S = len(PROMPT), 8
    rows = [2, 0, 1]
    table = torch.full((3, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
    ids, pos, sm = [], [], []
    for r, s in enumerate(rows):
        table[r, 0] = blocks[s]
        ids += PROMPT + [0] * (S - n)
        pos += list(range(n)) + [n - 1] * (S - n)
        sm += [blocks[s] * BLOCK_SIZE + p for p in range(n)] + [-1] * (S - n)
    out = model(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        positions=torch.tensor(pos, dtype=torch.int32),
        attn_metadata=_metadata(model, torch.tensor(sm), table, S, request_slots=rows),
        sampling_positions=torch.tensor([r * S + n - 1 for r in range(3)]),
    )
    for r in range(3):
        ref = _hf_greedy(hf, mels[r : r + 1], 1)[-1]
        assert int(out[r].argmax()) == ref, r


def test_embed_multimodal_needs_slots_for_batch(hf_checkpoint):
    path, _ = hf_checkpoint
    model = _build_native(path)
    with pytest.raises(ValueError, match="request_slots"):
        model.embed_multimodal(
            input_features=torch.zeros(2, 128, 3000),
            encoder_cache=_FakeEncoderCache(),
            mm_hashes=["a", "b"],
        )


def test_warmup_inputs(hf_checkpoint):
    path, _ = hf_checkpoint
    model = _build_native(path)
    inputs = model.build_encoder_synthetic_inputs(1500, None, torch.device("cpu"))
    assert inputs["input_features"].shape == (1, 128, 3000)
    out = model.get_encoder()(
        **inputs, encoder_cache_buffer=torch.zeros(2, 1500, D_MODEL),
        write_block_ids=torch.zeros(1, dtype=torch.int64),
    )
    assert out.shape == (1, 1500, D_MODEL)
    spec = model.get_kv_spec()
    assert [s.name for s in spec.layers] == [
        f"decoder.layers.{i}.self_attn" for i in range(LAYERS)
    ]


def test_fused_decode_matches_torch_path(hf_checkpoint, monkeypatch):
    """WHISPER_FUSED_DECODE=1 (attention_decode, torch fallback on CPU) gives the HF greedy tokens."""
    from vllm_neuron.functional.attention import attention_decode as _ad

    if not hasattr(_ad, "_fold_gqa_block_kv_mask"):
        pytest.skip("The fused decode needs the GQA decode-mask fold (#4817).")
    path, hf = hf_checkpoint
    monkeypatch.setenv("WHISPER_FUSED_DECODE", "1")
    model = _build_native(path)
    assert model.decoder.layers[0].self_attn.fused
    torch.manual_seed(1)
    mel = torch.randn(1, 128, 3000)
    model.embed_multimodal(input_features=mel, encoder_cache=_FakeEncoderCache(), mm_hashes=["c"])
    blocks = [3, 5]
    # The NKI decode mask needs a context of a multiple of 128 positions.
    table = torch.full((1, 128 // BLOCK_SIZE), -1, dtype=torch.int32)
    table[0, :2] = torch.tensor(blocks, dtype=torch.int32)

    def slot(pos):
        return blocks[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE

    n = len(PROMPT)
    out = model(
        input_ids=torch.tensor(PROMPT, dtype=torch.int32),
        positions=torch.arange(n, dtype=torch.int32),
        attn_metadata=_metadata(model, torch.tensor([slot(p) for p in range(n)]), table, n),
        sampling_positions=torch.tensor([n - 1]),
    )
    tokens = list(PROMPT) + [int(out[0].argmax())]
    for _ in range(9):
        pos = len(tokens) - 1
        out = model(
            input_ids=torch.tensor([tokens[-1]], dtype=torch.int32),
            positions=torch.tensor([pos], dtype=torch.int32),
            attn_metadata=_metadata(model, torch.tensor([slot(pos)]), table, 1),
            sampling_positions=torch.tensor([0]),
        )
        tokens.append(int(out[0].argmax()))
    assert tokens == _hf_greedy(hf, mel, len(tokens) - n)


@pytest.mark.parametrize(
    "mode,fused,kt",
    [("kernel", "0", "0"), ("kernel", "1", "0"), ("perm", "1", "0"), ("perm", "1", "1"),
     ("mask", "1", "1"), ("perm-active", "1", "1")],
)
def test_kernel_cross_mode_batched(hf_checkpoint, monkeypatch, mode, fused, kt):
    """WHISPER_CROSS_MODE=kernel (blocked cross-KV, attention_decode) and perm give the HF tokens.

    Two requests in slots 2 and 0, decoded as one batch in reverse slot order
    with a padded third row. On CPU the kernel call runs the torch fallback.
    """
    from vllm_neuron.functional.attention import attention_decode as _ad

    if not hasattr(_ad, "_fold_gqa_block_kv_mask"):
        pytest.skip("The kernel cross mode needs the GQA decode-mask fold (#4817).")
    path, hf = hf_checkpoint
    if mode == "perm-active":
        # The batch has 3 rows and the slots are 2, 0 and 3 (scratch): every
        # slot is below the first 4, so the active window covers them.
        mode = "perm"
        monkeypatch.setenv("WHISPER_CROSS_ACTIVE", "1")
    monkeypatch.setenv("WHISPER_CROSS_MODE", mode)
    monkeypatch.setenv("WHISPER_FUSED_DECODE", fused)
    monkeypatch.setenv("WHISPER_CROSS_KT", kt)
    model = _build_native(path, max_num_seqs=3)
    assert model.audio_tower.blocked == (mode == "kernel")
    from vllm_neuron.model.whisper.model_bf16 import CROSS_BLOCKS_PER_SLOT

    assert model.audio_tower.cross_k_0.shape[0] == (
        4 * CROSS_BLOCKS_PER_SLOT if mode == "kernel" else 4
    )

    torch.manual_seed(2)
    mels = [torch.randn(1, 128, 3000), torch.randn(1, 128, 3000)]
    req_slots = [2, 0]
    blocks = [[3, 4], [7, 8]]
    width = 128 // BLOCK_SIZE
    for r in range(2):
        model.embed_multimodal(
            input_features=mels[r], encoder_cache=_FakeEncoderCache(), mm_hashes=[f"c{r}"],
            request_slots=[req_slots[r]],
        )

    def slot(r, pos):
        return blocks[r][pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE

    tokens = [list(PROMPT), list(PROMPT)]
    n = len(PROMPT)
    for r in range(2):
        table = torch.full((1, width), -1, dtype=torch.int32)
        table[0, :2] = torch.tensor(blocks[r], dtype=torch.int32)
        out = model(
            input_ids=torch.tensor(PROMPT, dtype=torch.int32),
            positions=torch.arange(n, dtype=torch.int32),
            attn_metadata=_metadata(
                model, torch.tensor([slot(r, p) for p in range(n)]), table, n,
                request_slots=[req_slots[r]],
            ),
            sampling_positions=torch.tensor([n - 1]),
        )
        tokens[r].append(int(out[0].argmax()))

    order = [1, 0]
    table = torch.full((3, width), -1, dtype=torch.int32)
    for row, r in enumerate(order):
        table[row, :2] = torch.tensor(blocks[r], dtype=torch.int32)
    for _ in range(6):
        pos = [len(tokens[r]) - 1 for r in order]
        out = model(
            input_ids=torch.tensor([tokens[r][-1] for r in order] + [0], dtype=torch.int32),
            positions=torch.tensor(pos + [0], dtype=torch.int32),
            attn_metadata=_metadata(
                model,
                torch.tensor([slot(r, p) for r, p in zip(order, pos)] + [-1]),
                table,
                1,
                request_slots=[req_slots[r] for r in order] + [3],
            ),
            sampling_positions=torch.arange(3),
        )
        for row, r in enumerate(order):
            tokens[r].append(int(out[row].argmax()))

    for r in range(2):
        assert tokens[r] == _hf_greedy(hf, mels[r], len(tokens[r]) - n), r
