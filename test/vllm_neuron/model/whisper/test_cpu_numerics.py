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
    with torch.device("meta"):
        # The runner builds the model on meta; load_weights materializes it.
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
     ("mask", "1", "1"), ("perm-active", "1", "1"), ("mqa", "0", "0"), ("mqa", "1", "0"), ("mqa2", "1", "0"), ("mqa4pad", "1", "0"),
     ("xkernel", "0", "0"), ("xkernel", "1", "0")],
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
    if mode == "mqa2":
        # Two MQA head groups of 2 heads (the test model has 4 heads).
        mode = "mqa"
        monkeypatch.setenv("WHISPER_CROSS_MQA_GROUP", "2")
    if mode == "mqa4pad":
        # One group of 4 heads (6144 positions), padded to 8192 as a span above
        # one flash tile would be (the test forces the pad at this size).
        mode = "mqa"
        monkeypatch.setenv("WHISPER_CROSS_MQA_GROUP", "4")
        from vllm_neuron.model.whisper import model_bf16 as _mb

        monkeypatch.setattr(_mb, "mqa_span", lambda g: 8192)
    if mode == "perm-active":
        # The batch has 3 rows and the slots are 2, 0 and 3 (scratch): every
        # slot is below the first 4, so the active window covers them.
        mode = "perm"
        monkeypatch.setenv("WHISPER_CROSS_ACTIVE", "1")
    monkeypatch.setenv("WHISPER_CROSS_MODE", mode)
    monkeypatch.setenv("WHISPER_FUSED_DECODE", fused)
    monkeypatch.setenv("WHISPER_CROSS_KT", kt)
    model = _build_native(path, max_num_seqs=3)
    assert model.audio_tower.blocked == (mode in ("kernel", "mqa"))
    from vllm_neuron.model.whisper.model_bf16 import CROSS_BLOCKS_PER_SLOT

    groups = getattr(model.audio_tower, "mqa_groups", 1)
    expect = {"kernel": 4 * CROSS_BLOCKS_PER_SLOT, "mqa": 4 * 128 * groups}.get(mode, 4)
    assert model.audio_tower.cross_k_0.shape[0] == expect
    if mode == "xkernel":
        # K^T [slots, h, d, 1536] and V [slots, h, 1536, d].
        assert model.audio_tower.cross_k_0.shape[1:] == (HEADS, D_MODEL // HEADS, 1536)
        assert model.audio_tower.cross_v_0.shape[1:] == (HEADS, 1536, D_MODEL // HEADS)

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


@pytest.mark.parametrize(
    "write,cross,in_out,self_attn",
    [("index", "mask", "decoder,lm_head", "torch"), ("where", "mask", "decoder,lm_head", "torch"),
     ("index", "mqa", "decoder,lm_head,encoder", "torch"), ("where", "mqa", "none", "torch"),
     ("index", "xkernel", "decoder,lm_head", "torch"), ("index", "xkernel", "decoder,lm_head", "kernel"),
     ("index", "mask", "none", "kernel")],
)
def test_slot_self_kv_batched(hf_checkpoint, monkeypatch, write, cross, in_out, self_attn):
    """WHISPER_SELF_KV=slot (model-owned contiguous self-KV) gives the HF greedy tokens.

    Two requests in slots 2 and 0: one prefill each, then batched decode in the
    reverse slot order with a padded third row. The decode table is 2 blocks wide
    (a 16-position context bucket) and then 6 blocks, so both context widths run.
    The paged caches hold noise and are never read.
    """
    from vllm_neuron.functional.attention import attention_decode as _ad

    if cross == "mqa" and not hasattr(_ad, "_fold_gqa_block_kv_mask"):
        pytest.skip("The MQA cross mode needs the GQA decode-mask fold (#4817).")
    path, hf = hf_checkpoint
    monkeypatch.setenv("WHISPER_SELF_KV", "slot")
    monkeypatch.setenv("WHISPER_SELF_WRITE", write)
    monkeypatch.setenv("WHISPER_CROSS_MODE", cross)
    monkeypatch.setenv("WHISPER_IN_OUT", in_out)
    monkeypatch.setenv("WHISPER_SELF_ATTN", self_attn)
    model = _build_native(path, max_num_seqs=3)
    sa = model.decoder.layers[0].self_attn
    # The runner builds on meta and load_weights calls to_empty: the slots must be zero.
    assert all(int(lyr.self_attn.slot_k.abs().sum()) == 0 for lyr in model.decoder.layers)
    if self_attn == "kernel":
        assert sa.self_kernel and sa.slot_k.shape == (4, 448, sa.kernel_heads * (D_MODEL // HEADS))
    else:
        assert sa.slot_kv and sa.slot_k.shape == (4 * HEADS, 448, D_MODEL // HEADS)
    if "decoder" in in_out:
        assert model.decoder.layers[0].mlp.fc1.weight_layout == "in_out"
    torch.manual_seed(2)
    mels = [torch.randn(1, 128, 3000), torch.randn(1, 128, 3000)]
    req_slots = [2, 0]
    for r in range(2):
        model.embed_multimodal(
            input_features=mels[r], encoder_cache=_FakeEncoderCache(), mm_hashes=[f"c{r}"],
            request_slots=[req_slots[r]],
        )
    tokens = [list(PROMPT), list(PROMPT)]
    n = len(PROMPT)
    # The paged slot mapping only marks real (>= 0) and pad (-1) tokens in slot mode.
    for r in range(2):
        table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
        out = model(
            input_ids=torch.tensor(PROMPT + [0] * 4, dtype=torch.int32),
            positions=torch.tensor(list(range(n)) + [n - 1] * 4, dtype=torch.int32),
            attn_metadata=_metadata(
                model, torch.tensor(list(range(n)) + [-1] * 4), table, n + 4,
                request_slots=[req_slots[r]],
            ),
            sampling_positions=torch.tensor([n - 1]),
        )
        tokens[r].append(int(out[0].argmax()))
    order = [1, 0]
    for step in range(8):
        width = 2 if step < 4 else MAX_BLOCKS_PER_SEQ
        table = torch.full((3, width), -1, dtype=torch.int32)
        pos = [len(tokens[r]) - 1 for r in order]
        assert max(pos) < width * BLOCK_SIZE
        out = model(
            input_ids=torch.tensor([tokens[r][-1] for r in order] + [0], dtype=torch.int32),
            positions=torch.tensor(pos + [0], dtype=torch.int32),
            attn_metadata=_metadata(
                model, torch.tensor([1, 1, -1]), table, 1,
                request_slots=[req_slots[r] for r in order] + [3],
            ),
            sampling_positions=torch.arange(3),
        )
        for row, r in enumerate(order):
            tokens[r].append(int(out[row].argmax()))
    for r in range(2):
        assert tokens[r] == _hf_greedy(hf, mels[r], len(tokens[r]) - n), r


def test_in_out_matches_out_in(hf_checkpoint, monkeypatch):
    """WHISPER_IN_OUT=decoder,lm_head,encoder gives the same logits as WHISPER_IN_OUT=none."""
    path, _ = hf_checkpoint
    torch.manual_seed(5)
    mel = torch.randn(1, 128, 3000)
    outs = []
    for parts in ("none", "decoder,lm_head,encoder"):
        monkeypatch.setenv("WHISPER_IN_OUT", parts)
        model = _build_native(path)
        model.embed_multimodal(input_features=mel, encoder_cache=_FakeEncoderCache(), mm_hashes=["c"])
        table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
        table[0, 0] = 3
        n = len(PROMPT)
        outs.append(model(
            input_ids=torch.tensor(PROMPT, dtype=torch.int32),
            positions=torch.arange(n, dtype=torch.int32),
            attn_metadata=_metadata(model, torch.tensor([24 + p for p in range(n)]), table, n),
            sampling_positions=torch.tensor([n - 1]),
        ))
    lyr = model.encoder.layers[0]
    assert lyr.self_attn.q_proj.weight_layout == "in_out" and lyr.mlp.fc2.in_out
    torch.testing.assert_close(outs[0], outs[1], atol=1e-5, rtol=1e-5)


def test_encoder_nf_matches_plain(hf_checkpoint, monkeypatch):
    """WHISPER_ENCODER_NF=1 (fused LN + QKV, flash, o_proj; torch fallbacks on CPU) matches the plain encoder."""
    path, _ = hf_checkpoint
    torch.manual_seed(6)
    mel = torch.randn(1, 128, 3000)
    outs = []
    for nf in ("0", "1"):
        monkeypatch.setenv("WHISPER_ENCODER_FLASH", nf)
        monkeypatch.setenv("WHISPER_ENCODER_NF", nf)
        model = _build_native(path)
        assert model.encoder.layers[0].self_attn.nf == (nf == "1")
        with torch.no_grad():
            outs.append(model.encoder(mel))
    torch.testing.assert_close(outs[0], outs[1], atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("cross", ["mask", "mqa", "xkernel"])
def test_cross_kv_fused_matches_per_layer(hf_checkpoint, monkeypatch, cross):
    """WHISPER_CROSS_KV_FUSED=1 (one K|V GEMM for all layers) writes the same cross-KV."""
    from vllm_neuron.functional.attention import attention_decode as _ad

    if cross == "mqa" and not hasattr(_ad, "_fold_gqa_block_kv_mask"):
        pytest.skip("The MQA cross mode needs the GQA decode-mask fold (#4817).")
    path, _ = hf_checkpoint
    monkeypatch.setenv("WHISPER_CROSS_MODE", cross)
    torch.manual_seed(3)
    mel = torch.randn(1, 128, 3000)
    bufs = {}
    for fused in ("0", "1"):
        monkeypatch.setenv("WHISPER_CROSS_KV_FUSED", fused)
        model = _build_native(path, max_num_seqs=3)
        model.embed_multimodal(
            input_features=mel, encoder_cache=_FakeEncoderCache(), mm_hashes=["c"],
            request_slots=[1],
        )
        bufs[fused] = [getattr(model.audio_tower, f"cross_{k}_{i}").clone()
                       for i in range(LAYERS) for k in ("k", "v")]
    for a, b in zip(bufs["0"], bufs["1"]):
        torch.testing.assert_close(a, b, atol=2e-2, rtol=2e-2)


def test_fp8_decoder_mlp_close_to_bf16(hf_checkpoint, monkeypatch):
    """WHISPER_FP8=decoder_mlp: per-channel fp8 fc1 / fc2 give logits close to bf16 (CPU dequant path)."""
    path, _ = hf_checkpoint
    torch.manual_seed(4)
    mel = torch.randn(1, 128, 3000)
    n = len(PROMPT)
    logits = {}
    for fp8 in ("", "decoder_mlp"):
        monkeypatch.setenv("WHISPER_FP8", fp8)
        model = _build_native(path)
        if fp8:
            mlp = model.decoder.layers[0].mlp
            assert mlp.fp8 and mlp.w1_q.dtype == torch.float8_e4m3fn and mlp.fc1.weight.numel() == 0
        model.embed_multimodal(input_features=mel, encoder_cache=_FakeEncoderCache(), mm_hashes=["c"])
        table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
        table[0, 0] = 1
        out = model(
            input_ids=torch.tensor(PROMPT, dtype=torch.int32),
            positions=torch.arange(n, dtype=torch.int32),
            attn_metadata=_metadata(model, torch.arange(BLOCK_SIZE, BLOCK_SIZE + n), table, n),
            sampling_positions=torch.tensor([n - 1]),
        )
        logits[fp8] = out
    a, b = logits[""].float().flatten(), logits["decoder_mlp"].float().flatten()
    assert torch.nn.functional.cosine_similarity(a, b, dim=0) > 0.99


@pytest.mark.parametrize("mode", ["bf16w", "bf16"])
def test_lm_head_bf16(hf_checkpoint, monkeypatch, mode):
    """WHISPER_LM_HEAD=bf16w stores the tied LM head in bf16 and runs an fp32 GEMM over it;
    bf16 also runs the GEMM in bf16 (close logits)."""
    path, _ = hf_checkpoint
    ref = _build_native(path)
    monkeypatch.setenv("WHISPER_LM_HEAD", mode)
    m = _build_native(path)
    assert m.lm_head.weight.dtype == torch.bfloat16
    w = ref.lm_head.weight.to(torch.bfloat16).float()
    assert torch.equal(m.lm_head.weight.float(), w)
    x = torch.randn(3, D_MODEL).to(torch.bfloat16)
    want = x.float() @ (w if m.lm_head.weight.shape[0] == D_MODEL else w.t())
    got = m._logits_local(x)
    assert got.dtype == torch.float32
    tol = 1e-5 if mode == "bf16w" else 3e-2
    torch.testing.assert_close(got, want, rtol=tol, atol=tol)


def test_fp8_e4m3_codes_round_trip():
    """fp8_e4m3_encode gives the trn2 e4m3 codes (round to nearest even, clamp 240); decode inverts it."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    import numpy as np

    from vllm_neuron._kernels.whisper.cross_attention_decode import (
        fp8_e4m3_codes,
        fp8_e4m3_decode,
        fp8_e4m3_encode,
        fp8_e4m3_values,
    )

    torch.manual_seed(0)
    x = torch.cat([torch.randn(20000) * s for s in (0.003, 0.05, 1.0, 8.0, 60.0)]
                  + [torch.tensor([0.0, 240.0, -240.0, 300.0, -1000.0, 2**-9, 2**-10, 15.5])])
    x = x.to(torch.bfloat16)
    want = np.clip(x.float().numpy(), -240, 240).astype(ml_dtypes.float8_e4m3).view(np.uint8)
    got = fp8_e4m3_codes(x).numpy()
    assert ((got == want) | ((want == 0x80) & (got == 0))).all()
    dec = fp8_e4m3_values(torch.from_numpy(want), torch.float32).numpy()
    assert np.array_equal(dec, want.view(ml_dtypes.float8_e4m3).astype(np.float32))
    # Packed: two codes per bf16 element; decode inverts encode.
    xp = x[: x.numel() // 2 * 2].view(-1, 2)
    packed = fp8_e4m3_encode(xp)
    assert packed.dtype == torch.bfloat16 and packed.shape == (xp.shape[0], 1)
    assert torch.equal(fp8_e4m3_decode(packed, torch.float32), fp8_e4m3_values(fp8_e4m3_codes(xp), torch.float32))


def test_cross_fp8_xkernel_close_to_bf16(hf_checkpoint, monkeypatch):
    """WHISPER_CROSS_FP8=1 stores the xkernel cross-KV as e4m3 codes; the batched decode logits stay close."""
    path, _ = hf_checkpoint
    monkeypatch.setenv("WHISPER_SELF_KV", "slot")
    monkeypatch.setenv("WHISPER_CROSS_MODE", "xkernel")
    outs = []
    for fp8 in ("0", "1"):
        monkeypatch.setenv("WHISPER_CROSS_FP8", fp8)
        model = _build_native(path, max_num_seqs=3)
        k0 = model.audio_tower.cross_k_0
        assert k0.shape[-1] == (768 if fp8 == "1" else 1536)
        torch.manual_seed(2)
        mel = torch.randn(1, 128, 3000)
        model.embed_multimodal(input_features=mel, encoder_cache=_FakeEncoderCache(), mm_hashes=["c"],
                               request_slots=[1])
        n = len(PROMPT)
        table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
        out = model(
            input_ids=torch.tensor(PROMPT + [0] * 4, dtype=torch.int32),
            positions=torch.tensor(list(range(n)) + [n - 1] * 4, dtype=torch.int32),
            attn_metadata=_metadata(model, torch.tensor(list(range(n)) + [-1] * 4), table, n + 4,
                                    request_slots=[1]),
            sampling_positions=torch.tensor([n - 1]),
        )
        outs.append(out[0].float())
    cos = torch.nn.functional.cosine_similarity(outs[0], outs[1], dim=-1).min()
    assert cos > 0.99


def test_xkernel_dma_rejects_one_column_pv(monkeypatch):
    """VB=0 (one-column PV) is wrong on trn2 hardware, so the config rejects it at load."""
    from vllm_neuron.model.whisper import model_bf16

    monkeypatch.delenv("WHISPER_XKERNEL_DMA", raising=False)
    assert model_bf16.xkernel_dma() == (5, 10, 5, 0)
    monkeypatch.setenv("WHISPER_XKERNEL_DMA", "5,10,0")
    with pytest.raises(ValueError, match="VB must be >= 1"):
        model_bf16.xkernel_dma()
    monkeypatch.setenv("WHISPER_XKERNEL_DMA", "5,10,5,1")
    assert model_bf16.xkernel_dma() == (5, 10, 5, 1)


@pytest.mark.parametrize(
    "cross,self_attn,rows_mode",
    [("xkernel", "kernel", "kernel"), ("xkernel", "kernel", "repeat"), ("xkernel", "torch", "kernel"),
     ("mask", "torch", "kernel")],
)
def test_slot_verify_step_matches_one_token_steps(hf_checkpoint, monkeypatch, cross, self_attn, rows_mode):
    """A verify-shaped step (S tokens per row) on the slot self-KV equals S one-token steps.

    Two requests in slots 2 and 0 and a pad row. After the prefill, one step with S = 3
    tokens per row (the HF greedy tokens) must give the same logits as three one-token
    decode steps, for every row and token. This is the target verify step of speculative
    decoding on the throughput layout (slot self-KV, xkernel cross-attention).
    """
    path, hf = hf_checkpoint
    monkeypatch.setenv("WHISPER_SELF_KV", "slot")
    monkeypatch.setenv("WHISPER_CROSS_MODE", cross)
    monkeypatch.setenv("WHISPER_SELF_ATTN", self_attn)
    monkeypatch.setenv("WHISPER_SPEC_ROWS", rows_mode)
    S = 3
    torch.manual_seed(3)
    mels = [torch.randn(1, 128, 3000), torch.randn(1, 128, 3000)]
    req_slots = [2, 0]
    n = len(PROMPT)
    ref = [_hf_greedy(hf, mels[r], S + 1) for r in range(2)]

    def prefill(model):
        for r in range(2):
            model.embed_multimodal(
                input_features=mels[r], encoder_cache=_FakeEncoderCache(), mm_hashes=[f"c{r}"],
                request_slots=[req_slots[r]],
            )
            table = torch.full((1, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
            model(
                input_ids=torch.tensor(PROMPT + [0] * 4, dtype=torch.int32),
                positions=torch.tensor(list(range(n)) + [n - 1] * 4, dtype=torch.int32),
                attn_metadata=_metadata(
                    model, torch.tensor(list(range(n)) + [-1] * 4), table, n + 4,
                    request_slots=[req_slots[r]],
                ),
                sampling_positions=torch.tensor([n - 1]),
            )

    order = [1, 0]
    table = torch.full((3, 2), -1, dtype=torch.int32)
    rs = [req_slots[r] for r in order] + [3]
    # S one-token steps.
    one = _build_native(path, max_num_seqs=3)
    prefill(one)
    step_logits = []
    for j in range(S):
        out = one(
            input_ids=torch.tensor([ref[r][n + j] for r in order] + [0], dtype=torch.int32),
            positions=torch.tensor([n + j] * 2 + [0], dtype=torch.int32),
            attn_metadata=_metadata(one, torch.tensor([1, 1, -1]), table, 1, request_slots=rs),
            sampling_positions=torch.arange(3),
        )
        step_logits.append(out[:2])
    # One step with S tokens per row (token order: row 0 tokens, row 1 tokens, pad row).
    ver = _build_native(path, max_num_seqs=3)
    prefill(ver)
    ids = [ref[r][n + j] for r in order for j in range(S)] + [0] * S
    pos = [n + j for _ in order for j in range(S)] + [0] * S
    meta = _metadata(ver, torch.tensor([1] * (2 * S) + [-1] * S), table, S, request_slots=rs)
    for m in meta.values():
        m["decode_token_threshold"] = S
    out = ver(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        positions=torch.tensor(pos, dtype=torch.int32),
        attn_metadata=meta,
        sampling_positions=torch.arange(3 * S),
    )
    for row in range(2):
        for j in range(S):
            torch.testing.assert_close(out[row * S + j], step_logits[j][row], atol=2e-4, rtol=2e-4)
            assert int(out[row * S + j].argmax()) == ref[order[row]][n + j + 1]
