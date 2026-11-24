# Ministral 3

The text decoder of the Ministral 3 family on the Neuron backend, e.g.
[`mistralai/Ministral-3-8B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16).

Ministral 3 checkpoints declare the multimodal `Mistral3ForConditionalGeneration` architecture: a
`ministral3` text decoder under `text_config` and a Pixtral vision tower under `vision_config`. The
text decoder has the Llama architecture (GQA attention without bias, RMSNorm, SiLU-gated MLP), so
this package builds the [`llama3`](../llama3/) implementation with two additions:

* **YaRN RoPE.** `LlamaRotaryEmbedding` implements `rope_type: "yarn"` (transformers
  `_compute_yarn_parameters`, including `attention_factor` on cos/sin). It changes 43 of the 64
  rotary frequencies of Ministral-3-8B, so it matters at every position, not only past the original
  context.
* **Checkpoint prefix.** The decoder weights are under `language_model.`.
* **Images (optional).** With images enabled, `multimodal.py` adds the Pixtral vision tower and
  the Mistral3 projector (`vision.py`) on the encoder TP / DP groups. With
  `--language-model-only` only the text decoder is built.

## Architecture (Ministral-3-8B-Instruct-2512)

| Parameter | Value |
|---|---|
| hidden_size / num_hidden_layers / vocab_size | 4096 / 34 / 131072 |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 8 / 128 |
| intermediate_size | 14336 |
| RoPE | YaRN: rope_theta 1e6, factor 16, original_max_position_embeddings 16384, beta_fast 32, beta_slow 1, mscale = mscale_all_dim = 1 (attention_factor 1.0) |
| Llama 4 query scaling | beta 0.1 above position 16384 (implemented; see "Long context") |
| RMSNorm eps | 1e-5 |
| tie_word_embeddings | False |
| sliding_window | null |

`config.py` reads every value above from `text_config` and checks it against the built
`LlamaConfig` at load time. The llama3 defaults differ for rope_theta (5e5) and
tie_word_embeddings (True).

## Differences from llama3

| Area | llama3 | ministral3 |
|---|---|---|
| Config | flat Llama config | nested `text_config` of the `mistral3` wrapper; serving dtype is vLLM's `--dtype` |
| RoPE | default / llama3 | + YaRN |
| Weight names | `model.*`, `lm_head.weight` | `language_model.model.*`, `language_model.lm_head.weight` |
| Architecture names | `LlamaForCausalLM` | `Mistral3ForConditionalGeneration`, `Ministral3ForCausalLM` |

The vLLM release in this environment cannot import its own `Mistral3ForConditionalGeneration`
class with the installed transformers (`PixtralRotaryEmbedding` was renamed), and model config
validation inspects that class before the worker registers the Neuron models. The platform plugin
therefore registers the architecture name for validation (`platform.pre_register_and_update`).

## Optimized stack

`examples/vllm_neuron/models/ministral3/setup_optimized.sh` sets up:

- `VLLM_NEURON_FUSE_DECODE_NORMS=1`: the BF16 decode runs both RMSNorms inside the attention and MLP
  kernels (decode TPOT -8 % at batch 1 and 128).
- nkilib from `jimburtoft/nki-library@ministral3/unified-2.33` (overlay on the SDK 2.33 nkilib):
  - The batched block-KV decode attention writes the active token once per DMA-batch group and keeps the fast K
    gather at the 512 decode bucket. With `"decode_context_length_buckets": [512, 1024]` the decode step at
    batch 64-128 is 11 % faster than the SDK nkilib.
  - Unpacked FP8 K loads through the DMA-transpose gather when a core holds at least 4 KV heads (TP=2).
  - The per-token query scale for long context.
  - These changes go upstream through review.
  - Against the previous branch (`qwen3-asr/2.33-champion`) the unified branch is equal at batch 1 and up to
    8 % faster at batch 128 (BF16 TP=4 decode TPOT 29.9 -> 27.6 ms).
- `VLLM_NEURON_DECODE_VENDORED_ATTN=1` routes BF16 decode through the vendored attention block too (single
  V-extraction DMA). It measured within 2 % of the default, so it is off.
- Private NKI / vLLM caches per nkilib revision (the NKI trace-cache key does not hash nkilib source).
- `VLLM_NEURON_SWA_REAL_ADMISSION=1`, KV cap 0.6.

Recommended servers with it (one trn2.3xlarge, BF16, output tok/s at the throughput peak):

FP8 KV: use the packed layout (`--kv-cache-dtype fp8_e4m3` with `"fp8_packed_kv": true` and
`UNSAFE_FP8FNCAST=1`). It keeps BF16 accuracy (GSM8K 89.4 %) and is the fastest KV layout at 1k contexts and
beyond. The unpacked layout is slower (3,661 vs 4,097 tok/s at 1024 / 1024) and needs single-shot prefill.

| Use | Layout and settings | 128 / 128 | 1024 / 1024 | 4096 / 512 | 8192 / 1024 |
|---|---|---|---|---|---|
| Short prompts | TP=2 x DP=2; `--max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 128`, packed prefill (`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8`), `num_seqs_buckets` to 128, `decode_context_length_buckets` [512, 1024], KV cap 0.4, `on_device_sampling_config` `{"max_top_k": 32}` | **5,151** | 3,121 | | |
| ~1k contexts | as "Short prompts" plus `--kv-cache-dtype fp8_e4m3` and `"fp8_packed_kv": true` (`UNSAFE_FP8FNCAST=1`), BF16 weights | 4,074 | 4,097 | | |
| ~1k contexts, FP8 weights | same with the FP8 checkpoint (`VLLM_NEURON_FP8_INPUT_SCALE_COMPENSATION=1`) | 4,304 | **4,415** | | |
| Long prompts | TP=4; `--max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 128`, `num_batched_tokens_buckets` [256, 1024, 2048, 4096, 8192], `num_seqs_buckets` [1, 4, 8, 16, 32, 64, 96, 128], `decode_context_length_buckets` [2048, 4096, 8192], KV cap 0.5 | | 2,947 | 900 | 723 |
| Long prompts, packed FP8 KV | as "Long prompts" plus `--kv-cache-dtype fp8_e4m3`, `"fp8_packed_kv": true`, `max_top_k` 32 | | 3,224 | 1,158 | 1,012 |
| Long prompts, FP8 weights + packed FP8 KV | same with the FP8 checkpoint | | 3,865 | **1,420** | **1,206** |

- GSM8K (1319): 89.2 % (short), 89.4 % (packed FP8 KV), 89.2 % (FP8 weights + packed FP8 KV), 89.6 % (long),
  89.3 % / 89.0 % (long with packed FP8 KV, BF16 / FP8 weights); 89.4 % before these changes. Unpacked FP8 KV
  (without `fp8_packed_kv`) is slower: 3,661 tok/s at 1024 / 1024, because 1-byte K takes a PE-transpose path in the decode kernel.
- Keep a decode bucket at about 2 x the typical context: with [1024, 4096] instead of [2048, 4096, 8192]
  the long server drops to 2,015 tok/s at 1024 / 1024.
- KV cap: 0.5 on TP=2 x DP=2 and 0.6 on the 10k-context server with 44 decode graphs run out of device
  memory under load (`Allocation Failure`); the caps above are the largest that passed.
- `max_top_k` 32 supports `top_k` <= 32; it is 3 % faster than the default 256 (131k vocabulary).
- FP8 checkpoint, short server at TP=2 x DP=2 (same settings): 1024 / 1024 **3,667** tok/s (the best 1k / 1k
  configuration: FP8 KV fits twice the requests), 128 / 128 4,295; GSM8K 88.9 %. At TP=4: 4,124 / 3,333.

## Serving (trn2.3xlarge, TP=4)

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6
vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --config-format hf --load-format safetensors --language-model-only \
  --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32],
     "decode_context_length_buckets": [2048, 4096]}}'
```

- `--config-format hf --load-format safetensors` are required when serving the Hugging Face model id.
  For a `mistralai/...` hub id with a `params.json`, vLLM's `auto` picks the Mistral-native format and
  inspects `PixtralForConditionalGeneration`, which fails to import with this transformers
  (`cannot import name 'PixtralRotaryEmbedding'`). A local download directory resolves to `config.json` on
  its own; the flags make both cases the same.
- `--tokenizer-mode mistral` is required: the repository has no Jinja chat template for the
  HuggingFace tokenizer. The tekken tokenizer gives the same chat-templated token ids as
  transformers' `apply_chat_template`.
- `--language-model-only` builds the text decoder only (no vision weights). For images see "Images".
- Decode uses the fused attention kernel at every `--max-model-len`; see "Long context" for > 16384.
- `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (default 0.30) caps the KV cache.
- `VLLM_NEURON_SWA_REAL_ADMISSION=1` admits requests by their real KV peak instead of `max_model_len`;
  with `num_seqs_buckets` up to 128 (`--max-num-seqs 128`) the long server reaches 2,849 tok/s at
  1024 / 1024 (1,852 without) and 891 at 4096 / 512 (719).
- `decode_context_length_buckets` selects Model Runner V1 on this release.

## Accuracy (BF16, TP=4)

| Check | Result | HF reference |
|---|---|---|
| First token, 6 chat prompts | = fp32 argmax 6/6 | |
| Teacher-forced argmax vs HF fp32 (6 prompts x 64 tokens) | 99.7 %, 1 non-tie mismatch | HF bf16 vs fp32: 99.4 %, 1 non-tie (same position) |
| Logprob error vs fp32 on the fp32 top-5, per position (median / p95) | 0.075 / 0.161 | HF bf16: 0.078 / 0.152 |
| Same, needle prompts of 2k / 8k / 15k tokens | 0.101 / 0.215 | HF bf16: 0.084 / 0.254; default RoPE instead of YaRN: 1.88 / 3.84 |
| Needle retrieval at 2k / 8k / 15k | 3/3 | HF fp32: 3/3 |
| GSM8K (1319, zero-shot CoT, greedy) | 89.4 % (TP=4 and TP=2 x DP=2) | HF bf16, first 250: 90.8 % (Neuron on the same 250: 90.0 %) |
| GSM8K, FP8 checkpoint with input-scale compensation | 88.9 % (89.6 % on the packed server) | |

## Performance (trn2.3xlarge, one device, LNC=2; `vllm bench serve`, random data, output tok/s)

Peak throughput (concurrency doubled until the gain is below 5 %):

| Input / output | Server | Peak | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 128 / 128 | short, BF16, TP=4 | 3,110 | 516 ms | 37.8 ms |
| 128 / 128 | short, BF16, TP=2 x DP=2 | 3,484 | 509 ms | 33.3 ms |
| 128 / 128 | short, FP8, TP=4 | 3,770 | 427 ms | 31.4 ms |
| 1024 / 1024 | short, BF16, TP=4 | 2,950 | 3.1 s | 40.4 ms |
| 1024 / 1024 | short, FP8, TP=4 | 3,316 | 2.3 s | 36.4 ms |
| 1024 / 1024 | long, BF16, TP=4, real-peak admission | 2,849 | 48.5 s (c256 queueing); 3.8 s at c128 | 41.3 ms |
| 4096 / 512 | long, BF16, TP=4, real-peak admission | 891 | 6.4 s | 59.5 ms |
| 8192 / 1024 | long, BF16, TP=4, real-peak admission | 705 | 6.7 s | 38.8 ms |

Single request (long server): 1024 / 1024 at 102 tok/s, TTFT 63 ms, TPOT 9.8 ms. FP8 decode TPOT at batch 1: 7.3 ms.

- *long*: the serving command above (max-model-len 10240). Its KV cache holds 39 sequences of 10240 tokens.
- *short*: `--max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 128 --no-enable-prefix-caching`,
  `VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` (packed prefill), `num_batched_tokens_buckets` [128, 256, 512, 1024, 2048],
  `num_seqs_buckets` [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128], `decode_context_length_buckets` [1024].
  TP=2 x DP=2 uses `--max-num-seqs 64` per engine and `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.4` (0.6 runs out of
  device memory under load).

## Speculative decoding (EAGLE-3)

The public EAGLE-3 draft head
[`flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512`](https://huggingface.co/flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512)
(Apache-2.0) is published in the economize format. Convert it to the vLLM EAGLE-3 layout first. The
plugin's draft loader reads the token embedding from the draft folder, so pass the target checkpoint
too:

```bash
hf download flaitenberger/economize-eagle3-Ministral-3-8B-Instruct-2512 --local-dir economize-draft
python vllm_neuron/model/ministral3/convert_economize_eagle3.py economize-draft \
  Ministral-3-8B-Instruct-2512-BF16/config.json m3-eagle3 Ministral-3-8B-Instruct-2512-BF16
vllm serve Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 --max-model-len 2048 \
  --max-num-batched-tokens 2048 --max-num-seqs 32 --language-model-only --tokenizer-mode mistral --config-format hf --load-format safetensors \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [2048], "decode_context_length_buckets": [1024], "on_device_sampling_config": {"all_greedy": "true"}}}' \
  --speculative-config '{"method": "eagle3", "model": "m3-eagle3", "num_speculative_tokens": 3}'
```

TP=4, 256 output tokens, greedy, output tok/s relative to the same server without speculation:

| prompts | concurrency 1 | 4 | 8 |
|---|---|---|---|
| GSM8K | 1.89x (2.67 tokens per step) | 1.48x | 1.31x |
| code | 1.71x (2.33) | 1.32x | 1.14x |
| chat (UltraChat) | 1.40x (1.88) | 1.09x | 0.94x |

At concurrency 16 and 32 (chat) it is 0.79x and 0.70x, so use it for latency-bound serving at low
concurrency. GSM8K-256 accuracy is unchanged (90.6 %). `num_speculative_tokens` 4 compiles only
with `--max-num-seqs` 16 or less (the draft sampler's 32,768-entry vocabulary limit) and is not
faster than 3.

**With the FP8 checkpoint and packed FP8 KV** (add `--kv-cache-dtype fp8_e4m3`, `"fp8_packed_kv": true`,
`UNSAFE_FP8FNCAST=1 VLLM_NEURON_FP8_INPUT_SCALE_COMPENSATION=1`) the draft head stacks on the faster
target. Output tok/s, 256 tokens, greedy:

| prompts | c | FP8 + packed KV | + EAGLE-3 k=3 | BF16 + EAGLE-3 k=3 |
|---|---|---|---|---|
| GSM8K | 1 / 4 / 8 | 125 / 316 / 473 | **195 / 523 / 728** | 196 / 453 / 630 |
| code | 1 / 4 / 8 | 127 / 374 / 585 | **178 / 516 / 803** | 178 / 446 / 648 |
| chat | 1 / 4 / 8 | 127 / 328 / 571 | **149 / 425 / 651** | 146 / 363 / 527 |
| chat | 16 / 32 | 780 / 1,008 | 149 (see below) / 1,007 | |

- FP8 + EAGLE-3 is the best configuration at concurrency 1-8 for every prompt type. At 8 it is 1.14-1.54x over
  FP8 alone, where BF16 + EAGLE-3 is already slower than plain BF16 on chat.
- Median TPOT at c=1 is 4.6 ms on GSM8K (7.2 ms without EAGLE-3).
- GSM8K-1319: 89.1 % with EAGLE-3, 89.6 % without (FP8 checkpoint).
- At c=32 it equals plain FP8 (1,007 tok/s).
- At c=16 one run on a 32-sequence server collapsed to 149 tok/s (TPOT 104 ms) with no preemption or KV
  pressure. That is the same unexplained high-concurrency stall seen once with k=2 on BF16, so keep EAGLE-3
  servers at `--max-num-seqs` 8.
- Multi-token decode (the verify step) on the FP8 path uses the fused decode kernel
  (`VLLM_NEURON_FP8_MULTITOKEN_KERNEL=0` restores single-token only). Before this change the server failed to
  start, because the PyTorch fallback has no FP8 QKV.

## Long context (> 16384)

Past `original_max_position_embeddings` (16384) Ministral 3 scales the queries by
`1 + 0.1 * log(1 + floor(pos / 16384))` after RoPE. The llama3 attention applies it when
`max_model_len` exceeds 16384. Prefill scales the query rows. Decode passes the per-token scale to
the fused decode kernel as a `q_scale` input. That input comes from a vendored `attention_block_tkg`
(`functional/vendored_kernels/attention_block_tkg_qscale.py`, from
`jimburtoft/nki-library@ministral3/tkg-q-scale-2.33`) until it ships in nkilib. The static-FP8
attention applies the same scaling.

At `--max-model-len 32768`, TP4, BF16:
- Single-request decode TPOT is 12.4 ms (22.5 ms on the PyTorch path); batch 8 is 55.7 ms (202 ms).
- Needle retrieval at 20k and 24k tokens: 2/2.
- Teacher-forced logprob error vs HF fp32 is 0.066 / 0.166 (median / p95), against 0.089 / 0.168 for
  HF bf16 and 0.33 / 0.53 for HF fp32 without the query scaling.

FP8 weights with packed FP8 KV at 32768: TPOT 9.0 ms (batch 1) and 18.2 ms (batch 8); needle 2/2.
`VLLM_NEURON_QSCALE_TORCH=1` puts decode back on the PyTorch path.

An unpacked FP8 KV cache (`--kv-cache-dtype fp8_e4m3` without `"fp8_packed_kv": true`, which the
ModelOpt FP8 checkpoint selects by default) cannot use segmented prefill. Set
`--max-num-batched-tokens` to at least `--max-model-len` so prefill is single-shot; the server
refuses a smaller value, because a chunked prompt would otherwise ignore its earlier chunks.

## Images

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5 VLLM_NEURON_SWA_REAL_ADMISSION=1
vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --config-format hf --load-format safetensors --limit-mm-per-prompt '{"image": 2}' \
  --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048, 4096],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 48, 64], "decode_context_length_buckets": [2048]},
   "encoder_neuron_config": {"tp_size": 4, "encoder_attention_block_size": 12288,
     "num_encoder_input_buckets": [1024, 2048, 3072, 4096, 6144, 8192, 12288],
     "encoder_cache_num_blocks": 256}}'
```

- **Encoder TP 4.** The default (encoder data parallel over the 4 ranks) runs one real block and three
  padding blocks for a single image (640x896 image: 100.5 ms at DP 4, 33.8 ms at TP 4).
- **Image-size buckets.** With encoder TP, a bucket smaller than `encoder_attention_block_size` is
  compiled as one block of `bucket` patches, and every image runs in the smallest bucket that holds
  it. 640x896 (2,944 patches) runs the 3072 bucket: encoder 25 ms, single-request TTFT 141 ms (155 ms
  with one 4096 block); 448x448 runs the 1024 bucket (10 ms, TTFT 93 ms); a 1540 px image (up to
  12,100 patches) runs the 12288 bucket instead of being rejected.
- Images are sized by mistral-common (the chat template in `--tokenizer-mode mistral`, largest edge
  1540); `--mm-processor-kwargs` `size` does not change them. The grid and pixels come from the same
  encoder, so the `[IMG]` count always equals the encoder output.
- Encoder-cache blocks hold 256 merged rows (`VLLM_NEURON_ENC_CACHE_BLOCK`, 0 = one attention block),
  so an image takes ceil(tokens / 256) blocks (256 blocks = 512 MB per core at hidden 4096). Set
  `encoder_cache_num_blocks` to about 2 x the token budget / 256 + 1.
- **Multi-image packing** (opt-in, `VLLM_NEURON_PIXTRAL_PACK=2`): the images of one request share one
  encoder block of equal slots. The slot layout is passed to the attention kernel as
  `segment_cu_seqlens`, which skips the cross-image tiles (nkilib attention_cte sequence-packing tile
  pruning; vendored here from `jimburtoft/nki-library@attention-cte-seqpack-tile-prune`, based on nki-library
  2.33, until it ships in a release). Two 448x448 images: encoder
  15.9 ms instead of 2 x 10 ms; two 640x896 images: 46 ms instead of 2 x 24 ms (53-60 ms packed
  without the pruning). On its own this helps multi-image prompts only.
- **Image requests in packed prefill** (opt-in, `VLLM_NEURON_PACK_MULTIMODAL=1` with
  `VLLM_NEURON_MAX_PREFILLS_PER_BATCH=N`): up to N image prompts share one prefill step, and their
  images are encoded in the same step. The encoder budget and the per-step cache-block window scale
  by N. This needs `max_num_batched_tokens >= max_model_len` and prefill buckets no larger than
  `VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS` (default 2048); otherwise packing stays off.
  640x896 images, TP4, mml 4096, max_num_seqs 128, N=4: plateau 1,136 tok/s against 1,100 without
  packing; single-request TTFT unchanged.
- The upstream vLLM `Mistral3ForConditionalGeneration` does not import with the transformers in
  this environment, so the processor is vendored (`processing.py`) and the platform registers a
  stand-in class (`validation.py`) for config validation.
- Measured and left off (opt-in env vars): `VLLM_NEURON_PIXTRAL_FUSED_QKV=1` (NF.qkv_proj with the
  RMSNorm fused + NF.o_proj; 1-18 ms slower), `VLLM_NEURON_PIXTRAL_NKI_MLP=1` (nkilib MLP; slower),
  `VLLM_NEURON_ENCODER_O3=1` (encoder -O3; device -5 %, TTFT unchanged).

Online, 640x896 random images, 128 output tokens, server above: c=1 91 tok/s (TTFT 148 ms,
TPOT 9.9 ms), c=32 792 tok/s, plateau 944 tok/s.

Accuracy (TP=4, 5 image prompts incl. one with two images, 48 tokens, teacher-forced against HF fp32
on the same pixels): argmax 98.0 % with 1 non-tie mismatch, against 97.5 % / 1 for HF bf16;
logprob error 0.091 / 0.210 (median / p95) against 0.092 / 0.241 for HF bf16.

## FP8

Accuracy: FP8 **weights** raise the per-token logprob error against HF fp32 about 5x over BF16 (median 0.36 vs 0.075),
whatever the scheme. An HF fake-quant study (per-tensor or per-channel weights, static or dynamic activations, MLP-only)
found none within the BF16 noise floor, although GSM8K stays within 0.5 points. FP8 **KV** (packed, scale 1.0) keeps BF16
accuracy: K/V stay below 18 in every layer, far inside the FP8 range. For accuracy-sensitive use: BF16 weights + packed
FP8 KV. On trn2 the plugin re-scales FP8 weights by 0.5 (exact) instead of 240/448 to fit the 240-range FP8 format.


The `mistralai/Ministral-3-8B-Instruct-2512` checkpoint is static per-tensor FP8 (`quant_method: fp8`,
`weight_scale_inv` / `activation_scale`). It runs on the llama3 static-FP8 path after renaming the scales to
`weight_scale` / `input_scale` and setting a ModelOpt `quantization_config`
(`{"quant_method": "modelopt", "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8",
"exclude_modules": ["lm_head"]}}`); `weight_scale_inv` is a dequantization multiplier, so the values are used as is.

On trn2 the static-FP8 kernels quantize activations into the 240-range fp8 format, while this checkpoint's input scales
are calibrated for `float8_e4m3fn` (448). Set `VLLM_NEURON_FP8_INPUT_SCALE_COMPENSATION=1` to scale the input scales by
448/240; without it activations clip and GSM8K drops from 88.9 % to 88.3 %.

## Feature status

| Feature | Status | Notes |
|---|---|---|
| TP | ✅ | TP=4 validated; TP=2 validated on a 4-layer slice and with DP=2 |
| TP=1 | ❌ | the decode MLP kernel fails with `NCC_INKI016 Stack out of memory` at intermediate size 14336 |
| DP | ✅ | TP=2 x DP=2 |
| Prefix caching, segmented prefill | ✅ | validated at BS 6 |
| Packed prefill | ✅ | short server |
| Vision tower (Pixtral) + projector | ✅ | one image per encoder block; see Images |
| FP8 checkpoint | ✅ (after scale renaming) | see FP8 |
| EAGLE-3 speculative decoding | ✅ | public economize draft head after conversion (see above) |
| Llama 4 query scaling (positions >= 16384) | ✅ | fused decode kernel with a per-token query scale (vendored kernel) |
| FP8 KV cache | ✅ | packed (`"fp8_packed_kv": true`, recommended) or unpacked; BF16 or FP8 weights, scales 1.0 |
