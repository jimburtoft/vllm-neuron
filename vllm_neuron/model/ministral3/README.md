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

## Serving (trn2.3xlarge, TP=4)

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6
vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --language-model-only \
  --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32],
     "decode_context_length_buckets": [2048, 4096]}}'
```

- `--tokenizer-mode mistral` is required: the repository has no Jinja chat template for the
  HuggingFace tokenizer. The tekken tokenizer gives the same chat-templated token ids as
  transformers' `apply_chat_template`.
- `--language-model-only` builds the text decoder only (no vision weights). For images see "Images".
- Up to `--max-model-len 16384` decode uses the fused attention kernel. Above it see "Long context".
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

## Long context (> 16384)

Past `original_max_position_embeddings` (16384) Ministral 3 scales the queries by
`1 + 0.1 * log(1 + floor(pos / 16384))` after RoPE. The llama3 attention applies it when
`max_model_len` exceeds 16384: prefill scales the query rows, and decode passes a per-token query
scale to `NF.attention_decode`. The fused decode kernel only takes a scalar softmax scale, so decode
attention then runs the PyTorch path: at `--max-model-len 32768` the single-request decode TPOT is
22.3 ms (9.8 ms on the fused kernel at <= 16384). Needle retrieval at 20k and 24k tokens: 2/2; the
teacher-forced logprob error vs HF fp32 is 0.086 / 0.190 (median / p95), against 0.089 / 0.168 for HF
bf16 and 0.33 / 0.53 for HF fp32 without the query scaling.

## Images

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5 VLLM_NEURON_SWA_REAL_ADMISSION=1
vllm serve mistralai/Ministral-3-8B-Instruct-2512-BF16 --tensor-parallel-size 4 \
  --tokenizer-mode mistral --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 8192 --max-num-batched-tokens 4096 --max-num-seqs 64 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048, 4096],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 48, 64], "decode_context_length_buckets": [2048]},
   "encoder_neuron_config": {"encoder_attention_block_size": 4096, "tp_size": 4,
     "num_encoder_input_buckets": [4096, 8192], "encoder_cache_num_blocks": 160}}'
```

- Each image runs in one encoder attention block, so `encoder_attention_block_size` must hold the
  image's patches: an image of up to 896 px is at most 64 x 64 = 4096 patches; the full 1540 px needs
  12,100 patches (block 12,288). With `--tokenizer-mode mistral` the image size is set by mistral-common
  (largest edge 1540), not by `--mm-processor-kwargs` `size`; an image whose patches exceed the block
  fails with a clear error, so size the block for the largest image you serve (or resize client-side).
- The image grid and pixels come from mistral-common (the chat template in `--tokenizer-mode
  mistral`), which rounds the merged grid up and resizes the image to it, so the `[IMG]` count always
  equals the encoder output.
- Use encoder `tp_size` 4. The default (encoder DP over the 4 ranks) runs one real block and three
  padding blocks for a single image: 640x896 image encoder 100.5 ms at DP4, 33.8 ms at TP4, 23.9 ms at
  TP4 with `encoder_attention_block_size` 3072 (single-request image TTFT 246 -> 149 ms).
- Set `encoder_cache_num_blocks`: every image takes a whole cache block, and the default size (from
  the scheduler's merged-token budget) can be only a few blocks.
- The upstream vLLM `Mistral3ForConditionalGeneration` does not import with the transformers in
  this environment, so the processor is vendored (`processing.py`) and the platform registers a
  stand-in class (`validation.py`) for config validation.

Online, 640x896 random images, 128 output tokens, encoder TP=4, `--max-num-seqs 64`,
`VLLM_NEURON_SWA_REAL_ADMISSION=1`: c=1 90 tok/s (TTFT 168 ms, TPOT 9.8 ms), c=32 752 tok/s, plateau about 848 tok/s.

Accuracy (TP=4, 5 image prompts incl. one with two images, 48 tokens, teacher-forced against HF fp32
on the same pixels): argmax 98.1 % with 0 non-tie mismatches, against 98.6 % / 0 for HF bf16;
logprob error 0.085 / 0.225 (median / p95) against 0.089 / 0.236 for HF bf16.

## FP8

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
| Llama 4 query scaling (positions >= 16384) | ✅ | decode attention uses the PyTorch path when `max_model_len > 16384` |
| FP8 KV cache | ✅ | with the FP8 checkpoint (scales 1.0) |
| Eagle3 | not tested | |
