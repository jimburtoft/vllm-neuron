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
* **Checkpoint prefix.** The decoder weights are under `language_model.`; the vision tower
  (`vision_tower.*`) and projector (`multi_modal_projector.*`) are not loaded.

## Architecture (Ministral-3-8B-Instruct-2512)

| Parameter | Value |
|---|---|
| hidden_size / num_hidden_layers / vocab_size | 4096 / 34 / 131072 |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 8 / 128 |
| intermediate_size | 14336 |
| RoPE | YaRN: rope_theta 1e6, factor 16, original_max_position_embeddings 16384, beta_fast 32, beta_slow 1, mscale = mscale_all_dim = 1 (attention_factor 1.0) |
| Llama 4 query scaling | beta 0.1 above position 16384 (not implemented, see below) |
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
- `--language-model-only` (or `--limit-mm-per-prompt '{"image": 0}'`) is required; image inputs
  are rejected otherwise.
- `--max-model-len` must be at most 16384 (see "Not supported").
- `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (default 0.30) caps the KV cache.
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
| GSM8K, FP8 checkpoint with input-scale compensation | 88.9 % | |

## Performance (trn2.3xlarge, one device, LNC=2; `vllm bench serve`, random data, output tok/s)

Peak throughput (concurrency doubled until the gain is below 5 %):

| Input / output | Server | Peak | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 128 / 128 | short, BF16, TP=4 | 3,110 | 516 ms | 37.8 ms |
| 128 / 128 | short, BF16, TP=2 x DP=2 | 3,484 | 509 ms | 33.3 ms |
| 128 / 128 | short, FP8, TP=4 | 3,825 | 381 ms | 30.9 ms |
| 1024 / 1024 | short, BF16, TP=4 | 2,950 | 3.1 s | 40.4 ms |
| 1024 / 1024 | short, FP8, TP=4 | 3,349 | 2.2 s | 36.2 ms |
| 1024 / 1024 | long, BF16, TP=4 | 1,851 | 942 ms | 16.4 ms |
| 4096 / 512 | long, BF16, TP=4 | 719 | 3.2 s | 38.3 ms |
| 8192 / 1024 | long, BF16, TP=4 | 707 | 6.7 s | 38.7 ms |

Single request (long server): 1024 / 1024 at 102 tok/s, TTFT 63 ms, TPOT 9.8 ms. FP8 decode TPOT at batch 1: 7.3 ms.

- *long*: the serving command above (max-model-len 10240). Its KV cache holds 39 sequences of 10240 tokens.
- *short*: `--max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 128 --no-enable-prefix-caching`,
  `VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` (packed prefill), `num_batched_tokens_buckets` [128, 256, 512, 1024, 2048],
  `num_seqs_buckets` [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128], `decode_context_length_buckets` [1024].
  TP=2 x DP=2 uses `--max-num-seqs 64` per engine and `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.4` (0.6 runs out of
  device memory under load).

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
| Vision tower (Pixtral) | ❌ | text only |
| FP8 checkpoint | ✅ (after scale renaming) | see FP8 |
| Llama 4 query scaling (positions >= 16384) | ❌ | `max_model_len > 16384` is rejected |
| FP8 KV cache | ✅ | with the FP8 checkpoint (scales 1.0) |
| Eagle3 | not tested | |
