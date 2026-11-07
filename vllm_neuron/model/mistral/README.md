# Mistral

`MistralForCausalLM` on the Neuron backend, for checkpoints without a sliding window
(`sliding_window: null`), e.g. [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).

Mistral with full attention has the same architecture and HuggingFace weight names as Llama
(GQA attention without bias, RoPE, RMSNorm, SiLU-gated MLP; transformers' `MistralForCausalLM`
subclasses the Llama classes). This package registers the architecture and builds the
[`llama3`](../llama3/) implementation. A checkpoint with a sliding window shorter than
`max_position_embeddings` (Mistral-7B-v0.1) is rejected with `NotImplementedError`, because the
llama3 attention has no sliding window and would silently run full attention.

## Architecture (Mistral-7B-Instruct-v0.3)

| Parameter | Value |
|---|---|
| hidden_size / num_hidden_layers / vocab_size | 4096 / 32 / 32768 |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 8 / 128 |
| intermediate_size | 14336 |
| RoPE | default, rope_theta 1e6 |
| RMSNorm eps | 1e-5 |
| tie_word_embeddings | False |
| sliding_window | null |

The llama3 config reads every value above from the checkpoint (`LlamaConfig.from_configs`); note that
its defaults differ for rope_theta (5e5) and tie_word_embeddings (True).

## Serving (trn2.3xlarge, TP=4)

Long context (10k), segmented prefill:

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32],
     "decode_context_length_buckets": [2048, 4096]}}'
```

Short prompts (2k context), packed prefill:

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 64 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32, 48, 64],
     "decode_context_length_buckets": [1024]}}'
```

- `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (default 0.30) caps the KV cache; at 0.30 only about 21
  sequences of 10240 tokens are admitted at TP=4.
- `decode_context_length_buckets` selects Model Runner V1.
- Packed prefill needs single-shot prefill (`max_num_batched_tokens >= max_model_len`) and no prefix caching.

## Accuracy (BF16, TP=4)

| Check | Result | HF reference |
|---|---|---|
| Teacher-forced argmax vs HF fp32 (32 chat prompts, ~3.5k tokens) | 99.4 % | HF bf16 vs fp32: 99.35 % |
| GSM8K, 1319, zero-shot CoT, greedy | 59.6-60.9 % | HF bf16: 60.1 % |
| Needle retrieval, 1.5k-8.6k tokens | 8/8 | |

## Performance (trn2.3xlarge, one device, TP=4, BF16; output tok/s, `vllm bench serve`, random data)

| Input / output | Concurrency | Throughput | TTFT p50 | TPOT p50 |
|---|---|---|---|---|
| 1024 / 1024 (10k server) | 1 | 112 | 63 ms | 8.8 ms |
| 1024 / 1024 (10k server) | 32 | 2,042 | 897 ms | 14.8 ms |
| 8192 / 1024 (10k server) | 1 | 99 | 409 ms | 9.7 ms |
| 8192 / 1024 (10k server) | 32 | 764 | 6.3 s | 35.8 ms |
| 128 / 128 (2k packed server) | 32 | 2,135 | 158 ms | 13.8 ms |
| 128 / 128 (2k packed server) | 64 | 3,087 | 276 ms | 18.8 ms |

## Not supported

- Sliding-window checkpoints (Mistral-7B-v0.1).
- TP=1 for 7B: the decode MLP kernel fails with `NCC_INKI016 Stack out of memory` at intermediate size 14336.
