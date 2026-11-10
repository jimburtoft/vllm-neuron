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

Long context (10k), segmented prefill, real-peak KV admission:

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_SWA_REAL_ADMISSION=1
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 128 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
     "decode_context_length_buckets": [2048, 4096, 8192]}}'
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

- `VLLM_NEURON_SWA_REAL_ADMISSION=1` (opt-in) admits a request only while the real KV peak of every live
  request fits the pool. The real peak is prompt + max_tokens + in-flight tokens, not `max_model_len`. Without
  it, the scheduler charges every request `max_model_len`, so the 10k server runs only 42 requests at once
  (`max_num_seqs ... exceeds worst-case KV cache capacity (42 concurrent requests`). With it, all 128 run, and 0
  requests are preempted. The variable name comes from the sliding-window models the feature was written for;
  dense models use it unchanged.
- `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (default 0.30) caps the KV cache. 0.6 is the largest value that
  still leaves HBM for the 8192-token prefill graph; 0.7 fails at load with `Allocation Failure`.
- `decode_context_length_buckets` selects Model Runner V1 on this release. On the 10k server they double
  1k/1k decode throughput. On the 32k server without them, every decode step reads the full 32k KV window
  (see Full 32k context).
- Packed prefill runs on both model runners. It needs single-shot prefill (`max_num_batched_tokens >=
  max_model_len`) and no prefix caching.
- Model Runner V2, packed N=8, 128/128 (2k server, no decode context buckets): 1,223 / 1,919 / 2,573 tok/s at
  c=16 / 32 / 64, compared with 1,101 / 1,642 / 2,071 unpacked. GSM8K is the same (59.7 % / 59.8 %).

## Accuracy (BF16, TP=4)

| Check | Result | HF reference |
|---|---|---|
| Teacher-forced argmax vs HF fp32 (32 chat prompts, ~3.5k tokens) | 99.4 % | HF bf16 vs fp32: 99.35 % |
| GSM8K, 1319, zero-shot CoT, greedy | 59.6-60.9 % | HF bf16: 60.1 % |
| Needle retrieval, 1.5k-8.6k tokens | 8/8 | |

## Performance (trn2.3xlarge, one device, TP=4, BF16; output tok/s, `vllm bench serve`, random data)

| Input / output | Server | Concurrency | Throughput | TTFT p50 | TPOT p50 |
|---|---|---|---|---|---|
| 1024 / 1024 | 10k | 1 | 112 | 63 ms | 8.8 ms |
| 1024 / 1024 | 10k | 32 | 2,043 | 887 ms | 14.8 ms |
| 1024 / 1024 | 10k | 128-512 (peak) | **3,164** | | 37.2 ms |
| 4096 / 512 | 10k | peak | **993** | | |
| 8192 / 1024 | 10k | 1 | 99 | 409 ms | 9.7 ms |
| 8192 / 1024 | 10k | 32 | 764 | 6.3 s | 35.8 ms |
| 128 / 128 | 10k | 256 (peak) | 2,261 | | |
| 128 / 128 | 2k packed | 32 | 2,135 | 158 ms | 13.8 ms |
| 128 / 128 | 2k packed | 64 | 3,087 | 276 ms | 18.8 ms |
| 128 / 128 | 2k packed, mns 256 | peak | 4,591 | | |

The 10k peaks are with real-peak admission. Without it, they are 2,050 (1k/1k) and 866 (4k/512).

## Full 32k context

`max_position_embeddings` is 32768, and the full context serves correctly: a needle is retrieved 6/6 at
920 / 15k / 30k prompt tokens, and GSM8K is 59.5 % on the same server. Real-peak admission is required here:
without it, a 32k server charges every request 32k tokens and runs only about 13 at once, even for short
prompts.

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_SWA_REAL_ADMISSION=1
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 32 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 32],
     "decode_context_length_buckets": [2048, 4096, 8192, 16384]}}'
```

- `decode_context_length_buckets` must not include max_model_len, because max_model_len is the implicit top
  bucket. The server rejects the config otherwise.
- Keep the decode context buckets. Measured on this release without them (Model Runner V2, mns 128): 1k/1k
  peaks at 465 tok/s, because every decode step reads the full 32k KV window. With them, a 32k server serves 1k
  requests as fast as the 10k server.
- Keep `--max-num-seqs` at 32 or below at 32k. The cold compile of batch 64-128 decode graphs at ctx 16384-32768
  ran a trn2.3xlarge out of host memory (124 GB RAM + 64 GB swap).
- One device: 16k-token prompts give 82-84 tok/s at c=1 (TTFT 0.9 s), and 30k-token prompts give 75 tok/s
  (TTFT 2.4 s). The peaks are 281 tok/s at 16k/1k and 181 tok/s at 30k/1k. The KV pool holds about 13 live 31k-token
  requests.

## Not supported

- Sliding-window checkpoints (Mistral-7B-v0.1).
- TP=1 for 7B: the decode MLP kernel fails with `NCC_INKI016 Stack out of memory` at intermediate size 14336.
