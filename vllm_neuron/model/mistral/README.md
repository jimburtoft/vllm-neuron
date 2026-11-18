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
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_SWA_REAL_ADMISSION=1 NEURON_SKIP_EFA_AFFINITY=1
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 128 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048, 4096, 8192],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
     "decode_context_length_buckets": [2048, 4096, 8192]}}'
```

Short prompts (2k context), packed prefill:

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8 NEURON_SKIP_EFA_AFFINITY=1
vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tensor-parallel-size 4 \
  --max-model-len 2048 --max-num-batched-tokens 2048 --max-num-seqs 64 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {
     "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
     "num_seqs_buckets": [1, 2, 4, 8, 16, 24, 32, 48, 64],
     "decode_context_length_buckets": [1024]}}'
```

- `NEURON_SKIP_EFA_AFFINITY=1` is required on a trn2.3xlarge, which has no EFA device. Without it, engine start fails with `No EFA device found ...`.
- `VLLM_NEURON_SWA_REAL_ADMISSION=1` (opt-in) admits a request only while the real KV peak of every live
  request fits the pool. The real peak is prompt + max_tokens + in-flight tokens, not `max_model_len`. Without
  it, the scheduler charges every request `max_model_len`, so the 10k server runs only 42 requests at once
  (`max_num_seqs ... exceeds worst-case KV cache capacity (42 concurrent requests`). With it, all 128 run, and 0
  requests are preempted. The variable name comes from the sliding-window models the feature was written for;
  dense models use it unchanged.
- `VLLM_NEURON_FUSE_DECODE_NORMS=1` (opt-in) runs the two decode RMSNorms inside the attention and MLP kernels.
  Decode time per token is 6 % lower at batch 1-32 (8.79 -> 8.22 ms at batch 1). GSM8K is the same (60.5 %).
- For short prompts, a 512 decode context bucket (`"decode_context_length_buckets": [512, 2048]`) cuts batch-64
  decode from 20.4 to 16.4 ms. With it, the optional NKI Library branch
  [`jimburtoft/nki-library@qwen3-asr/2.33-champion`](https://github.com/jimburtoft/nki-library/tree/qwen3-asr/2.33-champion)
  gives another -9 % at batch 32 and -11 % at batch 64. Put the branch's `src/nkilib_src` first on `PYTHONPATH`, and use
  private NKI caches (`NKI_TRACE_CACHE_URL`, `NKI_COMPILE_CACHE_URL` and `VLLM_CACHE_ROOT` in new directories).
- Greedy-only deployments: add `"on_device_sampling_config": {"all_greedy": true}` to `neuron_config`. The default
  on-device sampler runs a 256-way top-k over every vocabulary shard on each step, even for greedy requests (~280 us).
  With `all_greedy` it is a distributed argmax: decode is 2-4 % faster (7.89 -> 7.71 ms at batch 1). It makes **every**
  request greedy, so do not use it when requests sample with a temperature.
- Mixed greedy and sampled traffic: `VLLM_NEURON_PER_STEP_GREEDY=1` compiles a second (argmax) graph per decode
  bucket and uses it on decode steps where every request is greedy and none asks for logprobs or structured output.
  Other steps use the full sampler. Decode is 1-3 % faster on all-greedy steps (7.89 -> 7.74 ms at batch 1,
  12.68 -> 12.24 ms at batch 32), and sampled requests are unchanged. It doubles the decode compile time.
- `VLLM_NEURON_REPLICATED_DECODE_EMBEDDING=1` keeps the full embedding table on each rank for decode (+256 MiB per
  rank in bf16) and skips the embedding all-reduce: -2.5 % at batch 32, no gain at batch 1-8.
- FP8 without a special checkpoint: `VLLM_NEURON_FP8_ROW_MLP=1` quantizes the MLP weights (about 80 % of the weight
  bytes) to FP8 at load, with one scale per output channel, from the BF16 checkpoint. Attention stays BF16. Decode
  multiplies BF16 activations by the FP8 weights. Prefill quantizes activations per token and runs the NKI Library
  FP8 prefill MLP kernel. This branch includes a fix to that kernel's tile sizing, without which it runs out of
  on-chip memory at this model size for prompts of 2048 tokens or more. Measured on stock Llama-3.1-8B-Instruct
  (same MLP shape as Mistral-7B), against the BF16 model on the 10k recipe:
  - decode 13 % faster at batch 1, 7-10 % at batch 8-128;
  - peak throughput +3 % at 128/128 (2,202 -> 2,266 tok/s) and +13 % at 1k/1k (3,106 -> 3,497 tok/s);
  - 8k-token prompts: time to first token 16-19 % lower;
  - GSM8K 85.7 % (BF16 85.4 %); on Mistral-7B 59.1 % (BF16 59.2-60.9 %). Token-level agreement with an fp32
    reference is about 0.5 points lower than BF16.
  Prefix caching and segmented prefill keep working. `VLLM_NEURON_FP8_ROW_CTE=0` falls back to dequantizing the
  weights for prefill (slower prefill, same decode).
- Per-tensor FP8 checkpoints (ModelOpt, or `RedHatAI/Mistral-7B-Instruct-v0.3-FP8` relabelled to ModelOpt) also
  load, with or without an FP8 KV cache. Decode is 18 % faster at batch 1, but GSM8K drops to about 56.5 %: the
  per-tensor weight scales and the static per-tensor activation quantization each cost about 2 points.
- `VLLM_NEURON_FP8_LM_HEAD=1` stores the LM head in FP8 with one scale per vocabulary row: -0.6 / -5.5 / -8.2 %
  decode time at batch 1 / 8 / 32, but GSM8K drops by 1-2 points (59.0 %). Off by default.
- EAGLE3 speculative decoding (`--speculative-config '{"method": "eagle3", "model": <draft>,
  "num_speculative_tokens": 3}'`) with a Mistral-7B-v0.3 EAGLE3 draft cuts decode time 1.75x at concurrency 1
  (4.5 ms per token), 1.39x at 4 and 1.15x at 8. From concurrency 16 up, the plain recipe above is faster (1,405
  vs 1,286 tok/s at 16, 2,371 vs 1,614 at 32), so use EAGLE3 for latency-bound serving only. Under speculative
  decoding, packed prefill is turned off by the scheduler, and fused decode norms make it 2-4 % slower, so leave
  both off with EAGLE3.
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
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_SWA_REAL_ADMISSION=1 NEURON_SKIP_EFA_AFFINITY=1
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
- TP=1 / TP=2 data-parallel layouts are slower per device than TP=4: decode is weight-read bound, so each engine
  streams the whole model for its own small batch. Per-engine decode at batch 8: TP1 29.6 ms, TP2 16.6 ms, TP4
  10.2 ms. At concurrency 32, TP1 x 4 engines gives 1,080 tok/s against 2,432 for TP4. TP1 compiles on this branch
  (the decode MLP turns column tiling off above 7168 intermediate per rank), but the 10k-context server takes more
  than an hour to compile. TP2 x 2 does not fit the 10k server's prefill and decode graphs at any KV cap that
  leaves room for the KV cache.
