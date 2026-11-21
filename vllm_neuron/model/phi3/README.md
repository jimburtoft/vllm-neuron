# Phi-3 family (Phi-4, Phi-4-mini) on vLLM Neuron 0.29

`Phi3ForCausalLM` (`microsoft/phi-4`, `microsoft/Phi-4-mini-instruct`) runs on the Llama 3 implementation of this plugin.
Stock vLLM Neuron 0.29.0.1.2.0 has no Phi-3 registration and fails with
`'Phi3ForCausalLM' has no attribute 'from_configs'`.

What this module adds (no kernel changes):

- Fused `qkv_proj` / `gate_up_proj` checkpoint tensors are exposed under the Llama names, so the Llama 3 loaders and TP
  sharding are reused.
- Partial rotary (`partial_rotary_factor` 0.75 on Phi-4-mini) through a per-head permutation of the Q and K weight rows
  onto the Llama 3 rotate-half layout. The pass-through dims get frequency 0.
- LongRoPE. The factor set is chosen from `max_model_len`, as in upstream vLLM: short factors up to 4096, long factors
  for every position above that.
- KV heads that do not divide TP are replicated at load (Phi-4: 10 KV heads -> 20 at TP4).

Tested on one trn2.3xlarge (TP4, SDK 2.33 DLAMI, venv `aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`).

## Install

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@phi3-trn2-0.29"
export NEURON_SKIP_EFA_AFFINITY=1   # required on trn2.3xlarge (no EFA device)
```

## Phi-4 (14B)

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5 VLLM_NEURON_SWA_REAL_ADMISSION=1 \
       VLLM_NEURON_FUSE_DECODE_NORMS=1 VLLM_NEURON_FP8_ROW_MLP=1
vllm serve microsoft/phi-4 --tensor-parallel-size 4 --max-model-len 10240 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048],
     "num_seqs_buckets": [1, 4, 16, 32, 64, 128], "decode_context_length_buckets": [2048, 4096]}}'
```

The cold compile takes about 45 minutes.

- Batch-1 decode: 11.7 ms per token.
- Peak output throughput: 1,019 tok/s at 128/128 and 1,228 tok/s at 1k/1k. Both are about 4x the defaults.
- GSM8K (first 400 problems): 96.75 %. HF bf16 on CPU scores 97.0 % on the first 200.
- Long prompts (4k-8k) are limited by KV capacity and peak at 225-295 tok/s. The replicated KV heads double the KV
  cache size.
- `fp8_packed_kv` does not compile for this model at TP4: `NCC_IGCA030` in neuronx-cc 2.28.7631.
- With KV cap 0.6 and larger bucket grids, warmup fails with `Allocation Failure`.

## Phi-4-mini-instruct (3.8B), 4k context

```bash
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.6 VLLM_NEURON_SWA_REAL_ADMISSION=1 VLLM_NEURON_FUSE_DECODE_NORMS=1
vllm serve microsoft/Phi-4-mini-instruct --tensor-parallel-size 4 --max-model-len 4096 \
  --max-num-batched-tokens 2048 --max-num-seqs 192 --no-enable-prefix-caching --kv-cache-dtype fp8_e4m3 \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048],
     "num_seqs_buckets": [1, 4, 16, 32, 64, 128, 192], "decode_context_length_buckets": [1024, 2048],
     "on_device_sampling_config": {"max_top_k": 32}, "fp8_packed_kv": true}}'
```

- Batch-1 decode: 5.7 ms per token.
- Peak output throughput: 3,054 tok/s at 128/128, 5,252 tok/s at 1k/1k and 2,503 tok/s at 3k/512.
- With `--max-num-seqs 512` and a seq ladder up to 512, the peaks rise to 3,383 tok/s (128/128) and 6,238 tok/s (1k/1k).
  The cold compile then takes about 4 hours.
- GSM8K (first 400 problems): 92.25 %. HF bf16 scores 92.75 %.
- For the lowest latency, keep the KV cache in BF16 and set `VLLM_NEURON_FP8_ROW_MLP=1`. Batch-1 decode is then 5.1 ms.

## Phi-4-mini, long context

Use `--max-model-len 16384` or more. Above 4096 tokens the long LongRoPE factors are used for every position.

- Needle retrieval: 12/12, at prompt lengths up to 14.6k tokens.
- Token agreement with HF fp32 (long factors forced in HF) is at the bf16 floor.
- GSM8K drops to 88.75 %. This is the model's own behavior with long factors; upstream vLLM makes the same choice.
  Use 4096 unless you need longer context.

## Known issue

On vLLM 0.29, `/v1/completions` requests that ask for `logprobs` return HTTP 500 for Phi-4-mini: `IndexError` in
`_create_completion_logprobs`. Offline `LLM(...).generate(..., SamplingParams(logprobs=5))` works.
