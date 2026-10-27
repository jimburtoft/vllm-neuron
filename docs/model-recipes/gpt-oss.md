# GPT-OSS Model Recipe

<!-- meta: description: Model recipe for deploying gpt-oss with vLLM on Neuron,
including supported checkpoints, feature support, accuracy results, and a link to
the end-to-end deployment tutorial for gpt-oss 20B and 120B on Trn3 (MXFP4) or
Trn2 (BF16). -->
<!-- meta: keywords: vLLM, Neuron, gpt-oss, gpt-oss-20b, gpt-oss-120b, MoE,
MXFP4, model recipe, model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-10-16 -->
<!-- Content type: model-card -->
<!-- Jira: NDOC-185 -->

## Introduction

[gpt-oss](https://huggingface.co/openai/gpt-oss-120b) is an open-weight
Mixture-of-Experts (MoE) language model family released by OpenAI, designed for
strong reasoning, tool use, and long-context generation. The published
checkpoints ship with MXFP4-quantized expert weights.

gpt-oss is supported for inference serving with
[vLLM](https://github.com/vllm-project/vllm) using the Neuron SDK on AWS
Trainium2 (`trn2`) and Trainium3 (`trn3`) hardware.

**Compatible model checkpoints:**

| Model | HuggingFace | Hardware | Quantization |
|-------|-------------|----------|--------------|
| gpt-oss-20b | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | Trn2, Trn3 | BF16 (Trn2), MXFP4 (Trn3) |
| gpt-oss-120b | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | Trn2, Trn3 | BF16 (Trn2), MXFP4 (Trn3) |

> On Trn3, MXFP4-quantized weights are auto-selected from the published
> checkpoints. On Trn2, both models run in BF16. MXFP4 is Trn3 only.

## Features

Per-model feature availability for gpt-oss. See the
[features guide](../guides/features-guide.md) for configuration details.

| Category | Feature | Status |
|---|---|---|
| **Inputs** | Text | ✅ |
| **Quantization** | MXFP4 weights (Trn3) | ✅ |
| | BF16 weights (Trn2) | ✅ |
| | KV cache FP8 | ✅ (see [Known issues](#known-issues)) |
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Data parallelism (DP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | Speculative decoding (EAGLE3) | ✅ |
| | Speculative decoding (P-EAGLE) | ✅ |
| | Disaggregated inference (1P1D / xPyD) | ✅ |
| | Decode megakernel at LNC=1 (Trn3, DI decode, 120B) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| **Serving** | Structured outputs / tool calling | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |
| **Runtime** | Model Runner V2 | ❌ (runs on MRv1) |

**Status legend:**

- ✅ Supported: integrated and tested for gpt-oss
- ❌ Not supported: may be considered for future releases

This model runs on Model Runner V1 (MRv1) because it is a Mixture-of-Experts model with sliding-window attention. [Model Runner V2](../guides/features-guide.md#model-runner-v2) supports dense text-generation models only.

The [deployment tutorial](../tutorials/tutorial-gpt-oss.md) walks through
deploying gpt-oss 20B and 120B on Trn3 (MXFP4) or Trn2 (BF16) via two paths: a
single-instance server and a disaggregated-inference (separate prefill and
decode) deployment.

## Trn2 performance options

On Trn2 the published checkpoints load with no `--hf-overrides`. Instances without
EFA (for example trn2.3xlarge) log a warning and continue. If `max_num_batched_tokens`
does not enable segmented prefill, default prefix caching is turned off with a warning.

| `neuron_config` option | Effect on Trn2 |
|---|---|
| `"quantization": "fp8"` | Requantizes the MXFP4 expert weights to FP8 E4M3 (row scales, power-of-two, exact). Decode reads half the expert bytes. Prefill uses the same FP8 weights when the installed NKI Library supports it (the companion branch below); otherwise a BF16 copy of the experts is kept for prefill and `gpu_memory_utilization` is capped at 0.80. |
| `"gpt_oss_fp8_prefill": "auto"` (default) \| `"fp8"` \| `"bf16_copy"` | With `quantization: "fp8"`: `"auto"` drops the BF16 prefill copy when the NKI Library has FP8 prefill (`bwmm_shard_on_I.FP8_DIRECT_MATMUL` and `SUPPORTS_GATE_UP_MAJOR`) and the per-rank expert intermediate size is at least 720 (TP <= 4), else keeps it. FP8 without the copy is supported at TP=4 and TP=2; at TP=8 the FP8 prefill kernel does not compile, so `"auto"` keeps the copy there. `"bf16_copy"` always keeps the copy (1-3 % faster served throughput at 4-32 concurrent requests, half the KV cache). The legacy `GPTOSS_HYBRID_PREFILL=1/0` is honoured when this key is not set. |
| `"quantization": "bf16"` (default) | MXFP4 dequantized to BF16 |
| `"decode_context_length_buckets": [1024, 2048, 4096, 7168]` | Compiles decode at several context lengths, so short requests do not pay for `max_model_len` on every step |
| `"on_device_sampling_config": {"all_greedy": true}` | Greedy-only serving skips the general sampler |

The defaults also:
- write only the new token slots into the paged KV cache during prefill, so TTFT does
  not grow with the KV cache size;
- use the `shard_on_i` prefill MoE kernel;
- run decode attention for both full and sliding-window layers on the NKI
  `attention_block_tkg` kernel (per-KV-head block table);
- give prefill padding tokens no MoE routing.

With the companion NKI Library branch
(`pip install "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.33"`,
then `rm -rf ~/.cache/neuron_libtorch`), the model also uses a gate/up-major FP8 decode
weight layout and the hybrid static/dynamic prefill MoE kernel, which skips empty blocks.
With it, FP8 runs prefill from the FP8 weights without a BF16 expert copy (the default,
`gpt_oss_fp8_prefill: "auto"`): time to first token 127.5 ms at 2048 tokens (129.0 with the copy), decode
5.20 ms at batch 1 (5.20), and 2.1x the KV cache (756K vs 353K tokens), so served throughput at about 64
concurrent requests is 10-18 % higher. `"bf16_copy"` is 1-3 % faster at 4-32 concurrent requests.
The model detects the installed kernels, and works unchanged with the bundled NKI Library.

**FP8 needs the NKI Library branch at 33 or more concurrent requests.** The bundled NKI
Library applies the per-channel FP8 down-projection scales to the wrong columns in the
decode MoE kernel when the batch rounds up to the 64-request decode graph, and the output
is incoherent (GSM8K 0.005 vs 0.880 at 64 concurrent requests). The branch has the fix
from `fc5fec9` (2026-10-26) on; if you installed it earlier, reinstall it and run
`rm -rf ~/.cache/neuron_libtorch`. With the bundled NKI Library, use FP8 only with
`max_num_seqs=32` or lower. BF16 is not affected.

Recommended starting point on trn2.3xlarge:

```python
LLM(model="openai/gpt-oss-20b", tensor_parallel_size=4, max_model_len=8192,
    block_size=128, max_num_batched_tokens=2048, enable_prefix_caching=False,
    additional_config={"neuron_config": {
        "quantization": "fp8",
        "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
        "decode_context_length_buckets": [1024, 2048, 4096, 7168],
        "on_device_sampling_config": {"all_greedy": True}}})
```

`num_batched_tokens_buckets` compiles smaller prefill graphs, so short prompts do not pay for the full
2048-token chunk: time to first token 44 ms at 128 tokens (77 ms without the buckets), and 130 / 254 ms at
2048 / 4096 tokens (197 / 406 ms with `max_num_batched_tokens=512`). Each entry must be a multiple of 128 and
the last must equal `max_num_batched_tokens`. `max_num_batched_tokens=512` without buckets keeps about 30 %
more KV cache, which is 7 % faster for short prompts only at 64 concurrent requests.

Use the default compile route and optimization level: leave `VLLM_NEURON_BACKEND` unset
and do not pass `--optimization-level`. With these changes the default (native) route at
its default `-O2` matches the XLA route (`VLLM_NEURON_BACKEND=vllm_neuron`) within a few
percent on prefill and decode, and compiles about 2x faster. Forcing `-O1` on the native
route makes decode about 15-25% slower. (Without these changes, stock GPT-OSS BF16 decode
on the native route is about 3x slower than on the XLA route on Trn2; the GQA decode
attention change above avoids that.)
Every component can be switched back to the original path with an environment
variable (`GPTOSS_KV_WRITE=kernel`, `GPTOSS_PREFILL_IMPL=shard_on_block`,
`GPTOSS_DECODE_ATTN=torch`, `GPTOSS_PREFILL_NSB=-1`, `GPTOSS_MASK_PAD_ROUTE=0`,
`GPTOSS_GU_MAJOR=0`, `GPTOSS_DECODE_OPT=base`, neuron_config `gpt_oss_fp8_prefill: "bf16_copy"`); see the top of
`vllm_neuron/model/gpt_oss/model_bf16.py`. Decode graphs compile at `-O3` by default
(`GPTOSS_DECODE_OPT`). On the XLA route this lifts the decode graph from `-O1`; on the
native route `-O2` decode is already as fast, so the hook has no measurable effect there.
An explicit `--optimization-level` turns the per-graph hook off.

## Known issues

- **Intermittent HTTP 500 errors with an FP8 KV cache on Trn3.** A race
  condition can cause some requests to fail with an HTTP 500 error. The server
  log shows an `openai_harmony.HarmonyError` for each failed request. A retry
  of the same request succeeds. The issue occurs on Trn3 with
  `--kv-cache-dtype fp8` when the decode does not use the decode megakernel.
  Trn2 is not affected. The configuration in the
  [8K/1K Trn3 tutorial](../tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md) uses
  the decode megakernel and is not affected. **Mitigation:** set
  `VLLM_NEURON_ATTN_MASK_TORCH=1` in the server environment. For a
  disaggregated deployment, set it on the decode server. This flag builds the
  decode attention mask in PyTorch, in place of an NKI kernel.

## Accuracy Evaluation

Accuracy measured on real hardware with **MXFP4** weights (Trn3), medium
reasoning effort.

| Benchmark | gpt-oss-120b (MXFP4) |
|-----------|:--------------------:|
| GSM8K-CoT | 88.8% |
| AIME25 (avg@8, medium) | 78.75% |
| GPQA-diamond (medium) | 72.22% |

## Tutorials

- [Tutorial: Deploy gpt-oss with vLLM Neuron](../tutorials/tutorial-gpt-oss.md)
  — End-to-end single-instance and disaggregated-inference deployment recipe for
  gpt-oss 20B and 120B.
- [Prefix caching benchmark tutorial](../tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.md)
  — Benchmark prefix caching with gpt-oss across open-source datasets.
- [High-performance GPT-OSS 120B implementation for 8K/1K serving on Trn3](../tutorials/tutorial-gpt-oss-120b-8k1k-trn3.md)
  — The 6P1D configuration on two Trn3 instances with the decode megakernel,
  and the scripts that run it and its benchmark.
