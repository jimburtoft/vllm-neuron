# GPT-OSS Model Recipe

<!-- meta: description: Model recipe for deploying gpt-oss with vLLM on Neuron,
including supported checkpoints, feature support, accuracy results, and a link to
the end-to-end deployment tutorial for gpt-oss 20B and 120B on Trn3 (MXFP4) or
Trn2 (BF16). -->
<!-- meta: keywords: vLLM, Neuron, gpt-oss, gpt-oss-20b, gpt-oss-120b, MoE,
MXFP4, model recipe, model card, LLM serving, Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-07-15 -->
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
| **Parallelism** | Tensor parallelism (TP) | ✅ |
| | Data parallelism (DP) | ✅ |
| | Expert parallelism (EP) | ✅ |
| | Pipeline parallelism (PP) | ❌ |
| **Performance** | Continuous batching | ✅ |
| | Segmented prefill | ✅ |
| | Prefix caching (APC) | ✅ |
| | Speculative decoding (EAGLE3) | ✅ |
| | Disaggregated inference (1P1D / xPyD) | ✅ |
| | On-device sampling (greedy, top-k, top-p) | ✅ |
| **Serving** | Structured outputs / tool calling | ✅ |
| **Compilation** | torch.compile (XLA backend) | ✅ |
| | CPU mode (testing) | ✅ |

**Status legend:**

- ✅ Supported: integrated and tested for gpt-oss
- ❌ Not supported: may be considered for future releases

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
| `"quantization": "fp8"` | Requantizes the MXFP4 expert weights to FP8 E4M3 (row scales, power-of-two, exact). Decode reads half the expert bytes. By default a BF16 copy of the experts is kept for prefill (`GPTOSS_HYBRID_PREFILL=0` drops it and gives more KV cache). |
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
(`pip install "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.32"`,
then `rm -rf ~/.cache/neuron_libtorch`), the model also uses a gate/up-major FP8 decode
weight layout and the hybrid static/dynamic prefill MoE kernel, which skips empty blocks.
The model detects the installed kernels, and works unchanged with the bundled NKI Library.

Recommended starting point on trn2.3xlarge:

```python
LLM(model="openai/gpt-oss-20b", tensor_parallel_size=4, max_model_len=8192,
    block_size=128, max_num_batched_tokens=512, enable_prefix_caching=False,
    additional_config={"neuron_config": {
        "quantization": "fp8",
        "decode_context_length_buckets": [1024, 2048, 4096, 7168],
        "on_device_sampling_config": {"all_greedy": True}}})
```

Use `max_num_batched_tokens=2048` when most prompts are longer than about 500 tokens.
Every component can be switched back to the original path with an environment
variable (`GPTOSS_KV_WRITE=kernel`, `GPTOSS_PREFILL_IMPL=shard_on_block`,
`GPTOSS_DECODE_ATTN=torch`, `GPTOSS_PREFILL_NSB=-1`, `GPTOSS_MASK_PAD_ROUTE=0`,
`GPTOSS_GU_MAJOR=0`); see the top of `vllm_neuron/model/gpt_oss/model_bf16.py`.

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
