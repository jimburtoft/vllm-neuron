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

### Packed prefill for short prompts (opt-in, Trn2)

`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=16` lets one prefill step carry several whole, fresh prompts (up to the
largest prefill bucket in total). Attention stays per prompt (per-token segment bounds), and padding tokens get no
MoE routing. It works with `quantization` `"fp8"` and `"bf16"`, and next to segmented prefill: with
`max_num_batched_tokens` < `max_model_len` (the configuration above), whole short prompts pack and chunks of long
prompts run alone. Prefix caching, LoRA, structured outputs, prompt embeddings and data parallelism turn it off
with a log line. Each prefill bucket compiles a second (packed) graph, which adds 5-10 minutes to the first start.

Served on trn2.3xlarge (the FP8 configuration above, `max_num_seqs` 64, random prompts, 128 output tokens), output
tokens/s at 64 concurrent requests, without / with packing:

| Input tokens | 64 | 128 | 256 | 512 | 1024 | 2048 | > 2048 |
|---|---|---|---|---|---|---|---|
| Without | 1,298 | 1,264 | 1,211 | 1,078 | 897 | 658 | |
| With packing | 2,160 (+66 %) | 2,064 (+63 %) | 1,844 (+52 %) | 1,498 (+39 %) | 1,078 (+20 %) | 692 (+5 %) | no change |

Time to first token at 64 concurrent requests and 64 input tokens drops from 1.5 s to 0.29 s. The gain grows with
concurrency (+25 % at 16 concurrent requests for 64-token prompts, +0-6 % at one request) and ends at the packed
token budget: a prompt longer than the largest bucket never packs. `max_num_batched_tokens=4096` with buckets up
to 4096 and `VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS=4096` keeps +15 % at 2048 tokens and +8 % at 3072 tokens.
GSM8K-CoT (250 questions): 0.896 with packing, 0.876 without. Packing changes which prompts share a prefill
step, so greedy output can differ from the unpacked run in the last bits (near ties); answers do not leak between
prompts.

Knobs: `VLLM_NEURON_PACKED_PREFILL_MAX_TOKENS` (real-token budget, default the largest bucket),
`VLLM_NEURON_PACKED_PREFILL_MIXED=0` (pack only when prefill is single shot), `VLLM_NEURON_PACKED_PREFILL_SPEC=0`
(no packing with EAGLE3).

### EAGLE3 speculative decoding (gpt-oss-20b, Trn2)

The public EAGLE3 draft [`RedHatAI/gpt-oss-20b-speculator.eagle3`](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3)
works with the FP8 configuration above. It needs the companion NKI Library branch at `e9b52c6` or later: older
versions fail to compile the verify step (`cannot align DMA operands ... partition count (8)` in
`attention_block_tkg`).

```bash
vllm serve openai/gpt-oss-20b --tensor-parallel-size 4 --max-model-len 4096 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 --block-size 128 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"quantization": "fp8",
      "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
      "decode_context_length_buckets": [1024, 2048], "num_seqs_buckets": [1, 4, 16],
      "on_device_sampling_config": {"all_greedy": true}}}' \
  --speculative-config '{"method": "eagle3", "model": "RedHatAI/gpt-oss-20b-speculator.eagle3", "num_speculative_tokens": 1}'
```

On trn2.3xlarge (chat prompts, 256 output tokens): output tokens/s 187 / 402 / 907 at 1 / 4 / 16 concurrent
requests, against 167 / 410 / 707 without speculation (+12 % / -2 % / +28 %). The draft acceptance rate was
71-73 %. GSM8K-CoT (1,319 questions) was 0.876 with EAGLE3 and 0.873 without. `num_speculative_tokens: 2` was
slower than `1` at every concurrency. The first compile takes about 48 minutes instead of 19.

**Not recommended (engine hang, see Known issues):** EAGLE3 with packed prefill (`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` on the command above); the draft
model attends and writes its KV cache per prompt. Same instance, chat prompts, 256 output tokens, median of 6-10
runs, output tokens/s without / with packing: 186 / 175 (-6 %) at 1 concurrent request, 407 / 414 (+2 %) at 4,
914 / 1,010 (+10 %) at 16. The acceptance rate is the same (0.68-0.69). Use it when there are usually several
requests in flight; at one request the decode steps after a packed prefill are slightly slower.

### Packed FP8 KV cache for long requests (opt-in, Trn2)

Add `--kv-cache-dtype fp8` and `"fp8_packed_kv": true` (in `neuron_config`) to the FP8 command above. The KV cache
holds twice as many tokens (1,511,925 vs 755,920 on trn2.3xlarge at `max_model_len` 8192). Decode reads the packed
cache in the GQA attention kernel, and prefill writes only the new token pairs. Decode step time (5.4 / 8.8 / 18.4 ms
at batch 1 / 4 / 16) and time to first token (60 / 141 / 275 ms at 128 / 2,000 / 3,900 tokens) match the BF16 KV
cache within 3 %.

Served on trn2.3xlarge, random prompts, output tokens/s, BF16 KV / packed FP8 KV:

| Prompt / output tokens, concurrency | BF16 KV | FP8 KV |
|---|---:|---:|
| 512 / 128, 64 | 1,096 | 1,109 (+1 %) |
| 4,000 / 128, 64 | 383 | 393 (+3 %) |
| 6,000 / 1,024, 64 (`--max-num-seqs 128`) | 979 | 1,129 (+15 %) |
| 6,000 / 1,024, 128 (`--max-num-seqs 128`) | 843 | 1,202 (+43 %) |

The gain comes when the BF16 cache cannot hold every running request. GSM8K-CoT (250 questions): 0.892. Greedy
output differs from the BF16 KV cache after a few dozen tokens, as expected for an 8-bit cache. Packed prefill
(`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=16`) also works: +51 % at 64-token prompts and 64 concurrent requests.
**EAGLE3 works with it at BF16 speed** (EAGLE3 command above plus the two FP8 settings; chat prompts, 256 output tokens,
tokens/s BF16 KV / FP8 KV: 185-190 / 178-186 at 1 request, 411-413 / 411-414 at 4, 927-933 / 956-963 at 16; same
acceptance, GSM8K 0.876). **Packed prefill + EAGLE3 + FP8 KV is not recommended**: time to first token is about 245 ms
instead of 65 ms (packed steps rewrite the whole packed K cache), and one engine hang (`RPC call to sample_tokens timed
out`) was seen in about 1,500 requests, also with BF16 KV (Known issues).
Knobs: `VLLM_NEURON_LLAMA3_GQA_MULTI_TOKEN=0` (EAGLE3 draft decode back on the PyTorch path), `GPTOSS_FP8_PACKED_GQA=0` (PyTorch decode, about 20x slower), `GPTOSS_SEG_PACKED_K_WRITE=reswizzle` (the old
whole-cache prefill write, 4-5x slower), `VLLM_NEURON_PREFILL_KV_BLOCK_WRITE=packed|all|off` (single-step prefill).

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
- **`HarmonyError` (HTTP 500) when a benchmark forces the output length.** `/v1/chat/completions` with
  `ignore_eos: true` and a `max_tokens` past the model's natural end fails with `openai_harmony.HarmonyError:
  Unexpected token 200002 while expecting start token 200006`. Forced on, gpt-oss repeats `<|return|>` (200002), and
  the chat parser rejects a token after the end of the final message. This is vLLM / Harmony behaviour on any backend,
  not a Neuron fault, and a normal request cannot reach it. For fixed-length benchmarks use `/v1/completions` (with the
  chat template applied) or `vllm bench serve` with random prompts.
- **Engine hang with packed prefill + EAGLE3 on Trn2.** The engine can stop with `RPC call to sample_tokens timed out`
  on a packed prefill step of several fresh prompts; rank 0's device result for that step never completes. In tests it
  hit 3 of 6 fresh server starts whose first batch was 64 concurrent prompts, with BF16 and FP8 KV. Not seen without
  packing (0 of 4) or with `--no-async-scheduling` (0 of 4). **Do not combine `VLLM_NEURON_MAX_PREFILLS_PER_BATCH`
  with EAGLE3** until this is fixed; packed prefill without EAGLE3 and EAGLE3 without packing are not affected.

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
