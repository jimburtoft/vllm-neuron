# Contrib Model: Qwen3.5 (dense and MoE, text-only)

This folder serves the Qwen3.5 family with vLLM on AWS Trainium through the
`vllm-neuron` plugin:

| Checkpoint | Architecture | Layers (DeltaNet / attention) | MLP |
| --- | --- | --- | --- |
| Qwen3.5-2B | `Qwen3_5ForConditionalGeneration` | 24 (18 / 6) | dense |
| Qwen3.5-27B | `Qwen3_5ForConditionalGeneration` | 64 (48 / 16) | dense |
| Qwen3.5-35B-A3B | `Qwen3_5MoeForConditionalGeneration` | 40 (30 / 10) | 256 experts, top-8, gated shared expert |
| Qwen3.5-397B-A17B | `Qwen3_5MoeForConditionalGeneration` | 60 (45 / 15) | 512 experts, top-10, gated shared expert |

Qwen3.5 is a **hybrid** decoder. Most layers are gated DeltaNet, a linear
recurrence that keeps a fixed-size state per sequence. The other layers are
full attention with a paged KV cache. vLLM puts the two kinds of layer in
separate KV cache groups (one attention group and `MambaSpec` state groups) that
share one block pool.

> **Note:** Contrib models are community contributions and may have limited
> testing or performance optimization compared to the officially-supported
> models under `vllm_neuron/model/`. The compatibility matrix below records
> exactly what was tested.

## Requirements

**This model needs the recurrent-state (hybrid) support in `vllm-neuron`** from
gregnetau's `qwen3_5-upstream` branch ("Recurrent-state (hybrid) models on the V1
runner, with MTP", and the three commits under it), plus the block-table
alignment fix for non-power-of-2 blocks. That change adds `StateLayerSpec` /
`KVSpec.state_layers`, the `HasRecurrentState` / `ServesTextOnly` model
interfaces, and a dedicated state pool in `NeuronModelRunner`, which keeps the
recurrent state out of the shared KV buffers. Without it the model fails at
`bind_state_cache`.

It runs on the native compile route (the default; set
`VLLM_NEURON_BACKEND=neuron_native` to also get the Precompile Cache).

Hybrid models run on the V1 model runner. `model_runner_v2_policy` already routes
them there; do not set `VLLM_USE_V2_MODEL_RUNNER=1`.

## Usage

Install `vllm-neuron` in your Neuron environment (see the top-level `README.md`).
Then register the two architectures with vLLM (see
[contrib/README.md](../../README.md), "How do I register a custom model with
vLLM?"). The registered classes are `src/factory.py`.

### Offline inference with the Python API (Option 1)

Register the factories from your script, before you construct `LLM(...)`:

```python
from vllm import LLM, ModelRegistry, SamplingParams

# The package is this folder's src/, copied into your application or installed
# with `pip install -e contrib/models/qwen3_5`.
from qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)

ModelRegistry.register_model(
    "Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration
)
ModelRegistry.register_model(
    "Qwen3_5MoeForConditionalGeneration", Qwen3_5MoeForConditionalGeneration
)

llm = LLM(
    model="Qwen/Qwen3.5-2B",
    tensor_parallel_size=4,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    max_num_seqs=4,
    enable_prefix_caching=False,
    limit_mm_per_prompt={"image": 0, "video": 0},
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": True},
        }
    },
)
outputs = llm.generate(["The color of the sky is"], SamplingParams(max_tokens=32))
print(outputs[0].outputs[0].text)
```

`scripts/run.py` does the same for all four checkpoints, including the
expert-parallel and head-replication settings that 397B-A17B needs:

```bash
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-2B
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-27B \
    --gpu-memory-utilization 0.65
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-35B-A3B \
    --gpu-memory-utilization 0.72
python contrib/models/qwen3_5/scripts/run.py --model <path>/Qwen3.5-397B-A17B \
    --tensor-parallel-size 64 --expert-parallel --ep-degree 8 \
    --gpu-memory-utilization 0.7
```

The registration in your script reaches the engine and worker processes when
vLLM forks them. If they are spawned, use the package install below.

### Serve with the OpenAI-compatible server

`vllm serve` runs no script of yours, so install this folder as a package. Its
`vllm.general_plugins` entry point (`src/plugin.py`) registers both
architectures in every vLLM process, with no change to the plugin's core code:

```bash
pip install -e contrib/models/qwen3_5

vllm serve Qwen/Qwen3.5-2B \
    --tensor-parallel-size 4 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [1024], "num_seqs_buckets": [4]}}'
```

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen3.5-2B", "prompt": "The color of the sky is", "max_tokens": 32}'
```

### Settings this model needs

- **`limit_mm_per_prompt={"image": 0, "video": 0}`.** The checkpoints carry a
  `vision_config`, but this port is text-only; with every limit at 0 the
  platform skips vision bucket resolution (core branch), which otherwise fails
  for `max_model_len` below the encoder block size (2048).
- **`enable_prefix_caching=False`.** Prefix caching over recurrent state needs
  per-block state copies, which are not implemented. The runner raises if
  `mamba_cache_mode` is not `"none"`.
- **`max_num_batched_tokens` equal to `max_model_len`.** The model does not
  support chunked prefill: each prefill would start DeltaNet from a zero state
  and attend only to its own tokens, which gives wrong output without an error.
  The 0.29 runner turns segmented prefill on whenever `max_num_batched_tokens <
  max_model_len`, so the factory raises at load in that case.
- **No explicit `--optimization-level`.** The model's `compiler_args_for`
  compiles the dense decode graph at `-O3` and without
  `--modular-flow-mac-threshold=10`. On vLLM-neuron 0.24 the dense decode graph
  failed codegen with that flag (`NCC_IBTN006`), and 27B's decode graph failed
  ISA validation at `-O1` (`NCC_IINAR001`). The runner calls this policy only
  when you do not set the level. The sparse checkpoints keep the base
  arguments: their MoE block calls NKI kernels, which is what the threshold is
  for.
- **`--gpu-memory-utilization` below the default for the large checkpoints.**
  HBM on Trn2 is 24 GB per logical NeuronCore. The block planner fills the
  whole KV budget, which for a hybrid model means recurrent-state blocks far
  past `max_num_seqs` (`mamba_block_size == max_model_len`, so each sequence
  uses one state block). Too many blocks fail compilation with `NCC_EVRF009`
  (tensors exceed the HBM limit). Earlier runs used 0.65 for 27B and 0.72 for
  35B-A3B at TP=4, and 0.7 for 397B-A17B at TP=64.
- **397B-A17B only:** `enable_expert_parallel=True` with
  `neuron_config.ep_degree=8` at TP=64, and the head-count relaxation
  (`relax_head_divisibility` in `src/plugin.py`, applied by the plugin in every
  process, because vLLM rejects 64 ranks for 32 query heads). Its top-10 routing is above the fused decode kernel's
  limit (`moe_block_tkg` calls `nkilib`'s `router_topk`, which asserts K <= 8),
  so decode routes in torch and runs the experts through `moe_tkg`
  (`_forward_decode_unfused` in `src/moe.py`); prefill's `NF.router` already
  falls back to torch above 8. No patched `nkilib` is needed.

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29) | 2.32 and earlier (vllm-neuron 0.24) |
| ---------------------- | ----------------------- | ----------------------------------- |
| Trn3                   | Not tested              | Not tested                          |
| Trn2                   | Working (2B, 27B, 35B-A3B, 397B-A17B; see below) | Working (see below) |

On SDK 2.33 (trn2.48xlarge, TP=4, with the core branch): Qwen3.5-2B passes the
logit-validation integration test (`test/integration/test_model.py`, batch 1 and 4,
`max_model_len` 1024, every token within tolerance); Qwen3.5-27B and 35B-A3B give
coherent greedy output from `scripts/run.py`; Qwen3.5-397B-A17B (TP=64, EP=8,
stock `nkilib`) gives coherent greedy output from `scripts/run.py`.

### Measured on SDK 2.33

Setup:

- trn2.48xlarge, bf16, greedy, native PyTorch route
  (`VLLM_NEURON_BACKEND=neuron_native`).
- 1024-token input, 128-token output, `max_model_len` 2048.
- `scripts/benchmark_latency.py`: streamed TTFT, then TPOT over the remaining
  tokens; 3 rounds after a warmup round.
- Batch = `--max-num-seqs` = the number of concurrent requests.
- The TP=4 checkpoints used one Neuron device each, four runs side by side
  (`NEURON_VISIBLE_DEVICES=0-3`, `4-7`, ...). `--gpu-memory-utilization` 0.65
  for 27B and 0.72 for 35B-A3B.

| Checkpoint | Parallelism | Batch | TTFT mean | TPOT mean | Output tok/s (aggregate) |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B | TP=4 | 1 | 47.9 ms | 6.89 ms | 138.7 |
| Qwen3.5-2B | TP=4 | 4 | 109.3 ms | 10.89 ms | 342.3 |
| Qwen3.5-2B | TP=4 | 8 | 202.2 ms | 33.32 ms | 230.7 |
| Qwen3.5-27B | TP=4 | 1 | 291.8 ms | 35.41 ms | 26.7 |
| Qwen3.5-27B | TP=4 | 4 | 755.7 ms | 50.40 ms | 71.4 |
| Qwen3.5-27B | TP=4 | 8 | 1697.5 ms | 478.85 ms | 16.4 |
| Qwen3.5-35B-A3B | TP=4 | 1 | 228.9 ms | 18.17 ms | 50.5 |
| Qwen3.5-35B-A3B | TP=4 | 4 | 561.1 ms | 33.99 ms | 104.9 |
| Qwen3.5-35B-A3B | TP=4 | 8 | 1035.2 ms | 65.76 ms | 109.0 |

Against the same checkpoints on the XLA route (`VLLM_NEURON_BACKEND=vllm_neuron`,
earlier on SDK 2.33, each run alone on the instance):

| Checkpoint | Batch 1 TPOT | Batch 4 TPOT | Batch 8 TPOT |
| --- | --- | --- | --- |
| Qwen3.5-2B | 7.01 ms | 14.53 ms | 25.60 ms |
| Qwen3.5-27B | 53.60 ms | 173.32 ms | 161.67 ms |
| Qwen3.5-35B-A3B | 34.54 ms | 77.40 ms | 143.79 ms |

- TTFT is within a few percent of the XLA route (2B 49.3 / 125.6 / 201.2 ms,
  27B 298.0 / 821.0 / 1348.1 ms, 35B-A3B 232.9 / 599.2 / 1058.3 ms).
- **Open:** batch 8 on the dense checkpoints is slower than the XLA route
  (2B 33.3 vs 25.6 ms, 27B 478.9 vs 161.7 ms). 27B is 50.4 ms at batch 4, so
  something changes between 4 and 8; not yet profiled.
- **Open:** 397B-A17B (TP=64, EP=8) has not completed on the native route: its
  batch-1 decode warmup times out in a collective. On the XLA route it measured
  TTFT 225.8 / 563.0 / 959.8 ms and TPOT 51.77 / 59.50 / 69.07 ms at batch
  1 / 4 / 8 with `--prefill-buckets 1024,2048`.

Two changes account for most of the native-route decode time:

- **DeltaNet state pages** (`src/kernels/state_cache.py`). Inside the full
  decode graph, `index_select` over the state pages becomes ~37k small DMAs per
  page (27B batch 1 was 603.7 ms). `index_copy_` makes the prefill state write
  slow (35B-A3B TTFT 508 ms). The kernels move each half-page with one
  hardware-DGE DMA. By default (`QWEN3_5_STATE_KERNEL=auto`) the write kernel
  is always used and the read kernel only on the dense checkpoints: on the
  sparse ones the torch gather is fast and the kernel's output costs a spilled
  HBM transpose per layer (35B-A3B batch 1: 18.2 ms vs 47.7 ms). Overrides:
  `1` (both), `read`, `write`, `0` (neither).
- **Decode attention** gathers each KV head's context once and carries the
  group's query heads in the matmul (2B batch 8: 53.0 -> 33.3 ms). Gathering per
  query head fetched the same pages 2x (2B) to 6x (27B).

The results below are from the
earlier in-tree port on vllm-neuron 0.24.0.1.1.0, measured on trn2.3xlarge at
TP=4, bf16, greedy, 1024-token input and 128-token output, `max_model_len`
2048:

| Checkpoint | Batch | TTFT mean | TPOT | Output tok/s (aggregate) |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | 1 | 145.3 ms | 3.88 ms | 200.5 |
| Qwen3.5-2B | 4 | 353.5 ms | 7.62 ms | 385.5 |
| Qwen3.5-2B | 8 | 624.3 ms | 12.35 ms | 464.8 |
| Qwen3.5-35B-A3B | 1 | 328.2 ms | 36.17 ms | 26.0 |
| Qwen3.5-35B-A3B | 4 | 860.4 ms | 77.15 ms | 47.9 |
| Qwen3.5-35B-A3B | 8 | 1544.4 ms | 142.69 ms | 52.0 |

- 27B at batch 1 on a 1024-token prompt, TP=4, 1024 prefill bucket: TTFT
  284 ms, TPOT 48.5 ms.
- 35B-A3B accuracy: 137/160 greedy tokens equal to HF bf16 (85.6%), no
  first-token mismatch.
- 397B-A17B: functional validation only (correct greedy output) on a
  trn2.48xlarge at `world_size=64, ep_degree=8`, 11.93 GiB used per rank, with a
  patched `nkilib` for top-k > 8 (no longer needed, see above).

### Optional: decode megakernel (35B-A3B, TP=4)

`QWEN3_5_DECODE_MEGAKERNEL=1` runs each one-token-per-request decode step of
Qwen3.5-35B-A3B as one NKI kernel per TP rank (`src/kernels/megakernel/`, called
from `src/megakernel.py`). The kernel covers the 40 decoder layers, the final
norm and, with on-device sampling, the vocab-parallel lm_head. The kernel replaces
the per-layer graph ops and the two all-reduces per layer with in-kernel
collectives. It reads and writes the model's own paged KV caches and DeltaNet
state pages in place. Prefill and the fallback path therefore see the same
state. A few weights are copied once at load into the kernel's layout: the
per-core DeltaNet in_proj / conv weights, fused qkv, fp32 router / norm weights,
fused shared-expert gate/up and the lm_head tiles. The routed experts are read
as loaded.

- **Coverage.** The 35B-A3B shapes, TP=4, no expert parallelism, LNC=2, KV
  block size 256 and one token per request. Pass `--block-size 256`. Batch
  buckets up to 128 run in the kernel when the runner uses state-slot rows
  (below), up to 8 otherwise. The DeltaNet state must be in the dedicated
  state pool (the default without prefix caching); with
  `VLLM_NEURON_SHARED_STATE_POOL=1` decode keeps the default path. Any other
  checkpoint or configuration logs a
  warning and keeps the default path, as does any other graph (prefill). With
  the flag unset none of this code runs.
- **Large batches.** Routed experts are deduplicated from batch 8 up: each
  expert the step routes to is loaded once and applied to all rows. Above 128
  DeltaNet state columns (batch 64 and up) the recurrence runs in groups of
  requests.
- **State-slot rows.** With the dedicated recurrent-state pool (no prefix
  caching, no speculative decoding) and a largest decode bucket of at least
  `--max-num-seqs`, the runner asks the model whether decode batch row k should
  be state slot k (`use_state_slot_rows`, true only with the flag on). If so it
  hands out the lowest free slot and pads a decode batch up to the bucket that
  covers its largest slot. The kernel then reads and writes the DeltaNet state of
  the batch as contiguous blocks instead of one indexed DMA per request; batch
  rows are permuted into slot order in the graph (per-row tensors only, the
  state never moves), and rows with no request keep their state unchanged.
- **Long context.** Context buckets above 2048 tokens use a flash-decode
  attention core (online softmax over KV-block chunks).
- **Recommended serving config** (trn2.3xlarge, validated below):
  ```bash
  QWEN3_5_DECODE_MEGAKERNEL=1 MAX_MODEL_LEN=2048 CHUNK=2048 MAX_NUM_SEQS=64 GMU=0.85 \
  PREFILL_BUCKETS="[128, 256, 512, 1024, 2048]" DECODE_BUCKETS="[1, 4, 8, 16, 32, 64]" \
  bash contrib/models/qwen3_5/scripts/serve.sh /path/to/Qwen3.5-35B-A3B --block-size 256
  ```
  Small prefill buckets keep short prompts from padding to 2048 tokens. For
  longer contexts raise `MAX_MODEL_LEN` and keep `CHUNK` 2048; `serve.sh` then
  adds the KV segment bucket (the long-context table below). Each state slot
  takes about 30 MiB per rank in the dedicated recurrent-state pool (30 DeltaNet
  layers x one padded 1 MiB page), taken out of the KV budget: at
  `max_num_seqs` 64 and GMU 0.85 the KV cache holds 176,500 tokens (86
  requests of 2048 tokens). `max_num_seqs` 128 (decode buckets up to 128) also
  runs, but at GMU 0.85 its KV cache holds 33,512 tokens, which caps concurrency
  at 16 for 2048-token requests, and GMU 0.9 does not leave enough HBM to load
  the graphs.
  `scripts/benchmark_latency.py` takes `--block-size`.

Measured on trn2.3xlarge, SDK 2.33 (vllm-neuron 0.29), native route, bf16,
greedy, TP=4: `max_model_len` 2048, `max_num_batched_tokens` 2048, prefill
buckets `[128, 256, 512, 1024, 2048]`, `max_num_seqs` 64, decode buckets
`[1, 4, 8, 16, 32, 64]`, `--block-size 256`. The two arms differ only in the
flag. They ran at GMU 0.8 (KV cache 85,085 tokens), because with the flag off
the batch-64 decode graph fails at run time at GMU 0.85 ("Failed to schedule
neff execution ... Allocation Failure"). The last two columns are the
recommended config (flag on, GMU 0.85, KV cache 176,500 tokens).

| | Flag off | Flag on |
| --- | --- | --- |
| Greedy tokens equal to HF bf16 (8 prompts x 32) | 205/256 | 190/256 |
| Prompts identical to HF over all 32 tokens | 5/8 | 5/8 |
| GSM8K, 100 questions, 8-shot chat, 100 concurrent, strict / flexible | 0.93 / 0.95 | 0.92 / 0.95 (GMU 0.85: 0.93 / 0.95) |
| Passkey retrieval, ~1.8K-token context (5 depths x 2 keys) | 10/10 | 10/10 |

Every greedy divergence from HF in either arm is at a near-tie (top-2 logit
margin 0.0 with the flag on). The GSM8K difference is one question.

`vllm bench serve`, random input / output lengths, `--ignore-eos`; output
tok/s and mean TPOT (ms):

| In / out | Concurrency | tok/s off | tok/s on | TPOT off | TPOT on | tok/s on, GMU 0.85 | TPOT on, GMU 0.85 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 / 128 | 1 | 48.6 | 132.4 | 17.81 | 6.31 | 132.0 | 6.32 |
| 256 / 128 | 16 | 126.2 | 406.6 | 116.77 | 29.13 | 403.7 | 29.27 |
| 256 / 128 | 32 | 83.7 | 502.3 | 364.70 | 44.00 | 495.6 | 44.39 |
| 256 / 128 | 64 | 28.5 | 458.5 | 1323.01 | 53.43 | 547.4 | 77.20 |
| 256 / 128 | 128 | 22.5 | 456.6 | 1793.34 | 60.16 | 548.3 | 77.25 |
| 256 / 1024 | 64 | 29.2 | 876.8 | 1305.05 | 35.99 | 1322.6 | 43.40 |
| 256 / 1024 | 128 | not run | 867.4 | not run | 40.07 | 1321.9 | 43.43 |
| 1024 / 128 | 16 | 115.6 | 313.1 | 122.59 | 34.72 | 311.8 | 34.76 |
| 1024 / 128 | 64 | 27.9 | 344.7 | 1335.00 | 65.65 | 394.5 | 99.58 |

With the flag off, decode at batch 16 and above is slow on this stack; the
flag-off 256 / 1024 run at concurrency 128 was not run (about 25 tok/s). With
the flag on at GMU 0.8 the KV cache caps concurrency at 41, which is why
concurrency 64 and 128 are higher at GMU 0.85. Prefill does not use the
kernel: mean TTFT at concurrency 16 is 1399 ms off and 1334 ms on (256-token
prompts), 2147 ms and 2128 ms (1024-token prompts).

`scripts/benchmark_latency.py` (1024-token input, 128-token output,
`max_model_len` 2048, `--gpu-memory-utilization` 0.72, `--block-size 256`,
`--max-num-seqs` = batch, one decode bucket per run), means over 3 rounds:

| Batch | TTFT off | TTFT on | TPOT off | TPOT on | Output tok/s off | Output tok/s on |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 268.8 ms | 268.1 ms | 18.30 ms | 6.33 ms | 49.4 | 119.3 |
| 4 | 650.8 ms | 647.0 ms | 36.36 ms | 12.78 ms | 97.1 | 225.4 |
| 8 | 1171.4 ms | 1148.7 ms | 61.80 ms | 20.21 ms | 113.4 | 275.4 |

The flag-off latency matches the previous base (18.29 / 36.33 / 61.81 ms).

Longer context (`max_model_len` 9472, `max_num_batched_tokens` 2048 with KV
segments, prefill bucket `[2048]`, `max_num_seqs` 8, decode buckets `[1, 4, 8]`,
GMU 0.85, `--block-size 256`; context buckets above 2048 tokens use the
flash-decode core). `vllm bench serve` medians:

| In / out | Concurrency | TPOT off | TPOT on | Output tok/s off | Output tok/s on |
| --- | --- | --- | --- | --- | --- |
| 1024 / 128 | 1 | 21.17 ms | 7.18 ms | 41.7 | 99.1 |
| 1024 / 128 | 4 | 49.62 ms | 16.04 ms | 70.6 | 172.3 |
| 1024 / 128 | 8 | 82.52 ms | 27.89 ms | 84.2 | 196.5 |
| 8192 / 256 | 1 | 21.15 ms | 7.16 ms | 37.2 | 77.2 |
| 8192 / 256 | 4 | 53.70 ms | 20.12 ms | 59.0 | 116.4 |

Passkey retrieval at ~6.7K tokens (5 depths x 2 keys): 10/10 off and on. Greedy
vs HF with the flag on at this config: 218/256, 5/8 prompts identical.


## Example Checkpoints

- <https://huggingface.co/Qwen/Qwen3.5-2B>
- <https://huggingface.co/Qwen/Qwen3.5-27B>
- <https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>

## Testing

The integration test needs the hybrid KV cache core change (see
"Requirements") and Neuron hardware. It registers the architectures itself,
serves the model through vLLM and validates its logits against HuggingFace.
The defaults (Qwen3.5-2B, TP=4) fit a trn2.3xlarge.

```bash
pytest contrib/models/qwen3_5/test/integration/test_model.py \
    -v -s --timeout=3600
```

Set `QWEN35_MODEL_PATH` (a local path or a Hub id) and `QWEN35_TP` to test
another checkpoint or tensor-parallel size.

### Optional: unit and module tests

There are no unit or module tests yet. Add them under `test/unit/` and
`test/module/` and run them in CPU mode:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_5/test/unit/
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/qwen3_5/test/module/ -v -s --timeout=600
```

## Known limitations

- **Text-only.** The vision tower is not ported. The registered classes do not
  implement `SupportsMultiModal`, so vLLM builds no multimodal processor and
  the checkpoint's `vision_config` is ignored. The earlier in-tree port reused
  the Qwen3-VL encoder from `vllm_neuron/model/qwen3_vl`, which a contrib model
  must not import.
- **No chunked prefill, no prefix caching, no speculative decoding.** The MTP
  head in the checkpoint is ignored.
- **One batch bucket per process.** With several `num_seqs_buckets` warmup hung
  on vllm-neuron 0.24.
- **Decode gathers the full block-table width** rather than only the occupied
  pages. This is the largest known performance item on the dense path.
- **`head_dim = 256`** (27B, 35B-A3B, 397B-A17B) exceeds the flash-attention
  kernel's `MAX_HEAD_DIM = 128`, so attention runs in torch. Measured at 6-8% of
  prefill.
- **Sparse decode is gather-bound.** 35B-A3B decodes about 6x slower than its
  active-parameter count predicts.
- **Expert parallelism compiles one graph per EP partition** (`ep_rank` is a
  trace constant) and cannot span data-parallel replicas.

## Source layout

```text
src/
  __init__.py      exports the two factories and the configs
  factory.py       the registered classes (and vLLM's hybrid-model contract)
  plugin.py        vllm.general_plugins entry point for `vllm serve`
  config.py        HF config parsing (dense and MoE)
  model.py         decoder, attention, KV/state cache binding, weights, compiler args
  deltanet.py      gated DeltaNet mixer (prefill chunked delta rule, decode step)
  moe.py           sparse MoE block (NKI moe_cte / moe_block_tkg kernels)
  megakernel.py    opt-in 35B-A3B decode megakernel: coverage check, load-time weight copies, call
  flags.py         debug and experiment environment switches
  kernels/         NKI kernels: DeltaNet prefill (opt-in), state-page gather / scatter
  kernels/megakernel/  the decode megakernel (DeltaNet, attention, MoE, lm_head blocks)
scripts/run.py     offline text generation for all four checkpoints
test/integration/test_model.py
```

## Attribution

This work derives from the Qwen3.5 port in
[`qingzwang/vllm-neuron`](https://github.com/qingzwang/vllm-neuron), branch
`model/Qwen3.5-2B` (Apache-2.0), which is where the 2B implementation
originates. `kernels/nki_deltanet.py`, `kernels/nki_deltanet_fused.py` and
`flags.py` are that fork's files unchanged, and the bulk of `model.py`,
`deltanet.py` and `config.py` is its work as well. The 27B, 35B-A3B and
397B-A17B support and `moe.py` were added in the in-tree port for
vllm-neuron 0.24, and this folder ports that code to vllm-neuron 0.29.
