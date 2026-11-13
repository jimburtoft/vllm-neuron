# Mellum2.1 on Trainium2: run recipe and accuracy gate

Scripts for [`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
on one Trainium2 device (4 logical NeuronCores at LNC=2) with vLLM 0.29 + this vllm-neuron branch. The stock
NKI Library of the SDK 2.33 DLAMI is enough.

| File | What |
|---|---|
| `env.sh` | Environment for every run (sliding-window table alignment and the SWA scheduler features) |
| `fp32_gate.py` | Accuracy gate: fp32 Hugging Face log-probs at the first greedy divergence of each prompt |
| `bench.py` | Offline throughput on GSM8K questions (chat template, thinking on) |
| `gsm8k_eval.py` | GSM8K accuracy (greedy, thinking on) with vLLM or Hugging Face transformers |
| `data/` | fp32 reference outputs (token ids + top-20 fp32 log-probs) for 5 short, 2 long and 2 very long (20k / 40k-token) prompts |

## Settings

```bash
source tools/mellum/env.sh
vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 8192 --max-num-seqs 32 --max-num-batched-tokens 8192 --no-enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes
```

- `VLLM_NEURON_SWA_ALIGN_TOKENS=1024`: at LNC=2 the decode attention kernel needs the sliding-window block table
  to be a multiple of 256 tokens. Without it the 1024-token window gives a 1,152-token table and compilation
  fails with `NCC_INKI016 ... Number of blocks per batch must be a multiple of (sprior_n_prgs * p_max)`.
- `VLLM_NEURON_SWA_RING=1`, `VLLM_NEURON_SWA_REAL_ADMISSION=1`, `VLLM_NEURON_SWA_EARLY_FREE=1`: the 21
  sliding-window layers keep a fixed ring of window blocks per request and are charged their real KV peak. The ring
  refuses segmented prefill and prefix caching: set `VLLM_NEURON_SWA_RING=0` to use them (and then pick a
  `--max-num-batched-tokens` below `--max-model-len`).
- `--enable-expert-parallel`: 16 experts per rank with the full expert width (896). Without it each rank holds all
  64 experts at width 224; the prefill expert kernel then runs with fp32 hidden-state I/O (see the model README).
- The thinking traces are long (GSM8K: about 550 tokens on average, up to several thousand). Size
  `--max-model-len` for the output you allow.

## Accuracy

```bash
python tools/mellum/fp32_gate.py --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --enable-expert-parallel
python tools/mellum/gsm8k_eval.py --backend vllm --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep \
  --n 100 --max-tokens 4096 --max-model-len 4608 --mnbt 4608 --out gsm8k.json
```

| Check | Result (TP=4, EP=4, one trn2.3xlarge) |
|---|---|
| fp32 first-divergence gate, 5 short prompts (to EOS, up to 449 tokens) | PASS, max gap 0.229 |
| fp32 first-divergence gate, 2 long prompts (3,512 / 2,209 input tokens, 1,024 output) | PASS, max gap 0.192 (also with segmented prefill, `--max-num-batched-tokens 1024`) |
| fp32 first-divergence gate, 2 very long prompts (20,549 / 40,232 input tokens, 256 output; `--sets xlong`) | PASS, max gap 0.154 |
| Hugging Face bf16 on CPU, same gate (precision floor) | max gap 0.493 |
| GSM8K, first 100 test questions, greedy, thinking on | 98 % |

Packed prefill (`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8`) changes which prompts share a step, so greedy tokens move at
bf16 near-ties; GSM8K is unchanged within noise (96-97 %).

## Prefill

With `--enable-expert-parallel` the prefill MoE uses nkilib's `shard_on_i_hybrid` kernel (default,
`MELLUM_MOE_PREFILL_IMPL=auto`). It runs only the expert blocks that hold tokens. One request, `vllm serve`,
time to first token:

| Prompt | TTFT (ms) |
|---|---:|
| 1,024 | 80 |
| 4,096 | 207 |
| 8,192 (segmented, see Long context) | 480 |
| 16,384 | 951 |
| 30,000 | 1,953 |

## Online serving (`vllm bench serve`)

`--max-model-len 8192 --max-num-batched-tokens 8192 --no-enable-prefix-caching`, TP=4, EP=4, random prompts, EOS
ignored, one prompt per prefill step unless noted.

| Input / output | Concurrency | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 1,024 / 1,024 | 16 | 999 | 629 | 15.4 |
| 1,024 / 1,024 | 64 | 1,866 | 2,318 | 32.1 |
| 4,096 / 1,024 | 16 | 896 | 1,628 | 16.3 |
| 4,096 / 1,024 | 64 | 1,536 | 6,164 | 35.7 |
| 128 / 1,024 | 64 | 1,969 | 1,384 | 31.2 |
| 1,024 / 4,096 | 64 | 2,069 | 2,307 | 30.4 |

### Decode context buckets

By default decode compiles only at `--max-model-len`, so every decode step attends over the full 8k bucket. Adding a
smaller decode bucket lets requests whose prompt + output fits use a cheaper graph:

```bash
--additional-config '{"neuron_config": {"decode_context_length_buckets": [2048]}}'
```

(each bucket must be smaller than `--max-model-len`, which stays as the fallback). 8 prompts per prefill step, EP=4:

| Input / output | Concurrency | Output tok/s, 8k only -> + 2048 bucket | TPOT p50 (ms) |
|---|---:|---|---|
| 128 / 1,024 | 64 | 2,100 -> 3,268 (+56 %) | 30.1 -> 19.2 |
| 128 / 1,024 | 128 | 2,829 -> 3,231 (+14 %) | 44.6 -> 38.9 |
| 128 / 1,024 | 256 | 3,338 -> 3,899 (+17 %) | 75.4 -> 64.3 |
| 1,024 / 1,024 | 128 | 2,573 -> 2,906 (+13 %) | 46.7 -> 41.0 |

GSM8K-100 with the 2048 bucket: 98 %. Each extra bucket adds decode graphs (compile time and HBM).

### Low-latency decode with FP8 experts (no expert parallelism)

For latency-sensitive serving at up to ~32 concurrent requests, run without `--enable-expert-parallel` and turn on
the FP8 routed-expert decode kernel (an extra FP8 copy of the experts; prefill keeps bf16):

```bash
export MELLUM_MOE_FP8_ROUTED=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.2
vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 64 \
  --max-num-batched-tokens 8192 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"decode_context_length_buckets": [2048]}}'
```

| 128 / 1,024, concurrency | bf16 (no EP) tok/s, TPOT | FP8 routed kernel tok/s, TPOT |
|---:|---|---|
| 1 | 163.2, 6.07 ms | **177.2, 5.59 ms** |
| 4 | 379.6, 10.4 ms | **522.9, 7.52 ms** (+38 %) |
| 8 | 465.1, 17.0 ms | **723.2, 10.8 ms** (+56 %) |
| 16 | 488.7, 32.3 ms | **934.2, 16.7 ms** (+91 %) |
| 32 | 664.9, 47.3 ms | **1,095, 28.4 ms** (+65 %) |
| 48 / 64 (64-token decode bucket) | 2,072 / 2,670, 22 ms | same (the dense bf16 path is faster at 64; the kernel stops at 32) |

GSM8K-100: 97 % (bf16 97-98 %). The FP8 weights move greedy tokens more than bf16 noise on one of the 7 gate
prompts (fp32 gap 1.14 at token 181, the token still in the fp32 top 5), so treat it as an opt-in accuracy trade.
For throughput (32+ concurrent requests) use `--enable-expert-parallel` without FP8.

### High concurrency

`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` (packed prefill), TP=4, EP=4. The default KV budget
(`VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.30`, 6.6 GiB per core) holds about 256 requests at 1k/1k; for more,
raise it to 0.42 and cap `--max-num-seqs 512` (higher fractions do not fit HBM next to the decode graphs).

| Input / output | Concurrency | KV budget | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---|---:|---:|---:|
| 128 / 1,024 | 128 | default | 2,828 | 703 | 44.6 |
| 128 / 1,024 | 256 | default | 3,333 | 1,411 | 75.5 |
| 128 / 1,024 | 512 | 0.42, `--max-num-seqs 512` | 3,994 | 2,664 | 125.6 |
| 1,024 / 1,024 | 128 | default | 2,529 | 3,174 | 47.5 |
| 1,024 / 1,024 | 256 | default | 2,985 | 6,230 | 79.7 |
| 1,024 / 1,024 | 512 | 0.42, `--max-num-seqs 512` | 3,483 | 12,095 | 135.3 |


## Packed prefill

`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` lets one prefill step take up to 8 waiting prompts (whole-prompt prefill
only: keep `--max-num-batched-tokens` = `--max-model-len` and prefix caching off; the model rejects other
configurations). It mostly cuts time to first token for short prompts under load:

| Input / output | Concurrency | TTFT p50, 1 -> 8 per step (ms) | Output tok/s, 1 -> 8 per step |
|---|---:|---|---|
| 128 / 1,024 | 64 | 1,438 -> 388 | 1,960 -> 2,095 |
| 128 / 1,024 | 256 | 5,619 -> 1,367 | 3,007 -> 3,337 |
| 1,024 / 1,024 | 64 | 2,333 -> 1,576 | 1,864 -> 1,954 |
| 1,024 / 1,024 | 256 | 9,427 -> 6,277 | 2,771 -> 2,988 |

## Long context

`VLLM_NEURON_SWA_RING=0`, `--max-model-len 32768` (or 49152) `--max-num-batched-tokens 4096 --no-enable-prefix-caching`
(segmented prefill in 4,096-token chunks), TP=4, EP=4, 1,024 output tokens.

| Input | Concurrency | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 8,192 | 1 | 93.2 | 480 | 10.3 |
| 8,192 | 4 | 251 | 1,119 | 14.8 |
| 8,192 | 16 | 644 | 3,724 | 21.2 |
| 16,384 | 1 | 89.4 | 951 | 10.3 |
| 16,384 | 4 | 225 | 2,300 | 15.5 |
| 30,000 | 1 | 82.3 | 1,953 | 10.3 |

Decode cost barely grows with context (TPOT 9.7 ms at 1k, 10.3 ms at 30k for one request): 21 of the 28 layers only
read a 1,024-token window. A single-shot prefill graph of 24k+ tokens needs more host memory to compile than a
trn2.3xlarge has, so use segmented prefill for long prompts.

## Offline throughput

`python tools/mellum/bench.py JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep --n 128 --max-num-seqs 64 --fixed 1024 --max-model-len 2048`
(GSM8K questions, about 70 input tokens, 1,024 output tokens, EOS ignored):

| Batch | Output tok/s |
|---|---:|
| 1 | 102.8 |
| 32 | 1,818 |
| 64 | 2,728 |
| 128 | 2,683 |

Without `--enable-expert-parallel`, single-request decode is faster (163.8 tok/s) but decode at 16-32 concurrent
requests is about half as fast; EP=4 is the recommended layout.
