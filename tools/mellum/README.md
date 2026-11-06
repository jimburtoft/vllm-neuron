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
| fp32 first-divergence gate, 2 long prompts (3,512 / 2,209 input tokens, 1,024 output) | PASS, max gap 0.346 (0.192 in other compiles; bf16 near-ties move between compiles) |
| fp32 first-divergence gate, 2 very long prompts (20,549 / 40,232 input tokens, 256 output; `--sets xlong`) | PASS, max gap 0.154 |
| Hugging Face bf16 on CPU, same gate (precision floor) | max gap 0.493 |
| GSM8K, first 100 test questions, greedy, thinking on | 97 % |

## Throughput

`python tools/mellum/bench.py JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep --n 128 --max-num-seqs 64 --fixed 1024 --max-model-len 2048`

| Batch | Output tok/s (one device, TP=4, EP=4) |
|---|---|
| 1 | 102.8 |
| 32 | 1,818 |
| 64 | 2,728 |
| 128 | 2,683 |

GSM8K questions (about 70 input tokens), 1,024 output tokens per request (`--fixed`, EOS ignored), BF16, SDK 2.33
DLAMI, second of two runs after warm-up. With 4,096 output tokens (`--max-model-len 4608`), batch 64 gives 2,067.

Without `--enable-expert-parallel`, single-request decode is faster (163.8 tok/s) because each token reads 8
expert slices at 1/4 width, and long-prompt prefill is faster (`vllm serve`, 4,096-token prompts, 64 concurrent:
TTFT p50 6.6 s vs 16.8 s with EP=4), but decode at 16-32 concurrent requests is about half as fast. Pick EP=4 for
mixed traffic and short prompts, no EP for long prompts or single-stream latency.

## Packed prefill

`VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8` lets one prefill step take up to 8 waiting prompts (whole-prompt prefill
only: keep `--max-num-batched-tokens` = `--max-model-len` and prefix caching off). Same server as below, TP=4,
EP=4, 64 concurrent requests unless noted:

| Input / output | Output tok/s, 1 -> 8 per step | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---:|---:|
| 128 / 128 | 946 -> 1,542 (+63 %) | 2,472 -> 816 | 48.7 -> 35.4 |
| 512 / 128 | 728 -> 1,014 (+39 %) | 3,783 -> 2,216 | 58.8 -> 46.2 |
| 128 / 1,024 | 1,852 -> 2,046 (+10 %) | 2,465 -> 808 | 32.2 -> 30.5 |
| 128 / 1,024, concurrency 16 | 993 -> 1,040 (+5 %) | 666 -> 312 | 15.5 -> 15.1 |
| 512 / 1,024 | 1,728 -> 1,885 (+9 %) | 3,780 -> 2,220 | 33.4 -> 31.8 |
| 1,024 / 1,024 | 1,575 -> 1,699 (+8 %) | 5,646 -> 4,161 | 35.1 -> 33.6 |
| 4,096 / 1,024 | 1,032 -> 1,054 (+2 %) | 16,761 -> 16,330 | 45.7 -> 44.8 |

GSM8K-100 97 % -> 96 % (within noise; 3 vs 4 misses, 2 in common). Packing changes which requests share a step,
so greedy tokens move at bf16 near-ties (fp32 gate max gap 0.66 vs 0.35 unpacked; every divergence is a near-tie
with the token in the fp32 top 5).

## Online serving (`vllm bench serve`)

One server, `--max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching`,
TP=4, EP=4, random prompts, EOS ignored.

| Input / output | Concurrency | Output tok/s | TTFT p50 (ms) | TPOT p50 / p99 (ms) |
|---|---:|---:|---:|---:|
| 128 / 1,024 | 1 | 102.8 | 82 | 9.7 / 9.7 |
| 128 / 1,024 | 16 | 996 | 659 | 15.4 / 16.0 |
| 128 / 1,024 | 64 | 1,859 | 2,471 | 32.0 / 34.3 |
| 1,024 / 1,024 | 1 | 101.8 | 184 | 9.7 / 9.7 |
| 1,024 / 1,024 | 16 | 910 | 1,493 | 16.1 / 17.4 |
| 1,024 / 1,024 | 64 | 1,581 | 5,638 | 35.0 / 40.3 |
| 4,096 / 1,024 | 1 | 98.4 | 534 | 9.7 / 9.7 |
| 4,096 / 1,024 | 16 | 698 | 4,400 | 18.6 / 22.4 |
| 4,096 / 1,024 | 64 | 1,034 | 16,780 | 45.5 / 61.2 |
| 1,024 / 4,096 | 1 | 103.1 | 183 | 9.7 / 9.7 |
| 1,024 / 4,096 | 16 | 1,030 | 1,487 | 15.2 / 15.5 |
| 1,024 / 4,096 | 64 | 1,979 | 5,646 | 31.0 / 32.3 |

## Long context

`VLLM_NEURON_SWA_RING=0`, `--max-model-len 49152 --max-num-batched-tokens 4096 --no-enable-prefix-caching`
(segmented prefill in 4,096-token chunks), TP=4, EP=4, 1,024 output tokens.

| Input | Concurrency | Output tok/s | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---:|---:|---:|
| 16,384 | 1 | 77.4 | 2,422 | 10.6 |
| 16,384 | 4 | 167.6 | 5,933 | 18.1 |
| 32,768 | 1 | 65.2 | 4,924 | 10.6 |
| 32,768 | 4 | 119.1 | 12,175 | 21.7 |
| 45,056 | 1 | 57.9 | 6,909 | 10.5 |
| 45,056 | 4 | 97.1 | 17,066 | 24.6 |

Decode cost barely grows with context (TPOT 9.7 ms at 1k, 10.5 ms at 45k for one request): 21 of the 28 layers
only read a 1,024-token window. Use segmented prefill for long prompts. A single-shot prefill graph of 24k+
tokens needs more host memory to compile than a trn2.3xlarge has (128 GB RAM + swap was not enough).
