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
| `data/` | fp32 reference outputs (token ids + top-20 fp32 log-probs) for 5 short and 2 long prompts |

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
- `--enable-expert-parallel`: 16 experts per rank with the full expert width (896), which lets the prefill
  expert kernel run. Without it each rank holds all 64 experts at width 224, and prefill uses the torch MoE.
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
| fp32 first-divergence gate, 2 long prompts (3,512 / 2,209 input tokens, 1,024 output) | PASS, max gap 0.192 |
| Hugging Face bf16 on CPU, same gate (precision floor) | max gap 0.493 |
| GSM8K, first 100 test questions, greedy, thinking on | 97 % |

## Throughput

`python tools/mellum/bench.py JetBrains/Mellum2.1-12B-A2.5B-Thinking --ep --n 128 --max-num-seqs 64 --fixed 1024 --max-model-len 2048`

| Batch | Output tok/s (one device) |
|---|---|
| 1 | 102.6 |
| 32 | 1,743 |
| 64 | 1,978 |

GSM8K questions (about 70 input tokens), 1,024 output tokens per request (`--fixed`, EOS ignored), BF16, SDK 2.33
DLAMI, median of 2 runs after warm-up.
