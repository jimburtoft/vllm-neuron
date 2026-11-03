# Gemma3-12B on Trainium2: run recipe and long-document workload

Scripts for `google/gemma-3-12b-it` on one Trainium2 device (4 logical NeuronCores at LNC=2, TP=4) with
vLLM 0.29 + this vllm-neuron branch. They need the NKI Library branch
[`jimburtoft/nki-library@gemma3-trn2-perf-2.33`](https://github.com/jimburtoft/nki-library/tree/gemma3-trn2-perf-2.33)
for full speed. The model also runs on the stock NKI Library, more slowly.

| File | What |
|---|---|
| `run.sh` | The recommended settings: `offline-2k`, `offline-4k`, `serve-2k`, `serve-4k`, `online-2k`, `online-4k` |
| `long_workload.py` | Builds the Wikipedia prompts, runs the offline benchmark (`target`), builds the fp32 CPU reference (`reference`) |
| `fp32_gate.py` | Accuracy gate: fp32 HF logits at the first greedy divergence of each prompt |
| `online_client.py` | Streaming `/v1/completions` client at fixed concurrency (token-ID prompts) |
| `data/` | The 2,048- and 4,096-token prompts and their fp32 reference outputs (Wikipedia text, CC BY-SA 4.0; see `data/README.md`) |

## Settings

| | 2k (2,048 in / up to 2,048 out) | 4k (4,096 in / up to 4,096 out) |
|---|---|---|
| `--max-model-len` | 4096 | 8192 |
| `--max-num-seqs` | 83 | 63 |
| `--max-num-batched-tokens` | 2176 | 4224 |
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` | 0.55 | 0.60 (the highest that loads at 8192) |
| `decode_context_length_buckets` | [2560, 3072, 3584] | [5120, 6144, 7168] |

Both use `--gpu-memory-utilization 0.90`, `--no-enable-prefix-caching`, BF16, and
`neuron_config` `{"on_device_sampling_config": {"all_greedy": true}, "max_logprobs": 0}`, plus:

```bash
export NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD=1000000000000000   # in-kernel d256 KV write
export VLLM_NEURON_SWA_REAL_ADMISSION=1 VLLM_NEURON_SWA_EARLY_FREE=1 VLLM_NEURON_SWA_RING=1
export VLLM_NEURON_SWA_CTX_BUCKETS=1 VLLM_NEURON_SWA_ALIGN_TOKENS=1024
export GEMMA3_PACKED_PREFILL=1 VLLM_NEURON_MAX_PREFILLS_PER_BATCH=1
```

What the sliding-window settings do (all off by default in the plugin, so other models are unchanged):

- `VLLM_NEURON_SWA_REAL_ADMISSION=1`: vLLM charges every request its worst-case KV for all layers. Gemma3's 40
  sliding-window layers only ever hold a 1,024-token window, so the scheduler admits by each request's real peak.
- `VLLM_NEURON_SWA_EARLY_FREE=1`: frees sliding-window blocks that fall out of the window once prefill is done.
- `VLLM_NEURON_SWA_RING=1`: each sliding-window layer keeps a fixed ring of window blocks per request, so decode
  reads a fixed-size table. Without it, 48 requests fit at 2k instead of 83, and throughput halves.
- `"max_logprobs": 0`: skips the all-gather of the full logits every step. Remove it if you need logprobs.
- `decode_context_length_buckets` + `VLLM_NEURON_SWA_CTX_BUCKETS=1`: decode graphs at several context lengths;
  the buckets must be multiples of 256.

The ring refuses segmented prefill, prefix caching and speculative decoding.

## Long context (8K / 16K input)

`run.sh` also has `l8`, `l16`, `s8` and `s16` modes (`offline-l8`, `serve-l16`, `online-s8`, ...), measured on one trn2 device on the public SDK 2.33 DLAMI
(trn2.3xlarge), greedy, BF16, natural EOS at the cap:

| mode | in / out cap | `--max-model-len` | batch | KV fraction | decode ctx buckets | out tok/s (1 device) | fp32 gate |
|---|---|---|---|---|---|---:|---|
| `l8` | 8,192 / 8,192 | 16384 | 37 | 0.58 | [9216,10240,...,15360] (step 1024) | 924.4 | PASS (max 0.243) |
| `l16` | 16,384 / 16,384 | 32768 | 20 | 0.55 | [24576] | 409.5-442.8 | PASS (max 0.188, first 2,048 tokens) |
| `s8` | 8,192 / 512 | 8704 | 60 | 0.60 | - | 531.0 | PASS (max 0.098) |
| `s16` | 16,384 / 512 | 16896 | 32 | 0.50 | - | 252.5 | PASS (max 0.189) |

The prompts are 8 longer Wikipedia articles (the 2k/4k articles are too short for 16K):
`python tools/gemma3/long_workload.py prompts-wiki --lengths 8192 16384 --out-prefix <dir>/wiki`, then `PROMPTS_DIR=<dir> bash tools/gemma3/run.sh offline-l8`.

Notes:
- Odd decode batches are supported. Before the GQA-folded decode mask fix (`GEMMA3_MASK_FOLD_KV`, default on), odd batch sizes
  with 2 KV heads per rank (TP4) gave wrong tokens; the error was largest at >= 8K context.
- **Compiling the 16K graphs needs a lot of host memory**: about 35-65 GB per compiler process, one per TP rank. On a trn2.3xlarge (128 GB) set `PCW=1`
  and add a large swap file (the instance NVMe works). The first compile takes several hours. The KV fraction above 0.60 (8K) / 0.50 (16K, short output)
  fails at model load.
- The short-output shapes are prefill-bound: one 8K prefill takes about 0.54 s and one 16K prefill about 1.2 s, and prefills run one per step.
