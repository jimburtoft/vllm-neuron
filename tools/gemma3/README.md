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
