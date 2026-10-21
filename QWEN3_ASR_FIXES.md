# Qwen3-ASR fixes for vllm-neuron 0.29.0.1.2.0

This branch is the public `release-0.29.0.1.2.0` (vLLM 0.29.0, SDK 2.33) plus three Qwen3-ASR fixes. The same fixes are proposed for the vllm-neuron development line.

## The fixes

| commit | change | effect |
|---|---|---|
| LNC launch grids | `flash_attention`, `qkv_proj`, `o_proj` and `merge_multimodal_embeddings` launch with the configured LNC instead of a fixed grid of 2 | **LNC=1 works**: 8 replicas (TP1 × DP8) on one trn2.3xlarge |
| TP1 argmax | The TP1 NKI argmax reads its uint32 index through a no-op `torch.gather` (works around `NCC_IDDT901` at batch ≥ 2), plus a stage free-size guard (`nc_find_index8` limit) | **TP1 batching compiles** at `max_num_seqs` ≥ 2 |
| Encoder shard order | At encoder DP > 1, shard the audio blocks across DP ranks before the conv frontend | +17 % (8 s) / +5 % (30 s) at encoder DP 2. Unchanged at DP 1 |

Files: `vllm_neuron/functional/{argmax,merge_multimodal_embeddings}.py`, `vllm_neuron/functional/attention/{attention_cte,qkv,o_proj}.py`, `vllm_neuron/model/qwen3_asr/audio_encoder_bf16.py`.

## Results

**Setup**
- **Hardware:** one trn2.3xlarge (1 Trainium2 device).
- **Image:** SDK 2.33 release image (`neuron_native`).
- **Client:** `vllm bench serve --backend openai-audio --endpoint /v1/audio/transcriptions`.
- **Workload:** 2048 Earnings22 requests, greedy decoding, 256-token cap.
- **Metric:** throughput, in audio seconds per wall second.

| configuration | 30 s clips | 8 s clips |
|---|---|---|
| release, unpatched: best layout that runs (TP2 × DP2, LNC=2) | 963–965 | 442 |
| this branch: TP1 × DP4, LNC=2, max_num_seqs 48 | **1,355–1,384** | **950–981** |
| this branch: TP1 × DP8, LNC=1, max_num_seqs 48 | **1,533–1,548** | **1,072–1,162** |
| this branch: TP2 × DP2, LNC=2 | 1,027–1,030 | 528–530 |

- **Earnings22 WER:** 6.01–6.06 % (upstream `run_accuracy_asr`, 341 chunks). Reference gate: 6.24 %.
- **Single stream (TP4, LNC=2, 30 s):** TTFT 46–48 ms, E2E p50 361–365 ms.
- **Without this branch:** the unpatched release can't compile TP1 at `max_num_seqs` ≥ 2, and can't run at LNC=1.

## Recommended settings
```bash
export OMP_NUM_THREADS=1        # REQUIRED on trn2.3xlarge (12 vCPU): otherwise TTFT rises 5-10x
# LNC=2 (default), 4 replicas:
vllm serve Qwen/Qwen3-ASR-1.7B --tensor-parallel-size 1 --data-parallel-size 4 --max-num-seqs 48 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --no-enable-prefix-caching --dtype bfloat16 \
  --additional-config '{"neuron_config":{"quantization":"bf16","num_batched_tokens_buckets":[128,256,512,1024],"num_seqs_buckets":[1,2,4,8,16,32,48]},"encoder_neuron_config":{"num_encoder_input_buckets":[1,2,4],"encoder_attention_block_size":128,"dp_size":1}}'
# LNC=1, 8 replicas: NEURON_LOGICAL_NC_CONFIG=1 VLLM_NEURON_RESERVED_HEADROOM_GIB=2 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1, --data-parallel-size 8
```

## Known issue: uneven API-server load (vLLM 0.29)

With more than one API server process, which is the default when data parallel is above 1, vLLM 0.29 accepts every connection from one shared socket.

- **Symptom:** a client's keep-alive pool can land mostly on one API process. That process becomes CPU-bound, so throughput drops at random, by up to about 30 %. With TP1 × DP8, 5 of 22 benchmark runs were affected.
- **Upstream fix:** vLLM 0.32 accepts each connection on the least-loaded worker (<https://github.com/vllm-project/vllm/pull/60723>).
- **On 0.29:**
  - Check each API server's request share in the server log. If one takes more than about 50 %, restart the server before measuring.
  - Or use fewer API servers.
