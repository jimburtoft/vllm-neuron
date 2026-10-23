# Qwen3-ASR fixes for vllm-neuron 0.29.0.1.2.0

This branch is the public `release-0.29.0.1.2.0` (vLLM 0.29.0, SDK 2.33) plus the Qwen3-ASR fixes below. The same fixes are proposed for the vllm-neuron development line.

**Champion stack (clone both):**
- vllm-neuron: `https://github.com/jimburtoft/vllm-neuron`, branch `qwen3-asr/0.29-champion` (this branch)
- nkilib: `https://github.com/jimburtoft/nki-library`, branch `qwen3-asr/2.33-champion`. It holds two single-token decode kernel improvements and one batched-decode improvement; see `QWEN3_ASR_CHAMPION.md` there.

```bash
git clone -b qwen3-asr/0.29-champion https://github.com/jimburtoft/vllm-neuron.git
git clone -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git
source vllm-neuron/qwen3_asr/setup_champion.sh $PWD/nki-library   # sets PYTHONPATH, private NKI caches, OMP_NUM_THREADS
```

Run inside an SDK 2.33 vLLM environment, either the DLAMI venv `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0` or the 2.33 vLLM container. The script checks that `vllm_neuron` and `nkilib` resolve to the clones.

## The fixes

| commit | change | effect |
|---|---|---|
| LNC launch grids | `flash_attention`, `qkv_proj`, `o_proj` and `merge_multimodal_embeddings` launch with the configured LNC instead of a fixed grid of 2 | **LNC=1 works**: 8 replicas (TP1 × DP8) on one trn2.3xlarge |
| TP1 argmax | The TP1 NKI argmax reads its uint32 index through a no-op `torch.gather` (works around `NCC_IDDT901` at batch ≥ 2), plus a stage free-size guard (`nc_find_index8` limit) | **TP1 batching compiles** at `max_num_seqs` ≥ 2 |
| Encoder shard order | At encoder DP > 1, shard the audio blocks across DP ranks before the conv frontend | +17 % (8 s) / +5 % (30 s) at encoder DP 2. Unchanged at DP 1 |
| Encoder GELU in fp32 | Compute the encoder GELU in fp32 and cast back. The bf16 `1 + erf(x)` loses its digits for x < −2 | Fixes the LNC=1 encoder TP2 accuracy miss (BC 0.9896 → 0.9986). Encoder −7 % (LNC=1 25.6 → 23.7 ms, LNC=2 20.1 → 18.6 ms). WER unchanged |
| API-server listeners | Each internal-DP API worker binds its own SO_REUSEPORT listener (`VLLM_NEURON_API_SERVER_REUSEPORT`, default on) | Removes the random up-to-30 % throughput drops where one API process took most connections |

Files: `vllm_neuron/functional/{argmax,merge_multimodal_embeddings}.py`, `vllm_neuron/functional/attention/{attention_cte,qkv,o_proj}.py`, `vllm_neuron/model/qwen3_asr/audio_encoder_bf16.py`, `vllm_neuron/envs.py`, `vllm_neuron/vllm/patches/dp_load_balancer_patch.py`.

## Results

These are for the full champion stack: this branch plus the champion nkilib. They were measured 2026-10-23 (throughput rows re-measured with the batched-decode attention change, interleaved against the stack without it: 1,366–1,383 and 1,517–1,526) on one trn2.3xlarge in the SDK 2.33 release vLLM image (vllm 0.29.0, nki 0.7.0, neuronx-cc 2.28.7631), set up with `qwen3_asr/setup_champion.sh`.
- **Client:** `vllm bench serve --backend openai-audio --endpoint /v1/audio/transcriptions`.
- **Workload:** 2048 Earnings22 requests, greedy decoding, 256-token cap, `OMP_NUM_THREADS=1`.
- **Metric:** throughput in audio seconds per wall second. Each result is 2 runs on one server.

| configuration | 30 s clips | 8 s clips |
|---|---|---|
| release, unpatched: best layout that runs (TP2 × DP2, LNC=2) | 963–965 | 442 |
| **champion**: TP1 × DP4, LNC=2, max_num_seqs 48 | **1,412–1,413** | **978** |
| **champion**: TP1 × DP8, LNC=1, max_num_seqs 48 | **1,568–1,577** | — |

| single stream (TP4, LNC=2, 30 s clip, 256 requests) | TTFT p50 | ITL p50 | E2E p50 |
|---|---|---|---|
| release, unpatched | 46–48 ms | 3.60 ms | 359–365 ms |
| **champion** | **44.9–47.5 ms** | **3.13–3.14 ms** | **319–324 ms** (−11 %) |

- **Earnings22 WER** (upstream `run_accuracy_asr`, 341 chunks, champion TP1 × DP4): **5.99 %**. Reference gate: 6.24 %.
- **API-server balance:** the largest per-worker request share was 0.26–0.29 with 4 servers and 0.15 with 8, so the load is even (fair share is 0.25 and 0.125).
- **`decode_context_length_buckets`:** leave it unset. With the champion nkilib, `[512]` is slower (E2E 328–331 vs 323–324 ms).
- **Without this branch:** the unpatched release can't compile TP1 at `max_num_seqs` ≥ 2 and can't run at LNC=1.

## Recommended settings
```bash
export OMP_NUM_THREADS=1        # REQUIRED on trn2.3xlarge (12 vCPU): otherwise TTFT rises 5-10x
# LNC=2 (default), 4 replicas:
vllm serve Qwen/Qwen3-ASR-1.7B --tensor-parallel-size 1 --data-parallel-size 4 --max-num-seqs 48 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --no-enable-prefix-caching --dtype bfloat16 \
  --additional-config '{"neuron_config":{"quantization":"bf16","num_batched_tokens_buckets":[128,256,512,1024],"num_seqs_buckets":[1,2,4,8,16,32,48]},"encoder_neuron_config":{"num_encoder_input_buckets":[1,2,4],"encoder_attention_block_size":128,"dp_size":1}}'
# LNC=1, 8 replicas: NEURON_LOGICAL_NC_CONFIG=1 VLLM_NEURON_RESERVED_HEADROOM_GIB=2 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1, --data-parallel-size 8
```

## API-server load (vLLM 0.29)

vLLM 0.29 has every API process accept from one shared socket. A client's keep-alive pool can then land mostly on one process, which becomes CPU-bound, and throughput drops at random. With TP1 × DP8, 5 of 22 runs were up to 30 % low.

- **On this branch** each API worker binds its own SO_REUSEPORT listener, so the kernel spreads connections across the workers. Set `VLLM_NEURON_API_SERVER_REUSEPORT=0` to get the old behavior.
- **Upstream** vLLM 0.32 fixes the same problem with a least-loaded accept (<https://github.com/vllm-project/vllm/pull/60723>). That fix spreads connections more evenly than kernel hashing.
- **When benchmarking**, still record each API server's request share and flag runs where one takes > 50 %.
