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
| Fused encoder attention | One block-local attention kernel per encoder layer (`encoder_window_attention_kernel.py`). It reads the fused QKV output, folds the segment mask into QKᵀ, and writes the o_proj layout. Default on; `VLLM_NEURON_QWEN3_ASR_ENC_ATTN=cte` restores the previous path | Encoder 29.9 → 22.4 ms per 30 s clip (LNC1); **+3.8 % RTFx** |
| Fused encoder conv frontend (opt-in) | `VLLM_NEURON_QWEN3_ASR_ENC_CONV=nki`: conv1–conv3 + bias + fp32 GELU in one NKI kernel, intermediates in SBUF | Encoder −0.6 to −1.7 ms per clip (30 s: 20.7 → 19.4 ms p50); full model within noise (+0.4 %). WER 6.03 %. Off by default |
| Fused lm_head + greedy argmax (opt-in) | `VLLM_NEURON_QWEN3_ASR_LMHEAD=fused`: one NKI kernel streams the lm_head weight and keeps a running argmax, so it never writes logits. TP1 all-greedy only, and needs `--max-logprobs 0` | +0.4 % RTFx (within noise; WER 5.98 %). Off by default |
| FP8 KV cache (opt-in) | Packed FP8 KV for the text decoder: `--kv-cache-dtype fp8_e4m3` + `"fp8_packed_kv": true`. The prefill write quantizes, and K is stored in token pairs; decode passes the scales and packed layout to the kernel. Scale 1.0, since the checkpoint has none (`UNSAFE_FP8FNCAST=1` needed on trn2). | **1,784–1,792 RTFx** at TP1 × DP8 LNC1 with the champion nkilib (vs 1,717–1,721 bf16 KV), WER 6.07 %. KV capacity doubles |
| API-server listeners | Each internal-DP API worker binds its own SO_REUSEPORT listener (`VLLM_NEURON_API_SERVER_REUSEPORT`, default on) | Removes the random up-to-30 % throughput drops where one API process took most connections |

Files: `vllm_neuron/functional/{argmax,merge_multimodal_embeddings}.py`, `vllm_neuron/functional/attention/{attention_cte,qkv,o_proj}.py`, `vllm_neuron/model/qwen3_asr/audio_encoder_bf16.py`, `vllm_neuron/envs.py`, `vllm_neuron/vllm/patches/dp_load_balancer_patch.py`.

## Results

These are for the full champion stack: this branch plus the champion nkilib. They were measured 2026-10-23/24. The throughput rows were re-measured with the champion nkilib's batched-decode attention changes and `decode_context_length_buckets [512]`; the stack without them gave 1,366–1,383 and 1,517–1,526. All runs on one trn2.3xlarge in the SDK 2.33 release vLLM image (vllm 0.29.0, nki 0.7.0, neuronx-cc 2.28.7631), set up with `qwen3_asr/setup_champion.sh`.
- **Client:** `vllm bench serve --backend openai-audio --endpoint /v1/audio/transcriptions`.
- **Workload:** 2048 Earnings22 requests, greedy decoding, 256-token cap, `OMP_NUM_THREADS=1`.
- **Metric:** throughput in audio seconds per wall second. Each result is 2 runs on one server.

| configuration | 30 s clips | 8 s clips |
|---|---|---|
| release, unpatched: best layout that runs (TP2 × DP2, LNC=2) | 963–965 | 442 |
| **champion**: TP1 × DP4, LNC=2, max_num_seqs 48, decode bucket [512] | **1,508–1,516** | **1,042–1,058** |
| **champion**: TP1 × DP8, LNC=1, max_num_seqs 48, decode bucket [512] | **1,712–1,722** | **1,297** |
| **champion + FP8 KV**: TP1 × DP8, LNC=1, max_num_seqs 48, decode bucket [512] | **1,825–1,845** (fused encoder attention; 1,784–1,792 before it) | — |
| **champion + FP8 KV + `NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=0`**: same | **1,953–1,961** (same session without it: 1,815–1,883) | — |
| **champion + FP8 KV**, later session (different instance): same | **1,864–1,917**; `--max-logprobs` 20 vs 0 makes no difference (same-session A/B); the fused lm_head adds ~0.4 % | — |

| single stream (TP4, LNC=2, 30 s clip, 256 requests) | TTFT p50 | ITL p50 | E2E p50 |
|---|---|---|---|
| release, unpatched | 46–48 ms | 3.60 ms | 359–365 ms |
| **champion** | **44.9–47.5 ms** | **3.13–3.14 ms** | **319–324 ms** (−11 %) |

- **Earnings22 WER** (upstream `run_accuracy_asr`, 341 chunks): champion TP1 × DP4 **6.01 %** with the decode bucket [512] (5.99 % without). With FP8 KV: 6.01 % (TP1 × DP4), 6.07 % (TP1 × DP8 MNS48), 6.06 % (MNS96); with FP8 KV + fused encoder attention 5.97 % (TP1 × DP8 MNS48); with `--max-logprobs 0` and the fused lm_head 5.98 %. Reference gate: 6.24 %.
- **API-server balance:** the largest per-worker request share was 0.26–0.29 with 4 servers and 0.15 with 8, so the load is even (fair share is 0.25 and 0.125).
- **`decode_context_length_buckets`:** depends on the workload.
  - **Throughput:** use `[512]`. The champion nkilib keeps the fast K gather at the 512 bucket, giving +5.7 % (LNC2) / +8.4 % (LNC1) on 30 s clips.
  - **Single stream (TP4, B=1):** leave it unset. `[512]` is ~5 ms slower there (328–331 vs 323–324 ms), because the batched-decode path does not apply at B=1.
- **Without this branch:** the unpatched release can't compile TP1 at `max_num_seqs` ≥ 2 and can't run at LNC=1.

## Recommended settings
```bash
export OMP_NUM_THREADS=1        # REQUIRED on trn2.3xlarge (12 vCPU): otherwise TTFT rises 5-10x
export NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=0   # throughput serving: frees host CPU (+4.1 % RTFx, TTFT -65 %)
# LNC=2 (default), 4 replicas:
vllm serve Qwen/Qwen3-ASR-1.7B --tensor-parallel-size 1 --data-parallel-size 4 --max-num-seqs 48 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --no-enable-prefix-caching --dtype bfloat16 \
  --additional-config '{"neuron_config":{"quantization":"bf16","num_batched_tokens_buckets":[128,256,512,1024],"num_seqs_buckets":[1,2,4,8,16,32,48],"decode_context_length_buckets":[512]},"encoder_neuron_config":{"num_encoder_input_buckets":[1,2,4],"encoder_attention_block_size":128,"dp_size":1}}'
# Throughput: keep "decode_context_length_buckets":[512] (above). Single stream (TP4): remove it.
# Highest throughput: add FP8 KV: UNSAFE_FP8FNCAST=1, --kv-cache-dtype fp8_e4m3, and "fp8_packed_kv":true in neuron_config
#   (TP1 x DP8 LNC1 MNS48: 1,825-1,845 RTFx with the default fused encoder attention, WER 5.97 %).
# --max-logprobs 0 is needed only for VLLM_NEURON_QWEN3_ASR_LMHEAD=fused. Same-session A/B: no throughput effect
#   by itself (default 20: 1,874-1,910 RTFx; 0: 1,864-1,909). Larger max_num_seqs (64/96) fit in the doubled KV but are not faster.
# LNC=1, 8 replicas: NEURON_LOGICAL_NC_CONFIG=1 VLLM_NEURON_RESERVED_HEADROOM_GIB=2 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1, --data-parallel-size 8
```

## API-server load (vLLM 0.29)

vLLM 0.29 has every API process accept from one shared socket. A client's keep-alive pool can then land mostly on one process, which becomes CPU-bound, and throughput drops at random. With TP1 × DP8, 5 of 22 runs were up to 30 % low.

- **On this branch** each API worker binds its own SO_REUSEPORT listener, so the kernel spreads connections across the workers. Set `VLLM_NEURON_API_SERVER_REUSEPORT=0` to get the old behavior.
- **Upstream** vLLM 0.32 fixes the same problem with a least-loaded accept (<https://github.com/vllm-project/vllm/pull/60723>). That fix spreads connections more evenly than kernel hashing.
- **When benchmarking**, still record each API server's request share and flag runs where one takes > 50 %.
