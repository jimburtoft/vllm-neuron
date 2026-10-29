# MuseGlimmer on vLLM-Neuron

Serves [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) (image + video + text in, text out) on AWS Trainium2 with vLLM-Neuron.

- **Base:** vllm-neuron `0.29.0.1.2.0`, Neuron SDK 2.33, vLLM 0.29.0.
- **Target hardware:** `trn2.3xlarge` (one Trainium2 device, 4 logical NeuronCores at LNC=2). Text runs at TP=4 and the vision encoder at TP=4.

**What is supported:**
- Text, single-image and video prompts, plus mixed image+video prompts.
- BF16, or FP8 MLP weights self-quantized at load time.
- An optional FP8 KV cache.
- Optional DFlash speculative decoding with [`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant).

Upstream vLLM 0.29 ships its own GPU `MuseGlimmerForConditionalGeneration`. This plugin registers the Neuron class over it in the front-end, the engine core and the workers (see `platform.py`).

## Quick start (trn2.3xlarge, Neuron SDK 2.33 DLAMI)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
pip install --target ~/vn_muse --no-deps git+https://github.com/jimburtoft/vllm-neuron.git@contrib/muse-glimmer-0.29
export PYTHONPATH=~/vn_muse
export NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_SWA_ALIGN_TOKENS=4096
```

Add 64 GB of swap first (`sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && sudo mkswap --force /swapfile && sudo swapon /swapfile`). The compiler can use a lot of host memory while it builds the TP=4 graphs.

### Recommended serving config (FP8 MLP + FP8 KV cache)

```bash
export MUSE_GLIMMER_FP8=mlp
export MUSE_GLIMMER_KV_SCALES=/path/to/kv_scales.json     # per-layer V scales, see "FP8 KV cache"
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.27
export VLLM_NEURON_BARRIER_TIMEOUT=14400                  # FP8 prefill graphs compile slowly
export MUSE_GLIMMER_VIS_SMALL_BLOCK=1 MUSE_GLIMMER_VIS_MLP=nki MUSE_GLIMMER_VIS_LN=fused

vllm serve meta-models/Muse-Glimmer-30B --tensor-parallel-size 4 \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 --max-logprobs 0 --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' --mm-processor-kwargs '{"max_image_tokens": 1024}' \
  --additional-config '{
    "neuron_config": {"num_batched_tokens_buckets": [512, 1024, 2048, 4096, 8192],
                      "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 48, 64],
                      "decode_context_length_buckets": [4096],
                      "on_device_sampling_config": {"all_greedy": true}},
    "vision_neuron_config": {"vision_attention_block_size": 4096,
                             "num_vision_tokens_buckets": [1024, 2048, 3072, 4096],
                             "encoder_cache_num_blocks": 16, "tp_size": 4}}' \
  --port 8000
```

**BF16 (no quantization):**
- Drop `MUSE_GLIMMER_FP8`, `MUSE_GLIMMER_KV_SCALES` and `--kv-cache-dtype`.
- Set `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.15`, `--max-num-seqs 16`, `num_seqs_buckets [1, 2, 4, 8, 16]`.
- Use `num_vision_tokens_buckets [4096]`. The extra vision buckets need HBM that the BF16 weights do not leave.

An offline example is in `examples/vllm_neuron/models/muse_glimmer/run.py`.

## Performance (trn2.3xlarge, one device, recommended config)

Measured with `vllm bench serve` (random dataset, `--ignore-eos`, greedy). Concurrency was doubled until throughput stopped improving.

| Workload (in / out tokens) | Peak output tok/s (concurrency) | c=1 TTFT / TPOT (ms) |
|---|---|---|
| 128 / 128 | 747 (128) | 76 / 21.1 |
| 128 / 512 | 1,061 (256) | 76 / 21.1 |
| 1024 / 256 | 773 (128) | 136 / 21.1 |
| 2048 / 128 | 380 (128) | 247 / 21.0 |
| 2048 / 512 | 787 (256) | 247 / 21.1 |
| 896x672 image + 256 / 128 | 304 (128) | 343 / 21.1 |

- 58.5 sequences fit in the KV cache at `max_model_len` 8192, and throughput plateaus there.
- Decode step at batch 1 / 13 / 32 / 48 / 58 is 21.0 / 26.2 / 33.3 / 41.1 / 47.1 ms.
- The in-kernel decode gate (`MUSE_GLIMMER_DECODE_FUSE=gate`, now the default) is not included in the table above. Measured separately on a trn2 device against the same config, it gave:
  - a further 4-16 % faster decode step at batch 13-58;
  - +5 to +16 % peak throughput on the text workloads;
  - +3.5 % peak throughput on images.
- **BF16:**
  - TPOT 27.2 ms at c=1;
  - peak output tok/s 184 (image) to 455 (128 / 512);
  - 16 sequences fit in the KV cache.
- **DFlash** (natural-language prompts, 256 output tokens, acceptance length about 4.1):

  | | c=1 tok/s | Speedup at c=1 | c=4 tok/s | c=8 tok/s |
  |---|---|---|---|---|
  | FP8 MLP | 132.9 | 2.76x | 291 | 355 |
  | BF16 | 108.1 | 2.93x | 263 | 355 |

  DFlash is a latency option. It runs with a smaller KV cap and helps up to about c=8.

## Accuracy

Greedy decoding was compared with Hugging Face transformers in BF16 on the same prompts.

| Config | 66-prompt set (text, image, video): all 16 tokens equal | Notes |
|---|---|---|
| BF16 | 66/66 (measured on vllm-neuron 0.24 / SDK 2.32; on 0.29 the BF16 long-form text checks are unchanged at 32/32) | prefill logit cosine >= 0.99997; text 32/32 tokens on 3 long-form prompts, including a 5,246-token prompt |
| FP8 MLP | 61-63/66 | every divergence is a near-tie (HF top-2 logit gap below the FP8/BF16 noise) |
| FP8 MLP + FP8 KV | 58-59/66 | every divergence is a near-tie |

The first generated token matched HF on every prompt in all configs. FP8 is close to BF16 but not identical. Run your own task-level eval before you rely on it.

## Architecture notes

**Text decoder:**
- 52 layers, hidden 6656, 32 query / 2 KV heads, head_dim 128.
- Sliding-window (2048) and global layers in a 3:1 pattern. RoPE applies only on sliding layers; global layers use no positional encoding.
- Weightless QK-norm with a fixed Q scale.
- Sigmoid output gate before o_proj.
- Centered sandwich RMSNorms, computed as `x * (1 + w)`.
- Logit softcap of 20.

**Perception encoder:**
- 1.8B-parameter ViT: 50 layers, width 1536, 16 heads.
- Windowed attention with full attention every fourth layer.
- 2D RoPE, then a 2x2 pixel shuffle and an adapter.

**TP layout:** at TP=4 the 2 KV heads are replicated to 4.

| File | Purpose |
|---|---|
| `config.py` | Text, vision and top-level configs, with load-time asserts on every model-specific field |
| `factory.py` | Registry class, vLLM model stubs and processor for the front-end |
| `processing.py` | Multimodal processor; image and video prompt expansion identical to HF `MuseGlimmerProcessor` |
| `model_bf16.py` | Text decoder, LM head + softcap, KV spec, weight loading, `embed_multimodal` |
| `vision_encoder_bf16.py` | Perception encoder, adapter, projection, encoder-cache write |
| `fp8.py` | Load-time FP8 ROW self-quantization (per-output-channel weight scales) |
| `kv_fp8.py` | FP8 KV cache: analytic K scale and per-layer V scales |
| `decode_fusions.py`, `gate_decode.py` | Decode fusion knobs and the gated decode attention call |
| `nki_attn_gate_tkg.py` | NKI kernel, see below |
| `dflash_model.py` | DFlash drafter for the assistant checkpoint |
| `weight_loaders.py`, `utils/vision_preprocessing.py` | Vision weight loaders; CPU-side pack order, window bounds and RoPE tables |

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `MUSE_GLIMMER_FP8` | unset | `mlp`: FP8 MLP weights (ROW, per-output-channel scales) quantized from the BF16 checkpoint at load. `mlp,attn` also quantizes the attention projections; it is not recommended (no gain, slightly slower decode). |
| `MUSE_GLIMMER_KV_SCALES` | unset | Per-layer V scale JSON, required with `--kv-cache-dtype fp8`. K uses an analytic scale: K is stored after the weightless QK-norm, so it is bounded by sqrt(head_dim). |
| `MUSE_GLIMMER_DECODE_FUSE` | `gate` | `gate`: sigmoid gate and o_proj inside the decode attention kernel. `off`: the gate runs in torch. `norm`, `ffn`, `early_gate`: other folds, measured slower or neutral. |
| `MUSE_GLIMMER_DECODE_FUSE_GATE_MIN_BS` | 16 | Smallest decode batch bucket that uses the in-kernel gate. It is about 3 % slower at batch 1. |
| `MUSE_GLIMMER_DECODE_OPT` | 3 | neuronx-cc `-O` level for the decode graph only. `-O3` is up to 12.6 % faster at batch >= 32. `off` keeps the engine level. |
| `MUSE_GLIMMER_KV_DEFER` | 2 | Decode KV writes run after the layer loop, one write per physical KV buffer. This avoids a per-step copy of the shared hybrid KV buffers. `0` restores per-layer writes. |
| `MUSE_GLIMMER_MLP_TKG_MAX_T` | 128 | FP8 MLP decode batches of 97 to this many rows use the TKG ROW kernel (bf16 activations, the same kernel as <= 96 rows) instead of the prefill CTE ROW kernel. The 128-row decode step is about 11 % faster. `96` restores the old path. |
| `MUSE_GLIMMER_FP8_CTE_CHUNK` | 512 | FP8 MLP prefill row chunk. It works around an nkilib 2.33 kernel-validation failure ("Stack out of memory") at > 512 rows. The output is bit-exact with the un-chunked kernel. |
| `MUSE_GLIMMER_FP8_CTE_SHARED_ABS` | 0 | `1`: patch the nkilib CTE ROW kernel at build time so that one scratch buffer is shared (see below). Prefill then runs un-chunked: about 8 % lower TTFT at 1k-4k tokens. Opt-in. |
| `MUSE_GLIMMER_VIS_SMALL_BLOCK` | 0 | `1`: single-block images run the smallest vision bucket that fits, instead of the full block. Needs `num_vision_tokens_buckets` below the block size. |
| `MUSE_GLIMMER_VIS_MLP` | torch | `nki`: nkilib MLP kernel with exact GELU and the fc1 bias, used while I/rank < 4096. |
| `MUSE_GLIMMER_VIS_LN` | torch | `fused`: LayerNorms fused into the QKV and MLP kernels. |
| `MUSE_GLIMMER_VIS_WIN_SPLIT`, `MUSE_GLIMMER_VIS_ROPE`, `MUSE_GLIMMER_VIS_OPT` | off | Exact but measured slower or neutral. Kept for experiments. |
| `MUSE_GLIMMER_DECODE_TORCH`, `MUSE_GLIMMER_COMPUTE_DTYPE`, `MUSE_GLIMMER_STRICT_CONFIG` | off | Debug and CPU-test knobs |
| `VLLM_NEURON_SWA_ALIGN_TOKENS` | 0 (set 4096) | Pads the trimmed sliding-window KV table to 4096 tokens, so the decode kernel moves 16-token KV DMA chunks instead of 2-token ones. +12-14 % throughput at batch 13. Must be below `max_model_len`. |
| `VLLM_NEURON_MLP_CTE_I_OVERRIDE` | 0 | Mirrors nkilib's I-sharding rule (I > 4096) in the plugin's MLP kernel gate |

**Config rules:**
- **KV cap.** The default `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (0.30) leaves no HBM for the compiled graphs. Use 0.27 with FP8 MLP and 0.15 with BF16.
- **Validate any new cap or bucket set with a sweep that runs every prefill, decode and vision bucket.** DMA rings are allocated the first time each graph runs, so a short test can pass while a full sweep fails with `status=4`.
- **Don't count batch slots that aren't admitted.** Concurrency above the admitted sequence count only queues requests. When you time a decode batch, make sure every sequence was admitted.
- **`decode_context_length_buckets` below 4096** is supported: sliding-window groups trim to the bucket. It is 5-14 % faster at batch 13-58 for short contexts, but it costs about 0.9 GiB per core. That does not fit at cap 0.27 / MNS 64.

## NKI kernels in this directory

- **`nki_attn_gate_tkg.py`** is a patched copy of nkilib `experimental/transformer/attention_block_tkg.py` from nkilib 0.7.0 (SDK 2.33).
  - It adds the model's output gate inside the decode kernel: `out = (attn * sigmoid(X @ W_gate)) @ W_out`.
  - The gate projection reads the same normalized input as QKV. It is issued before attention, so its weight DMA overlaps the KV traffic.
  - It keeps the attention output in SBUF through the output projection.
  - New keyword-only arguments: `W_gate`, `W_gate_scale` (FP8 ROW) and `gate_dq`.
- **`_install_cte_shared_abs`** (`model_bf16.py`) replaces one allocation in nkilib `core/mlp/mlp_cte/basic/mlp_cte_basic_quantization.py`:
  - The `row_quant_reduction_res` scratch is a discarded destination; only its row absmax is used. It is allocated once per 128-row subtile, which overflows the stack at 1024-row tiles with I/core = 2560.
  - Sharing one buffer is exact.

Both changes, plus three kernel issues found during this port, are proposed for nki-library on the branch [`jimburtoft/nki-library@contrib/muse-glimmer-2.33`](https://github.com/jimburtoft/nki-library/tree/contrib/muse-glimmer-2.33):
- FP8 ROW CTE MLP gives wrong output when the per-core intermediate size is not a multiple of 128. The model works around it by zero-padding I/rank 4992 to 5120.
- `attention_segmented_cte` returns wrong rows when the prior token count is not a multiple of the KV block size. The plugin aligns the chunk size.
- The MLP kernel drops the projection bias when it I-shards (I >= 4096). The model falls back to torch in that case.

## Plugin changes outside this directory

- **`vllm/platform.py`:** registers MuseGlimmer over upstream vLLM's GPU class in the front-end, the engine core and the workers.
- **`vllm/worker/neuron_model_runner.py`:**
  - The trimmed sliding-window block count is aligned for LNC=2 (`s_prior` sharding).
  - The `VLLM_NEURON_SWA_ALIGN_TOKENS` knob is added.
  - Sliding-window trimming is corrected for multi-token (speculative verify) decode steps.
  - SWA groups honour `decode_context_length_buckets`.
  - The decode graph gets a per-graph compiler level.
- **`vllm/core/scheduler.py`:** under speculative decoding, the chunked-prefill chunk size is rounded down to a multiple of the KV block size. `attention_segmented_cte` needs an aligned prefix.
- **`functional/mlp.py`, `functional/rmsnorm_quant.py`:** ROW FP8 torch fallback, the ROW packed-activation width, the I-shard rule, and `NO_NORM` quantization.
- **`functional/argmax.py`:** falls back to `torch.max` when a `cascaded_max` stage would exceed the `nc_find_index8` limit of 16,384 elements. This happens at 256 verify rows over a 50k vocab shard.
- **`vllm/spec_decode/dflash.py`, `model/qwen3/dflash_model.py`:** target-model hooks used by the MuseGlimmer drafter.

## Known limitations

- Tested only on `trn2.3xlarge` at TP=4, `max_model_len` 8192, with prefix caching off.
- Images and video are not tested together with DFlash.
- DFlash with the FP8 KV cache runs (drafter KV scales are analytic) but does not fit in HBM on trn2: FP8 KV needs the 8192-token prefill graph loaded, and with DFlash only a 0.05 KV cap loads, which fails at runtime at 8-16 concurrent streams. Use DFlash with a BF16 KV cache, prefill buckets 512/2048, cap 0.10, `max_num_seqs` 16, plus `VLLM_NEURON_SWA_REAL_ADMISSION=1 VLLM_NEURON_SWA_EARLY_FREE=1` (not the SWA ring). It beats the ring server up to about 8 concurrent streams per device.
- FP8 prefill graphs compile slowly. Raise `VLLM_NEURON_BARRIER_TIMEOUT`.
- Any change to the KV cap or `max_num_seqs` recompiles every graph.
- Windowed vision layers still compute full masked blocks. A tile-skip in nkilib `attention_cte` would remove that work.
