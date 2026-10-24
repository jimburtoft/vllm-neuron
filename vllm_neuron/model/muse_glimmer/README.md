# MuseGlimmer (vLLM-Neuron)

Port of `MuseGlimmerForConditionalGeneration` (Muse-Glimmer-30B: image + video +
text in, text out) to vLLM-Neuron on trn2. This branch targets vllm-neuron
0.29.0.1.2.0 (SDK 2.33); the 0.24 branch is `muse-glimmer`.

**0.29 rebase (2026-10-21, SDK 2.33 DLAMI 20261020, vllm 0.29.0, transformers 5.17.0):**
text 32/32 on 3 goldens (BF16; FP8 MLP 32/32, 32/32, 22/32 near-tie), image 32/32,
video_short 25/32 (exact HF tie at step 25, margin 0.000). Decode BS=1 / BS=13:
BF16 27.1 / 32.8 ms, FP8 MLP 20.5 / 27.1 ms. HBM after load unchanged
(14.21 / 9.92 GiB per core); validated caps unchanged.

**Status (2026-10-18, 0.24): text, image, video and mixed image+video verified on trn2.**
Setup: trn2.3xlarge, LNC=2, SDK 2.32 (DLAMI 20260818), vllm 0.24.0 + vllm-neuron
0.24.0.1.1.0 + this branch, text TP=4, BF16, against HF BF16 (transformers 5.15.0):
- prompt ids equal HF on all 10 golden prompts; prefill logit cos 0.99999-1.00000;
- text greedy 32/32 on 3 prompts, including 5246 tokens (past the 2048 window);
- image / video / mixed greedy 24-32/32, every divergence a near-tie (HF top-2 margin 0.19-0.31);
- vision DP=4 and vision TP=4 both verified.

Perf, same setup, `vllm serve`: text BS=1 TPOT 31.9 ms, TTFT 320 ms at 1k input;
216 output tok/s at concurrency 16; image (896x672) TTFT 479 ms with vision TP=4.

Files:

| File | Purpose |
|---|---|
| `config.py` | Text/vision/top-level configs, plus load-time asserts for every special field |
| `factory.py` | Registry class: `from_configs` for the worker, vLLM-model stubs and processor for the front-end |
| `processing.py` | Front-end vLLM multimodal processor (image and video prompt expansion identical to HF `MuseGlimmerProcessor`) |
| `model_bf16.py` | Text decoder, LM head + softcap, KV spec, weight loading, `embed_multimodal`, `get_encoder` + `build_encoder_synthetic_inputs` |
| `vision_encoder_bf16.py` | Perception encoder + adapter + projection + weightless RMSNorm, encoder-cache write, vision DP/TP |
| `weight_loaders.py` | 3-tensor (q/k/v) fused vision QKV loaders, transposed replicated loader |
| `utils/vision_preprocessing.py` | CPU: pack order, window/full bounds, merged-order gather, axial RoPE, bilinear taps (vendored HF math, transformers 5.15/5.20 agnostic) |

## Architecture (text)

| Parameter | Value (released checkpoint) |
|---|---|
| Layers / hidden / intermediate | 52 / 6656 / 19968 (SwiGLU, SiLU) |
| Q heads / KV heads / head_dim | 32 / 2 / 128 (q width 4096 differs from hidden) |
| Vocab | 202048, untied `lm_head` |
| Layer pattern | `[sliding, sliding, sliding, full] x 13` |
| Sliding layers | window 2048 (`kv > q - 2048`), RoPE theta 5e5 (rotate_half, full head_dim) |
| Full layers | full causal attention, **NoPE** |
| QK-norm | weightless RMSNorm (eps 1e-5) on Q and K, then Q x 3.87 |
| Softmax scale | 3.87 / sqrt(128) = 0.342 (the Q multiplier is folded in exactly, so bf16(3.87) = 3.875 never appears) |
| Output gate | `attn * sigmoid(x @ W_gate)` before `o_proj` (gate sharded like Q) |
| Norms | sandwich: input / post-attn / pre-FFN / post-FFN, all centered `x * rsqrt(ms+eps) * (1+w)` with fp32 `w` and fp32 `1+w`. eps is 1e-5 for the pre norms and 1e-8 for the post norms |
| Embedding | lookup, then weightless RMSNorm (`embed_norm`); vision rows are scattered in after the norm |
| Final norm | weighted RMSNorm, eps 1e-5 |
| Logits | `20 * tanh(logits * 0.19611613513818404 / 20)`, in fp32, applied before sampling and on the gathered-logits (logprobs) path |

## Architecture (perception encoder)

| Parameter | Value (released checkpoint) |
|---|---|
| Layers / width / heads / head_dim / MLP | 50 / 1536 / 16 / 96 / 8960 (exact erf GELU) |
| Patch embed | Linear 1176 -> 1536, no bias; patch flattened (temporal 2, RGB, 14, 14) |
| Position table | learned 32x32, bilinear resample, `align_corners=False`, zero padding (4 taps, CPU indices/weights) |
| Norms | `ln_pre`, `norm1`/`norm2` (eps 1e-5 hard-coded in HF), `ln_post`; all LayerNorm with bias |
| Attention | q/k/v/proj with bias, non-causal; 2D axial RoPE (theta 1e4, positions `(w+1, h+1)`, layout `[w,h,w,h]`, fp32 rotation) |
| Layer schedule | `full_attention` at 3,7,...,47 and 49; `window_attention` (32x32-patch windows) elsewhere |
| Merge | 2x2 pixel shuffle, **channel-interleaved** (`out[c*4+j] = tok_j[c]`) -> 6144 |
| Adapter / projection | fc1 6144->4096, GELU, fc2 4096->4096, GELU (no biases); projection 4096->6656 (no bias); weightless RMSNorm (eps 1e-5) |

Window handling: the CPU packs each frame window-major, then 2x2-block-major,
so every HF window is a contiguous run and window vs full layers differ only in
their `[bound_min, bound_max)` tensors (selected per layer by the static
`layer_types`). The pixel shuffle is then a reshape, and a per-block
`merged_order` gather restores the HF merged-token order before the cache
write. Bounds are a dynamic mask in `attention_cte` (no compute skip), so a
window layer costs as much as a full layer over the block (perf item).

Prompt format: the chat template renders one `<|patch|>` per image; the
processor expands it to `<|image_start|>` + `<|patch|>` x (t*h*w/4) +
`<|image_end|>`; only `<|patch|>` rows receive vision features. vLLM tokenizes
multimodal prompts with special tokens; the processor collapses the resulting
doubled BOS so prompt ids equal HF `apply_chat_template` (verified for
`LLM.chat` and raw-text prompts with and without the leading BOS).

## Differences from the reference models

- Derived from `qwen3_vl` (VLM plumbing and on-device encoder-cache merge), with gemma3-style sandwich norms and per-layer SWA, trinity-style NoPE and gate, and gpt_oss-style SWA decode.
- **No M-RoPE, no deepstack, no QK-norm gammas.** Text RoPE is plain 1D.
- Decode megakernel is called with `W_out=None`. The gate and `o_proj` run outside the kernel, and the KV scatter is manual (`update_cache=False`).
- Decode SWA start is `pos - window + 1` in the runner's trimmed block-table frame (`swa_kv_pos_offset`).
- KV heads 2 < TP 4, so KV replication x2 (`num_kv_replicas`): rank r holds KV head `r // 2`.

## Front-end registration

vLLM 0.24 has no native MuseGlimmer and resolves the architecture to
`TransformersMultiModalForCausalLM`. vLLM 0.29 ships its own
`MuseGlimmerForCausalLM` and maps `MuseGlimmerForConditionalGeneration` to it;
its processor emits different kwargs (`image_pixel_values`,
`*_feature_sizes`) and rejects `max_image_tokens`. In both cases the plugin
model would not be used. `register_plugin_model_architectures()`
(`vllm_neuron/vllm/platform.py`) overrides the entry with
`vllm_neuron.model.muse_glimmer.factory:MuseGlimmerForConditionalGeneration`.
It runs in every process that resolves the class: the front-end
(`pre_register_and_update`), the EngineCore (Neuron scheduler, which builds a
`MultiModalBudget`) and the workers (`_init_encoder_cache`). The factory carries
`@MULTIMODAL_REGISTRY.register_processor`, the vLLM model-interface stubs and
`SupportsEncoderBudget` (0.29: without it the prefill gather is sized to 0 blocks).

## Config asserts

`MuseGlimmerTextConfig.from_hf_config` fails if any of the following is missing:
`sliding_window`, `layer_types`, `layer_rope_theta`, `qk_scale_factor`,
`output_multiplier`, `final_logit_softcapping`, `post_norm_eps`, `rms_norm_eps`,
`rope_parameters`.

`validate()` always checks:
- full layers are exactly the NoPE layers;
- no per-layer theta differs from the global theta;
- default rope type, SiLU, no attention bias, untied embeddings.

On the released 52-layer / 6656 geometry (or with `MUSE_GLIMMER_STRICT_CONFIG=1`)
it also checks every released value: window 2048, 3.87, 1e-8, the multiplier,
softcap 20, theta 5e5, the head counts, the vocab, and the layer pattern.
Truncated test checkpoints use the structural checks only.

## Environment knobs

| Variable | Effect |
|---|---|
| `MUSE_GLIMMER_DECODE_TORCH=1` | Decode attention uses the plain-torch reference (`_torch_attention_decode_impl`) instead of the NKI megakernel. This is the escape hatch for kernel-suspect issues. |
| `MUSE_GLIMMER_FP8=mlp` (or `mlp,attn`) | FP8 weights self-quantized from the BF16 checkpoint at load time (`fp8.py`): nkilib `QuantizationType.ROW`, per-output-channel e4m3 weight scales (240 range), per-token dynamic activation quant on prefill, weight-only on decode. No calibration. `mlp` frees 4.3 GiB/core (14.21 -> 9.92 GiB used after load, server config with vision TP=4) and allows `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.27` / `max_num_seqs` 32 (29.3 admitted; 0.30 is text-only: an image stress run hit `nrt_tensor_allocate status=4`). `attn` adds qkv/gate/o_proj but is slower at BS=13 (Task 021); use `mlp`. FP8 prefill graphs compile about 2.5x slower: set `VLLM_NEURON_BARRIER_TIMEOUT=14400` on a cold cache |
| `MUSE_GLIMMER_FP8_CTE_CHUNK` | Default 512. FP8 MLP prefill (CTE ROW) is split into row chunks of this size: nkilib 2.33 fails kernel validation (`NCC_INKI016 Stack out of memory`) at T >= 1024 for H=6656 / I=5120 per rank. Bit-exact with the un-chunked 2.32 kernel (simulator). 0 disables |
| `MUSE_GLIMMER_DECODE_OPT` | Default 3. neuronx-cc `-O` level for the plain decode graph only (runner per-graph `compiler_args_for`); prefill, vision and spec-verify keep the engine level (`-O2` on the 0.29 native route). `-O3` turns off modular flow: decode step -10 to -13 % at BS >= 32 (FP8 MLP, BF16 or FP8 KV), unchanged at BS <= 16, same greedy tokens and logits as `-O2` (goldens, 66-prompt eq set, batch 48/58 token check; Task 029). `off` keeps the engine level for every graph. `1` is 2.2x slower at BS=1 on 0.29; do not use |
| Decode bucket layout (Task 029) | With FP8 KV at cap 0.27 / `max_num_seqs` 64 (58 admitted), use `num_seqs_buckets` `[1,2,4,8,16,32,48,64]`: without the 48 bucket, BS 33-48 pads to the 64-row graph (BS=48 53.2 ms vs 44.8). Do not make the top bucket the admitted count (58): `model_code` +1.2 GiB/core and the 6000/256 c=8 point fails with `status=4 Allocation Failure`. Check admitted sequences (`Maximum concurrency ... x` in the log) against `max_num_seqs`: requests above it run in a second wave, which looks like a slow top bucket |
| `MUSE_GLIMMER_STRICT_CONFIG=0/1` | Force-disable or force-enable the released-value asserts |
| Vision encoder knobs (Task 020) | All default off. Measured on trn2 (Kaizen, one device, text TP=4, vision TP=4, FP8 MLP), same-pod A/B. **Recommended for image serving:** `MUSE_GLIMMER_VIS_SMALL_BLOCK=1 MUSE_GLIMMER_VIS_MLP=nki MUSE_GLIMMER_VIS_LN=fused` with `num_vision_tokens_buckets [1024,2048,3072,4096]` (block 4096). The 896x672 image TTFT at c=1 goes from 395 to 339 ms, image tok/s goes up 10 % at c=16 and 13 % at c=64, and greedy tokens are identical on the goldens and the 23 eq image/video prompts |
| `MUSE_GLIMMER_VIS_SMALL_BLOCK=1` | When a single-block request fits a bucket smaller than the block size, it runs ONE block of `bucket` rows ("block = image size"), for example 3072 for 896x672 and 1024 for 448x448. Only acts if such buckets are configured. Vision DP must be 1 (vision TP). VE device time: 896x672 123 -> 92 ms, 448 124 -> 35 ms. Multi-block requests (video) keep the block size |
| `MUSE_GLIMMER_VIS_MLP=nki\|fp32gelu` | `nki`: nkilib MLP kernel, `skip_gate`, exact GELU (`nl.gelu`), with the fc1 bias. If I/rank >= 4096 (vision TP <= 2), it falls back to torch, because nkilib forces I-sharding there and drops the bias (simulator cos 0.90; this is the qwen3-asr "cos 0.19" failure). `fp32gelu`: no gain, so it is not recommended |
| `MUSE_GLIMMER_VIS_LN=fused` | Fuses norm1 into `NF.qkv_proj` (LAYER_NORM + bias) and, with `MLP=nki`, norm2 into the MLP kernel. Together with `MLP=nki` it saves about 15 ms per 3072-row encode |
| `MUSE_GLIMMER_VIS_WIN_SPLIT=1` | Window layers attend in 1024-row chunks: windows are FFD-packed on the CPU, then gathered/scattered in the graph. It is exact (fp32 cos 1.0, simulator OK), but **slower** on trn2 (+6 to +13 ms): the gathers cost more than the masked tiles saved. Not recommended |
| `MUSE_GLIMMER_VIS_ROPE=bf16`, `MUSE_GLIMMER_VIS_OPT=<n>` | RoPE rotation in bf16: no gain. Encoder graph `-O3`: neutral at 1024/3072 rows, +72 ms at 4096 rows. Neither is recommended |
| `MUSE_GLIMMER_VIS_TIMING=1` (or `MUSE_GLIMMER_VIS_TIMING_FILE` + `<file>.on`) | Per-VE-call CPU / H2D / device split. It adds syncs and a cache-row readback, because `torch.neuron.synchronize()` returns before the encoder graph finishes. Use it for diagnostics only |
| `MUSE_GLIMMER_COMPUTE_DTYPE=float32` | CPU-mode test knob: fp32 compute, for an exact structural comparison against HF fp32 (use with `LLM(dtype="float32")`) |

## Feature status

"trn2" = verified on trn2.3xlarge against HF BF16 (2026-10-17/18). "CPU" = verified only
in `VLLM_NEURON_CPU_MODE=1` on a truncated checkpoint against HF fp32.

| Feature | Status | Notes |
|---|---|---|
| TP (text) | trn2 (TP=4) | TP=4 uses KV replication x2. TP=1 CPU only (does not fit HBM for the 30B) |
| SP (prefill sequence parallel) | trn2 | Embedding reduce-scatter; all-gather/reduce-scatter around attention and MLP |
| Single-shot prefill (`flash_attention`) | trn2 | sliding_window per layer |
| Chunked/segmented prefill (`segmented_attention`) | CPU | 512-token segments with SWA, fp32 exact |
| Decode megakernel (`attention_decode`) | trn2 | NoPE via `cos=None`, SWA via `swa_start_pos_ids`, weightless QK-norm |
| Hybrid KV cache (SWA + full groups) | trn2 | `LayerSpec.sliding_window_size` per layer |
| On-device sampling / logprobs gather | trn2 | Softcap applied before both |
| `mm_language_model_only` (PD pool) | trn2 | Text-only serving without building the vision tower |
| Images | trn2 | 1-window and multi-window; multi-image per request CPU only |
| Vision DP (default) / vision TP | trn2 | DP=4 and `encoder_neuron_config.tp_size=4` at text TP=4 |
| Video | trn2 (16 frames) | One ViT item per temporal group. 96 frames: CPU only |
| Mixed image + video | trn2 | |
| `vllm serve` (text, image) | trn2 | `harness/perf_serve.sh` in the project |
| `mm_encoder_only` (VE pool) | No | Raises |
| Prefix caching (APC) | Untested | Run with `enable_prefix_caching=False` |
| DP / EP / attention-DP / DCP | N/A | Dense model, single node |
| DFlash speculative decoding | trn2 (text, mml 4096 and 8192) | `speculative_config={"method": "dflash", "model": <Muse-Glimmer-30B-assistant>, "num_speculative_tokens": 15}`, greedy only, MNS <= 16. Drafter `dflash_model.py` (`DFlashMuseGlimmerAssistantModel`, inclusive bidirectional SWA 2048). Target aux taps via `SupportsEagle3`. Acceptance length 4.1, c=1 2.7-3.0x on the 43 natural prompts. KV cap 0.07 (BF16) / 0.12 (FP8 MLP) at mml 4096. **mml 8192**: prefill buckets `512,2048` (a 4096 bucket fails to load the 8-seq decode NEFF), cap 0.08 (BF16, c=8 fits) / 0.10 (FP8 MLP); greedy spec vs no-spec divergences are all target near-ties (both tokens in the top 2). Needs the scheduler's block-aligned chunk budget (Task 027): without it the chunk boundary at `mnbt - 15` gives the segmented prefill kernel an unaligned prior and wrong attention. Images untested |
| Eagle3 | Untested | The target hooks are shared with DFlash; no Muse Glimmer Eagle3 head exists |
| FP8 KV cache, quantized weights | No | Only `quantization in (None, "bf16")` |
| Prefill MLP NKI kernel | No (torch fallback) | At TP=4, I/rank 4992 needs 10 PSUM banks and H=6656 < 7168 blocks I-sharding. Perf only |

## Required and recommended config (trn2.3xlarge, TP=4)

| Knob | Value | Why |
|---|---|---|
| `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (env) | `0.12` | The default 0.30 (266k KV tokens) leaves no HBM for the NEFFs: `Allocation Failure status=4` at decode warmup. 0.12 gives 106,543 tokens (13x at 8192) |
| `NEURON_SKIP_EFA_AFFINITY` (env) | `1` | trn2.3xlarge topology |
| `--max-model-len` | 8192 (tested) | 96-frame video needs >= 8192 |
| `neuron_config.decode_context_length_buckets` | unset, or only values > 2560 (e.g. `[4096]`) | A bucket below the trimmed SWA window (80 blocks = 2560 tokens) is warmed with the wrong SWA shape and the first concurrent decode fails `fail_on_recompile` |
| `--no-enable-prefix-caching` | set | APC untested |
| `encoder_neuron_config.tp_size` | `4` under concurrency | Saves 2.6 GiB/core vs DP and is 26 % faster at image BS=1. DP=4 with a 32-block encoder cache OOMed HBM at concurrency 4 |
| `encoder_neuron_config.encoder_attention_block_size` | 4096 (tested); 16384 for HF-default max images | Must be >= patches of the largest image / temporal group |
| `encoder_neuron_config.num_encoder_input_buckets` | `[4096]` images, `[4096, 8192]` short video | 96-frame video: 28672 at block 4096 (untested on trn2) |
| `encoder_neuron_config.encoder_cache_num_blocks` | 8 (single request), 16 (concurrency 4) | The derived default can be too small; each item takes whole blocks |
| `mm_processor_kwargs.max_image_tokens` | `vision_block // 4` | Caps images to the compiled block |

## Known limits

- **Cold compile:** about 12 min for text at `max_num_seqs` 1, about 24 min at 16, plus about 5 min for the vision buckets. A warm restart takes about 9 min.
- **Not run on trn2:** 96-frame video and the largest image bucket (block 16384).
- **Window layers pay full-block attention.** Bounds are a mask with no compute skip.
- The gate matmul runs outside the QKV kernel (an extra `[T, H] x [H, 1024]` per layer).
- The bf16 encoder cache rounds vision features. This explains the residual cos gap vs HF fp32; it is the same rounding as the runner's other VLMs.
- `LLM.chat` with a video content part needs a URL or data URI (vLLM's chat parser). Offline use goes through `prompt` + `multi_modal_data`.
- Changes outside this directory: the architecture registration (platform, scheduler, runner `_init_encoder_cache`), the SWA block alignment to 2 x 128 for LNC=2 plus `VLLM_NEURON_SWA_ALIGN_TOKENS` (`_compute_swa_num_blocks`), and in `functional/`: the opt-in MLP CTE I-shard override, the ROW-packed `[T, H+4]` MLP gate + ROW torch fallback, and `rmsnorm_quant(norm_type=NO_NORM)`. The o_proj ROW layout fix of the 0.24 branch is upstream in 0.29.
- Legacy option names (`vision_neuron_config`, `num_vision_tokens_buckets`, `vision_attention_block_size`) still work on 0.29 with a deprecation warning; use `encoder_neuron_config` / `num_encoder_input_buckets` / `encoder_attention_block_size`.

## Running

```bash
# trn2.3xlarge, TP=4, LNC=2 (text only)
NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.12 python examples/vllm_neuron/models/muse_glimmer/run.py \
    --model /home/ubuntu/models/Muse-Glimmer-30B --tp 4 --max-model-len 8192
# one image (vision DP; add --vision-tp 4 for vision TP). --vision-block = max patches per image
NEURON_SKIP_EFA_AFFINITY=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.12 python examples/vllm_neuron/models/muse_glimmer/run.py \
    --model /home/ubuntu/models/Muse-Glimmer-30B --tp 4 --max-model-len 8192 \
    --image cat.jpg --vision-block 4096
```

Video prompts: the chat template renders one `<|video|>`; the processor
expands it to `<|vid_start|>Time: 0.0s<|video|>*n<|vid_frame_separator|>Time:
1.0s ... <|vid_end|>` (one timestamp per temporal group of 2 frames, HF
`replace_video_token`); only `<|video|>` rows get vision features. Pass
`multi_modal_data={"video": (frames_uint8[T,H,W,3], {"fps": src_fps,
"total_num_frames": T})}`: the HF processor samples it at `fps` (default 2) up
to `num_frames` (default 96). Frames already sampled by vLLM's video loader
(`do_sample_frames=False`, original `frames_indices`) keep source-time
timestamps. A temporal group (`h*w` <= 576 patches at the default
`max_video_frame_tokens=144`) must fit in one vision block; a block holds
`block // (h*w)` groups, so a 96-frame 16:9 video (48 groups) needs 7 blocks at
block 4096 (bucket 28672) or 2 blocks at block 16384 (bucket 32768).

Vision config notes: `encoder_attention_block_size` must be >= the patch count
of the largest image (`max_image_tokens * 4`; the HF default 4096 merged tokens
needs 16384). Pass `mm_processor_kwargs={"max_image_tokens": block // 4}` to
cap images to the compiled block. Set `encoder_cache_num_blocks` explicitly
(each image occupies whole cache blocks).

CPU-mode tests on a truncated random checkpoint live in the project harness
(`working/muse_glimmer/harness/`: `build_tiny_glimmer.py`,
`run_tiny_glimmer.py`, `compare_tiny_glimmer.py`, `run_t003_matrix.sh`; image path:
`build_tiny_glimmer_vision.py`, `t005_module_test.py`, `t005_e2e.py`, `run_t005_matrix.sh`; video:
`build_tiny_glimmer_video_goldens.py`, `t007_processor_test.py`, `t007_module_test.py`, `t007_e2e.py`,
`run_t007_matrix.sh`).
