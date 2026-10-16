# MuseGlimmer (vLLM-Neuron)

Port of `MuseGlimmerForConditionalGeneration` (Muse-Glimmer-30B: image + video +
text in, text out) to vLLM-Neuron 0.24 on trn2. **Status: text decoder only.**
The perception encoder and multimodal inputs are the next step; until then
`embed_multimodal` raises `NotImplementedError`.

Files:

| File | Purpose |
|---|---|
| `config.py` | Text/vision/top-level configs, plus load-time asserts for every special field |
| `factory.py` | Registry class: `from_configs` for the worker, vLLM-model stubs and processor for the front-end |
| `processing.py` | Front-end multimodal processing info and limits (image replacement only matters once the vision tower exists) |
| `model_bf16.py` | Text decoder, LM head + softcap, KV spec, weight loading |

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

## Differences from the reference models

- Derived from `qwen3_vl` (VLM plumbing and on-device encoder-cache merge), with gemma3-style sandwich norms and per-layer SWA, trinity-style NoPE and gate, and gpt_oss-style SWA decode.
- **No M-RoPE, no deepstack, no QK-norm gammas.** Text RoPE is plain 1D.
- Decode megakernel is called with `W_out=None`. The gate and `o_proj` run outside the kernel, and the KV scatter is manual (`update_cache=False`).
- Decode SWA start is `pos - window + 1` in the runner's trimmed block-table frame (`swa_kv_pos_offset`).
- KV heads 2 < TP 4, so KV replication x2 (`num_kv_replicas`): rank r holds KV head `r // 2`.

## Front-end registration

vLLM 0.24 has no native MuseGlimmer. Without help, `ModelConfig` resolves the
architecture to `TransformersMultiModalForCausalLM` and the plugin model is
never used, even for text. `NeuronPlatform.pre_register_and_update`
(`vllm_neuron/vllm/platform.py`) therefore registers
`vllm_neuron.model.muse_glimmer.factory:MuseGlimmerForConditionalGeneration` in
the front-end. The factory carries `@MULTIMODAL_REGISTRY.register_processor`
and the vLLM model-interface stubs.

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
| `MUSE_GLIMMER_STRICT_CONFIG=0/1` | Force-disable or force-enable the released-value asserts |
| `MUSE_GLIMMER_COMPUTE_DTYPE=float32` | CPU-mode test knob: fp32 compute, for an exact structural comparison against HF fp32 (use with `LLM(dtype="float32")`) |

## Feature status

| Feature | Status | Notes |
|---|---|---|
| TP (text) | Yes | TP=1 and TP=4 verified in CPU mode. TP=4 uses KV replication x2 |
| SP (prefill sequence parallel) | Yes | Embedding reduce-scatter; all-gather/reduce-scatter around attention and MLP |
| Single-shot prefill (`flash_attention`) | Yes | sliding_window per layer |
| Chunked/segmented prefill (`segmented_attention`) | Yes (CPU) | Verified at 512-token segments with SWA |
| Decode megakernel (`attention_decode`) | Yes (CPU and NKI simulator) | NoPE via `cos=None`, SWA via `swa_start_pos_ids`, weightless QK-norm |
| Hybrid KV cache (SWA + full groups) | Yes | `LayerSpec.sliding_window_size` per layer |
| On-device sampling / logprobs gather | Yes | Softcap applied before both |
| `mm_language_model_only` (PD pool) | Yes | Text-only runs with and without it |
| Vision encoder / images | No | Next step: `embed_multimodal` raises |
| Video | No | Next step after images |
| `mm_encoder_only` (VE pool) | No | Raises (needs the vision tower) |
| Prefix caching (APC) | Untested | Run with `enable_prefix_caching=False` |
| DP / EP / attention-DP / DCP | N/A | Dense model, single node |
| Eagle3 / speculative decoding | No | Not wired |
| FP8 KV cache, quantized weights | No | Only `quantization in (None, "bf16")` |
| Prefill MLP NKI kernel | Partial | At TP=4, I/rank 4992 needs 10 PSUM banks and H=6656 < 7168 blocks I-sharding, so prefill MLP falls back to torch on trn2. Correctness is unaffected; this is a perf item |

## Running

```bash
# trn2.3xlarge, TP=4, LNC=2 (text only)
NEURON_SKIP_EFA_AFFINITY=1 python examples/vllm_neuron/models/muse_glimmer/run.py \
    --model /home/ubuntu/models/Muse-Glimmer-30B --tp 4 --max-model-len 8192
```

CPU-mode tests on a truncated random checkpoint live in the project harness
(`working/muse_glimmer/harness/`: `build_tiny_glimmer.py`,
`run_tiny_glimmer.py`, `compare_tiny_glimmer.py`, `run_t003_matrix.sh`).
