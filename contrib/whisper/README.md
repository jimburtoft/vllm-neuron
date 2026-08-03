<!-- SPDX-License-Identifier: Apache-2.0 -->
# Whisper large-v3 on the vllm-neuron NATIVE XLA backend

A **clean-room, OpenAI-aligned Whisper large-v3** implementation for the
vllm-neuron **native** backend (`torch.compile(backend="vllm_neuron")`) on AWS
Trainium2. This is a **reference implementation** (voxtral-style contrib), not a
plugin-integrated serving path — see [Status / limitations](#status--limitations).

- **Target container**: `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`
  (vllm-neuron `0.21.0.1.0.0`, NKI `0.5.0`, neuronx-cc `2.26.6360`, SDK 2.31)
- **Hardware**: `trn2.3xlarge`, TP=4, LNC=2, bf16
- **License**: Apache-2.0. Clean-room — no NxDI (`neuronx-distributed-inference`)
  modeling code was copied; NxDI's `modeling_whisper` was referenced for design only.

---

## What this is

`whisper_neuron.py` is a self-contained set of `nn.Module`s (encoder + decoder)
built from the plugin's `ColumnParallelLinear` / `RowParallelLinear` plus plain
`torch` attention math. It is driven by a **direct model-forward harness**: the
model is instantiated, moved to `neuron:0`, and each of its four sub-graphs is
compiled with `torch.compile(module, backend="vllm_neuron", fullgraph=True)`. The
plugin's registered Dynamo backend lowers each FX graph → HLO → NEFF and returns a
device-executable callable. No vLLM `LLM`/`EngineCore`/scheduler is involved.

The four compiled graphs are:

1. **encoder** — mel `[1, 128, 3000]` → encoder hidden `[1, 1500, 1280]`
2. **`precompute_cross_kv`** — project the encoder output through every decoder
   layer's cross-attention K/V **once** into persistent HBM buffers
3. **`prefill`** — causal self-attention over the 4-token SOT prompt; populates the
   self-KV cache slots `[0:4)`
4. **`decode_step`** — the steady-state `[1,1]` graph: appends one token, reads the
   cross-KV + self-KV caches, writes the new token's self-KV in place

## Why it's notable

- **First Whisper (an encoder-decoder model with real cross-attention) on the
  native XLA backend.** Prior native attempts stalled on the cross-attention KV
  cache and on per-kernel dispatch cost.
- **The whole decode step fuses to ONE XLA NEFF.** The plain-torch attention math
  lowers entirely to neuronx-cc HLO ops and the plugin fuses the full step into a
  single NEFF, so there is **no per-kernel dispatch tax** (host dispatch measured at
  ~0.06 ms/step, 0.7% of the step). This is *why* it is fast despite using **zero
  custom NKI kernels**. On the older SDK 2.30 / vLLM 0.19 stack the same model
  dispatched ~167 NKI kernels/step through Dynamo FX; the 0.21 + NKI 0.5.0 stack
  eliminates that.
- **Byte-identical greedy to OpenAI-whisper large-v3** (5/5 clip correctness gate,
  TP=1 and TP=4).

## Results

Single 30-s clip, whisper-large-v3, greedy (decode-to-EOS), warm:

| Path | Instance | Config | 30-s latency |
|------|----------|--------|-------------|
| **this (native XLA)** | trn2.3xlarge | **TP=4 LNC=2 bf16** | **611.7 ms** |
| stock NxDI | trn2.3xlarge | TP=4 LNC=2 | 785 ms (this is **22% faster**) |
| PyTorch-Native | trn2.3xlarge | TP=4 | 540.6 ms |
| A100 CT2 reference | 1× A100 | bf16 | ~573–600 ms (~parity) |

Per-file latency scales with decoded-token count (decode is ~94% of the wall), so
shorter clips are proportionally faster; the 30-s clip above is the worst case within
Whisper's native window.

**TP=4 is the architectural maximum** for whisper-large-v3: it has 20 attention
heads, and `20 % 8 != 0`, so TP=8 is not expressible (`assert num_heads % tp == 0`).
Valid TP ∩ power-of-2 on trn2 = {1, 2, 4}.

Decode dominates the wall (~94%): it is textbook M=1 autoregressive
memory-bandwidth-bound weight streaming (~1.6 GB/step re-read of the decoder
weights). Encoder + cross-KV precompute + prefill together are ~5%.

## Design

- **Cross-attention KV cache.** Whisper cross-attention attends to the fixed encoder
  output, so K/V are a function of the encoder output only. They are projected
  **once** at prefill (`precompute_cross_kv`) into persistent
  `register_buffer(persistent=False)` HBM buffers `[1, n_kv_heads, 1500, 64]` per
  layer and read (Q-only) every decode step — no per-step reprojection.
- **Self-attention KV cache.** A growing `register_buffer` `[1, n_kv_heads, 448, 64]`
  written per step via `index_copy_` at a **0-D device-tensor** position, so the
  decode-step graph compiles **once** at shape `[1,1]` and is reused for every
  position (no per-length recompile). A `-inf` additive mask forbids positions
  `> cur_pos`, so stale cache slots are never attended.
- **HBM persistence without `input_output_aliases`.** `input_output_aliases` is
  XLA-only and **unavailable** on the `torch.compile(backend=...)` path. Instead the
  `register_buffer` + in-place `index_copy_`/`.copy_()` pattern is auto-recognized by
  the plugin's `aliasing_output_rewrite` FX pass, which builds a 64-entry `io_map`
  (32 layers × 2 self-KV buffers) aliasing each in-place write back to its HBM
  allocation — the native-backend equivalent of NxDI aliasing.
- **Sharded LM head + on-device greedy argmax.** The LM head (tied to
  `embed_tokens`) is a vocab-parallel `ColumnParallelLinear`; each rank computes only
  its vocab slice, then a tiny `(max, global_index)` all-gather picks the global
  greedy token on device — avoiding both the unsharded 132.8 MB/step LM-head re-read
  and a full-vocab CPU gather. `vocab_size=51866` is padded up to a multiple of TP;
  padded columns are masked to `-inf` so they can never win. This is byte-identical
  to `torch.argmax` over the full vocab, including lowest-index tie-breaking.
- **TP=4 on a single chip.** A bare (non-serving-stack) harness needs the
  `neuron_worker.py` 1-chip-per-rank handling (map 1 visible device → 4 ranks;
  per-rank `NEURON_RT_VISIBLE_CORES`) plus an explicit `rendezvous_ccom_bootstrap()`
  call after the gloo init — otherwise the RowParallel all-reduce NEFF fails with
  `failed to init NCCL comm`.

## Footguns documented

These were hit and resolved while bringing the model up on the native backend:

1. **`F.gelu` graph-breaks under `fullgraph=True`** (`torch._C._nn.gelu` is a skipped
   C builtin intercepted by `libtorch_neuronx_lite`). Fixed with an explicit
   erf-GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
2. **`RowParallelLinear` rejects a non-fp32 bias at `tp_size > 1`** (XLA
   constant-inlining limitation). Fixed with a `RowParallelLinearFP32Bias` wrapper:
   `RowParallelLinear(bias=False)` + a separate fp32 bias added after the all-reduce
   (applies to `out_proj` and `mlp.fc2`).
3. **The SDK-2.31 encoder-LayerNorm `[NCC_ISAU902]` rejection does NOT fire on this
   backend.** It is NxDI-specific (that path injects `--verify-hlo=true` and lowers
   LayerNorm to a rejected `batch-norm-training` op). The native backend lowers
   `nn.LayerNorm` through torch_xla → HLO with a different decomposition and does not
   set `--verify-hlo=true`, so no `--expand-batch-norm-training` fix is needed. (The
   knob is available via
   `torch.compile(..., options={"compiler_args": "--internal-hlo2tensorizer-options=--expand-batch-norm-training"})`
   if ever required.)
4. **`.float()` directly on a bf16 neuron tensor raises
   `Expected self.dtype()==dst.dtype()`.** Materialize to CPU first
   (`.to("cpu").float()`) for the final greedy argmax / EOS check.

## Status / limitations

This is a **direct model-forward harness** (standalone `nn.Module`s +
`torch.compile`), **NOT** integrated into the plugin's model registry or the vLLM
serving stack. The native plugin has **no encoder-decoder scheduler and no
cross-attention KV-cache manager**, so the full vLLM serving path cannot drive
Whisper today. This reference exists to prove out the model architecture, the
cross-KV `register_buffer` / auto-aliasing pattern, and the latency characteristics.

Full plugin integration (a `model/whisper/{factory,model_bf16,weight_loaders}` +
`registry.py` wiring, plus an encoder-decoder scheduler) is planned as a **follow-up
upstream PR** on a separate branch.

## Layout

```
contrib/whisper/
  README.md                 <- this file
  whisper_neuron.py         <- clean-room OpenAI-aligned Whisper large-v3 model
  weight_loaders.py         <- HF safetensors -> module state-dict remap
  harnesses/
    harness_encoder.py          <- encoder correctness (TP=1)
    harness_encoder_tp4.py      <- encoder correctness (TP=4 LNC=2)
    harness_kv_decode.py        <- KV-cache decode + 5-clip correctness gate (TP=1)
    harness_kv_decode_tp4.py    <- KV-cache decode end-to-end (TP=4 LNC=2)
    harness_opt.py              <- sharded-LM-head greedy path: gate + 30-s bench (TP=4)
    harness_bench_tp4.py        <- 30-s latency bench + encoder/prefill/decode split (TP=4)
  reference/
    ref_openai_whisper.py       <- OpenAI-whisper CPU reference (single clip)
    ref_all_clips.py            <- OpenAI-whisper greedy references for a clip set
    validate_kv_cpu.py          <- CPU-eager KV-cache vs naive-recompute equivalence
    extract_libri.py            <- LibriSpeech dev-clean clip extraction (public data)
  experiments/
    nki_decode_mlp.py           <- NKI 0.5.0 fused decoder-MLP spike (DMA investigation)
    test_nki_mlp.py             <- correctness test for the spike (nki.simulate / device)
```

## Usage

Run inside the target container on a `trn2.3xlarge`. Set `MODEL_DIR` to a local
`openai/whisper-large-v3` snapshot directory (must contain `config.json` and the
safetensors weights). The harnesses read a reference directory (default
`/large/work/ref`) built by the `reference/` tooling — edit the `REF` constant or
mirror that layout.

**1. Build the OpenAI-whisper references** (CPU, in a venv with `openai-whisper`):

```bash
python reference/extract_libri.py        # extract LibriSpeech dev-clean clips
python reference/ref_all_clips.py        # write mel npy + greedy token IDs per clip
```

**2. CPU-eager algebraic check** (no Neuron device needed — validates the KV-cache
math against naive recompute):

```bash
python reference/validate_kv_cpu.py <MODEL_DIR>
```

**3. Encoder correctness on device**:

```bash
# TP=1
python harnesses/harness_encoder.py <MODEL_DIR>
# TP=4 LNC=2
torchrun --nproc_per_node=4 harnesses/harness_encoder_tp4.py <MODEL_DIR>
```

**4. KV-cache decode + correctness gate** (5 clips, byte-identical to OpenAI-whisper):

```bash
# TP=1 gate
python harnesses/harness_kv_decode.py <MODEL_DIR>
# TP=4 LNC=2 end-to-end
torchrun --nproc_per_node=4 harnesses/harness_kv_decode_tp4.py <MODEL_DIR>
```

**5. Optimized path (sharded LM head) — gate + 30-s latency bench**:

```bash
# correctness gate (5 clips) at TP=1
WX_MODE=gate torchrun --nproc_per_node=4 harnesses/harness_opt.py <MODEL_DIR>
# 30-s warm latency bench at TP=4
WX_MODE=bench WX_CLIP=libri_3 torchrun --nproc_per_node=4 harnesses/harness_opt.py <MODEL_DIR>
# phase-split latency bench
torchrun --nproc_per_node=4 harnesses/harness_bench_tp4.py <MODEL_DIR>
```

The correctness gate passes when all clips produce transcripts byte-identical to the
OpenAI-whisper large-v3 greedy reference (`CORRECTNESS_GATE 5/5 byte-identical`).

## experiments/

`experiments/nki_decode_mlp.py` is a feasibility spike: a fused NKI 0.5.0
decoder-MLP kernel that loads the `fc1`/`fc2` weight matrices as large contiguous
SBUF blocks, built to test whether contiguous loading raises HBM memory-bandwidth
utilization above the ~30% baseline of the plain-torch XLA lowering. It is
correctness-tested (`test_nki_mlp.py`, cos-sim 0.99999779 vs plain torch) but was a
**NO-GO** for latency: contiguous loading did not produce the large HBM bursts
needed to move the memory wall — the decode is bound by per-sublayer weight-DMA
barriers, not by fragment size. Kept here to document the DMA investigation; it is
**not** part of the model path.
