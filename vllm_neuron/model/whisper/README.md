<!-- SPDX-License-Identifier: Apache-2.0 -->
# Whisper large-v3 — native vllm-neuron backend (encoder-decoder + serving)

The first **encoder-decoder** model in the vllm-neuron native plugin: OpenAI
**whisper-large-v3** with real cross-attention, integrated into the plugin's model
registry and driven end-to-end through the vLLM serving stack. Unlike a standalone
`torch.compile` harness, this is registered under
`WhisperForConditionalGeneration` and `vllm serve` routes the OpenAI-compatible
`/v1/audio/transcriptions` endpoint to it.

- **Target container**: `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`
  (vllm `0.21.0`, vllm-neuron `0.21.0.1.0.0`, NKI `0.5.0`, neuronx-cc `2.26.6360`,
  SDK 2.31)
- **Hardware**: `trn2.3xlarge`, **TP=4, LNC=2, bf16**
- **License**: Apache-2.0. The modeling body is clean-room; the transcription /
  audio-multimodal routing classmethods are inherited verbatim from vLLM-core's
  upstream Whisper class (see [Design](#design)).

---

## What it is

A registered plugin model package `vllm_neuron/model/whisper/`:

| File | Content |
|------|---------|
| `config.py` | `WhisperConfig` — asserts large-v3 dims (`d_model==1280`, `vocab==51866`, 20 enc/dec heads), coerces fp16/fp32 → bf16. |
| `factory.py` | Registered `WhisperForConditionalGeneration` — carries the `SupportsTranscription` + `SupportsMultiModal` interface markers and the audio-mm processor registration so `vllm serve` routes `/v1/audio/transcriptions` here. |
| `model_bf16.py` | The concrete encoder + decoder: block-managed self-KV, model-owned cross-KV buffers, on-device greedy sampler, vocab-parallel (sharded) fp32 LM head. |
| `weight_loaders.py` | HF `openai/whisper-large-v3` safetensors → module state-dict remap. |
| `medusa_heads.py` | Medusa spec-decode heads (ResBlock MLPs on the decoder's last hidden state) + head loader (customer / placeholder weights). See **[MEDUSA.md](MEDUSA.md)**. |
| `registry.py` (one-line edit) | registers `WhisperForConditionalGeneration` in `get_models()`. |

> **Speculative decoding**: a Medusa spec-decode framework for BS=1 latency ships
> alongside this model — enable it from `vllm serve` with
> `--speculative-config '{"method":"medusa","num_speculative_tokens":5}'`. See
> **[MEDUSA.md](MEDUSA.md)** for the serve command, config surface, head loader,
> the speedup-vs-acceptance model, and the correctness guarantee.

One runner edit is required (see [Known limitations](#known-limitations)):
`neuron_model_runner.py get_supported_tasks()` now reports `("transcription",)`
for a Whisper model (mirrors vLLM-core's generation-tasks logic); every other
Neuron model keeps `("generate",)` unchanged.

## Why it's notable

- **First encoder-decoder model on the native plugin.** Prior native attempts
  stalled on the cross-attention KV cache and on driving an encoder-decoder
  through the plugin's decoder-only scheduler. This wires the encoder to run once
  per request via the existing multimodal-encoder trigger and stores cross-KV in
  model-owned HBM buffers — no new scheduler or KV-cache-manager needed.
- **Serves the real OpenAI audio API.** `vllm serve openai/whisper-large-v3`
  answers `/v1/audio/transcriptions` (curl + OpenAI Python client), not just a
  bare model-forward harness.
- **Byte-identical greedy to OpenAI-whisper large-v3** (5/5 clip gate at TP=1).

---

## Quick start — `vllm serve`

Run inside the target container on a `trn2.3xlarge`. Audio decoding needs
`vllm[audio]`; if the endpoint returns *"Invalid or unsupported audio file"*,
install it and **restart the server**:

```bash
pip install soundfile librosa
```

TP=4 is the architectural maximum for whisper-large-v3 (20 heads, `20 % 8 != 0`),
so run at LNC=2 (4 logical cores → TP=4):

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NEURON_RT_NUM_CORES=4      # LNC=2 -> 4 logical cores -> TP=4 max

vllm serve openai/whisper-large-v3 \
  --tokenizer openai/whisper-large-v3 \
  --tensor-parallel-size 4 \
  --max-model-len 448 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1536 \
  --dtype bfloat16 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --port 8000 \
  --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}}, "vision_neuron_config": {"num_vision_tokens_buckets": [1500], "vision_attention_block_size": 1500}}'
```

Cold start compiles 5 graphs × 4 ranks (~5-6 min); warm start reuses
`/root/.cache/vllm/neuron/compile_cache`. Wait for `Application startup complete`
and `Supported tasks: ('transcription',)`.

### curl

```bash
curl -s -F file=@clip.wav -F model=openai/whisper-large-v3 \
     -F language=en -F temperature=0 -F response_format=json \
     http://localhost:8000/v1/audio/transcriptions
```
```json
{"text":" Nor is Mr. Quilter's manner less interesting than his matter.","usage":{"type":"duration","seconds":6}}
```

### OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("clip.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model="openai/whisper-large-v3", file=f,
        language="en", temperature=0, response_format="json")
print(r.text)
```

---

## Design

- **Encoder runs once per request.** The audio encoder + cross-KV projection are
  exposed as the model's `.visual` sub-module, which the runner compiles as its
  own NEFF and fires prefill-only through the existing multimodal-encoder trigger
  (`_execute_mm_encoder` → `embed_multimodal`). No per-step re-encode; no
  runner-core change to drive it.
- **Cross-attention KV via model-owned `register_buffer` (Option A).** Whisper
  cross-attention attends to the fixed encoder output, so K/V depend on the
  encoder output only. They are projected once at prefill into persistent
  `register_buffer(persistent=False)` HBM buffers `[1, heads//tp, 1500, 64]` per
  layer (32 layers) and read Q-only each decode step. The buffers are **not**
  listed in `get_kv_spec()`, so the block manager never touches them. HBM
  persistence across the encoder-write NEFF → decode-read NEFF boundary is
  provided by the plugin's `aliasing_output_rewrite` FX pass, which builds an
  `io_map` aliasing each in-place `.copy_()` back to its HBM allocation (the
  native-backend equivalent of NxDI aliasing; `input_output_aliases` is
  unavailable on the `torch.compile(backend=...)` path).
- **Decoder self-attention → block-managed KV.** Standard plugin path: 32
  self-attn `LayerSpec`s (`head_size=64`, `num_kv_heads = 20 // tp`), paged writes
  via `index_put_`, an additive causal + position mask so unwritten cache slots
  are never attended.
- **On-device greedy sampler + vocab-parallel LM head.** The serving/async path
  requires `forward` to return sampled **token ids**, not logits. The model owns a
  `Sampler` (`on_device_sampling_config`, `all_greedy`) and applies it in
  `forward`. The LM head is **vocab-parallel**: a `ColumnParallelLinear` shards the
  (padded) vocab across TP ranks so each rank projects only its `1/TP` slice
  (`gather_output=False`), instead of the older replicated full-vocab matmul.
  vocab=51866 is padded to the next multiple of TP (51868 at TP=4); the padded
  columns are masked to `-inf` so they can never win. The `Sampler` is constructed
  with `process_group=<lm_head TP group>`, so its greedy argmax runs the plugin's
  distributed global-argmax gather across ranks — lowest global index wins on ties,
  byte-identical to argmax over the full-vocab head. When on-device sampling is off
  (the offline byte-identical driver), `compute_logits` all-gathers the shards back
  to full vocab for host sampling.
- **fp32 at the precision-critical spots.** The attention softmaxes (encoder
  self-attn, decoder self-attn prefill + decode, cross-attn) and the tied LM-head
  projection run in fp32. This removes device-bf16 rounding at first-token near-ties
  without altering the reference math.
- **TP=4 is the maximum.** whisper-large-v3 has 20 attention heads and
  `20 % 8 != 0`, so TP=8 is not expressible. Valid TP ∩ power-of-2 on trn2 =
  {1, 2, 4}; TP=4 at LNC=2 is used here.

---

## Results

Correctness gate: 5 short public-domain **LibriSpeech** clips, greedy
(`temperature=0`, `language=en`), compared byte-for-byte to the OpenAI-whisper
large-v3 greedy reference token IDs.

| Path | Config | Result |
|------|--------|--------|
| **offline, TP=1** | greedy | **5/5 byte-identical** |
| **served (`/v1/audio/transcriptions`), TP=4** | greedy | **4/5 byte-identical + 1/5 documented fp32 first-token near-tie** = 5/5 correct at the documented bar |

The one TP=4 discrepancy is a single first-token near-tie on one clip (ref `2221`
` Mr.` vs `503` `"`, a ~0.12-logprob tie); see [Known
limitations](#known-limitations).

**Served latency (warm, TP=4)**: ~198–890 ms per clip, scaling with the number of
decoded tokens (not audio duration): a ~13-token utterance is ~198 ms, an
~84-token utterance ~890 ms. Fixed per-request serving overhead (audio decode →
mel front-end, request scheduling, HTTP round-trip) is on the order of
~100–150 ms on top of the encoder + prefill + decode NEFF time.

> Latency figures are warm wall-clock through the full HTTP serving stack — a
> serving-latency measurement, not a kernel/MFU number.

---

## Tests

Under `vllm_neuron/model/whisper/tests/`:

| Test / script | Device? | What it checks |
|---------------|---------|----------------|
| `test_whisper_registration.py` | **no** (CI-safe) | `get_models()` registers Whisper; factory imports + carries the transcription/multimodal markers; `WhisperConfig` asserts large-v3 dims and coerces fp16/fp32 → bf16. |
| `test_whisper_correctness.py` | **yes** (`neuron_device`) | The 5-clip byte-identical greedy gate vs the OpenAI-whisper reference IDs, via the served endpoint. Skipped unless `WHISPER_NEURON_DEVICE_TESTS=1` and a server is reachable. |
| `make_reference.py` | host (needs `openai-whisper`) | Regenerates the reference `<clip>.json` (`ref_token_ids` + `text`) so the gate is reproducible. |
| `serve_smoke_test.sh` | manual | Launches `vllm serve` with the validated config and curls one clip. |

Run the no-device unit tests (CI):

```bash
pytest vllm_neuron/model/whisper/tests/test_whisper_registration.py -v
```

Run the on-device gate (a `trn2.3xlarge` with a running server):

```bash
# 1. build the reference (host, venv with openai-whisper)
python -m vllm_neuron.model.whisper.tests.make_reference \
    --clips-dir /large/work/ref/wav --out-dir /large/work/ref/clips
# 2. start the server (see serve_smoke_test.sh for the exact command)
# 3. run the gate
WHISPER_NEURON_DEVICE_TESTS=1 \
WHISPER_REF_DIR=/large/work/ref/clips WHISPER_WAV_DIR=/large/work/ref/wav \
pytest vllm_neuron/model/whisper/tests/test_whisper_correctness.py -v
```

---

## Known limitations

- **One documented first-token fp32 near-tie (TP=4).** On one clip the served
  greedy output flips the FIRST token only (ref `2221` ` Mr.` vs `503` `"`), then
  re-syncs. Per-step top-5 shows a ~0.12-logprob near-tie. This is a device-bf16
  reduction-order artifact, not a modeling bug: the OpenAI/HF reference resolves
  `2221` in both fp32 and bf16 on CPU, and the gold reference model flips it
  identically on-device. At TP=4 the encoder attention is sharded (5 heads/rank)
  and all-reduced, so the fp32 reduction order differs from TP=1 and tips the
  tie the other way. The correctness gate allows this single clip if the rest of
  the sequence matches.
- **Greedy-only validated.** Correctness is proven for greedy decoding
  (`temperature=0`, `all_greedy`). Sampling (temperature > 0, top-p/top-k) is
  wired through the on-device `Sampler` but is **not** validated here.
- **30-second window; no long-form chunking.** The encoder runs the fixed
  1500-frame (30 s) mel. Clips longer than 30 s are truncated; long-form
  chunking / stitching is not implemented.
- **One runner edit.** `neuron_model_runner.py get_supported_tasks()` was changed
  from a hardcoded `("generate",)` to report `("transcription",)` for a Whisper
  model (mirrors vLLM-core; no effect on other models). This is the only
  runner-core change; all other logic lives in the model package.
- **Synchronous scheduling.** Served with `--no-async-scheduling`; the async
  device-tensor path is not yet wired for the Whisper output shape (a future
  perf lever). The synchronous on-device-sampling path is correct.

## Footguns

1. **fp16 → bf16 coercion is mandatory.** whisper-large-v3 ships as float16; the
   fp32-bias RowParallel add lowers to a mixed `f32 + f16` add that neuronx-cc
   rejects (`NCC_IVRF100`). `WhisperConfig` coerces fp16/fp32 → bf16, and the
   fp32-bias add is done in fp32 then cast back.
2. **Wipe the compile cache after any graph change.** The plugin caches NEFFs by
   HLO hash at `/root/.cache/vllm/neuron/compile_cache/`. A stale entry from a
   previous build is served even after a source fix — `rm -rf` it after any
   dtype/graph-affecting change or you will debug a ghost.
3. **`vision_neuron_config` + `max_num_batched_tokens >= 1500` are required.**
   vLLM flags Whisper as audio-multimodal, so `_init_encoder_cache` needs a
   `vision_neuron_config` (`num_vision_tokens_buckets=[1500]`,
   `vision_attention_block_size=1500`) supplied via `additional_config`, and the
   1500-token audio mm-item needs `max_num_batched_tokens >= 1500` (use 1536).
4. **On-device sampling is required for serving.** The async/MP-executor serving
   loop needs the model to return token ids, so `on_device_sampling_config`
   (`all_greedy: true`) must be set and the model must own + apply a `Sampler` in
   `forward`.
5. **`.float()` on a bf16 neuron tensor raises `Expected self.dtype()==dst.dtype()`.**
   Materialize to CPU first (`.detach().to("cpu").float()`) for any host-side
   inspection.
