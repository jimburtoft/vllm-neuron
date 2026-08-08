# Voxtral-Mini-3B on the vllm-neuron native XLA backend

Port of `mistralai/Voxtral-Mini-3B-2507` to the `vllm-neuron` 0.21 native XLA
backend (`torch.compile(backend="vllm_neuron")`), following the
`feature/whisper-native-integration` template.

- **Model**: `mistralai/Voxtral-Mini-3B-2507`
- **Architecture**: Whisper-derived audio encoder (32 layers, hidden 1280,
  head_dim 64, 128 mel bins) → 2-layer AudioLanguageAdapter (5120→3072→3072
  with GELU) → Ministral-3B text decoder (Llama-family GQA 32/8, head_dim
  128, 30 layers, vocab 131072, RoPE theta=1e8, SwiGLU, no SWA)
- **Hardware**: trn2.3xlarge, TP=4 LNC=2
- **SDK**: 2.31 (DLAMI `Deep Learning AMI Neuron (Ubuntu 24.04) 20260708`)
- **Container**: `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`

## Baseline numbers

Customer 18-file TED benchmark (`voxtral-native` dataset), first-pass unoptimized:

| Metric | Value | Reference |
|---|---:|---:|
| Mean per-file latency | **0.727 s** | NxDI TP=4 SDK 2.31: 0.656 s; L40S GPU vLLM: 0.670 s |
| Median | 0.767 s | -- |
| Real-time factor | 19.86x | -- |
| Byte-identical greedy (26+65 tokens) | ✓ | vs CPU HF `VoxtralForConditionalGeneration.from_pretrained(..., torch_dtype=bfloat16)` |

## Quick start

```bash
# On a trn2.3xlarge instance with the stock SDK 2.31 DLAMI + the vllm-neuron
# 0.21 container running with this plugin installed:

vllm serve mistralai/Voxtral-Mini-3B-2507 \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --additional-config '{"neuron_config":{"quantization":"bf16","on_device_sampling_config":{"all_greedy":"true"},"kv_segment_size_buckets":[1024],"num_batched_tokens_buckets":[1024],"num_seqs_buckets":[1]},"vision_neuron_config":{"num_vision_tokens_buckets":[375],"vision_attention_block_size":384}}'

# In another shell:
curl -sf http://localhost:8000/v1/audio/transcriptions \
    -F "model=mistralai/Voxtral-Mini-3B-2507" \
    -F "language=en" \
    -F "temperature=0.0" \
    -F "response_format=json" \
    -F "file=@/path/to/audio.wav"
```

See `tests/serve_smoke_test.sh` for a scripted version.

## Design

The port composes three pieces:

1. **Audio encoder + adapter (`self.visual`)** — a single `_EncoderPlusAdapter`
   `nn.Module` that owns both `VoxtralAudioEncoder` (Whisper-derived body)
   and `AudioLanguageAdapter` (2-layer MLP projector 5120→3072). Registered
   as `self.visual` so the plugin runner's
   `torch.compile(inner_model.visual, backend="vllm_neuron_graph_capture")`
   wraps the entire encoder+adapter pipeline as a single compiled graph.
   The mel-spectrogram computation is folded into the encoder graph
   (unfold-avoiding gather-based STFT — see "torch.Tensor.unfold segfault"
   note below).

2. **Ministral-3B text decoder (`self.language_model`)** — a direct
   component reuse of `vllm_neuron.model.llama3.LlamaForCausalLM`.
   Ministral-3B is Llama-family (RMSNorm + GQA + SwiGLU + RoPE), so
   `_voxtral_text_to_llama_config()` translates our `VoxtralTextConfig`
   into the plugin's `LlamaConfig` and instantiates the same
   `LlamaForCausalLM` used by other Llama-arch models.

3. **Audio-token injection at prefill** — audio embeddings are scatter-written
   into the plugin's `encoder_cache_blocks` via `embed_multimodal(encoder_cache,
   mm_hashes, audio_arrays)`. At prefill, `forward()` receives
   `vision_embedding_blocks` + `vision_positions` kwargs from the runner and
   builds `inputs_embeds` by scattering audio embeds at `input_ids ==
   audio_token_id (=24)` positions. Then delegates to `self.language_model`
   with `is_token_ids` mask so `NF.merge_prompt_embeds` splices audio at the
   right positions.

## torch.Tensor.unfold segfault (upstream Neuron bug)

**`torch.Tensor.unfold(dim, size, step)` on Neuron segfaults `torch_xla`.**
`Tensor.unfold` uses `as_strided` internally, which the Neuron torch_xla
HLO lowering can't handle. Root-caused via minimal reproducer (see
`bug_report/repro_unfold_segfault.py` in the project workspace; filed with
AWS Neuron team). Workaround: framing via `torch.gather` (see
`audio_encoder_bf16.py::VoxtralAudioEncoder.compute_mel_spectrogram`).

Related upstream issues we hit + workarounds:

- **Neuron runtime strict `.copy_()` dtype check**: any device-side `.to(dtype=)`
  that would trigger `.copy_()` fails with `Expected self.dtype() == dst.dtype()`.
  Workarounds: (a) fold dtype casts inside compiled graphs (where compiler
  handles them); (b) CPU-roundtrip for the rare eager-mode cast we can't avoid
  (`embed_multimodal`'s cache scatter-write).
- **`nn.GELU` dynamo skip**: `torch._C._nn.gelu` isn't traceable. Use explicit
  erf: `x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))` (see
  `AudioLanguageAdapter.forward`).
- **`convolution_overrideable not implemented`** for eager conv1d on Neuron:
  all convolutions must go through the compiled graph. Wire the encoder as
  `self.visual` so the runner's `torch.compile(inner_model.visual, ...)`
  compiles it.

## Upstream vLLM bugs (patched via container-local scripts, filed as PRs)

Two upstream vLLM bugs affect Voxtral serving on ANY backend (not just Neuron):

1. **`VoxtralDummyInputsBuilder.get_dummy_text` returns `""`** — the mm-only
   processor path feeds `('', [audio])` to `MistralCommonVoxtralProcessor`
   which rejects. Fix: return `"[AUDIO]" * n_audio`.

2. **`MistralCommonFeatureExtractor.fetch_audio` missing** — newer
   `transformers.ProcessorMixin.prepare_inputs_layout` calls
   `feature_extractor.fetch_audio(audio, sampling_rate=...)` before
   dispatching to `__call__`. Fix: 5-line pass-through shim.

Both are 5-line patches to `vllm/model_executor/models/voxtral.py` and
`vllm/transformers_utils/processors/voxtral.py` respectively. They should
go upstream as a separate PR to vLLM main. Until they land, this plugin
requires the two patches to be applied to the container's installed
vLLM (see project workspace `apply_voxtral_dummy_text_patch.py` and
`apply_fetch_audio_patch.py`).

## Files

- `__init__.py` — re-exports
- `config.py` — `VoxtralConfig` composite (text + audio) with dim asserts
- `factory.py` — `VoxtralForConditionalGeneration` factory with
  SupportsTranscription markers + upstream classmethod delegation
- `model_bf16.py` — concrete implementation: `_EncoderPlusAdapter` composite,
  `AudioLanguageAdapter`, `VoxtralForConditionalGeneration` with `forward`,
  `embed_multimodal`, `build_vision_synthetic_inputs`, dual-path
  `load_weights`
- `audio_encoder_bf16.py` — Whisper-derived `VoxtralAudioEncoder` with
  on-device mel-spectrogram (gather-based framing, matmul-based DFT)
- `weight_loaders.py` — reference HF-key mapping table (used by
  `_remap_hf_key` in `model_bf16.py`)
- `tests/` — 7 no-device pytest tests + 2 device-gated correctness tests
  (`VOXTRAL_NEURON_DEVICE_TESTS=1` to run)

## Running the tests

```bash
# No-device tests (CI-friendly):
pytest vllm_neuron/model/voxtral/tests/test_voxtral_registration.py -v

# Device-gated correctness tests (requires vllm serve running on port 8000):
VOXTRAL_NEURON_DEVICE_TESTS=1 pytest \
    vllm_neuron/model/voxtral/tests/test_voxtral_correctness.py -v
```

Regenerate CPU HF references after any change to processor / tokenizer:

```bash
cd vllm_neuron/model/voxtral/tests
python make_reference.py  # writes reference/*.json
```
