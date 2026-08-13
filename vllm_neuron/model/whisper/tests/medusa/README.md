# Medusa framework — correctness gate + customer latency harness (Task 020 M3)

BS=1 customer-facing validation for the Whisper-native Medusa speculative-decoding
framework (`method="medusa"`, vllm-neuron 0.21 native XLA backend, NO NxDI).

The framework itself (the `MedusaProposer`, the four runner dispatch edits, the
verify-K NEFF, the `MedusaHeads` module + loader) lives in:
- `vllm_neuron/vllm/spec_decode/medusa.py`
- `vllm_neuron/vllm/worker/neuron_model_runner.py` (guarded by `is_medusa_spec`)
- `vllm_neuron/model/whisper/model_bf16.py` (verify-K branch in `forward`)
- `vllm_neuron/model/whisper/medusa_heads.py`

This directory holds the M3 validation tooling.

## Files

| File | Purpose |
|---|---|
| `medusa_correctness_gate.py` | The formal BS=1 correctness gate. Proves the Medusa accepted-token output is BYTE-IDENTICAL to plain greedy on the SAME served config across three head modes — including when tokens are genuinely accepted (the multi-token accept path). |
| `medusa_bench.py` | The **customer latency harness**. Measures per-clip end-to-end latency, mean per-iter latency, mean accepted tokens/iter, effective ms/token, and speedup vs greedy. Emits a clean table + the speedup-vs-acceptance model. This is the tool the customer runs once they load their TRAINED heads. |
| `medusa_synth_heads.py` | Builds a SYNTHETIC partially-correct head checkpoint (crude fit to the greedy continuation) used by the gate to EXERCISE the accept path. Not a real trained model — only a device for driving multi-token accepts. |
| `make_clip_mels.py` | Builds the 5-clip mel set (jfk + 4 LibriSpeech dummy utterances) the gate/harness consume. |

## Correctness invariant (M0 spec §5)

With greedy sampling, every emitted token equals the target's own greedy argmax at
that position, regardless of what the Medusa heads proposed (the rejection sampler
sets each accepted token to `target_argmax` and rejects at the first draft≠target
mismatch, then appends the bonus). So the transcript is **byte-identical to plain
greedy at ANY acceptance rate** — untrained heads only change the iteration count
(latency), never the output. The gate compares Medusa-vs-greedy on the SAME
TP/precision config (NOT vs OpenAI-whisper) to isolate the framework from the
target model's own fp32 near-tie behavior.

## Running (on a trn2 instance, DLAMI 20260721+, vllm 0.21 native)

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0/bin/activate
export NEURON_SKIP_EFA_AFFINITY=1

# 1. build the 5-clip mel set (once)
python make_clip_mels.py

# 2. (optional) build a synthetic partially-correct head set to exercise the accept path
python medusa_synth_heads.py --clips jfk,ls1,ls2,ls3 --steps 400

# 3. correctness gate: 5 clips x {zero, random, load-synthetic} head modes, TP=4
torchrun --nproc_per_node=4 medusa_correctness_gate.py     # or: python ... for TP=1

# 4. customer latency harness (load YOUR trained heads here)
torchrun --nproc_per_node=4 medusa_bench.py --medusa-heads /path/to/your_heads.pt \
    --num-speculative-tokens 5 --clips jfk,ls1,ls2,ls3,ls4
```

`--medusa-heads` accepts `zero`, `random`, or a checkpoint path (upstream
FasterDecoding / vLLM speculators / internal `medusa_heads.{i}.{j}.linear.*`
layouts — the ResBlock Linear tensors are loaded, per-head output projections are
dropped under the Medusa-1 lm_head tie).

## Speedup expectation (be honest with the customer)

Placeholder/untrained heads (`zero`/`random`) accept ~0 real drafts → Medusa is
~1.0x or SLOWER than greedy (you pay the ~1.116x verify overhead for no accepted
tokens). This harness is the MEASUREMENT TOOL, not a speedup by itself. On the
measured on-device verify/step ratio r≈1.116 (M0.5), the speedup scales with
acceptance:

```
speedup ≈ (mean tokens emitted per verify) / 1.116
```

| tokens/verify | speedup |
|--:|--:|
| 1 (placeholder) | 0.90x  (SLOWER) |
| 2 | 1.79x |
| 3 | 2.69x |
| 4 | 3.58x |

Heads must deliver > ~0.12 accepted drafts/verify just to break even; the win
grows with acceptance. The customer's trained heads determine the real speedup.
