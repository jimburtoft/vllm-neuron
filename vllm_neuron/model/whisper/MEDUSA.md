<!-- SPDX-License-Identifier: Apache-2.0 -->
# Medusa speculative decoding for native-backend Whisper (BS=1 latency)

A **Medusa speculative-decoding framework** for `openai/whisper-large-v3` on the
vllm-neuron **0.21 native XLA backend** (SDK 2.31 default; `torch.compile(
backend="vllm_neuron")`, **no NxDI**). It targets **single-stream (BS=1) latency**
for `/v1/audio/transcriptions`, the regime where spec-decode helps Whisper.

- **Target container**: `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04`
- **Hardware**: `trn2.3xlarge`, **TP=4, LNC=2, bf16** (TP=4 is the whisper-large-v3
  architectural max — 20 attention heads).
- **Status**: **framework complete and correctness-proven end-to-end.** `vllm serve`
  starts with Medusa enabled at TP=4 and serves real transcriptions. The **speedup
  is a property of the customer's TRAINED heads** — this framework is the plumbing +
  measurement tooling, and the correctness guarantee holds at any acceptance rate.

---

## What it is (and is not)

**Medusa** attaches N=K small MLP "heads" to the decoder's last hidden state; each
head *j* proposes the token *j+1* positions ahead. Every decode iteration the target
runs a single **verify-K** graph over `[anchor, draft_0 .. draft_{K-1}]` (a `[1, K+1]`
window), and a rejection check accepts the longest prefix of drafts that match the
target's own greedy tokens, plus one bonus token. Multiple tokens can be emitted per
graph call → fewer iterations → lower latency **when the heads are accurate**.

- **It IS**: a correct, runnable spec-decode framework for Whisper that accepts
  customer-trained heads (or placeholder heads), enabled from `vllm serve`, with a
  correctness gate + a latency harness the customer runs against their own heads.
- **It is NOT**: a speedup by itself. Untrained / placeholder heads accept ~0 real
  drafts → ~1 token/iter → **slower than greedy** (you pay a small verify overhead for
  no accepted tokens). The win requires trained heads and scales with their acceptance
  (see [Speedup model](#speedup--fyour-heads-acceptance)).

### Correctness guarantee (holds at ANY acceptance rate)

With greedy sampling, the emitted token sequence is **byte-identical to plain greedy
decode of the same target**, for *any* head weights — trained, `random`, or `zero`.
The rejection check sets each accepted position to the target's own greedy argmax and
rejects at the first draft≠target mismatch, so heads can only change *how many*
iterations the decode takes, never *what* tokens come out. Proven 5/5 clips × 3 head
modes (zero / random / synthetic-partial) byte-identical at TP=4 and TP=1, including
the exercised multi-token accept path (Task 020 M3).

---

## Enable Medusa in `vllm serve` (the customer entry point)

Two config surfaces, one for K and one for the heads source:

| Surface | Flag | Fields |
|---|---|---|
| **K (lookahead)** | `--speculative-config` | `method="medusa"` (required), `num_speculative_tokens=K` (default 5) |
| **Heads source** | `--additional-config` | `medusa_config.init` = `zero` \| `random` \| `load`; `medusa_config.heads_path` (when `init="load"`); `medusa_config.num_heads` (optional, defaults to K); `medusa_config.medusa_num_layers` (ResBlocks per head, default 1) |

The heads live **inside the target model** — there is no separate draft model. A
plugin shim teaches vLLM-core's `SpeculativeConfig` to accept `method="medusa"`
without demanding a separate draft-model repo (it points the "draft" config at the
target itself, exactly like ngram/mtp).

### Copy-pasteable serve command (TP=4, placeholder `zero` heads)

```bash
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NEURON_RT_NUM_CORES=4

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
  --speculative-config '{"method":"medusa","num_speculative_tokens":5}' \
  --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}}, "vision_neuron_config": {"num_vision_tokens_buckets": [1500], "vision_attention_block_size": 1500}, "medusa_config": {"init": "zero"}}'
```

Once `Application startup complete` appears:

```bash
curl -s -F "file=@/path/to/clip.flac" -F "model=openai/whisper-large-v3" \
     -F "language=en" -F "temperature=0" -F "response_format=json" \
     http://localhost:8000/v1/audio/transcriptions
```

To run with **your trained heads**, swap the `medusa_config`:

```bash
  --additional-config '{..., "medusa_config": {"init": "load", "heads_path": "/path/to/your_heads.pt"}}'
```

`--no-async-scheduling` and on-device greedy sampling (`all_greedy: true`) are the
validated BS=1 serving levers (sync scheduling + on-device sampling — the runner keeps
sampling on device, so the served path does NOT pay the offline harness's per-iter
logits→CPU copy; see [Latency framing](#offline-harness-vs-served-latency)).

---

## Supplying trained heads (the loader)

The head loader (`medusa_heads.py`) accepts customer weights in several external
formats and normalizes them to the internal `medusa_head_{i}.{j}.linear.{weight,bias}`
scheme. Medusa-1 ties each head's output projection to the model's (sharded) `lm_head`,
so a customer checkpoint only needs to carry the per-head **ResBlock** weights.

| `init` | Meaning | Checkpoint required |
|---|---|---|
| `zero` | ResBlocks zeroed (each head proposes `argmax(lm_head(h))` — the same token K×) | No |
| `random` | ResBlocks randomly initialized (seeded) | No |
| `load` | Load customer/trained heads from `heads_path` | Yes |

Supported external checkpoint schemes (auto-detected):
- Generic / internal: `medusa_heads.{i}.{j}.linear.*` or `medusa_head_{i}.{j}.linear.*`
- Upstream FasterDecoding Medusa, and vLLM "speculators" Medusa layouts (per-head
  output projections are dropped under the tied-lm_head variant).

Requirements: `hidden_size == 1280`, `vocab_size == 51866`, `n_heads >= K`. Forbidden
variants (Medusa-2 / Hydra / EAGLE) are rejected with a clear error.

---

## Speedup = f(your head's acceptance)

The served speedup is set entirely by **how many tokens your head lets the target emit
per verify step**. The framework's per-step cost is fixed and independent of the head:

```
served speedup ≈ (accepted tokens per verify) / R
```

where **`R` is the measured served verify-step / greedy-step cost ratio**. With
on-device greedy rejection (accepted token ids returned from the verify NEFF; no
per-step `[K+1, vocab]` logits→host copy), the measured **`R ≈ 1.31×`** at TP=4 LNC=2
BS=1. If your heads let the target emit `T` tokens per verify (accepted drafts + 1
bonus):

| tokens/verify T | served speedup (R≈1.31) |
|--:|--:|
| **1** (placeholder / untrained heads land here) | **0.76× — slower than greedy** |
| **1.31** | **1.00× (breakeven)** |
| 2 | 1.53× |
| 3 | 2.29× |
| 4 | 3.05× |
| 5 | 3.82× |
| 6 | 4.58× |

**Breakeven: your head must deliver > ~1.31 accepted tokens/verify** to beat greedy.
Well-trained Medusa heads in the literature commonly reach 2–3+ accepted/verify, i.e.
~1.5–2.3× here. The framework measures *your* heads — it does not manufacture
acceptance; placeholder heads (`zero`/`random`) exist only to prove the plumbing runs
and stays byte-identical (they land at T≈1 → slower, expected).

> The bundled example head (`jburtoft/whisper-large-v3-medusa-heads`) is a lightly
> trained *demonstration* head (~1.19 accepted/verify → ~0.92×, just below breakeven).
> It exists to exercise the accept path end-to-end, not to show a speedup — bring your
> own trained head.

### Head compatibility contract (match these or acceptance silently collapses)

Your trained head must match the framework's assumptions, or it will load but accept
~nothing:

- **Medusa-1 (tied output projection).** Each head's output projection is tied to the
  target's `proj_out` (LM head). Per-head output-projection tensors in your checkpoint
  are dropped on load. A Medusa-2 (untied) head will not use its own output projection.
- **Head architecture:** `L` stacked ResBlocks per head, `ResBlock(x) = x + SiLU(Linear(d,d))`,
  `d_model = 1280`. Set `medusa_config.medusa_num_layers = L` to match your `L`.
- **Shift convention `i + 2`:** `head_i` must be trained to predict the token at position
  `t + i + 2` from the decoder's last hidden state at `t` (the +1 slot is the target's own
  next-token argmax; the heads cover +2 onward). A head trained with a different shift will
  mispredict and be rejected.
- **Dims:** `hidden_size == 1280`, `vocab_size == 51866`, `n_heads >= K`
  (`num_speculative_tokens`).
- **Train against the target's own greedy outputs (pseudo-labels), not ground-truth
  transcripts** — acceptance measures agreement with what `whisper-large-v3` would greedily
  emit. (See the example repo's training/harvest scripts.)

---

## Validate: correctness gate + latency harness

Both live in `tests/medusa/` (see that folder's README for full usage):

```bash
# Correctness gate — 5 clips × {zero, random, synthetic} heads, byte-identical to greedy
torchrun --nproc_per_node=4 tests/medusa/medusa_correctness_gate.py     # TP=4
python  tests/medusa/medusa_correctness_gate.py                          # TP=1

# Latency harness — per-clip latency, accepted/iter, ms/token, speedup + the model.
# Load YOUR trained heads here; the acceptance they deliver sets the speedup.
torchrun --nproc_per_node=4 tests/medusa/medusa_bench.py --medusa-heads /path/to/heads.pt
```

### Offline harness vs served latency

- **Offline harness (`medusa_bench.py`)** is a **mechanism/correctness** tool. It uses
  **host-side** rejection (copies the `[K+1, vocab]` verify logits to CPU each iter),
  which dominates wall-clock (~0.5× regardless of head quality). Use it to confirm the
  loop runs and to read `accepted-tokens/iter`; do **not** read its absolute wall-clock
  as the served latency. It prints this caveat in-line and uses the speedup **model**
  (above) for the customer-facing projection.
- **Served path (`vllm serve`)** is the **real latency** the customer measures. With
  on-device greedy sampling (`all_greedy: true`) the runner keeps sampling on device
  and does **not** perform the per-iter full-logits CPU copy — so the served latency
  reflects the on-device verify/step ratio, not the harness's host-transfer bound.
  (Fully on-device rejection over the `[K+1, vocab]` logits is the next optimization
  lever; the framework is correct and served either way.)

---

## Design (how it wires into the runner)

- **`MedusaProposer`** (`vllm/spec_decode/medusa.py`): a thin drafter registered under
  `method=="medusa"` alongside EAGLE3. No separate draft model, no draft KV cache
  (`use_eagle()` is False for medusa → the runner's draft-KV alloc/bind auto-skips).
- **Verify-K NEFF** (`model_bf16.py`): a `[1, K+1]` decode bucket that returns K+1
  logit rows + the next-K drafts (heads run in-graph on the last candidate's hidden
  state). Reuses the existing per-position causal mask + trailing self-KV write; the
  static cross-KV is read read-only.
- **Runner dispatch** (`neuron_model_runner.py`): four `is_medusa_spec`-guarded edits
  (registration, model-load/target-bind, propose-block gate, output-tuple parser +
  `_propose_draft_token_ids`). The EAGLE path is untouched; the non-spec serving path
  is unchanged.
- **Config shim** (`vllm/patches/medusa_spec_config_patch.py`): lets `vllm serve
  --speculative-config method=medusa` start without a separate draft model.

This follows the Task 020 M0 spec (§1/§4/§6) and was proven end-to-end on device
(M0.5 → M4). See `tests/medusa/README.md` for the validation tooling.
