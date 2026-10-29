# Tencent Hy4-preview on Trn2

[`tencent/Hy4-preview`](https://huggingface.co/tencent/Hy4-preview)
(`HYV4ForCausalLM`, 770B total / 49B active) on one trn2.48xlarge, text only,
32K context. Single node, or prefill / decode disaggregated over two nodes (1P1D).

| | |
|---|---|
| layers | 78: layer 0 dense FFN (18432), 77 MoE |
| MoE | 256 routed experts of 2048, top-8, sigmoid + correction-bias routing, x2.827, SwiGLU clamp 10, 1 shared expert |
| attention | gated MLA, 64 heads, q_lora 2048, kv_lora 512, qk 192+64, v 256, learnable sink per head |
| sparse attention | DSA, top-2048 keys; 20 "full" layers run the indexer, 58 "shared" ones reuse it |
| residual | iHC, 4 streams |
| vocab / hidden | 120832 / 6144 |

> **Note:** a community contribution. See the compatibility matrix for exactly
> what was tested.

## Usage

Use the Neuron SDK 2.33 vLLM environment (vllm-neuron 0.29.0.1.2.0, vLLM 0.29,
neuronx-cc 2.28), for example the DLAMI's
`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0`. Then:

```bash
# 1. Register the architecture. This installs src/ as the ``hy4_preview`` package
#    with a ``vllm.general_plugins`` entry point, so the registration runs in every
#    process vLLM starts (engine, workers, ``vllm serve``), not only your script.
pip install -e contrib/models/hy4_preview

# 2. Once: MXFP8 -> Trn2 fp8 (~7 min with 24 workers on a trn2.48xlarge; see below)
python contrib/models/hy4_preview/scripts/convert_mxfp8_to_neuron_fp8.py \
    --src <dir>/Hy4-preview-FP8 --dst <dir>/Hy4-preview-neuron-fp8 --workers 24

# 3. Use the XLA compile route (see "Known limitations")
export VLLM_NEURON_BACKEND=vllm_neuron
# 4. Attention paths are chosen per platform (see "Layout"): key-sharded
#    decode (HY4_DEC_CP) and the sparse decode kernel (HY4_DEC_SPARSE_NKI) on
#    Trn2 and Trn3; the sparse prefill (HY4_PREFILL_SPARSE_NKI) on Trn2 only.
#    Each variable overrides its default (=1 on, =0 off).
# 5. Optional, serving: the plugin sizes the KV cache to 30% of the budget by
#    default; where HBM allows (e.g. Trn3), raising it lets max_num_seqs requests
#    run together at max_model_len 32768 instead of in waves
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.5
```

Recommended neuron_config additions (used for the "optimized" numbers under
"Measured"): decode context buckets, so short contexts decode against a shorter
span, and several KV segment sizes, which bound a prefill chunk's prefix:

```json
{"decode_context_length_buckets": [4096, 8192, 16384],
 "kv_segment_size_buckets": [2048, 4096, 8192]}
```

Pass them in `--additional-config`'s `neuron_config` for `vllm serve`, or as
`run.py --extra-neuron-config '<json>'` (or `$HY4_EXTRA_NEURON_CONFIG`).

### Serve with the OpenAI-compatible server

```bash
vllm serve <dir>/Hy4-preview-neuron-fp8 \
    --tensor-parallel-size 64 --max-model-len 32768 \
    --max-num-batched-tokens 2048 --max-num-seqs 4 \
    --gpu-memory-utilization 0.8 --no-enable-prefix-caching \
    --distributed-timeout-seconds 14400 \
    --additional-config '{"neuron_config": {"quantization": "bf16",
        "num_batched_tokens_buckets": [2048], "num_seqs_buckets": [1, 4],
        "decode_context_length_buckets": [4096, 8192, 16384],
        "kv_segment_size_buckets": [2048, 4096, 8192],
        "on_device_sampling_config": {"all_greedy": true}}}'

curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"model": "<dir>/Hy4-preview-neuron-fp8", "prompt": "The capital of France is", "max_tokens": 32}'
```

### Offline inference

```bash
python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8
# a pass key hidden at the start of ~30K tokens of filler
python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8 --needle 30000
# TTFT / TPOT at 31,744 in / 256 out, one request, median of 3
python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8 --bench
# also time N simultaneous requests (decode TPOT at batch N)
python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8 --bench --bench-concurrency 4
# NRT device profile (rank 0) of one --bench-input-len request
python contrib/models/hy4_preview/scripts/run.py --model <dir>/Hy4-preview-neuron-fp8 --profile-dir <dir>/profile
```

`scripts/run.py` builds `LLM(...)` with the arguments above; the demo prompts and
their outputs are under "Measured".

### Disaggregated prefill / decode (1P1D, two trn2.48xlarge)

`scripts/di/` runs a prefill server (port 8100) on one node, a decode server
(8200) on the other, and vLLM's toy proxy (8000) in front, with the KV cache
moved by NIXL (LIBFABRIC over EFA). The plugin's `NeuronNixlConnector` registers
two slabs (K, V) per layer; this model's cache is key-only, so
`src/kv_connector.py` subclasses it to register one (`Hy4NixlConnector`, loaded
through `kv_connector_module_path`).

```bash
# prefill node
MODEL=<dir>/Hy4-preview-neuron-fp8 bash contrib/models/hy4_preview/scripts/di/run_prefill.sh
# decode node
MODEL=<dir>/Hy4-preview-neuron-fp8 bash contrib/models/hy4_preview/scripts/di/run_decode.sh
# prefill node, once both report healthy: real addresses, not localhost -- the
# proxy hands PREFILL_HOST to the decode server as the NIXL side-channel host
PREFILL_HOST=<prefill private ip> DECODE_HOST=<decode private ip> \
    VLLM_NEURON_REPO=. bash contrib/models/hy4_preview/scripts/di/run_proxy.sh
```

With `PREFILL_HOST=127.0.0.1` every request fails with a KV load failure: the
decode server tries the handshake against its own loopback. `run_decode.sh`
takes `MAX_NUM_SEQS` (4) and `SEQ_BUCKETS` (`[1, 4]`).

## Compatibility Matrix

| Instance / SDK version | 2.33 (vllm-neuron 0.29.0.1.2.0) | 2.27 (vllm-neuron 0.24) | earlier |
| ---------------------- | ------------------------------- | ----------------------- | ---------- |
| Trn3                   | Not tested (see note)           | Not tested              | Not tested |
| Trn2 (trn2.48xlarge)   | Working, single node and 1P1D   | Working (previous port) | Not tested |

Trn3: an optional native-MXFP8 expert path is included
(`convert_mxfp8_to_neuron_fp8.py --keep-mx-experts`, `neuron_quantization.format`
`mxfp8_experts`). The published OCP e4m3fn weights and e8m0 exponents are loaded
unchanged. By default (`HY4_MX_BACKEND=torch`) the routed experts run the same
torch selective/dense paths as the fp8-row checkpoint, with each 32-element MX
block dequantised in the graph as `fp8 * 2^(e-127)` in bf16 -- exact, since an
e4m3 value times a power of two is representable in bf16 (the load checks the
exponent range and logs it). This backend is pure torch and loads on any
platform, though the format is intended for Trn3. `HY4_MX_BACKEND=bf16` does
the same exact dequantisation once at load and keeps the routed experts as bf16
(no scales in the graph; values bit-identical to `torch`). It doubles the
expert HBM: about 21.7 GiB of experts per rank at TP=64, ~25.5 GiB with the
rest of the model, so on Trn3 (36 GiB per logical core) it needs
`--gpu-memory-utilization 0.85` or more to leave room for a 32K KV cache.
`HY4_MX_BACKEND=kernel`
instead tiles the experts into nkilib's MX `moe_tkg` layouts (`src/mx_experts.py`)
and runs that kernel; it refuses to load on any platform but Trn3. Only the CPU
unit tests of both backends are reported here.

## Example Checkpoints

- <https://huggingface.co/tencent/Hy4-preview-FP8> (ModelOpt MXFP8; convert with
  `scripts/convert_mxfp8_to_neuron_fp8.py`)
- <https://huggingface.co/tencent/Hy4-preview> (BF16, 1.56 TB: does not fit one
  trn2.48xlarge)

## Testing

```bash
pip install -e contrib/models/hy4_preview
HY4_MODEL=<dir>/Hy4-preview-neuron-fp8 VLLM_NEURON_BACKEND=vllm_neuron \
    pytest contrib/models/hy4_preview/test/integration/test_model.py -v -s --timeout=7200
```

The integration test checks greedy generations against the recorded outputs
below and the ~30K-token pass-key retrieval.

CPU unit tests (no device, no vLLM install needed; the nkilib-reference MX test
is skipped unless `nkilib` is importable):

```bash
pytest contrib/models/hy4_preview/test/unit
```

They cover the selective decode MoE against the dense path and an fp32
reference, key-sharded decode attention against the replicated one (2/4/8 ranks,
shuffled all-gather order), the kv-segment prefill bound (bitwise identical),
the uint8 selection all-gather, the MXFP8 kernel-backend layouts and loaders, and
the MXFP8 torch backend (loaders, bit-exact in-graph dequantisation, selective and
dense outputs against an fp32 reference at 1 and 2 ranks). The template's logit validation is
not used: its transformers reference would need the 770B model in fp32 on the
host.

### Optional: one-layer check against transformers (CPU)

```bash
HY4_ROPE_STYLE=neox VLLM_NEURON_CPU_MODE=1 python contrib/models/hy4_preview/scripts/check_layer_vs_hf.py \
    --model <dir>/Hy4-preview-neuron-fp8 --layer 1
```

Loads one layer through this port's weight loaders, runs a 2600-token prefill in
2048-token chunks (the second reads its prefix back from the paged cache, and
its rows past 2048 attend to a DSA selection) and one paged decode step, and
compares the layer's update against `transformers.models.hy_v4.HYV4DecoderLayer`
in fp32 on all 2601 tokens at once, on the same (dequantised) weights. It also
checks that the query-split selection, emulated over 4 ranks with the
all-gather returning blocks in a shuffled order, equals the unsplit one, and how
many of the decode row's 2048 selected keys match transformers' indexer.

Layers 0 and 1, current code: chunk 1 / later chunks / decode per-token median
0.39-0.47% / 0.43-0.47% / 0.45-0.61%; 2044-2045 of 2048 selected keys agree
with transformers (the rest are ties at the boundary). The table below is the
dense build's, 40 tokens:

| layer | kind | prefill, per-token median | decode |
|---|---|---|---|
| 0 | dense, full indexer | 0.43% | 0.41% |
| 1 | MoE, full indexer | 0.42% (tokens 26, 32 flip an expert) | 0.40% |
| 2 | MoE, shared indexer | 0.42% | 0.42% |
| 5 | MoE, full indexer | 0.49% (token 23 flips) | 0.50% |
| 77 | MoE, full indexer | 0.66% (tokens 1, 29 flip) | 0.84% |

The attention sub-block alone agrees to 0.5%. The few outlier tokens appear
only in MoE layers, only in prefill, and at 20-45% rather than drifting, which
is what an 8th/9th-expert near-tie resolving differently in bf16 and fp32 looks
like; that explanation is inferred, not yet checked expert by expert.

## Measured

trn2.48xlarge, world 64, `max_model_len` 32768, 2048-token prefill chunks,
decode buckets `[1, 4]`, greedy, XLA compile route.

Throughput at a 31,744-token random prompt, 256 output tokens, one request
(only one 32K request fits the KV cache, so this single stream is also the
maximum at this length):

| stack | deployment | TTFT | input tok/s | TPOT | output tok/s |
|---|---|---|---|---|---|
| SDK 2.33, vllm-neuron 0.29 | single node, `run.py --bench` (median of 3) | 87.2 s | 364 | 191.8 ms | 5.21 |
| SDK 2.33, vllm-neuron 0.29 | 1P1D, `vllm bench serve` via the proxy (2 requests) | 88.9 s | 357 | 192.0 ms | 5.21 |
| SDK 2.27, vllm-neuron 0.24 | single node | 97.0 s | 327 | 240.6 ms | 4.16 |

At one request in flight 1P1D costs ~1.7 s of KV transfer on top of the
prefill and nothing per token; what it buys -- decode steps that a prefill
never stalls -- shows only under concurrency, which this table does not measure.

### With selective decode MoE and key-sharded decode attention

Same stack (SDK 2.33, vllm-neuron 0.29), same flags as above plus `HY4_DEC_CP=1`
and the recommended `decode_context_length_buckets` / `kv_segment_size_buckets`.
"Before" is this README's original build. `run.py --bench`, one request, 256
output tokens, median of 3:

| input tokens | before: TTFT | before: TPOT | optimized: TTFT | optimized: TPOT |
|---|---|---|---|---|
| 31,744 | 87.47 s | 192.4 ms | 76.36 s | 81.3 ms |
| 16,384 | | | 32.87 s | 81.4 ms |
| 8,192 | | | 11.12 s | 59.6 ms |
| 4,096 | 10.95 s | 194.7 ms | 3.61 s | 49.6 ms |

`vllm bench serve` (random dataset, `--ignore-eos`, seed 0), single node:

| input / output | concurrency | before: output tok/s | optimized: output tok/s | median TPOT before / optimized |
|---|---|---|---|---|
| 4096 / 256 | 1 | 4.23 | 15.74 | 194.4 / 49.6 ms |
| 1024 / 256 | 4 | 4.65 | 19.07 | |

At concurrency 4 the KV cache (40,384 tokens at `max_model_len` 32768) admits
requests about one at a time (vLLM logs "Prefill deferred"), so this is not a
batch-4 decode number.

Accuracy, before and after: GSM8K, first 200 test questions (chat template,
greedy, 320 tokens) 0.735 both, identical predictions on 145 of 200; the pass
key is found; the demo outputs below reproduce.

With the sparse decode kernel and the sparse prefill (the Trn2 defaults;
same flags, `run.py --bench`, one request, 256 output tokens, median of 3):

| input tokens | TTFT | TPOT |
|---|---|---|
| 31,744 | 37.77 s | 39.2 ms |
| 8,192 | 9.33 s | 38.5 ms |
| 4,096 | 3.62 s | 38.5 ms |

The 4,096 row is with the sparse decode only (below
`HY4_PREFILL_SPARSE_MIN_SPAN` the dense prefill runs). GSM8K-200: 0.735 with the
sparse decode, 0.720 with sparse decode and prefill (18 / 15 questions right in
only one of base / sparse; McNemar p = 0.73); the pass key is found.

### Other

Pass key at the start of 28,804 tokens (`--needle 30000`): found, 87.7 s end to end
(93.8 s in an earlier run on the same stack).
KV cache 40,384 tokens at `--gpu-memory-utilization 0.8` (704 values x 78 layers
per token, replicated on every rank); weights 15.08 GiB of each core's 24.
A cold start compiles one prefill and two decode graphs in ~28 min.

Demo prompts (`run.py`, 32 tokens; SDK 2.33 and 2.27 agree):

```
'The capital of France is'  -> ' __________.\nA. Sydney\nB. New York\nC. Paris\nD. Ottawa 【解析】句意：法国的首都是巴黎。Sydney 悉尼；New York 纽约；Paris 巴黎；Ottawa '
'I am gonna keep counting forever, 1 2 3 4 5' -> ' 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29'
'def fibonacci(n):'         -> '\n    """Return the nth value in the Fibonacci series."""\n    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n -'
'Once upon a time, there was a' -> ' king who was very rich. He had a lot of gold and silver, but he was not happy. He wanted to be the happiest man in the world. One day, he heard that there was a wise man in his kingdom who'
```

## Why the fp8 checkpoint, and why converted

BF16 is 1.56 TB; a trn2.48xlarge has 64 x 24 GB = 1.5 TB of HBM, partitioned per
logical core. The published `Hy4-preview-FP8` is ModelOpt **MXFP8** (one e8m0
exponent per 32 elements), which has no Trn2 kernel, and its values use the OCP
e4m3fn range (448) where Trn2's fp8 tops out at 240.
`convert_mxfp8_to_neuron_fp8.py` therefore:

* requantises the **routed experts** to fp8 with one scale per output row, inside
  the 240 range (`down_proj` also stored transposed as `down_proj_t [E, I, H]`
  so a TP shard is a contiguous read);
* dequantises **everything else** to bf16, which is lossless (e4m3 x 2^k fits
  bf16 exactly);
* drops `quantization_config` from `config.json` and adds
  `neuron_quantization`, which the model checks for.

Requantisation error, MX -> per-row, measured by the converter over 156 expert
tensors: **median 2.56%, max 2.70%** relative (Frobenius). That is on top of the
MXFP8 quantisation itself.

Per rank at world 64: ~11.6 GiB of fp8 experts + ~3.2 GiB bf16 (q_a/kv_a are
replicated) ~= 15 GiB.

## Layout

**Pure TP over 64 ranks.** One attention head per rank. Every rank holds all
256 experts and 32 of each one's 2048 intermediate columns. The MoE runs in
torch, independent of whether Trn2's MoE kernels accept fp8 weights
(`NF.moe_tkg` rejects `float8` today). Graphs of up to
`HY4_SELECTIVE_MOE_MAX_TOKENS` (2) rows -- decode -- gather only the `rows x 8`
selected experts' fp8 weights (stored expert-major, one contiguous read each)
and apply the row scales to the matmul outputs; larger graphs run dense over the
256 local experts with zero affinity on the unselected ones (the best
cut-over for 4-row decode is not yet measured on the full model). The router picks each slot's weight with a one-hot rather than
`torch.gather`, which made neuronx-cc fail with NCC_IESL002 next to the
selective path.
The kernel path (`moe_cte` shard_on_block FP8 / `moe_tkg`) needs
`2048 / tp_degree` to be a multiple of 128, so `ep_degree >= 4`; not done here.

**MLA in latent space, with the DSA key alongside.** Per token the cache holds
the normed 512-wide latent, the 64-wide rope key and the 128-wide indexer key --
704 values, one key-only cache entry per token (`LayerSpec(key_only=True)`:
a `[pages, 1, block, 704]` cache, so a page is one contiguous read); layers with
a shared indexer leave the last 128 zero. `kv_b_proj` is absorbed into the query and the output. Every rank holds
a full copy, which at one head per rank is ~10% *more* than an expanded per-head
cache would be; sharding the latent's tokens across ranks is what would make
MLA's compression pay here.

**DSA.** A full layer scores every visible key with
`sum_h w_h * relu(q_h . k)` (32 heads of 128, LayerNorm'd key, rope on the last 64
dims) and keeps the top 2048 per query; the next shared layers reuse that mask.
The top-k is a per-row threshold found by bisection -- elementwise passes only,
no k=2048 sort and no scatter -- and is exact up to ties with the 2048th score.
In prefill the scoring and selection are **split by query** across the 64 ranks
and the mask all-gathered: done replicated, they were ~90% of the 32K prefill
graph and pushed it to 15.7M instructions, over neuronx-cc's 10M limit. The
attention itself is dense attention masked to the selection: exact, but its cost
still grows with the context. Prefill runs it over the span in 2048-key blocks
with an online softmax, so the `[chunk, span]` probabilities are never live at
once (that, plus 128-aligned slices of the cache entry, took the 28.8K needle
from 825 s to 129 s).

**Key-sharded decode attention (`HY4_DEC_CP=1`).** Without it each rank scores
the whole span for its one head. With it rank `r` gathers only its 1/64 of the
block table and handles those keys for all 64 heads: the absorbed queries and
the indexer scores are all-gathered, the global top-2048 threshold is bisected
over the gathered scores (order-independent, so identical on every rank), and
the softmax partials are merged with a max all-gather and one all-reduce of
numerators and denominators. The cache stays replicated; only the read is
split. Requires the block-table width to divide by the TP degree, otherwise
the replicated path runs.

**Sparse key-sharded decode (`HY4_DEC_SPARSE_NKI`, with `HY4_DEC_CP`; default on
on Trn2 and Trn3).** Each rank reads only the 128-wide indexer keys of its
shard (`gather_rows_cols`), and after the global threshold only the 576-wide
`[latent | rope]` slices of the keys DSA selected (`sparse_decode_attn`, adapted
from the workspace's absorbed-MLA DCP decode kernel): unselected keys and keys
past the prefix get an out-of-range row index that the indirect DMA skips, so
their bytes are never read. The kernel returns unnormalised local `(m, den, num)`
and the existing merge is unchanged. Both kernels are in `src/nki_sparse_decode.py`
and run on Neuron only; on CPU, torch references with the same contracts run
instead (`test_decode_cp_sparse_matches`). `test/sim/sim_sparse_decode.py`
checks the kernels in the NKI simulator. By default the per-rank key list is the
whole shard (always exact); `HY4_SPARSE_KLOC=<n>` compacts it to `n` (rounded
up to 128) slots, which is exact only while no rank holds more selected keys.

**Sparse prefill split by query (`HY4_PREFILL_SPARSE_NKI`; default on on Trn2,
off on Trn3, where its kernels are currently slower than the dense blocked
attention; graphs whose prefix span is below `HY4_PREFILL_SPARSE_MIN_SPAN`,
default 4096, keep the dense path).** Instead of each rank attending its own head over every query
and every key under the DSA mask, each rank attends its `C / world` queries for
all heads over only their selected keys. Full layers read only the 128-wide
indexer keys of the prefix from the paged cache (`gather_rows_cols`), score
the prefix and the chunk's own keys separately, and take the top-k threshold
over both. The rank's local selection becomes per-query lists of key positions
(`HY4_PREFILL_KSEL` slots, default 2176, rounded up to 128; on Neuron by the
NKI kernel `select_to_rows` in `src/nki_sparse_prefill.py`, GpSimd nonzero
compaction, and on CPU by a torch reference; a query with more selected keys
than slots keeps its first ones), so the mask all-gather is gone. The absorbed
queries are all-gathered over heads, and `sparse_decode_attn` in position mode
reads prefix keys straight from the paged cache and the chunk's keys from this
step's projections (each local query is one "request"). No concatenated tensor
feeds a kernel. The sink is folded in, and the bf16 result is all-gathered so
each rank takes its own heads back for `w_uv` / gate / `o_proj`. Shared layers
reuse the position lists. Needs `C % world == 0` and the span and chunk in
whole 128-key tiles, else the dense path runs. CPU check against the dense
path: `test_prefill_sparse.py`; simulator checks at prefill shapes:
`test/sim/sim_sparse_prefill.py` (attention) and `test/sim/sim_select_rows.py`
(position lists vs the torch reference, including tie-heavy selections).

**Prefill prefix bound.** With more than one `kv_segment_size_buckets` entry,
the plugin picks the smallest segment that holds the whole prefix, so a prefill
graph compiled for a smaller segment knows `cached_seq_len <= kv_segment_size`
and attends only to that many keys (rounded up to whole 2048-key blocks: the
result is bitwise identical). The model checks the plugin's picker at startup
(`HY4_PREFILL_SPAN_KVSEG=0` turns this off). `HY4_SEL_GATHER_DTYPE=uint8`
(opt-in) halves the bytes of the prefill DSA-mask all-gather; exact.

**Chunked prefill** is the plugin's segmented prefill: one chunk of one request
per step, attending to the `cached_seq_len` tokens already in the cache plus
causally to itself. `max_num_batched_tokens` (`--prefill-chunk`, default 2048)
must be one of 512/1024/2048/4096/8192.

## Known limitations

* **Attention cost still grows with the bucketed span.** Decode no longer
  reads every expert, and the decode-context and kv-segment buckets cut the span
  for short contexts, but a prefill chunk whose prefix exceeds the largest
  `kv_segment_size_buckets` entry still attends to (and its indexer scores) the
  full 32K span. A real sparse gather instead of a masked dense attention is
  next.
* **Zeroing the gathered context is a 2.8x speed matter, not a correctness
  one.** Keys beyond the prefix are masked out of the scores either way, but a
  build that left the context unmasked ran ~150 s per chunk (2306 s for the
  same needle) with a prefill graph twice the size (596K vs 284K instructions
  per subgraph) and an 85 min compile. Why is not understood.
* **Context is 32K, not 1M.** The latent cache is replicated on every rank, so
  KV capacity (40K tokens at gmu 0.8) is per rank, not per instance.
  `HY4_DEC_CP` splits the decode *read* across ranks but keeps the cache
  replicated, so the cap remains.
* **Tried and not kept:** bf16 attention-score inputs (slower end to end on
  device); `bmm` / `F.silu`
  formulations of the selective path (NCC_IESL002).
* **A neuronx-cc miscompile is worked around, not fixed.** In decode, masking
  the gathered `[rows, 32K, 704]` context with `in_prefix` (by `where` or by a
  multiply) made the device compute `in_prefix` *itself* wrongly wherever it was
  reused -- extra keys in 128-key blocks past the prefix, even on a dead row at
  position 0 -- so every decode at 32K attended to the wrong keys while the first
  token (from prefill) was right. The same ops compiled alone, and the CPU, are
  correct. `_zero_beyond` masks the context with a *separately computed*
  predicate (a float subtraction), which compiles correctly; checked on device
  in a standalone graph of the decode attention chain before it went in.
* **RoPE pairing: interleaved, and transformers disagrees.** vLLM's and SGLang's
  Hy4 ports rotate interleaved pairs; transformers' `hy_v4` rotates halves.
  Measured on device with the same weights, `HY4_ROPE_STYLE=neox` degrades
  within a few tokens -- `'... 1 2 3 4 5'` continues `' 6 7 8 9 10 13 20 5 6 8
  ...'`, `def fibonacci(n):` comes out broken -- while interleaved (the default)
  gives the outputs above. So the checkpoint wants interleaved. The CPU check
  below still runs `neox`, because it compares against transformers.
* No MTP (the `mtp_layers.0` tensors are ignored), no prefix caching, no mixed
  prefill/decode batches (the plugin runs one prefill request per step).
* **Use the XLA compile route (`VLLM_NEURON_BACKEND=vllm_neuron`).** On
  vllm-neuron 0.29's default native route the decode graphs compile once per rank
  (64 concurrent neuronx-cc processes, ~1.9 TB of host RAM, 2 h+) instead of
  once, and that route forces `-O2`. Not yet understood.
* `compiler_args_for` drops the runner's `--modular-flow-mac-threshold=10`
  (on graphs built mostly from torch ops it makes neuronx-cc fail codegen);
  `HY4_TENSORIZER_OPTIONS` (`run.py --tensorizer-options`) adds
  `--tensorizer-options`.
* The compile cache key does not cover this folder's source: after editing
  `src/`, clear `$VLLM_CACHE_ROOT` (or point it elsewhere), or the previous
  trace is compiled again.
* Rank 0 compiles while the other 63 wait in a gloo barrier whose default
  timeout is 3600 s; `run.py` and the DI scripts raise vLLM's
  `distributed_timeout_seconds` to 4 h.
* `HY4_DEBUG_CAPTURE=1` (with `HY4_DEBUG_LAYERS=0,1,...` and `run.py
  --capture-dir`) traces `capture_tensor` calls for those layers' attention and
  MLP into the graphs, for device-vs-CPU comparison. Off, nothing is traced.
