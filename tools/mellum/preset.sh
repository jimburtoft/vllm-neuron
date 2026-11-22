#!/bin/bash
# Mellum2.1 serving presets for one trn2 device (TP=4, EP=4, LNC=2), vllm-neuron branch mellum-trn2-0.29.
#
#   source tools/mellum/preset.sh <latency|mixed|throughput|long>
#   vllm serve JetBrains/Mellum2.1-12B-A2.5B-Thinking $MELLUM_SERVE_ARGS --additional-config "$MELLUM_ADDITIONAL_CONFIG"
#
# latency     1-16 concurrent requests, short prompts. No expert parallelism + the FP8 routed-expert decode kernel
#             (128/1k: 177 tok/s at c1, 934 at c16). TP=4 without EP.
# mixed       16-256 concurrent requests. EP + expert-major FP8 decode kernel with a bf16 copy for prefill
#             (128/1k: 1,228 at c16, 3,665 at c64, 4,294 at c256). KV budget 0.32, up to 256 sequences.
# throughput  256-512 concurrent requests. EP + one FP8 expert copy for prefill and decode (needs
#             jimburtoft/nki-library@mellum-trn2-2.33): 5,748 tok/s at 512 (128/1k), 4,671 at c256, 3,799 at c64. KV 0.42.
# long        prompts up to ~48k tokens. Segmented prefill (4,096-token chunks), ring off, bf16 experts.
#
# mixed and throughput use 8 prompts per prefill step (packed prefill; whole-prompt prefill only). Decode context
# buckets: [1024, 2048, 4096] for throughput (measured with FP8-only), [2048] for latency / mixed (measured there).
# Accuracy notes: FP8 presets score GSM8K-100 96-97 % (bf16 97-98 %); see tools/mellum/README.md.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_here/env.sh"

_p="${1:-throughput}"
unset MELLUM_MOE_EM MELLUM_MOE_FP8_ONLY MELLUM_MOE_FP8_PAD_I MELLUM_MOE_FP8_ROUTED VLLM_NEURON_MAX_PREFILLS_PER_BATCH \
      VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION
_buckets='[1024, 2048, 4096]'
case "$_p" in
  latency)
    export MELLUM_MOE_FP8_ROUTED=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.2  # measured without packed prefill
    MELLUM_SERVE_ARGS="--tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 --no-enable-prefix-caching"
    _buckets='[2048]'
    ;;
  mixed)
    export MELLUM_MOE_EM=1 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.32 VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8
    MELLUM_SERVE_ARGS="--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 8192 --max-num-seqs 256 --max-num-batched-tokens 8192 --no-enable-prefix-caching"
    _buckets='[2048]'
    ;;
  throughput)
    python - <<'PY' 2>/dev/null || echo "preset.sh: WARNING: the patched nkilib is not installed (pip install --no-deps --force-reinstall git+https://github.com/jimburtoft/nki-library.git@mellum-trn2-2.33)" >&2
import inspect, nkilib.core.moe.moe_cte.bwmm_shard_on_I as b
assert "_gu_coalesce_tail" in inspect.getsource(b)
PY
    export MELLUM_MOE_EM=1 MELLUM_MOE_FP8_ONLY=1 MELLUM_MOE_FP8_PAD_I=0 VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.42 \
           VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8
    MELLUM_SERVE_ARGS="--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 8192 --max-num-seqs 512 --max-num-batched-tokens 8192 --no-enable-prefix-caching"
    ;;
  long)
    export VLLM_NEURON_SWA_RING=0
    MELLUM_SERVE_ARGS="--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 49152 --max-num-seqs 16 --max-num-batched-tokens 4096 --no-enable-prefix-caching"
    _buckets='[2048, 4096, 8192, 16384, 32768]'
    ;;
  *) echo "preset.sh: unknown preset '$_p' (latency | mixed | throughput | long)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export MELLUM_SERVE_ARGS
export MELLUM_ADDITIONAL_CONFIG="{\"neuron_config\": {\"decode_context_length_buckets\": $_buckets}}"
echo "Mellum preset '$_p': $MELLUM_SERVE_ARGS --additional-config '$MELLUM_ADDITIONAL_CONFIG'" >&2
