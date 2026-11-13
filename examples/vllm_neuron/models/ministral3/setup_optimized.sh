#!/usr/bin/env bash
# Set up the optimized Ministral 3 stack on a trn2 host (SDK 2.33 vLLM DLAMI venv or the 2.33 vLLM container).
#   vllm-neuron: this checkout (jimburtoft/vllm-neuron, branch ministral3-trn2-0.29)
#   nkilib:      jimburtoft/nki-library, branch qwen3-asr/2.33-champion (batched block-KV decode attention:
#                group stitch + dt4; overlays the SDK 2.33 nkilib)
# Usage:  source examples/vllm_neuron/models/ministral3/setup_optimized.sh [nkilib_dir]
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
NKI_DIR=${1:-$HERE/../nki-library}
if [ ! -d "$NKI_DIR/src/nkilib_src/nkilib" ]; then
  git clone -q -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git "$NKI_DIR"
fi
export PYTHONPATH="$NKI_DIR/src/nkilib_src:$HERE${PYTHONPATH:+:$PYTHONPATH}"
# The NKI trace-cache key does not hash nkilib source: keep caches private per nkilib revision.
REV=$(git -C "$NKI_DIR" rev-parse --short HEAD)
export NKI_TRACE_CACHE_URL=${NKI_TRACE_CACHE_URL:-$HOME/.cache/ministral3/$REV/nki_trace}
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-$HOME/.cache/ministral3/$REV/nki_compile}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/.cache/ministral3/$REV/vllm}
mkdir -p "$NKI_TRACE_CACHE_URL" "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"
# BF16 decode: both RMSNorms inside the attention / MLP kernels.
export VLLM_NEURON_FUSE_DECODE_NORMS=1
# Admit requests by their real KV peak (long-context server).
export VLLM_NEURON_SWA_REAL_ADMISSION=1
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=${VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION:-0.6}
python - <<PY
import nkilib, vllm_neuron
assert nkilib.__file__.startswith("$NKI_DIR"), f"nkilib resolves to {nkilib.__file__}"
assert vllm_neuron.__file__.startswith("$HERE"), f"vllm_neuron resolves to {vllm_neuron.__file__}"
import nkilib.core.attention.attention_tkg as a
assert hasattr(a, "_use_batch_group_active_kv_stitch") or "stitch" in open(a.__file__).read(), "champion attention_tkg missing"
print("nkilib", nkilib.__file__)
PY
set +e
