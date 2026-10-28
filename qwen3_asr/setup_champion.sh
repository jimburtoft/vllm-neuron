#!/usr/bin/env bash
# Set up the Qwen3-ASR champion stack on a trn2 host or inside a vllm-neuron 0.29 / SDK 2.33 environment.
#   vllm-neuron: this checkout (jimburtoft/vllm-neuron, branch qwen3-asr/0.29-champion)
#   nkilib:      jimburtoft/nki-library, branch qwen3-asr/2.33-champion (overlay on the SDK 2.33 nkilib)
# Usage:  source qwen3_asr/setup_champion.sh [nkilib_dir]
# Requires: an SDK 2.33 Python env with vllm 0.29.0, nki 0.7.0, neuronx-cc 2.28 (e.g. the DLAMI venv
#           /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0, or the 2.33 vLLM container).
set -e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NKI_DIR=${1:-$HERE/../nki-library}
if [ ! -d "$NKI_DIR/src/nkilib_src/nkilib" ]; then
  git clone -q -b qwen3-asr/2.33-champion https://github.com/jimburtoft/nki-library.git "$NKI_DIR"
fi
python - <<PY
import nki, sys
v = nki.__version__
if not v.startswith("0.7."):
    sys.exit(f"nki {v}: the champion nkilib is built against SDK 2.33 (nki 0.7.x)")
print("nki", v)
PY
# nkilib ships inside the nki wheel as top-level package "nkilib"; put the champion copy first on the path.
export PYTHONPATH="$HERE:$NKI_DIR/src/nkilib_src${PYTHONPATH:+:$PYTHONPATH}"
# A modified nkilib needs PRIVATE NKI caches: the NKI trace-cache key does not hash nkilib source, so a
# shared cache silently serves the stock kernels. Keep these per nkilib revision.
REV=$(git -C "$NKI_DIR" rev-parse --short HEAD)
export NKI_TRACE_CACHE_URL=${NKI_TRACE_CACHE_URL:-$HOME/.cache/qwen3-asr-champion/$REV/nki_trace}
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-$HOME/.cache/qwen3-asr-champion/$REV/nki_compile}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/.cache/qwen3-asr-champion/$REV/vllm}
mkdir -p "$NKI_TRACE_CACHE_URL" "$NKI_COMPILE_CACHE_URL" "$VLLM_CACHE_ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # required on trn2.3xlarge (12 vCPU)
# FP8 KV (opt-in at serve time: --kv-cache-dtype fp8_e4m3 + "fp8_packed_kv": true) needs this on trn2 so NKI maps
# torch.float8_e4m3fn to the trn2 fp8 type. It only affects FP8 tensors, so it is safe to set for bf16 KV as well.
export UNSAFE_FP8FNCAST=${UNSAFE_FP8FNCAST:-1}
# Turn off the runtime's low-latency completion polling. Its per-worker polling thread uses ~0.64 vCPU each, and the
# 12-vCPU trn2.3xlarge host was saturated at 8 replicas. Measured +4.1 % RTFx, TTFT p50 -65 %, WER unchanged; also
# better single stream (TP4 C=1: E2E p50 322 -> 315 ms, ITL p99 5.4 -> 3.65 ms), so keep it for latency serving too.
# The in-graph async decode reorder (+7.4 % RTFx, identical transcripts) is on by default in this branch;
# VLLM_NEURON_MM_ASYNC_REORDER=0 turns it off. The async prefill->decode reorder (+1.6 % RTFx, device ~99.8 % busy,
# identical transcripts at C=1) is also on by default; VLLM_NEURON_ASYNC_PREFILL_REORDER=0 turns it off.
# Serve 30 s clips with "num_batched_tokens_buckets": [128,256,416,512,1024] (a 30 s prompt is 403 tokens; 416 instead of
# 512 makes each prefill 11 % faster, +3.9 % RTFx). See QWEN3_ASR_FIXES.md "Recommended settings".
export NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=${NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION:-0}
python - <<PY
import nkilib, vllm_neuron
assert nkilib.__file__.startswith("$NKI_DIR"), f"nkilib resolves to {nkilib.__file__}, not the champion"
assert vllm_neuron.__file__.startswith("$HERE"), f"vllm_neuron resolves to {vllm_neuron.__file__}"
import nkilib.core.qkv.qkv_tkg as q, nkilib.core.mlp.mlp_tkg.mlp_tkg as m
assert "_qkv_projection_weight_stationary_b1" in dir(q) and "_use_b1_down_column_tiling" in dir(m), "champion kernels missing"
import nkilib.core.attention.attention_tkg as at
assert "fp8st" in at._CR17_FLAGS and "dt4" in at._CR17_FLAGS, f"batched-decode flags {at._CR17_FLAGS}"
print("champion OK: vllm_neuron", vllm_neuron.__file__, "| nkilib", nkilib.__file__, "| rev $REV")
PY
set +e
