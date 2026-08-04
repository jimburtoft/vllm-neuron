#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve smoke-test for the native-plugin Whisper large-v3.
#
# This is a documented MANUAL step (a full automated on-device serve test is too
# heavy for CI). It (1) launches `vllm serve` with the exact validated config and
# (2) curls the /v1/audio/transcriptions endpoint for one clip.
#
# Run inside the target container on a trn2.3xlarge. Set CLIP to a 16 kHz mono
# WAV. The automated correctness gate is test_whisper_correctness.py (which hits
# an already-running server); this script is the quickest way to bring that
# server up and eyeball a single transcript.
#
# Usage:
#   ./serve_smoke_test.sh            # launch server, then in another shell curl
#   CLIP=/path/clip.wav ./serve_smoke_test.sh --curl-only   # just curl a clip
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3}"
PORT="${PORT:-8000}"
CLIP="${CLIP:-/large/work/ref/wav/libri_1.wav}"
URL="http://localhost:${PORT}/v1/audio/transcriptions"

if [[ "${1:-}" == "--curl-only" ]]; then
  echo "curl ${URL}  file=${CLIP}"
  curl -s -F "file=@${CLIP}" -F "model=${MODEL}" \
       -F "language=en" -F "temperature=0" -F "response_format=json" \
       "${URL}"
  echo
  exit 0
fi

# TP=4 is the architectural maximum for whisper-large-v3 (20 heads, 20 % 8 != 0).
# LNC=2 -> 4 logical cores -> TP=4 max.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NEURON_RT_NUM_CORES=4

# NOTE: audio decoding needs vllm[audio] in the container:
#   pip install soundfile librosa   # then restart the server

echo "launching vllm serve ${MODEL} at TP=4 on port ${PORT} ..."
vllm serve "${MODEL}" \
  --tokenizer "${MODEL}" \
  --tensor-parallel-size 4 \
  --max-model-len 448 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1536 \
  --dtype bfloat16 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --port "${PORT}" \
  --additional-config '{"neuron_config": {"on_device_sampling_config": {"all_greedy": true}}, "vision_neuron_config": {"num_vision_tokens_buckets": [1500], "vision_attention_block_size": 1500}}'

# Once "Application startup complete" appears, in another shell:
#   ./serve_smoke_test.sh --curl-only
# expected: {"text":" Nor is Mr. Quilter's manner less interesting than his matter.", ...}
