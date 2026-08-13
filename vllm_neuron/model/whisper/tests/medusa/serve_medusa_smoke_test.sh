#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve smoke-test for native-plugin Whisper large-v3 WITH Medusa spec-decode.
#
# This brings up `vllm serve` with Medusa enabled at TP=4 and curls one clip
# through /v1/audio/transcriptions -- the REAL customer entry point (the offline
# medusa_bench.py / medusa_correctness_gate.py are the mechanism/measurement
# tools; this is the served path). See ../../MEDUSA.md for the full config
# surface and the speedup-vs-acceptance model.
#
# Run inside the target container on a trn2.3xlarge.
#
# Usage:
#   ./serve_medusa_smoke_test.sh                         # launch server, then curl in another shell
#   CLIP=/path/clip.flac ./serve_medusa_smoke_test.sh --curl-only
#
# Heads source is controlled by MEDUSA_INIT (zero|random) or MEDUSA_HEADS_PATH:
#   MEDUSA_INIT=zero                 ./serve_medusa_smoke_test.sh   # placeholder (default)
#   MEDUSA_HEADS_PATH=/path/heads.pt ./serve_medusa_smoke_test.sh   # your trained heads
set -euo pipefail

MODEL="${MODEL:-openai/whisper-large-v3}"
PORT="${PORT:-8000}"
CLIP="${CLIP:-/large/work/ref/jfk.flac}"
K="${K:-5}"
URL="http://localhost:${PORT}/v1/audio/transcriptions"

if [[ "${1:-}" == "--curl-only" ]]; then
  echo "curl ${URL}  file=${CLIP}"
  curl -s -F "file=@${CLIP}" -F "model=${MODEL}" \
       -F "language=en" -F "temperature=0" -F "response_format=json" \
       "${URL}"
  echo
  exit 0
fi

# TP=4 is the architectural maximum for whisper-large-v3 (20 heads). LNC=2 -> 4 cores.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NEURON_RT_NUM_CORES=4

# Heads source: MEDUSA_HEADS_PATH wins; else MEDUSA_INIT (default zero placeholder).
if [[ -n "${MEDUSA_HEADS_PATH:-}" ]]; then
  MEDUSA_CFG="\"medusa_config\": {\"init\": \"load\", \"heads_path\": \"${MEDUSA_HEADS_PATH}\"}"
else
  MEDUSA_CFG="\"medusa_config\": {\"init\": \"${MEDUSA_INIT:-zero}\"}"
fi

echo "launching vllm serve ${MODEL} + Medusa (K=${K}) at TP=4 on port ${PORT} ..."
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
  --speculative-config "{\"method\":\"medusa\",\"num_speculative_tokens\":${K}}" \
  --additional-config "{\"neuron_config\": {\"on_device_sampling_config\": {\"all_greedy\": true}}, \"vision_neuron_config\": {\"num_vision_tokens_buckets\": [1500], \"vision_attention_block_size\": 1500}, ${MEDUSA_CFG}}"

# Once "Application startup complete" appears, in another shell:
#   ./serve_medusa_smoke_test.sh --curl-only
# expected (jfk.flac): {"text":" he hoped there would be stew for dinner ...", ...}
