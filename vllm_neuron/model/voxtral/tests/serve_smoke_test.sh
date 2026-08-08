#!/bin/bash
# Quick manual smoke: start vllm serve on Voxtral, then curl /v1/audio/transcriptions.
# Requires: trn2.3xlarge instance with SDK 2.31 DLAMI, vllm-neuron 0.21 plugin
# container, ~30 min of time for cold compile.
#
# See ../README.md for the full launch procedure.

set -euo pipefail

MODEL="mistralai/Voxtral-Mini-3B-2507"
AUDIO="${1:-BillGates_2010_seg_0017.wav}"

export NEURON_SKIP_EFA_AFFINITY=1

# Additional-config JSON.
ADDITIONAL_CONFIG='{
  "neuron_config": {
    "quantization": "bf16",
    "on_device_sampling_config": {"all_greedy": "true"},
    "kv_segment_size_buckets": [1024],
    "num_batched_tokens_buckets": [1024],
    "num_seqs_buckets": [1]
  },
  "vision_neuron_config": {
    "num_vision_tokens_buckets": [375],
    "vision_attention_block_size": 384
  }
}'

# Launch serve.
vllm serve "$MODEL" \
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
    --additional-config "$ADDITIONAL_CONFIG" &
SERVE_PID=$!

# Wait for startup.
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "serve died before startup"
        exit 1
    fi
done

echo "serve up, sending audio"
curl -sf http://localhost:8000/v1/audio/transcriptions \
    -F "model=$MODEL" \
    -F "language=en" \
    -F "temperature=0.0" \
    -F "response_format=json" \
    -F "file=@$AUDIO"
echo

kill "$SERVE_PID" 2>/dev/null || true
