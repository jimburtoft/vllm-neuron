#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Gemma3-12B (google/gemma-3-12b-it) on one Trainium2 device (TP=4, LNC=2) with vLLM + vllm-neuron 0.29.
#
#   bash tools/gemma3/run.sh offline-2k    # wiki 2048 in / natural EOS <= 2048 out, max_model_len 4096, batch 83
#   bash tools/gemma3/run.sh offline-4k    # wiki 4096 in / natural EOS <= 4096 out, max_model_len 8192, batch 63
#   bash tools/gemma3/run.sh serve-2k      # vllm serve with the 2k settings on :8000 (serve-4k: the 4k settings)
#   bash tools/gemma3/run.sh online-2k     # serve-2k in the background + online_client.py at CONC (default "32 56 83")
#
# Inputs: PROMPTS_DIR (default tools/gemma3/data, shipped) holding wiki_2048.json / wiki_4096.json. To rebuild them:
# `python tools/gemma3/long_workload.py prompts-wiki --lengths 2048 4096 --out-prefix <dir>/wiki`.
# OUT (default ./out) receives the target JSONs and logs. MODEL_ID overrides the checkpoint.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?usage: run.sh offline-2k|offline-4k|serve-2k|serve-4k|online-2k|online-4k}
PROMPTS_DIR=${PROMPTS_DIR:-$HERE/data}
OUT=${OUT:-$PWD/out}; mkdir -p "$OUT"
export MODEL_ID=${MODEL_ID:-google/gemma-3-12b-it}

# ---- environment shared by every mode ----
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SKIP_EFA_AFFINITY=1          # trn2.3xlarge has no EFA device
export VLLM_ENABLE_V1_MULTIPROCESSING=0      # the setting all published numbers used
export VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD=1000000000000000  # partitioner off -> in-kernel d256 KV write
export VLLM_NEURON_SWA_REAL_ADMISSION=1     # admit by each request's real KV peak (SWA layers hold a window)
export VLLM_NEURON_SWA_EARLY_FREE=1         # free out-of-window SWA blocks after prefill
export VLLM_NEURON_SWA_RING=1               # SWA layers keep a fixed ring of window blocks per request
export VLLM_NEURON_SWA_CTX_BUCKETS=1        # decode context buckets also narrow the SWA tables
export VLLM_NEURON_SWA_ALIGN_TOKENS=1024
export GEMMA3_PACKED_PREFILL=1
export VLLM_NEURON_MAX_PREFILLS_PER_BATCH=1

case "$MODE" in
  *-2k) IN=2048; ML=4096; MNS=83; MNBT=2176; FRAC=0.55; BUCKETS="[2560,3072,3584]"; CONC_DEFAULT="32 56 83" ;;
  *-4k) IN=4096; ML=8192; MNS=63; MNBT=4224; FRAC=0.60; BUCKETS="[5120,6144,7168]"; CONC_DEFAULT="16 32 63" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=$FRAC
NC='{"decode_context_length_buckets":'$BUCKETS',"on_device_sampling_config":{"all_greedy":true},"max_logprobs":0}'

serve() {
  exec vllm serve "$MODEL_ID" --tensor-parallel-size 4 --dtype bfloat16 --max-model-len $ML \
    --max-num-seqs $MNS --max-num-batched-tokens $MNBT --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --additional-config "{\"neuron_config\":$NC}" --port ${PORT:-8000} "$@"
}

case "$MODE" in
  offline-*)
    python3 "$HERE/long_workload.py" target --prompts "$PROMPTS_DIR/wiki_$IN.json" --out-tokens $IN --tp 4 \
      --max-model-len $ML --max-num-seqs $MNS --max-num-batched-tokens $MNBT --batches $MNS --reps ${REPS:-3} \
      --neuron-config "$NC" --natural-eos --out-file "$OUT/target_wiki_$IN.json" 2>&1 | tee "$OUT/offline_$IN.log" ;;
  serve-*) serve ;;
  online-*)
    serve > "$OUT/server_$IN.log" 2>&1 &
    SP=$!
    trap 'kill $SP 2>/dev/null || true' EXIT
    until curl -sf http://127.0.0.1:${PORT:-8000}/v1/models >/dev/null; do
      kill -0 $SP 2>/dev/null || { echo "server exited; see $OUT/server_$IN.log"; exit 1; }; sleep 10
    done
    for c in ${CONC:-$CONC_DEFAULT}; do
      python3 "$HERE/online_client.py" --prompts "$PROMPTS_DIR/wiki_$IN.json" --max-tokens $IN --conc $c --waves 2 \
        --base-url http://127.0.0.1:${PORT:-8000} --out "$OUT/online_${IN}_c$c.json"
    done ;;
esac
