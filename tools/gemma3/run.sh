#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Gemma3-12B (google/gemma-3-12b-it) on one Trainium2 device (TP=4, LNC=2) with vLLM + vllm-neuron 0.29.
#
#   bash tools/gemma3/run.sh offline-2k    # wiki 2048 in / natural EOS <= 2048 out, max_model_len 4096, batch 83
#   bash tools/gemma3/run.sh offline-4k    # wiki 4096 in / natural EOS <= 4096 out, max_model_len 8192, batch 63
#   bash tools/gemma3/run.sh serve-2k      # vllm serve with the 2k settings on :8000 (serve-4k: the 4k settings)
#   bash tools/gemma3/run.sh online-2k     # serve-2k in the background + online_client.py at CONC (default "32 56 83")
#   Long context (offline-/serve-/online-): l8 = 8192 in / <= 8192 out (ml 16384, batch 38), l16 = 16384 / <= 16384 (ml 32768, batch 20),
#   s8 = 8192 / 512 (ml 8704, batch 60), s16 = 16384 / 512 (ml 16896, batch 32). Build their prompts first (they need longer articles):
#   python tools/gemma3/long_workload.py prompts-wiki --lengths 8192 16384 --out-prefix <dir>/wiki ; PROMPTS_DIR=<dir> bash tools/gemma3/run.sh offline-l8
#   The 16K graphs need a lot of host RAM to compile (about 35-65 GB per compiler process); on a 128 GB host set PCW=1 and add swap.
#
# Inputs: PROMPTS_DIR (default tools/gemma3/data, shipped) holding wiki_2048.json / wiki_4096.json. To rebuild them:
# `python tools/gemma3/long_workload.py prompts-wiki --lengths 2048 4096 --out-prefix <dir>/wiki`.
# OUT (default ./out) receives the target JSONs and logs. MODEL_ID overrides the checkpoint.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?usage: run.sh offline-|serve-|online- + 2k|4k|l8|l16|s8|s16}
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

# NSB (num_seqs_buckets): the long-context shapes compile a single decode batch size.
NSB=""; NBT=""
case "$MODE" in
  *-2k) IN=2048; OUTN=2048; ML=4096; MNS=83; MNBT=2176; FRAC=0.55; BUCKETS="[2560,3072,3584]"; CONC_DEFAULT="32 56 83" ;;
  *-4k) IN=4096; OUTN=4096; ML=8192; MNS=63; MNBT=4224; FRAC=0.60; BUCKETS="[5120,6144,7168]"; CONC_DEFAULT="16 32 63" ;;
  *-l8) IN=8192; OUTN=8192; ML=16384; MNS=37; MNBT=8320; FRAC=0.58; BUCKETS="[9216,10240,11264,12288,13312,14336,15360]"; NSB="[37]"; NBT="[8192,8320]"; CONC_DEFAULT="8 16 37" ;;
  *-l16) IN=16384; OUTN=16384; ML=32768; MNS=20; MNBT=16512; FRAC=0.55; BUCKETS="[24576]"; NSB="[20]"; NBT="[16512]"; CONC_DEFAULT="4 8 20" ;;
  *-s8) IN=8192; OUTN=512; ML=8704; MNS=60; MNBT=8320; FRAC=0.60; BUCKETS=""; NSB="[1,8,16,32,60]"; NBT="[8192,8320]"; CONC_DEFAULT="8 16 60" ;;
  *-s16) IN=16384; OUTN=512; ML=16896; MNS=32; MNBT=16512; FRAC=0.50; BUCKETS=""; NSB="[32]"; NBT="[16512]"; CONC_DEFAULT="4 8 32" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=$FRAC
SFX=$IN; [ "$OUTN" != "$IN" ] && SFX=${IN}_$OUTN   # output file suffix (2k/4k keep their old names)
if [ -n "${PCW:-}" ]; then export NEURON_PARALLEL_COMPILE_WORKERS=$PCW NEURON_LIBTORCH_PARALLEL_TRACE_WORKERS=$PCW; fi
NC='{"on_device_sampling_config":{"all_greedy":true},"max_logprobs":0'
[ -n "$BUCKETS" ] && NC="$NC"',"decode_context_length_buckets":'$BUCKETS
[ -n "$NSB" ] && NC="$NC"',"num_batched_tokens_buckets":'$NBT',"num_seqs_buckets":'$NSB
NC="$NC}"

serve() {
  exec vllm serve "$MODEL_ID" --tensor-parallel-size 4 --dtype bfloat16 --max-model-len $ML \
    --max-num-seqs $MNS --max-num-batched-tokens $MNBT --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --additional-config "{\"neuron_config\":$NC}" --port ${PORT:-8000} "$@"
}

case "$MODE" in
  offline-*)
    python3 "$HERE/long_workload.py" target --prompts "$PROMPTS_DIR/wiki_$IN.json" --out-tokens $OUTN --tp 4 \
      --max-model-len $ML --max-num-seqs $MNS --max-num-batched-tokens $MNBT --batches $MNS --reps ${REPS:-3} \
      --neuron-config "$NC" --natural-eos --out-file "$OUT/target_wiki_$SFX.json" 2>&1 | tee "$OUT/offline_$SFX.log" ;;
  serve-*) serve ;;
  online-*)
    serve > "$OUT/server_$SFX.log" 2>&1 &
    SP=$!
    trap 'kill $SP 2>/dev/null || true' EXIT
    until curl -sf http://127.0.0.1:${PORT:-8000}/v1/models >/dev/null; do
      kill -0 $SP 2>/dev/null || { echo "server exited; see $OUT/server_$SFX.log"; exit 1; }; sleep 10
    done
    for c in ${CONC:-$CONC_DEFAULT}; do
      python3 "$HERE/online_client.py" --prompts "$PROMPTS_DIR/wiki_$IN.json" --max-tokens $OUTN --conc $c --waves 2 \
        --base-url http://127.0.0.1:${PORT:-8000} --out "$OUT/online_${SFX}_c$c.json"
    done ;;
esac
