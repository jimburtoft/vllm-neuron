#!/bin/bash
# Serve Isaac-0.5 as a text (or VL) model with vLLM-Neuron 0.29 on one trn2 device (TP=4, LNC=2).
# usage: MODEL=<vllm view dir from make_vllm_dir.py> [PROFILE=t2k|t8k|img] serve_text.sh
# Profiles are the tuned configurations measured in contrib/isaac05_omni/README.md.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
export PYTHONPATH=${VLLM_NEURON_DIR:-$HOME/vllm-neuron} NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1
export QWEN35_DECODE_MK=1 QWEN35_NULL_ROUTING=1 VLLM_NEURON_KV_ADMIT=tokens VLLM_NEURON_MIN_KV_BUDGET_GIB=0.5
PROFILE=${PROFILE:-t2k}
NC='"num_batched_tokens_buckets": [128, 256, 512, 1024, 2048], "on_device_sampling_config": {"all_greedy": true}'
MMARGS=()
case $PROFILE in
  t2k) MML=2048; MNS=128; export VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8
       NC="$NC, \"num_seqs_buckets\": [1, 4, 8, 16, 32, 64, 128], \"decode_context_length_buckets\": [512, 1024]"
       AC="{\"neuron_config\": {$NC}}"; GMU=0.9 ;;
  t8k) MML=8192; MNS=64; export VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8 QWEN35_MK_FLASH=auto QWEN35_MK_FLASH_G=8
       NC="$NC, \"num_seqs_buckets\": [1, 4, 8, 16, 32, 64], \"decode_context_length_buckets\": [512, 2048]"
       AC="{\"neuron_config\": {$NC}}"; GMU=0.9 ;;
  img) MML=4096; MNS=128; export VLLM_NEURON_MAX_PREFILLS_PER_BATCH=4 QWEN35_MK_FLASH=auto QWEN35_MK_FLASH_G=8
       NC="$NC, \"num_seqs_buckets\": [1, 8, 16, 32, 64, 128], \"decode_context_length_buckets\": [512, 2048]"
       AC="{\"neuron_config\": {$NC}, \"encoder_neuron_config\": {\"num_encoder_input_buckets\": [1152, 2304], \"encoder_cache_num_blocks\": 320, \"encoder_attention_block_size\": 576, \"tp_size\": 1}}"
       MMARGS=(--limit-mm-per-prompt '{"image":2,"video":0}' --mm-processor-cache-gb 0); GMU=0.85 ;;
esac
exec vllm serve "$MODEL" --served-model-name isaac05 --tensor-parallel-size 4 --max-model-len $MML \
  --max-num-batched-tokens 2048 --max-num-seqs $MNS --no-enable-prefix-caching --block-size 256 \
  --gpu-memory-utilization $GMU --additional-config "$AC" "${MMARGS[@]}" --port ${PORT:-8000}
