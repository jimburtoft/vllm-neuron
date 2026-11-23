#!/bin/bash
# Serve Isaac-0.5 actions with stock vllm-omni 0.29.0rc1 + vllm-omni-neuron 0.29.0.0.2.0 (SDK 2.33 omni venv)
# plus this vllm-neuron branch (VLLM_NEURON_DIR on PYTHONPATH) and the isaac05_omni package + LeRobot/mharmony installed into ISAAC05_OVERLAY (pip --target).
# PROF=1: NRT device profiler on rank 0 (POST /start_profile, /stop_profile), small buffers.
ISAAC05_OMNI_DIR=${ISAAC05_OMNI_DIR:-$(cd "$(dirname "$0")" && pwd)}
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin:$PATH
export PYTHONPATH=${VLLM_NEURON_DIR:-$HOME/vllm-neuron}:${ISAAC05_OVERLAY:-$HOME/isaac05_overlay} NEURON_LOGICAL_NC_CONFIG=2 NEURON_SKIP_EFA_AFFINITY=1
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$HOME/.cache/isaac05_omni} QWEN35_DECODE_MK=0 ${XENV}
PROFARGS=()
if [ "${PROF:-0}" = 1 ]; then
  mkdir -p $HOME/prof_omni
  export NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=64 NEURON_RT_PROFILE_BUF_EVENT_MB=32 NEURON_RT_PROFILE_BUF_DMA_MB=64
  PROFARGS=(--profiler-config '{"profiler": "cuda"}' --additional-config "{\"neuron_profiler\": {\"output_dir\": \"$HOME/prof_omni\"}}")
fi
exec vllm serve ${ISAAC05_MODEL:-$HOME/models/Isaac-0.5-bf16} --omni --trust-remote-code --port 8000 \
  --deploy-config ${ISAAC05_OMNI_DIR}/examples/${DEPLOY:-isaac05_action_tp4.yaml} --init-timeout 7200 --stage-init-timeout 7200 \
  --disable-log-stats "${PROFARGS[@]}"
