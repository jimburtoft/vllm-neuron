# Mellum on one trn2 device (TP=4, LNC=2), vllm-neuron 0.29 branch mellum-trn2-0.29.
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_SKIP_EFA_AFFINITY=1
export VLLM_NEURON_SWA_ALIGN_TOKENS=${VLLM_NEURON_SWA_ALIGN_TOKENS:-1024}   # trimmed SWA tables a multiple of 256 at LNC=2
export VLLM_NEURON_SWA_RING=${VLLM_NEURON_SWA_RING:-1}
export VLLM_NEURON_SWA_REAL_ADMISSION=${VLLM_NEURON_SWA_REAL_ADMISSION:-1}
export VLLM_NEURON_SWA_EARLY_FREE=${VLLM_NEURON_SWA_EARLY_FREE:-1}
export VLLM_NEURON_SWA_CTX_BUCKETS=${VLLM_NEURON_SWA_CTX_BUCKETS:-1}
# Content-fingerprint the standalone NKI Library in the NKI trace cache. The cache only version-pins packages outside
# the kernel's own package, and installing nki-library (or a new commit of it) keeps every pinned version unchanged,
# so without this a trace of the stock kernels keeps being served.
_nkilib_src="$(python -c 'import importlib.util as u, os; s = u.find_spec("nkilib_src"); print(os.path.dirname(s.origin) if s and s.origin else "")' 2>/dev/null)"
[ -n "$_nkilib_src" ] && export NKI_TRACE_CACHE_ROOTS="${NKI_TRACE_CACHE_ROOTS:+$NKI_TRACE_CACHE_ROOTS:}$_nkilib_src"
unset _nkilib_src
