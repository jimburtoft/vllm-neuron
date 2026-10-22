# SPDX-License-Identifier: Apache-2.0
"""Keep raw audio waveforms in fp32 through the multimodal processor.

``InputProcessingContext._postprocess_output`` casts every floating-point
processor output to the model dtype, the same way ``transformers.BatchFeature``
does. Most processors return features (mel spectrograms, pixel values), so the
cast is harmless. The upstream Voxtral processor returns the raw 16 kHz
waveform (``audio_arrays``), and the plugin computes the log-mel from it. With a
bf16 model the waveform is rounded to bf16 before the mel: the mel then differs
from the fp32 reference by about 0.8% relative L2 (max 0.24 absolute).

This patch skips the cast for the raw-waveform keys only. Every other output
is cast as before. Upstream fix: vllm-project/vllm#60352. Remove this patch
when the plugin's vLLM version includes it.
"""

import logging

logger = logging.getLogger(__name__)

# Processor outputs that hold raw waveforms, not features.
FP32_KEYS = ("audio_arrays",)

_applied = False


def apply_audio_fp32_patch() -> None:
    global _applied
    if _applied:
        return
    try:
        from vllm.multimodal.processing import context as _ctx
    except ImportError:
        logger.debug("audio fp32 patch: vllm.multimodal.processing.context not found; skipped")
        return
    cls = getattr(_ctx, "InputProcessingContext", None)
    if cls is None or not hasattr(cls, "_postprocess_output"):
        logger.debug("audio fp32 patch: InputProcessingContext._postprocess_output not found; skipped")
        return
    if getattr(cls, "_neuron_audio_fp32_patched", False):
        _applied = True
        return
    orig = cls._postprocess_output

    def _postprocess_output(self, output, *args, **kwargs):
        if isinstance(output, dict):
            keep = {k: output[k] for k in FP32_KEYS if k in output}
            if keep:
                rest = {k: v for k, v in output.items() if k not in keep}
                result = orig(self, rest, *args, **kwargs)
                for k, v in keep.items():
                    result[k] = v.cpu() if hasattr(v, "cpu") else v
                return result
        return orig(self, output, *args, **kwargs)

    cls._postprocess_output = _postprocess_output
    cls._neuron_audio_fp32_patched = True
    _applied = True
    logger.info("audio fp32 patch applied (keys: %s)", ", ".join(FP32_KEYS))
