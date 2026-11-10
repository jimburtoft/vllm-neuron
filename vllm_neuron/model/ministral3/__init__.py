# SPDX-License-Identifier: Apache-2.0
"""Ministral 3 (``Mistral3ForConditionalGeneration``) text decoder on Llama 3.

Checkpoints: ``mistralai/Ministral-3-{3B,8B,14B}-Instruct-2512`` (and the
``-BF16`` / ``-Reasoning`` variants). The HuggingFace config is the multimodal
``mistral3`` wrapper: the text decoder (``ministral3``) is nested under
``text_config`` and a Pixtral vision tower under ``vision_config``.

The text decoder has the Llama architecture (GQA attention without bias,
RMSNorm, SiLU-gated MLP), with two differences:

* YaRN RoPE (``rope_parameters.rope_type == "yarn"``). The Llama 3 rotary
  implements it (:class:`~vllm_neuron.model.llama3.model.LlamaRotaryEmbedding`).
* Llama 4 query scaling: ``q *= 1 + beta * log(1 + floor(pos / original_max))``.
  The factor is exactly 1 for positions below ``original_max_position_embeddings``
  (16384). It is not implemented, so ``max_model_len`` above that is rejected.

With ``--language-model-only`` (or ``--limit-mm-per-prompt '{"image": 0}'``) only
the text decoder is built and the vision weights are not loaded. With images
enabled and an ``encoder_neuron_config``, ``multimodal.py`` adds the Pixtral
vision tower and the projector on the encoder TP / DP groups.
"""

from .factory import Ministral3ForCausalLM

__all__ = ["Ministral3ForCausalLM"]
