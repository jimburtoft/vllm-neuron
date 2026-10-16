# SPDX-License-Identifier: Apache-2.0
"""Offline text generation example for MuseGlimmer (Muse-Glimmer-30B) on Neuron.

Text-only for now (the vision tower is Task 005). Runs on trn2.3xlarge at
TP=4 / LNC=2, or in vllm-neuron CPU mode (VLLM_NEURON_CPU_MODE=1) on a
truncated checkpoint.

Usage (trn2):
    NEURON_SKIP_EFA_AFFINITY=1 python examples/vllm_neuron/models/muse_glimmer/run.py \
        --model /home/ubuntu/models/Muse-Glimmer-30B --tp 4 --max-model-len 8192

Arguments may also come from the environment: MUSE_GLIMMER_MODEL, MUSE_GLIMMER_TP,
MUSE_GLIMMER_MAX_MODEL_LEN, MUSE_GLIMMER_MAX_NUM_SEQS.

Notes:
- ``--language-model-only`` (default on) sets ``mm_language_model_only`` at the
  TOP level of ``additional_config`` so no vision tower is built; pass
  ``--no-language-model-only`` to keep the multimodal runner path (text prompts
  still work; image/video inputs raise until Task 005).
- Prefix caching is off: APC needs segmented prefill, not validated yet.
- ``--prompt-token-ids`` skips the tokenizer (useful for tiny random checkpoints).
"""

import argparse
import os

os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")
os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "1200")

from vllm import LLM, SamplingParams  # noqa: E402

TEXT_PROMPTS = [
    "The capital of France is",
    "I am gonna keep counting forever, 1 2 3 4 5",
    "Once upon a time, there was a",
    "def fibonacci(n):",
]


def build_chat_prompts(model_path: str, prompts: list[str]) -> list[str]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if getattr(tok, "chat_template", None) is None:
        return prompts
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env("MUSE_GLIMMER_MODEL", "/home/ubuntu/models/Muse-Glimmer-30B"))
    ap.add_argument("--tp", type=int, default=int(env("MUSE_GLIMMER_TP", "4")))
    ap.add_argument("--max-model-len", type=int, default=int(env("MUSE_GLIMMER_MAX_MODEL_LEN", "8192")))
    ap.add_argument("--max-num-seqs", type=int, default=int(env("MUSE_GLIMMER_MAX_NUM_SEQS", "4")))
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="default: max_model_len (single-shot prefill). Use 512..8192 for chunked prefill.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chat", action="store_true", help="wrap prompts with the chat template")
    ap.add_argument("--prompt-token-ids", default="",
                    help="comma-separated token ids; skips the tokenizer")
    ap.add_argument("--language-model-only", dest="lm_only", action="store_true", default=True)
    ap.add_argument("--no-language-model-only", dest="lm_only", action="store_false")
    ap.add_argument("--num-gpu-blocks-override", type=int, default=0,
                    help="CPU mode needs a small value (e.g. 256)")
    args = ap.parse_args()

    mnbt = args.max_num_batched_tokens or args.max_model_len
    neuron_config = {
        "quantization": "bf16",
        "num_seqs_buckets": [args.max_num_seqs],
        "on_device_sampling_config": {"all_greedy": True},
    }
    if mnbt >= args.max_model_len:
        # single-shot prefill: a small bucket for short prompts + the full length
        neuron_config["num_batched_tokens_buckets"] = sorted({min(1024, mnbt), mnbt})
    else:
        # chunked (segmented) prefill: the bucket must equal the segment size
        neuron_config["num_batched_tokens_buckets"] = [mnbt]

    additional_config = {
        "neuron_config": neuron_config,
        "vision_neuron_config": {
            "vision_attention_block_size": 4096,
            "num_vision_tokens_buckets": [4096],
        },
    }
    if args.lm_only:
        additional_config["mm_language_model_only"] = True  # top level, not in vision_neuron_config

    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=mnbt,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        enable_prefix_caching=False,
        additional_config=additional_config,
    )
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        kwargs["enforce_eager"] = True
    if args.num_gpu_blocks_override:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.prompt_token_ids:
        kwargs["skip_tokenizer_init"] = True

    llm = LLM(**kwargs)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0,
                        detokenize=not args.prompt_token_ids)

    if args.prompt_token_ids:
        ids = [int(x) for x in args.prompt_token_ids.split(",") if x.strip()]
        out = llm.generate([{"prompt_token_ids": ids}], sp)
        print(f"Prompt ids: {ids}")
        print(f"Generated ids: {list(out[0].outputs[0].token_ids)}")
        return

    prompts = build_chat_prompts(args.model, TEXT_PROMPTS) if args.chat else TEXT_PROMPTS
    outputs = llm.generate(prompts, sp)
    for p, o in zip(TEXT_PROMPTS, outputs):
        print(f"Prompt: {p!r}")
        print(f"Generated: {o.outputs[0].text!r}")
        print()


if __name__ == "__main__":
    main()
