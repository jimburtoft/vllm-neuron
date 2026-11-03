# SPDX-License-Identifier: Apache-2.0
"""Offline throughput: N chat prompts (GSM8K questions, thinking on, natural EOS capped), batch = max_num_seqs.

python bench.py MODEL --n 128 --max-num-seqs 32 --max-tokens 4096 [--ep] [--reps 2]
"""
import argparse, json, time
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--n", type=int, default=128); ap.add_argument("--max-num-seqs", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--ep", action="store_true"); ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--max-model-len", type=int, default=4608); ap.add_argument("--nc", default="{}"); ap.add_argument("--out", default=""); ap.add_argument("--fixed", type=int, default=0, help="ignore EOS, generate exactly this many tokens")
a = ap.parse_args()
from datasets import load_dataset
from vllm import LLM, SamplingParams
qs = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))]
llm = LLM(model=a.model, tensor_parallel_size=4, enable_expert_parallel=a.ep, max_model_len=a.max_model_len,
          max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_model_len, enable_prefix_caching=False,
          additional_config={"neuron_config": json.loads(a.nc)})
tok = llm.get_tokenizer()
ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
res = []
for r in range(a.reps):
    t0 = time.time()
    sp = SamplingParams(temperature=0.0, max_tokens=a.fixed, ignore_eos=True) if a.fixed else SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    outs = llm.generate(ps, sp)
    dt = time.time() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs); n_in = sum(len(o.prompt_token_ids) for o in outs)
    res.append({"wall_s": dt, "out_tokens": n_out, "in_tokens": n_in, "out_tok_s": n_out / dt})
    print(f"BENCH rep {r}: {n_out} out tok / {dt:.1f}s = {n_out / dt:.1f} out tok/s (in {n_in}, mean out {n_out / len(outs):.0f})", flush=True)
if a.out:
    json.dump({"args": vars(a), "reps": res}, open(a.out, "w"), indent=1)
