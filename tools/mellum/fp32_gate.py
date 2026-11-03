# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for Mellum on Neuron: fp32 log-probs at the first greedy divergence of each prompt.

Runs greedy generation with vLLM (or reads a results file) and compares each prompt's tokens with the fp32
Hugging Face reference in data/. At the first position where the tokens differ, the gap is
logprob_fp32(reference token) - logprob_fp32(target token). A divergence is a bf16 near-tie, and
passes, when the gap is below --max-gap and the target token is in the fp32 top 5. The Hugging Face
bf16 model on CPU reaches a gap of 0.49 on one of these prompts, so 0.35 is a strict threshold.

    python tools/mellum/fp32_gate.py --model JetBrains/Mellum2.1-12B-A2.5B-Thinking --enable-expert-parallel
    python tools/mellum/fp32_gate.py --results out.json          # gate a saved run
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def gate(refs, targets, max_gap):
    ok = True
    for i, (r, t) in enumerate(zip(refs, targets)):
        rg, tg = r["gen_ids"], t
        fd = next((j for j, (x, y) in enumerate(zip(rg, tg)) if x != y), None)
        if fd is None:
            print(f"prompt {i}: no divergence over {min(len(rg), len(tg))} tokens")
            continue
        ids, lps = r["top20_ids"][fd], r["top20_logprobs"][fd]
        ref_lp = lps[ids.index(rg[fd])]
        tgt_lp = lps[ids.index(tg[fd])] if tg[fd] in ids else float("-inf")
        gap = ref_lp - tgt_lp
        in_top5 = tg[fd] in ids[:5]
        row_ok = gap < max_gap and in_top5
        ok &= row_ok
        print(f"prompt {i}: first divergence at token {fd}, fp32 gap {gap:.4f}, target in fp32 top-5 {in_top5}"
              f" -> {'PASS' if row_ok else 'FAIL'}")
    print("GATE", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="JetBrains/Mellum2.1-12B-A2.5B-Thinking")
    ap.add_argument("--results", default="", help="JSON from a previous run (skip generation)")
    ap.add_argument("--save", default="", help="write the generated token ids here")
    ap.add_argument("--max-gap", type=float, default=0.35)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    a = ap.parse_args()
    sets = {k: json.load(open(os.path.join(HERE, "data", f"ref_fp32_{k}.json")))["rows"] for k in ("short", "long")}
    if a.results:
        out = json.load(open(a.results))
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(model=a.model, tensor_parallel_size=a.tensor_parallel_size,
                  enable_expert_parallel=a.enable_expert_parallel, max_model_len=8192, max_num_seqs=8,
                  max_num_batched_tokens=8192, enable_prefix_caching=False)
        out = {}
        for k, rows in sets.items():
            n = max(len(r["gen_ids"]) for r in rows)
            res = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in rows],
                               SamplingParams(temperature=0.0, max_tokens=n))
            out[k] = [list(o.outputs[0].token_ids) for o in res]
        if a.save:
            json.dump(out, open(a.save, "w"))
    ok = True
    for k, rows in sets.items():
        print(f"== {k} prompts")
        ok &= gate(rows, out[k], a.max_gap)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
