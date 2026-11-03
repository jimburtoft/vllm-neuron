# SPDX-License-Identifier: Apache-2.0
"""GSM8K subset eval (greedy, chat template, thinking on). Backends: vllm (offline) or hf (CPU batched).

python gsm8k_eval.py --backend vllm|hf --model PATH --n 100 --max-tokens 4096 --out out.json [--tp 4]
Score = last number in the text after </think> equals the gold answer.
"""
import argparse, json, re, time

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=4); ap.add_argument("--ep", action="store_true")
ap.add_argument("--max-num-seqs", type=int, default=32); ap.add_argument("--hf-batch", type=int, default=16)
ap.add_argument("--mnbt", type=int, default=1024); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--max-model-len", type=int, default=8192)
a = ap.parse_args()

from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(a.n))
SUFFIX = "\nPut the final numeric answer on the last line."
qs = [r["question"] + SUFFIX for r in ds]
gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in ds]


def extract(t):
    t = t.split("</think>")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


t0 = time.time()
if a.backend == "vllm":
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enable_expert_parallel=a.ep, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
              enable_prefix_caching=False, max_num_batched_tokens=a.mnbt)
    tok = llm.get_tokenizer()
    ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    t0 = time.time()
    outs = llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
    texts = [o.outputs[0].text for o in outs]
    ntok = [len(o.outputs[0].token_ids) for o in outs]
else:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).eval()
    texts, ntok = [], []
    for i in range(0, len(qs), a.hf_batch):
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs[i:i + a.hf_batch]]
        enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=a.max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in g[:, enc.input_ids.shape[1]:]:
            ids = row.tolist()
            if 28 in ids: ids = ids[: ids.index(28) + 1]
            texts.append(tok.decode(ids, skip_special_tokens=True)); ntok.append(len(ids))
        print(f"{i + len(ps)}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
preds = [extract(t) for t in texts]
correct = [p is not None and abs(float(p) - float(g)) < 1e-6 for p, g in zip(preds, gold)]
res = {"backend": a.backend, "n": a.n, "acc": sum(correct) / len(correct), "wall_s": time.time() - t0,
       "mean_out_tokens": sum(ntok) / len(ntok), "truncated": sum(n >= a.max_tokens for n in ntok),
       "rows": [{"pred": p, "gold": g, "ok": c, "ntok": n, "text": t} for p, g, c, n, t in zip(preds, gold, correct, ntok, texts)]}
json.dump(res, open(a.out, "w"), indent=1)
print(f"GSM8K backend={a.backend} n={a.n} acc={res['acc']:.3f} mean_out={res['mean_out_tokens']:.0f} truncated={res['truncated']} wall={res['wall_s']:.0f}s")
