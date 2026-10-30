# SPDX-License-Identifier: Apache-2.0
"""Long-document translation workload for Gemma3 on vLLM (Neuron or GPU).

Prompts: the first N English Wikipedia articles (``wikimedia/wikipedia`` 20231101.en, stream order,
skipping list/disambiguation pages), each wrapped in the model's chat template with the system prompt
"Translate the text below to Hindi." and token-trimmed so the FULL prompt is exactly IN tokens.
Outputs: greedy, natural EOS (``<eos>`` / ``<end_of_turn>``) capped at ``--out-tokens``; rows are cut
after the first EOS and right-padded with ``<eos>`` so every row has the same length.

  # 1. build the prompt files (needs `datasets`; downloads part of the Wikipedia dump)
  python long_workload.py prompts-wiki --lengths 2048 4096 --n 8 --out-prefix wiki
  # 2. vLLM run: correctness rows (the 8 prompts) + timed throughput windows of B requests
  python long_workload.py target --prompts wiki_2048.json --out-tokens 2048 --max-model-len 4096 \
      --max-num-seqs 83 --max-num-batched-tokens 2176 --batches 83 --natural-eos --out-file target.json
  # 3. fp32 HF reference on CPU (slow: ~15-60 min per row; use --rows to split across processes)
  python long_workload.py reference --prompts wiki_2048.json --out-tokens 2048 --natural-eos --out-file ref.json

Wikipedia text is CC BY-SA 4.0 (https://en.wikipedia.org/wiki/Wikipedia:Copyrights). The prompt files
record the article ids, titles and URLs that were used.
"""
import argparse, hashlib, json, os, sys, time

MODEL = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")
SYSTEM = "Translate the text below to Hindi."
EOS_IDS = [1, 106]  # Gemma3 generation_config.eos_token_id (<eos>, <end_of_turn>)
PAD_EOS = 1


def canonical_json(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(v):
    return hashlib.sha256(canonical_json(v).encode()).hexdigest()


def messages(txt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": txt}]


def eos_pad(gen, n):
    """Cut after the first EOS (kept), then right-pad with <eos> to exactly n tokens."""
    out = []
    for t in gen[:n]:
        out.append(int(t))
        if t in EOS_IDS:
            break
    return out + [PAD_EOS] * (n - len(out))


def chat_parts(tok):
    """Token IDs of the chat template before/after the user text (vLLM convention: add_special_tokens=True)."""
    mark = "XQZQX"
    t = tok.apply_chat_template(messages(mark), tokenize=False, add_generation_prompt=True)
    pre, post = t.split(mark)
    return tok(pre)["input_ids"], tok(post, add_special_tokens=False)["input_ids"]


def cmd_prompts_wiki(a):
    """The same articles are trimmed to each requested length. Prompt = pre_ids + text_ids[:k] + post_ids."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    pre, post = chat_parts(tok)
    need = max(a.lengths) - len(pre) - len(post)
    ds = load_dataset(a.dataset, a.config, split="train", streaming=True)
    picked = []
    for ex in ds:
        title, text = ex["title"], ex["text"]
        if title.startswith(("List of", "Lists of")) or "(disambiguation)" in title or len(text) < 4 * need:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < need + 64:
            continue
        picked.append({"id": ex["id"], "title": title, "url": ex.get("url"), "text_ids": ids})
        if len(picked) == a.n:
            break
    for L in a.lengths:
        k = L - len(pre) - len(post)
        rows = [pre + d["text_ids"][:k] + post for d in picked]
        assert all(len(r) == L for r in rows)
        out = {"input_tokens": L, "prompt_token_ids": rows, "digest": digest(rows),
               "source": {"dataset": a.dataset, "config": a.config, "split": "train", "license": "CC BY-SA 4.0",
                          "selection": "first N in stream order with >= need+64 text tokens, skipping List of / disambiguation",
                          "articles": [{"id": d["id"], "title": d["title"], "url": d["url"]} for d in picked]}}
        json.dump(out, open(f"{a.out_prefix}_{L}.json", "w"))
        print("prompts", L, "digest", out["digest"], [d["title"] for d in picked])


def cmd_reference(a):
    """HF transformers fp32 greedy reference on CPU (eager attention)."""
    import torch, transformers
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(int(os.environ.get("THREADS", str(os.cpu_count()))))
    rows = json.load(open(a.prompts))["prompt_token_ids"]
    sel = [int(x) for x in a.rows.split(",")] if a.rows else list(range(len(rows)))
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, a.dtype), attn_implementation="eager").eval()
    gen, t0 = {}, time.time()
    for idx in sel:
        with torch.no_grad():
            kw = dict(eos_token_id=EOS_IDS) if a.natural_eos else dict(eos_token_id=None, min_new_tokens=a.out_tokens)
            o = m.generate(torch.tensor([rows[idx]]), max_new_tokens=a.out_tokens, do_sample=False, pad_token_id=0, **kw)
        g = o[0, len(rows[idx]):].tolist()
        gen[str(idx)] = eos_pad(g, a.out_tokens) if a.natural_eos else g
        print("ref row", idx, f"{time.time() - t0:.0f}s", flush=True)
    out = {"model": MODEL, "precision": a.dtype, "torch": torch.__version__, "transformers": transformers.__version__,
           "python": sys.version.split()[0], "prompts_digest": digest(rows), "rows": gen}
    if not a.rows:
        out["prompt_token_ids"] = rows
        out["generated_token_ids"] = [gen[str(i)] for i in range(len(rows))]
    json.dump(out, open(a.out_file, "w"))
    print("reference written", a.out_file)


def cmd_target(a):
    """vLLM run. Correctness pass over the prompt set, then timed windows of B requests (prompts cycled)
    submitted at once; reports aggregate output tok/s per window."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = json.load(open(a.prompts))["prompt_token_ids"]
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs,
              max_num_batched_tokens=a.max_num_batched_tokens, enable_prefix_caching=False,
              gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tp)
    if a.neuron_config:
        kw["additional_config"] = {"neuron_config": json.loads(a.neuron_config)}
    t0 = time.time(); llm = LLM(**kw); build_s = time.time() - t0
    if a.natural_eos:
        sp = SamplingParams(temperature=0.0, max_tokens=a.out_tokens)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=a.out_tokens, min_tokens=a.out_tokens, ignore_eos=True)
    o = llm.generate([TokensPrompt(prompt_token_ids=r) for r in rows], sp, use_tqdm=False)
    gen = [list(x.outputs[0].token_ids) for x in o]
    if a.natural_eos:
        gen = [eos_pad(g, a.out_tokens) for g in gen]
    res = {"build_s": round(build_s, 1), "points": []}
    for b in a.batches:
        batch = [TokensPrompt(prompt_token_ids=rows[i % len(rows)]) for i in range(b)]
        llm.generate(batch[: min(b, 8)], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)
        for rep in range(a.reps):
            t = time.time(); oo = llm.generate(batch, sp, use_tqdm=False); dt = time.time() - t
            lens = [len(x.outputs[0].token_ids) for x in oo]
            pt = {"batch": b, "rep": rep, "wall_s": round(dt, 3), "out_tokens": sum(lens),
                  "out_tokens_per_req_min": min(lens), "out_tokens_per_req_max": max(lens),
                  "out_tok_s": round(sum(lens) / dt, 1), "in_tokens_per_req": len(rows[0]),
                  "out_tokens_cap": a.out_tokens, "natural_eos": bool(a.natural_eos)}
            res["points"].append(pt)
            print("POINT", json.dumps(pt), flush=True)
    out = {"model": MODEL, "prompt_token_ids": rows, "generated_token_ids": gen, "throughput": res,
           "config": {"tp": a.tp, "max_model_len": a.max_model_len, "max_num_seqs": a.max_num_seqs,
                      "max_num_batched_tokens": a.max_num_batched_tokens, "neuron_config": a.neuron_config},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("VLLM_NEURON_", "GEMMA3_", "NEURON_"))}}
    json.dump(out, open(a.out_file, "w"))
    print("target written", a.out_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="HF model id (default: $MODEL_ID or google/gemma-3-12b-it)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("prompts-wiki"); w.add_argument("--lengths", type=int, nargs="+", required=True)
    w.add_argument("--n", type=int, default=8); w.add_argument("--out-prefix", required=True)
    w.add_argument("--dataset", default="wikimedia/wikipedia"); w.add_argument("--config", default="20231101.en")
    r = sp.add_parser("reference"); r.add_argument("--prompts", required=True); r.add_argument("--out-tokens", type=int, required=True)
    r.add_argument("--dtype", default="float32"); r.add_argument("--out-file", required=True)
    r.add_argument("--natural-eos", action="store_true"); r.add_argument("--rows", default="")
    t = sp.add_parser("target"); t.add_argument("--prompts", required=True); t.add_argument("--out-tokens", type=int, required=True)
    t.add_argument("--tp", type=int, default=4); t.add_argument("--max-model-len", type=int, required=True)
    t.add_argument("--max-num-seqs", type=int, required=True); t.add_argument("--max-num-batched-tokens", type=int, required=True)
    t.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    t.add_argument("--batches", type=int, nargs="+", required=True); t.add_argument("--reps", type=int, default=2)
    t.add_argument("--neuron-config", default=""); t.add_argument("--out-file", required=True)
    t.add_argument("--natural-eos", action="store_true")
    a = ap.parse_args()
    if a.model:
        MODEL = a.model
    {"prompts-wiki": cmd_prompts_wiki, "reference": cmd_reference, "target": cmd_target}[a.cmd](a)
