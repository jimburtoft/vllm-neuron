# SPDX-License-Identifier: Apache-2.0
"""FP32 first-divergence gate for greedy bf16 outputs (Gemma3 long-document workload).

For each prompt, find the first position where the target's greedy tokens differ from the HF
transformers fp32 greedy reference (compared up to and including the reference's first EOS). Feed
prompt + reference prefix to the fp32 HF model (CPU, eager attention) and read the logits at that
position. The divergence is accepted as a bf16 near-tie if
  (i)   |logit(ref token) - logit(target token)| < EQ_NEAR_TIE (default 0.35),
  (ii)  the fp32 argmax is the reference token, and
  (iii) the target token is in the fp32 top-5.
The run passes if every prompt is identical or a near-tie. The 0.35 threshold is calibrated from the
spread of first-divergence gaps across recompiles of one unchanged bf16 configuration (max 0.277).

Usage: python fp32_gate.py REF.json TARGET.json OUT.json
  REF.json:    {"prompt_token_ids": [...], "generated_token_ids": [...]}  (long_workload.py reference)
  TARGET.json: the same keys (long_workload.py target)
Needs ~50 GB RAM for the 12B model in fp32. EQ_THREADS sets the torch thread count.
"""
import hashlib, json, os, sys, time

import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")
NEAR_TIE = float(os.environ.get("EQ_NEAR_TIE", "0.35"))
EOS_IDS = (1, 106)
ref_path, tgt_path, out_path = sys.argv[1:4]
ref, tgt = json.load(open(ref_path)), json.load(open(tgt_path))
assert ref["prompt_token_ids"] == tgt["prompt_token_ids"], "target and reference use different prompts"
torch.set_num_threads(int(os.environ.get("EQ_THREADS", str(os.cpu_count()))))
tok = AutoTokenizer.from_pretrained(MODEL)

divs = []
for i, (r, t) in enumerate(zip(ref["generated_token_ids"], tgt["generated_token_ids"])):
    ref_eos = next((j for j, a in enumerate(r) if a in EOS_IDS), None)
    span = len(r) if ref_eos is None else ref_eos + 1
    first = next((j for j, (a, b) in enumerate(zip(r[:span], t[:span])) if a != b), None)
    divs.append({"prompt": i, "first_div": first, "ref_first_eos": ref_eos, "compared_span": span})

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
print("model loaded", round(time.time() - t0, 1), "s", flush=True)
for d in divs:
    if d["first_div"] is None:
        d["verdict"] = "identical"
        continue
    i, p = d["prompt"], d["first_div"]
    r_tok, t_tok = ref["generated_token_ids"][i][p], tgt["generated_token_ids"][i][p]
    ids = torch.tensor([ref["prompt_token_ids"][i] + ref["generated_token_ids"][i][:p]])
    ts = time.time()
    with torch.no_grad():
        lg = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    top = torch.topk(lg, 5)
    top_ids = [int(k) for k in top.indices]
    gap = abs(lg[r_tok].item() - lg[t_tok].item())
    d.update({
        "context_len": ids.shape[1], "ref_token": r_tok, "target_token": t_tok,
        "ref_token_text": tok.decode([r_tok]), "target_token_text": tok.decode([t_tok]),
        "fp32_gap": gap, "fp32_top5": [{"id": k, "text": tok.decode([k]), "logit": v.item()} for v, k in zip(top.values, top_ids)],
        "cond_gap": gap < NEAR_TIE, "cond_argmax_is_ref": top_ids[0] == r_tok, "cond_target_in_top5": t_tok in top_ids,
        "forward_s": round(time.time() - ts, 1),
    })
    d["verdict"] = "near_tie_pass" if (d["cond_gap"] and d["cond_argmax_is_ref"] and d["cond_target_in_top5"]) else "fail"
    print(json.dumps({k: d[k] for k in ("prompt", "first_div", "ref_token_text", "target_token_text", "fp32_gap", "verdict")},
                     ensure_ascii=False), flush=True)

gaps = [d["fp32_gap"] for d in divs if "fp32_gap" in d]
out = {"model": MODEL, "precision": "float32", "torch": torch.__version__, "transformers": transformers.__version__,
       "near_tie_threshold": NEAR_TIE, "reference_sha256": hashlib.sha256(open(ref_path, "rb").read()).hexdigest(),
       "target_sha256": hashlib.sha256(open(tgt_path, "rb").read()).hexdigest(), "per_prompt": divs,
       "first_divergences": [d["first_div"] for d in divs], "max_gap": max(gaps) if gaps else 0.0,
       "passed": all(d["verdict"] in ("identical", "near_tie_pass") for d in divs)}
json.dump(out, open(out_path, "w"), ensure_ascii=False, indent=1)
print("wrote", out_path, "passed", out["passed"], "max_gap", round(out["max_gap"], 4), "first_div", out["first_divergences"])
