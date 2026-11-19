"""Task 036: convert the economize EAGLE-3 drafter for Ministral-3-8B to the vLLM EAGLE-3 draft folder.

Mirrors economize 0.4.0 ``backends/vllm/eagle3/export.py`` (Eagle3VllmExporter, format="eagle3") without
needing the base model loaded:
  * tensors ``eagle3.layer.*`` -> ``midlayer.*``; ``eagle3.{fc,norm,lm_head}.*`` -> same; ``d2t`` int64, ``t2d`` bool;
  * config: one-layer Llama with draft_vocab_size, target_hidden_size, eagle_aux_hidden_state_layer_ids
    (economize default (2, L // 2, L - 3) for L=34 -> (2, 17, 31), block outputs as vLLM reads them),
    the target's rope_parameters (economize config rope_theta=None -> the drafter uses the target's table).

  python convert_economize_eagle3.py <economize_dir> <target_config.json> <out_dir> [<target_dir_with_safetensors>]

  The vLLM-Neuron EAGLE-3 loader reads ``embed_tokens.weight`` from the draft checkpoint (it does not share the
  target's), so a target directory, when given, contributes a copy of the target embedding.
"""
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

src, tcfg, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
ecfg = json.load(open(os.path.join(src, "economize_config.json")))
tech = next(t for t in ecfg["techniques"] if t["technique"] == "eagle3")["config"]
t = json.load(open(tcfg))
text = t.get("text_config", t)
L = int(text["num_hidden_layers"])
aux = tech.get("aux_layers") or sorted({min(max(x, 0), L) for x in (2, L // 2, L - 3)})
assert tech["head"] == "own" and not tech["norm_before_residual"] and tech["rope_theta"] is None, tech

tensors = {}
with safe_open(os.path.join(src, "economize_model.safetensors"), "pt") as f:
    for k in f.keys():
        v = f.get_tensor(k)
        n = k.removeprefix("eagle3.")
        n = "midlayer." + n.removeprefix("layer.") if n.startswith("layer.") else n
        tensors[n] = v.contiguous()
H = int(text["hidden_size"])
q = tensors["midlayer.self_attn.q_proj.weight"]
k = tensors["midlayer.self_attn.k_proj.weight"]
head_dim = int(text.get("head_dim") or H // int(text["num_attention_heads"]))
nh, nkv = q.shape[0] // head_dim, k.shape[0] // head_dim
assert q.shape[1] == 2 * H and tensors["fc.weight"].shape == (H, len(aux) * H)
draft_vocab = tensors["lm_head.weight"].shape[0]
assert tensors["d2t"].shape[0] == draft_vocab and tensors["t2d"].shape[0] == int(text["vocab_size"])
assert int(tensors["t2d"].sum()) == draft_vocab

rp = dict(text["rope_parameters"])
rp.pop("type", None)
cfg = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "num_hidden_layers": 1,
    "hidden_size": H,
    "target_hidden_size": H,
    "intermediate_size": int(tensors["midlayer.mlp.gate_proj.weight"].shape[0]),
    "num_attention_heads": nh,
    "num_key_value_heads": nkv,
    "head_dim": head_dim,
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "rms_norm_eps": float(text["rms_norm_eps"]),
    "vocab_size": int(text["vocab_size"]),
    "draft_vocab_size": int(draft_vocab),
    "max_position_embeddings": int(text.get("max_position_embeddings") or 4096),
    "rope_parameters": rp,
    "tie_word_embeddings": False,
    "norm_before_residual": False,
    "use_aux_hidden_state": True,
    "eagle_aux_hidden_state_layer_ids": [int(i) for i in aux],
    "torch_dtype": "bfloat16",
}
for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
    if t.get(key, text.get(key)) is not None:
        cfg[key] = t.get(key, text.get(key))
if len(sys.argv) > 4:
    tdir = sys.argv[4]
    idx = json.load(open(os.path.join(tdir, "model.safetensors.index.json")))["weight_map"]
    key = next(k for k in idx if k.endswith("embed_tokens.weight") and "vision" not in k)
    with safe_open(os.path.join(tdir, idx[key]), "pt") as f:
        tensors["embed_tokens.weight"] = f.get_tensor(key).contiguous()
save_file(tensors, os.path.join(out, "model.safetensors"))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
print("aux", aux, "heads", nh, nkv, head_dim, "draft_vocab", draft_vocab, "tensors", len(tensors))
