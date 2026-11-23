"""Make a vLLM-loadable view of the BF16 checkpoint: symlink the weights, and rewrite config.json and processor_config.json.

- config.json: model_type `qwen3_5_moe`, so vLLM's own Qwen3_5MoeConfig parses it. architectures = the given view:
  Qwen3_5MoeForCausalLM (text only, default) or Qwen3_5MoeForConditionalGeneration (vision-language). `auto_map` is
  dropped. `text_config.isaac05_moe` is kept; the plugin keys on it (null-expert router).
- processor_config.json: `auto_map` (the Isaac05Processor that adds tensor_stream) is dropped, so the stock Qwen3-VL processor is used.
Usage: python make_vllm_dir.py <bf16_dir> <vllm_dir> [text|vl]
"""

import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
view = sys.argv[3] if len(sys.argv) > 3 else "text"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f in ("config.json", "processor_config.json") or f.endswith(".py"):
        continue
    p = os.path.join(dst, f)
    if not os.path.lexists(p):
        os.symlink(os.path.join(src, f), p)
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["model_type"] = "qwen3_5_moe"
cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration" if view == "vl" else "Qwen3_5MoeForCausalLM"]
cfg.pop("auto_map", None)
cfg["vision_config"]["model_type"] = "qwen3_5_moe"
json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
pc = json.load(open(os.path.join(src, "processor_config.json")))
pc.pop("auto_map", None)
pc["processor_class"] = "Qwen3VLProcessor"
json.dump(pc, open(os.path.join(dst, "processor_config.json"), "w"), indent=2)
print("ok", dst)
