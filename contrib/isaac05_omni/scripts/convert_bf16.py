"""Convert the Isaac-0.5 FP32 checkpoint to BF16 (same tensor names, same sharding layout, ~5 GB shards).

Usage: python convert_bf16.py <src_dir> <dst_dir>
All non-weight files (configs, code, tokenizer, manifests) are copied verbatim. config.json keeps
storage_dtype=float32 because Isaac05Config rejects anything else; the HF loader casts on load.
"""

import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for root, _, files in os.walk(src):
    rel = os.path.relpath(root, src)
    if rel.startswith(".cache"):
        continue
    for f in files:
        if f.endswith(".safetensors") and rel == "." and f.startswith("model-"):
            continue
        if f == "model.safetensors.index.json":
            continue
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        shutil.copy2(os.path.join(root, f), os.path.join(dst, rel, f))

index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
new_map, total = {}, 0
LIMIT = 5 * 2**30
buf, buf_bytes, n_out = {}, 0, 0
out_names = []


def flush():
    global buf, buf_bytes, n_out
    if not buf:
        return
    n_out += 1
    name = f"model-bf16-{n_out:05d}.safetensors"
    save_file(buf, os.path.join(dst, name), metadata={"format": "pt"})
    for k in buf:
        new_map[k] = name
    out_names.append(name)
    print(f"wrote {name} ({buf_bytes / 2**30:.2f} GiB, {len(buf)} tensors)", flush=True)
    buf, buf_bytes = {}, 0


for s in shards:
    with safe_open(os.path.join(src, s), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.is_floating_point():
                t = t.to(torch.bfloat16)
            t = t.contiguous()
            nb = t.numel() * t.element_size()
            if buf_bytes + nb > LIMIT:
                flush()
            buf[k] = t
            buf_bytes += nb
            total += nb
flush()
json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
          open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
print(f"DONE total {total / 2**30:.2f} GiB in {n_out} shards, {len(new_map)} tensors", flush=True)
