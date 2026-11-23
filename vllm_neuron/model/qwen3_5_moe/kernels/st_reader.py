#!/usr/bin/env python3
"""
Dependency-free safetensors reader.

The Native DLC image ships neither `safetensors` nor `transformers`, so the
weight loader reads .safetensors files directly. The format is simple:
  [8 bytes little-endian header length N][N bytes JSON header][raw tensor bytes]
The JSON header maps name -> {dtype, shape, data_offsets:[begin,end]} relative
to the start of the byte buffer (right after the header).

The file is mmapped and each `get_tensor` call copies that tensor's bytes into
its own buffer, so returned tensors are independent of the file (safe to mutate,
and still valid after `close()`).
"""
import json
import mmap
import os
import struct

import torch

_DT = {
    "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    # Keep FP8 tensors as raw bytes. The Native DLC may not expose a PyTorch
    # FP8 dtype, and Trn2 needs legacy E4M3 rather than safetensors' E4M3FN.
    "F8_E4M3": torch.uint8,
}


class SafeReader:
    """Lazy reader over one .safetensors file. Keeps the file mmapped."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "rb")
        n = struct.unpack("<Q", self._f.read(8))[0]
        self.header = json.loads(self._f.read(n).decode("utf-8"))
        self._data_start = 8 + n
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def get_dtype(self, name):
        """Return the safetensors dtype tag without translating it to torch."""
        return self.header[name]["dtype"]

    def get_tensor(self, name):
        meta = self.header[name]
        dtype_tag = meta["dtype"]
        if dtype_tag not in _DT:
            raise TypeError(f"unsupported safetensors dtype {dtype_tag!r} for {name!r}")
        dt = _DT[dtype_tag]
        b0, b1 = meta["data_offsets"]
        raw = self._mm[self._data_start + b0:self._data_start + b1]
        # Copy out of the read-only mmap: frombuffer needs a writable buffer.
        buf = bytearray(raw)
        t = torch.frombuffer(buf, dtype=dt)
        return t.reshape(meta["shape"])

    def close(self):
        try:
            self._mm.close(); self._f.close()
        except Exception:
            pass


def build_weight_map(ckpt):
    """Return {key: filename} across all .safetensors in ckpt (index or scan)."""
    import glob
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    wm = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        r = SafeReader(f)
        for k in r.keys():
            wm[k] = os.path.basename(f)
        r.close()
    return wm
