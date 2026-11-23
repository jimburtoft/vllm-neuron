"""OpenPI websocket client for the Isaac-0.5 Neuron server: sends the 3 reference observations (same tensors the reference
implementation used, from stream_{k}.pkl) and compares the returned normalized chunk with the reference one when the server
is given the reference initial noise. Then times 20 plans at concurrency 1.

Usage: python openpi_gate.py <streams_dir> [port]
"""
import asyncio, glob, os, pickle, sys, time
import numpy as np
import torch
import msgspec
import websockets

ST = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
URL = f"ws://127.0.0.1:{PORT}/v1/realtime/robot/openpi"


def enc_hook(o):
    if isinstance(o, np.ndarray):
        return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": list(o.shape)}
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def dec(obj):
    if isinstance(obj, dict):
        if obj.get(b"__ndarray__") or obj.get("__ndarray__"):
            g = lambda k: obj.get(k.encode(), obj.get(k))
            return np.frombuffer(g("data"), dtype=np.dtype(g("dtype"))).reshape(g("shape"))
        return {(k.decode() if isinstance(k, bytes) else k): dec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dec(x) for x in obj]
    return obj


def obs_of(k):
    s = pickle.load(open(os.path.join(ST, f"stream_{k}.pkl"), "rb"))["obs"]
    o = {}
    for key, v in s.items():
        if torch.is_tensor(v):
            o[key] = v[0].numpy().astype(np.float32)
            if os.environ.get("U8", "0") == "1" and key.startswith("observation.images."):
                # camera-native uint8 frames; exactly LeRobot's coerce_rgb_uint8 of the float frame (x*255, truncate)
                o[key] = np.clip(o[key] * 255.0, 0, 255).astype(np.uint8)
        elif isinstance(v, list):
            o[key] = v[0]
        else:
            o[key] = v
    r = torch.load(os.path.join(ST, f"action_ref_{k}.pt"), weights_only=False)
    o["noise"] = r["capture"]["states"][0][0].numpy().astype(np.float32)  # reference initial (dim-masked) noise [50, 64]
    return o, r["full"].numpy()


async def main():
    async with websockets.connect(URL, max_size=None) as ws:
        meta = dec(msgspec.msgpack.decode(await ws.recv()))
        print("server metadata:", meta, flush=True)
        for k in range(3):
            o, ref = obs_of(k)
            t0 = time.time()
            await ws.send(msgspec.msgpack.encode(o, enc_hook=enc_hook))
            out = dec(msgspec.msgpack.decode(await ws.recv()))
            dt = time.time() - t0
            a = out if isinstance(out, np.ndarray) else out.get("actions")
            a = np.asarray(a, dtype=np.float32)
            n = ref.shape[0]
            cos = float(np.dot(a[:n].ravel(), ref.ravel()) / (np.linalg.norm(a[:n]) * np.linalg.norm(ref)))
            print(f"OPENPI obs {k}: actions {a.shape} cos vs reference {cos:.5f} max|d| {np.abs(a[:n] - ref).max():.4f} | {dt*1e3:.0f} ms", flush=True)
        o, _ = obs_of(0)
        o.pop("noise")
        ts = []
        for _ in range(20):
            t0 = time.time(); await ws.send(msgspec.msgpack.encode(o, enc_hook=enc_hook)); await ws.recv(); ts.append(time.time() - t0)
        ts.sort()
        print(f"OPENPI plan latency (end-to-end, c=1): p50 {ts[10]*1e3:.0f} ms p90 {ts[18]*1e3:.0f} ms min {ts[0]*1e3:.0f} ms", flush=True)


asyncio.run(main())
