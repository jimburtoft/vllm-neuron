"""split one OpenPI plan into client->worker, forward(), worker->client using absolute timestamps
(client and server on the same host). Reads the matching rank-0 timing line from ~/omni_serve.log."""
import numpy as np
import asyncio, ast, os, re, sys, time, msgspec, websockets
sys.argv += [os.path.expanduser("~/streams")]
exec(open(os.path.expanduser("~/openpi_gate.py")).read().split("async def main():")[0])
async def one():
    rows = []
    async with websockets.connect(URL, max_size=None) as ws:
        await ws.recv(); o, _ = obs_of(0); o.pop("noise")
        if os.environ.get("U8", "0") == "1":
            for k in list(o):
                if k.startswith("observation.images."):
                    o[k] = np.clip(o[k] * 255.0, 0, 255).astype(np.uint8)
        for _ in range(8):
            ts = time.time(); await ws.send(msgspec.msgpack.encode(o, enc_hook=enc_hook)); await ws.recv(); tr = time.time()
            rows.append((ts, tr))
            await asyncio.sleep(0.3)
    log = open(os.path.expanduser("~/omni_serve.log")).read().splitlines()
    tl = [ast.literal_eval(re.search(r"\{.*\}", l).group(0)) for l in log if "timing_ms" in l and "t_enter" in l]
    for ts, tr in rows[2:]:
        m = [d for d in tl if ts <= d["t_enter"] <= tr]
        if not m:
            continue
        e = min(d["t_enter"] for d in m); x = max(d["t_exit"] for d in m)
        print(f"rtt {1e3*(tr-ts):6.1f}  send->enter {1e3*(e-ts):5.1f}  forward(max rank) {1e3*(x-e):6.1f}  exit->recv {1e3*(tr-x):5.1f}  ranks {len(m)}")
asyncio.run(one())
