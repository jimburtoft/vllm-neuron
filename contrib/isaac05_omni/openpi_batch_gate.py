"""accuracy under batching. C clients send the 3 reference observations WITH the reference initial noise at the
same time (so the server batches them); each returned chunk is compared with the reference implementation.
usage: python openpi_batch_gate.py <streams_dir> C"""
import asyncio, os, sys
import numpy as np
C = int(sys.argv[2])
os.environ.setdefault("U8", "1")
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open(os.path.expanduser("~/openpi_gate.py")).read().split("async def main():")[0])
import msgspec, websockets  # noqa: E402


async def one(i, res, go):
    k = i % 3
    o, ref = obs_of(k)
    async with websockets.connect(URL, max_size=None, ping_interval=None, open_timeout=600) as ws:
        await ws.recv()
        await go.wait()
        await ws.send(msgspec.msgpack.encode(o, enc_hook=enc_hook))
        out = dec(msgspec.msgpack.decode(await ws.recv()))
        a = np.asarray(out if isinstance(out, np.ndarray) else out.get("actions"), dtype=np.float32)
        n = ref.shape[0]
        res.append((i, k, float(np.dot(a[:n].ravel(), ref.ravel()) / (np.linalg.norm(a[:n]) * np.linalg.norm(ref)))))


async def main():
    res, go = [], asyncio.Event()
    ts = [asyncio.create_task(one(i, res, go)) for i in range(C)]
    await asyncio.sleep(3.0)
    go.set()
    await asyncio.gather(*ts)
    cs = [c for _, _, c in res]
    per = {k: [round(c, 5) for _, kk, c in sorted(res) if kk == k] for k in range(3)}
    print(f"BATCHGATE C={C} min_cos={min(cs):.5f} per_obs={per}", flush=True)

asyncio.run(main())
