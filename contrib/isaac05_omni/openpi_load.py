"""OpenPI throughput sweep. C concurrent websocket clients (one session each), each sends observation
(k % 3) back-to-back for DUR seconds after WARM warmup plans. Reports plans/s aggregate, p50/p90 per-plan latency.
Client frames are uint8 (camera-native) unless U8=0.
Usage: python openpi_load.py <streams_dir> C [DUR] [WARM]"""
import asyncio, os, sys, time
import numpy as np
C = int(sys.argv[2]); DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0; WARM = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.environ.setdefault("U8", "1")
sys.argv = [sys.argv[0], sys.argv[1]]  # the gate header reads argv[2] as the PORT
exec(open(os.path.expanduser("~/openpi_gate.py")).read().split("async def main():")[0])
import msgspec, websockets  # noqa: E402

OBS = []
for k in range(3):
    o, _ = obs_of(k)
    o.pop("noise")
    OBS.append(msgspec.msgpack.encode(o, enc_hook=enc_hook))


async def client(i, t_start, t_end, lat, done, ready):
    async with websockets.connect(URL, max_size=None, ping_interval=None, open_timeout=600) as ws:
        await ws.recv()
        n = 0
        for w in range(WARM):
            await ws.send(OBS[(i + w) % 3]); await ws.recv()
        ready.append(i)
        await t_start.wait()
        while time.time() < t_end[0]:
            t0 = time.time()
            await ws.send(OBS[(i + n) % 3]); await ws.recv()
            t1 = time.time()
            if t1 <= t_end[0]:
                lat.append(t1 - t0); done[0] += 1
            n += 1


async def main():
    lat, done = [], [0]
    t_start = asyncio.Event(); t_end = [0.0]
    ready = []
    tasks = [asyncio.create_task(client(i, t_start, t_end, lat, done, ready)) for i in range(C)]
    while len(ready) < C:  # every client connected + warmed
        await asyncio.sleep(0.05)
    t0 = time.time(); t_end[0] = t0 + DUR; t_start.set()
    await asyncio.gather(*tasks)
    el = DUR
    lat_ms = np.array(lat) * 1e3 if lat else np.array([float("nan")])
    print(f"LOAD C={C} plans={done[0]} plans_per_s={done[0] / el:.2f} p50_ms={np.median(lat_ms):.0f} "
          f"p90_ms={np.percentile(lat_ms, 90):.0f} dur_s={el:.0f}", flush=True)

asyncio.run(main())
