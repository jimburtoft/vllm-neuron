# SPDX-License-Identifier: Apache-2.0
# Online client: sends the wiki prompts as token-id lists to /v1/completions (streaming, greedy, natural EOS).
#   python online_client.py --prompts wiki_2048.json --max-tokens 2048 --conc 64 128 256 --out online_wiki2048.json
import argparse, asyncio, json, os, statistics, time

import aiohttp

MODEL = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")  # must match the served model name


def pct(v):
    v = sorted(v)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]
    return {"p50": round(q(50), 2), "p95": round(q(95), 2), "p99": round(q(99), 2), "mean": round(statistics.mean(v), 2)}


async def one(s, url, ids, max_tokens, rec):
    payload = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; n = 0
    async with s.post(url, json=payload) as resp:
        async for line in resp.content:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            j = json.loads(data)
            if j.get("usage") and j["usage"].get("completion_tokens"):
                n = j["usage"]["completion_tokens"]
            ch = j.get("choices") or []
            if ch and first is None:
                first = time.perf_counter()
    t1 = time.perf_counter()
    rec.append({"ttft_ms": (first - t0) * 1e3 if first else None, "e2e_ms": (t1 - t0) * 1e3, "out_tok": n,
                "tpot_ms": ((t1 - first) * 1e3 / (n - 1)) if first and n > 1 else None})


async def level(url, conc, prompts, max_tokens):
    rec = []; it = iter(prompts)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=3600),
                                     connector=aiohttp.TCPConnector(limit=0)) as s:
        async def worker():
            for p in it:
                await one(s, url, p, max_tokens, rec)
        t = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(conc)])
        dt = time.perf_counter() - t
    return rec, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True); ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--conc", type=int, nargs="+", required=True); ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = json.load(open(a.prompts))["prompt_token_ids"]
    url = a.base_url.rstrip("/") + "/v1/completions"
    R = {"mode": "online", "endpoint": "/v1/completions", "prompts": a.prompts, "max_tokens": a.max_tokens, "points": []}
    asyncio.run(level(url, 8, rows, 32))  # warmup
    for c in a.conc:
        n = c * a.waves
        rec, dt = asyncio.run(level(url, c, [rows[i % len(rows)] for i in range(n)], a.max_tokens))
        ok = [r for r in rec if r["ttft_ms"] is not None]
        ot = sum(r["out_tok"] for r in rec)
        pt = {"concurrency": c, "requests": len(rec), "wall_s": round(dt, 2), "out_tok_s": round(ot / dt, 1),
              "req_s": round(len(rec) / dt, 3), "out_tok_mean": round(ot / max(1, len(rec)), 1),
              "ttft_ms": pct([r["ttft_ms"] for r in ok]), "tpot_ms": pct([r["tpot_ms"] for r in ok if r["tpot_ms"]]),
              "e2e_ms": pct([r["e2e_ms"] for r in rec])}
        R["points"].append(pt); json.dump(R, open(a.out, "w"), indent=1); print(json.dumps(pt), flush=True)
    R["ok"] = True; json.dump(R, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
