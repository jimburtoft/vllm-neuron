"""Extract LibriSpeech dev-clean clips of varying length from the dummy parquet.
Writes N .flac files; durations computed via ffprobe (no soundfile needed)."""
import io
import json
import subprocess
import pyarrow.parquet as pq

PQ = "/large/work/ref/libri_dummy.parquet"
OUT = "/large/work/ref"


def dur_of(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path]).decode().strip()
    return float(out)


t = pq.read_table(PQ).to_pylist()
# write all to temp, measure, sort
tmp = []
for i, r in enumerate(t):
    b = r["audio"]["bytes"]
    p = f"{OUT}/_tmp_{i}.flac"
    with open(p, "wb") as f:
        f.write(b)
    d = dur_of(p)
    tmp.append((d, p, r["text"], r["id"]))

tmp.sort(key=lambda x: x[0])
tmp = [x for x in tmp if x[0] < 30.0]
n = len(tmp)
picks = [tmp[0], tmp[n // 3], tmp[2 * n // 3], tmp[-1]]

manifest = []
for i, (d, p, text, cid) in enumerate(picks):
    fn = f"{OUT}/libri_{i}.flac"
    subprocess.check_call(["cp", p, fn])
    print(f"libri_{i}.flac dur={d:.2f}s id={cid} text={text[:70]!r}", flush=True)
    manifest.append({"file": f"libri_{i}.flac", "dur": d, "id": cid, "text": text})

# cleanup temp
subprocess.call("rm -f " + OUT + "/_tmp_*.flac", shell=True)
json.dump(manifest, open(f"{OUT}/libri_manifest.json", "w"), indent=2)
print("wrote", len(manifest), "clips", flush=True)
