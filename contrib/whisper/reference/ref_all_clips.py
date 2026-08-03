"""Generate OpenAI-whisper greedy reference (mel npy + token IDs + text) for a
set of clips. This is the byte-identical correctness anchor for Task 004.

Uses openai-whisper (the mandated reference impl) in /large/refvenv.
Writes per-clip: <name>_mel.npy and appends to refs_all.json
{name: {tokens: [...], text: "...", dur: ...}}.

The greedy decode options mirror the Neuron harness: language=en, transcribe,
without_timestamps, temperature=0 (greedy). We capture res.tokens which are the
token IDs AFTER the SOT prompt up to and including EOS (matching what the harness
generates).
"""
import json
import os
import sys

import numpy as np
import torch
import whisper

OUT = "/large/work/ref"
CLIPS = [
    ("jfk", "/large/work/ref/jfk.flac"),
    ("libri_0", "/large/work/ref/libri_0.flac"),
    ("libri_1", "/large/work/ref/libri_1.flac"),
    ("libri_2", "/large/work/ref/libri_2.flac"),
    ("libri_3", "/large/work/ref/libri_3.flac"),
]


def main():
    model = whisper.load_model("large-v3", device="cpu")
    dims = model.dims
    assert dims.n_audio_state == 1280 and dims.n_vocab == 51866
    refs = {}
    for name, path in CLIPS:
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} missing", flush=True)
            continue
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=dims.n_mels)
        np.save(f"{OUT}/{name}_mel.npy", mel.float().cpu().numpy())
        opts = whisper.DecodingOptions(
            language="en", task="transcribe", without_timestamps=True,
            temperature=0.0, beam_size=None, fp16=False)
        res = whisper.decode(model, mel, opts)
        refs[name] = {"tokens": [int(t) for t in res.tokens], "text": res.text}
        print(f"{name}: {len(res.tokens)} tokens text={res.text!r}", flush=True)
        print(f"   tokens={res.tokens}", flush=True)
    json.dump(refs, open(f"{OUT}/refs_all.json", "w"), indent=2)
    print("wrote refs_all.json with", len(refs), "clips", flush=True)


if __name__ == "__main__":
    main()
