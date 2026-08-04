# SPDX-License-Identifier: Apache-2.0
"""Regenerate the OpenAI-whisper greedy reference for the Whisper correctness gate.

This produces the byte-identical anchor consumed by
``test_whisper_correctness.py``: one ``<clip>.json`` per input WAV containing the
OpenAI-whisper large-v3 greedy token IDs and text.

The eval clips are short public-domain LibriSpeech utterances (16 kHz mono WAV).
Point ``--clips-dir`` at a directory of ``<name>.wav`` files.

Run on the host in a venv with ``openai-whisper`` (CPU is fine)::

    pip install openai-whisper
    python -m vllm_neuron.model.whisper.tests.make_reference \
        --clips-dir /large/work/ref/wav \
        --out-dir   /large/work/ref/clips

Output per clip ``<name>.json``::

    {"ref_token_ids": [2221, 13, ...], "text": " Mr. Quilter is ..."}

``ref_token_ids`` are the IDs AFTER the SOT prompt up to and including EOS,
matching what a greedy transcription request returns.
"""

import argparse
import glob
import json
import os


def build_reference(clips_dir: str, out_dir: str) -> int:
    import numpy as np  # noqa: F401  (whisper pulls it; import to fail fast)
    import whisper

    os.makedirs(out_dir, exist_ok=True)
    model = whisper.load_model("large-v3", device="cpu")
    dims = model.dims
    assert dims.n_audio_state == 1280 and dims.n_vocab == 51866, (
        "loaded model is not whisper-large-v3"
    )

    opts = whisper.DecodingOptions(
        language="en",
        task="transcribe",
        without_timestamps=True,
        temperature=0.0,
        beam_size=None,
        fp16=False,
    )

    wavs = sorted(glob.glob(os.path.join(clips_dir, "*.wav")))
    if not wavs:
        raise SystemExit(f"no *.wav files found in {clips_dir}")

    written = 0
    for path in wavs:
        name = os.path.splitext(os.path.basename(path))[0]
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=dims.n_mels)
        res = whisper.decode(model, mel, opts)
        ref = {
            "ref_token_ids": [int(t) for t in res.tokens],
            "text": res.text,
        }
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(ref, f, indent=2)
        print(f"{name}: {len(res.tokens)} tokens  text={res.text!r}", flush=True)
        written += 1

    print(f"wrote {written} reference clip(s) to {out_dir}", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clips-dir",
        default="/large/work/ref/wav",
        help="directory of <name>.wav clips (public LibriSpeech utterances)",
    )
    ap.add_argument(
        "--out-dir",
        default="/large/work/ref/clips",
        help="output directory for <name>.json references",
    )
    args = ap.parse_args()
    build_reference(args.clips_dir, args.out_dir)


if __name__ == "__main__":
    main()
