# SPDX-License-Identifier: Apache-2.0
"""Make the OpenAI-whisper greedy reference for the Whisper correctness gate.

The script writes one ``<clip>.json`` for each WAV file in ``--clips-dir``. The
file holds the greedy token IDs and the text of OpenAI-whisper large-v3 in fp32
on CPU. The token IDs come after the SOT prompt and stop before EOT.

Run in an environment that has ``openai-whisper`` (CPU is sufficient)::

    pip install openai-whisper
    python test/vllm_neuron/model/whisper/make_reference.py \
        --clips-dir /data/wav --out-dir /data/ref

Output for each clip::

    {"ref_token_ids": [2221, 13, ...], "text": " Mr. Quilter is ..."}
"""

import argparse
import glob
import json
import os

EOT = 50257


def build_reference(clips_dir: str, out_dir: str) -> int:
    import whisper

    os.makedirs(out_dir, exist_ok=True)
    model = whisper.load_model("large-v3", device="cpu")
    if model.dims.n_audio_state != 1280 or model.dims.n_vocab != 51866:
        raise SystemExit("The loaded model is not whisper-large-v3.")

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
        raise SystemExit(f"No *.wav file is in {clips_dir}.")

    for path in wavs:
        name = os.path.splitext(os.path.basename(path))[0]
        audio = whisper.pad_or_trim(whisper.load_audio(path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        res = whisper.decode(model, mel, opts)
        ref = {
            "ref_token_ids": [int(t) for t in res.tokens if t != EOT],
            "text": res.text,
        }
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(ref, f, indent=2)
        print(f"{name}: {len(ref['ref_token_ids'])} tokens text={res.text!r}", flush=True)
    return len(wavs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-dir", required=True, help="Directory of <name>.wav clips.")
    ap.add_argument("--out-dir", required=True, help="Output directory for <name>.json.")
    args = ap.parse_args()
    build_reference(args.clips_dir, args.out_dir)


if __name__ == "__main__":
    main()
