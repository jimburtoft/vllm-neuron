# SPDX-License-Identifier: Apache-2.0
"""Build the 5-clip mel set for the Task 020 M3 Medusa correctness gate.

Clips: jfk (already present as jfk_mel.npy) plus four LibriSpeech dev-clean-dummy
utterances chosen for length/content variety. Each mel is [128, 3000] built with
the whisper-large-v3 WhisperFeatureExtractor (same convention as
make_jfk_mel.py).

Per the M0 spec (Task 020 §5) and the M0.5 de-risk, the correctness gate compares
Medusa-vs-GREEDY on the SAME served config -- the transcript CONTENT is incidental
(the gate is verify-vs-greedy byte-identity on the SAME model). So any five real
30-second clips exercise the gate; we use the standard cached LibriSpeech dummy set
so the clips are reproducible on this instance.

Saves /large/work/ref/{ls1,ls2,ls3,ls4}_mel.npy.
"""
import io
import os

import numpy as np
import soundfile as sf
from datasets import features, load_dataset
from transformers import WhisperFeatureExtractor

MODEL_DIR = os.environ.get("WX_MODEL", "/large/work/whisper-large-v3")
REF = "/large/work/ref"

# Dataset row indices into the sorted LibriSpeech dummy validation split, chosen
# for content/length variety (short + long utterances). Fixed for reproducibility.
CLIP_INDICES = {"ls1": 0, "ls2": 1, "ls3": 40, "ls4": 53}


def main():
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    ds = ds.cast_column("audio", features.Audio(decode=False))
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)
    print(f"fe n_mels={fe.feature_size} sr={fe.sampling_rate} n_ds={len(ds)}", flush=True)

    for name, idx in CLIP_INDICES.items():
        row = ds[idx]
        a = row["audio"]
        if a.get("bytes") is not None:
            audio, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            audio, sr = sf.read(a["path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        feats = fe(audio, sampling_rate=sr, return_tensors="np")
        mel = feats["input_features"][0]
        assert mel.shape == (128, 3000), mel.shape
        outp = os.path.join(REF, f"{name}_mel.npy")
        np.save(outp, mel)
        print(
            f"{name}: ds_idx={idx} dur={len(audio)/sr:.2f}s "
            f"text={row['text'][:55]!r} -> {outp} mean={mel.mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
