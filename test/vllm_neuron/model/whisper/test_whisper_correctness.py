# SPDX-License-Identifier: Apache-2.0
"""On-device correctness gate for Whisper large-v3 through ``vllm serve``.

The test sends each WAV file to ``/v1/audio/transcriptions`` with greedy
decoding. It compares the token IDs with the OpenAI-whisper reference that
``make_reference.py`` writes.

The test needs a Neuron device, the model weights, and a server. It runs only
when ``WHISPER_NEURON_DEVICE_TESTS=1`` is set. Set these variables:

* ``WHISPER_WAV_DIR``: the directory of ``<name>.wav`` clips.
* ``WHISPER_REF_DIR``: the directory of ``<name>.json`` references.
* ``WHISPER_SERVE_URL``: the endpoint. The default is
  ``http://localhost:8000/v1/audio/transcriptions``.
* ``WHISPER_MIN_EXACT``: the minimum fraction of byte-identical clips. The
  default is 1.0.

Run with::

    WHISPER_NEURON_DEVICE_TESTS=1 WHISPER_WAV_DIR=/data/wav WHISPER_REF_DIR=/data/ref \
        pytest test/vllm_neuron/model/whisper/test_whisper_correctness.py -v
"""

import glob
import json
import os
import urllib.request

import pytest

WAV_DIR = os.environ.get("WHISPER_WAV_DIR", "")
REF_DIR = os.environ.get("WHISPER_REF_DIR", "")
SERVE_URL = os.environ.get(
    "WHISPER_SERVE_URL", "http://localhost:8000/v1/audio/transcriptions"
)
MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
SERVED_NAME = os.environ.get("WHISPER_SERVED_NAME", "whisper")
MIN_EXACT = float(os.environ.get("WHISPER_MIN_EXACT", "1.0"))

pytestmark = pytest.mark.skipif(
    os.environ.get("WHISPER_NEURON_DEVICE_TESTS") != "1",
    reason="Set WHISPER_NEURON_DEVICE_TESTS=1 to run the on-device Whisper gate.",
)


def _transcribe(wav_path: str) -> str:
    import requests

    with open(wav_path, "rb") as f:
        r = requests.post(
            SERVE_URL,
            files={"file": (os.path.basename(wav_path), f, "audio/wav")},
            data={
                "model": SERVED_NAME,
                "language": "en",
                "temperature": "0",
                "response_format": "json",
            },
            timeout=600,
        )
    r.raise_for_status()
    return r.json()["text"]


def test_endpoint_reachable():
    health = SERVE_URL.rsplit("/v1/", 1)[0] + "/health"
    with urllib.request.urlopen(health, timeout=10) as r:
        assert r.status == 200


def test_greedy_matches_reference():
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained(MODEL)
    wavs = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    assert wavs, f"No *.wav file is in WHISPER_WAV_DIR={WAV_DIR!r}."

    exact, failures = 0, []
    for wav in wavs:
        name = os.path.splitext(os.path.basename(wav))[0]
        with open(os.path.join(REF_DIR, f"{name}.json")) as f:
            ref_ids = json.load(f)["ref_token_ids"]
        gen_ids = tok.encode(_transcribe(wav), add_special_tokens=False)
        if gen_ids == ref_ids:
            exact += 1
        else:
            first = next(
                (i for i, (a, b) in enumerate(zip(gen_ids, ref_ids)) if a != b),
                min(len(gen_ids), len(ref_ids)),
            )
            failures.append(f"{name}: first difference at token {first}")
    assert exact >= MIN_EXACT * len(wavs), (
        f"{exact}/{len(wavs)} clips are byte-identical. {failures}"
    )
