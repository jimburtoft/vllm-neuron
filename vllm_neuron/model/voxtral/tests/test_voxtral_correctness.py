# SPDX-License-Identifier: Apache-2.0
"""Device-gated byte-identical greedy correctness gate for Voxtral-Mini-3B.

Skipped unless VOXTRAL_NEURON_DEVICE_TESTS=1 (requires a trn2 device +
model weights + a running `vllm serve` on port 8000).

Compares Neuron output tokens to a pre-computed CPU HF reference on
two Bill Gates TED clips (short = 8.58s = 26 tokens, long = 16.04s
= 65 tokens). Both were verified byte-identical during initial port
(Task 004h).

Regenerate the reference with `make_reference.py` after any change to
the audio processor, tokenizer, or bf16 semantics upstream.
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("VOXTRAL_NEURON_DEVICE_TESTS") != "1",
    reason="requires Neuron device + running vllm serve; "
           "set VOXTRAL_NEURON_DEVICE_TESTS=1 to enable",
)


TESTS_DIR = Path(__file__).parent
REF_DIR = TESTS_DIR / "reference"


def _transcribe(audio_path: Path, url: str = "http://localhost:8000/v1/audio/transcriptions") -> str:
    """POST audio to /v1/audio/transcriptions and return the text."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    parts = []
    for name, value in [
        ("model", "mistralai/Voxtral-Mini-3B-2507"),
        ("language", "en"),
        ("temperature", "0.0"),
        ("response_format", "json"),
    ]:
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode())
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'.encode())
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["text"].strip()


def _get_reference(name: str) -> dict:
    ref_path = REF_DIR / f"{name}.json"
    if not ref_path.exists():
        pytest.skip(f"reference {ref_path} missing; regenerate with make_reference.py")
    with open(ref_path) as f:
        return json.load(f)


def test_serve_health():
    """/health must return 200 -- prereq for the transcription tests."""
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as resp:
            assert resp.status == 200
    except (urllib.error.URLError, ConnectionError) as e:
        pytest.skip(f"vllm serve not running on port 8000: {e}")


@pytest.mark.parametrize("clip_name", ["billgates_seg_0003", "billgates_seg_0017"])
def test_byte_identical_transcript(clip_name):
    """Neuron transcription must match CPU HF reference text byte-for-byte."""
    ref = _get_reference(clip_name)
    audio_path = REF_DIR / ref["audio_filename"]
    if not audio_path.exists():
        pytest.skip(f"audio {audio_path} missing")

    got = _transcribe(audio_path)
    expected = ref["text"].strip()

    assert got == expected, (
        f"Transcript mismatch for {clip_name}:\n"
        f"  neuron:  {got!r}\n"
        f"  cpu ref: {expected!r}"
    )
