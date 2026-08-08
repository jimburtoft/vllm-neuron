# SPDX-License-Identifier: Apache-2.0
"""Regenerate CPU HF greedy references for the Voxtral device-gated correctness tests.

Runs `VoxtralForConditionalGeneration.from_pretrained(..., torch_dtype=bfloat16)`
on CPU with `AutoProcessor.apply_transcription_request(...)` +
`model.generate(..., do_sample=False)` on each reference clip, then writes
the resulting text + token IDs to `reference/<name>.json`.

Usage
-----

    cd vllm_neuron/model/voxtral/tests
    python make_reference.py

The `reference/` directory must contain the audio files with the same names
listed in the `CLIPS` dict below. The test suite reads them back via
`test_voxtral_correctness.py` and compares against a live Voxtral serve.
"""

import json
import os
import sys
from pathlib import Path

os.environ.pop("PJRT_DEVICE", None)  # force CPU
os.environ.pop("NEURON_VISIBLE_DEVICES", None)
os.environ.pop("NEURON_RT_VISIBLE_CORES", None)

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

MODEL = "mistralai/Voxtral-Mini-3B-2507"
REF_DIR = Path(__file__).parent / "reference"
REF_DIR.mkdir(exist_ok=True)

# name -> (audio_filename, max_new_tokens)
CLIPS = {
    "billgates_seg_0003": ("BillGates_2010_seg_0003.wav", 26),
    "billgates_seg_0017": ("BillGates_2010_seg_0017.wav", 65),
}


def main():
    print(f"[ref] loading Voxtral-Mini-3B on CPU (bfloat16)")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
    ).to("cpu").eval()

    for name, (fname, max_new) in CLIPS.items():
        audio_path = REF_DIR / fname
        if not audio_path.exists():
            print(f"[ref] SKIP {name}: {audio_path} not present")
            continue
        print(f"[ref] {name}: {audio_path}")
        inputs = processor.apply_transcription_request(
            language="en", audio=str(audio_path), model_id=MODEL,
        )
        inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
            )
        tokens = out[0, prompt_len:].tolist()
        text = processor.batch_decode([tokens], skip_special_tokens=True)[0]
        print(f"[ref]   -> {len(tokens)} tokens: {text[:80]!r}")

        ref_path = REF_DIR / f"{name}.json"
        with open(ref_path, "w") as f:
            json.dump({
                "model": MODEL,
                "audio_filename": fname,
                "max_new_tokens": max_new,
                "prompt_len": int(prompt_len),
                "tokens": tokens,
                "text": text,
            }, f, indent=2)
        print(f"[ref]   wrote {ref_path}")


if __name__ == "__main__":
    main()
