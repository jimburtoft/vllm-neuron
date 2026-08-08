# Voxtral device-gated correctness tests -- audio files (BYO)

The device-gated test in `test_voxtral_correctness.py` compares live Voxtral
serve output against pre-computed CPU HF references stored as JSON in this
directory:

- `billgates_seg_0003.json` — expected tokens + text for
  `BillGates_2010_seg_0003.wav`
- `billgates_seg_0017.json` — expected tokens + text for
  `BillGates_2010_seg_0017.wav`

The audio files themselves are NOT shipped in the repository. Both are
publicly-available TED talk excerpts:

- **BillGates_2010_seg_0003.wav**: ~8.58s excerpt of Bill Gates' 2010 TED
  talk "Innovating to zero!"
- **BillGates_2010_seg_0017.wav**: ~16.04s excerpt of the same talk

Any downstream user can source their own audio (or replace with any English
audio clip) and regenerate the reference JSON via `make_reference.py`. The
test verifies **Neuron output matches CPU HF greedy reference for the same
audio + prompt**, so byte-identical requires (a) the exact same audio file
and (b) the exact same processor / tokenizer + `AutoProcessor.apply_transcription_request(language="en", ...)` prompt template.

To reproduce with fresh audio:

```bash
# Drop your audio files here as .wav (16 kHz mono ideally, but the Voxtral
# processor's mistral_common backend auto-resamples).
cp /path/to/BillGates_2010_seg_0003.wav .
cp /path/to/BillGates_2010_seg_0017.wav .

# Regenerate CPU HF reference JSONs.
python make_reference.py

# Start a Voxtral serve (see ../serve_smoke_test.sh).

# Run the device-gated correctness tests.
VOXTRAL_NEURON_DEVICE_TESTS=1 pytest test_voxtral_correctness.py -v
```
