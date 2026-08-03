"""OpenAI-whisper CPU reference for whisper-xla Task 003.

Produces:
  - encoder hidden states (saved as .npy) for cos-sim comparison
  - greedy transcript (the byte-identical correctness anchor for Task 004)
for a fixed 30-s audio clip, using the OpenAI `whisper` package (the mandated
reference implementation, NOT HuggingFace).
"""
import sys, json, numpy as np, torch, whisper

MODEL = "large-v3"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "/large/work/ref/jfk.flac"
OUT = "/large/work/ref"

def main():
    print(f"loading openai-whisper {MODEL} on CPU ...", flush=True)
    model = whisper.load_model(MODEL, device="cpu")
    dims = model.dims
    print(f"dims: n_audio_state={dims.n_audio_state} n_audio_head={dims.n_audio_head} "
          f"n_text_state={dims.n_text_state} n_text_head={dims.n_text_head} "
          f"n_vocab={dims.n_vocab} n_audio_ctx={dims.n_audio_ctx} n_text_ctx={dims.n_text_ctx} "
          f"n_audio_layer={dims.n_audio_layer} n_text_layer={dims.n_text_layer}", flush=True)
    # config asserts (Footgun 3 analog)
    assert dims.n_audio_state == 1280, dims.n_audio_state
    assert dims.n_vocab == 51866, dims.n_vocab

    audio = whisper.load_audio(AUDIO)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=dims.n_mels).to(model.device)
    print(f"mel shape {tuple(mel.shape)}", flush=True)

    with torch.no_grad():
        enc = model.encoder(mel.unsqueeze(0))  # [1, n_audio_ctx=1500, 1280]
    enc_np = enc.squeeze(0).float().cpu().numpy()
    np.save(f"{OUT}/enc_hidden_ref.npy", enc_np)
    np.save(f"{OUT}/mel_ref.npy", mel.float().cpu().numpy())
    print(f"encoder hidden saved: shape {enc_np.shape}, mean {enc_np.mean():.5f} std {enc_np.std():.5f}", flush=True)

    # greedy transcript
    opts = whisper.DecodingOptions(language="en", task="transcribe", without_timestamps=True,
                                   temperature=0.0, beam_size=None, fp16=False)
    res = whisper.decode(model, mel, opts)
    print(f"TRANSCRIPT: {res.text!r}", flush=True)
    print(f"TOKENS: {res.tokens}", flush=True)
    with open(f"{OUT}/transcript_ref.json", "w") as f:
        json.dump({"text": res.text, "tokens": res.tokens,
                   "dims": dims._asdict()}, f, indent=2)
    print("reference saved to", OUT, flush=True)

if __name__ == "__main__":
    main()
