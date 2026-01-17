# Voice Samples

Place voice samples here for backends that support voice cloning.

## File Format

**General**: Place `.wav`, `.mp3`, `.ogg` files here. They appear as voices by filename (without extension).

**pocket_tts** (strict requirements):
- **Format**: 24kHz mono WAV
- **Length**: 5-15 seconds of clean speech works best
- **Auth**: Requires HuggingFace login for cloning

Convert with:
```bash
ffmpeg -i input.wav -ar 24000 -ac 1 -t 15 voice_samples/myvoice.wav
```

Then use: `--voice myvoice`

## Per-Backend Notes

| Backend    | Cloning? | Requirements |
|------------|----------|--------------|
| pocket_tts | Yes      | 24kHz mono, HF auth, 5-15s recommended |
| chatterbox | Yes      | WAV, any sample rate (auto-converts) |
| vibevoice  | Yes      | WAV preferred |
| kokoro     | No       | Uses built-in voices only |
| glm_tts    | Limited  | Reference audio in backend config |