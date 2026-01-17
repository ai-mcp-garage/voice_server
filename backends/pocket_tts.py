"""
Pocket TTS Backend Adapter

Adapts Kyutai's Pocket TTS for the Voice Server.
CPU-optimized, lightweight model with streaming support.

Voice Cloning:
  - Requires HuggingFace auth (see error message for instructions)
  - Best results with 5-15s of clean speech
  - Convert to 24kHz mono WAV:
    ffmpeg -i input.wav -ar 24000 -ac 1 -t 15 output.wav
"""
import sys
import numpy as np
import yaml
from pathlib import Path
from typing import Iterator

# Load backend path from config
_config_path = Path(__file__).parent.parent / "backends.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
_backend_cfg = _config["backends"]["pocket_tts"]
POCKET_TTS_DIR = (Path(__file__).parent.parent / _backend_cfg["path"]).resolve()

# Add pocket-tts to path
sys.path.insert(0, str(POCKET_TTS_DIR))

# ============================================================================
# Backend State
# ============================================================================
_model = None
_voice_states = {}
_loaded = False

# Available predefined voices
_voices = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


def _load():
    global _loaded, _model
    if _loaded:
        return

    print("[pocket_tts] Loading model...")
    
    from pocket_tts import TTSModel
    
    _model = TTSModel.load_model()
    _loaded = True
    print(f"[pocket_tts] Model loaded (sample_rate={_model.sample_rate})")


def _get_voice_state(voice: str):
    """Get or create voice state for a voice."""
    global _voice_states
    
    if voice in _voice_states:
        return _voice_states[voice]
    
    # Determine voice prompt source
    is_custom_voice = False
    if voice in _voices:
        # Use predefined voice name directly - pocket_tts handles these internally
        voice_input = voice
    else:
        is_custom_voice = True
        # Check if it's a custom voice file path
        voice_path = Path(voice)
        if not voice_path.exists():
            # Try voice samples directory
            voice_samples = Path(__file__).parent.parent / "voice_samples"
            voice_path = voice_samples / f"{voice}.wav"
            if not voice_path.exists():
                print(f"[pocket_tts] Voice '{voice}' not found, using 'alba'")
                voice = "alba"
                voice_input = voice
                is_custom_voice = False
            else:
                voice_input = str(voice_path)
        else:
            voice_input = str(voice_path)
    
    # Check if voice cloning is available when using custom voice
    if is_custom_voice and not _model.has_voice_cloning:
        raise RuntimeError(
            f"[pocket_tts] Voice cloning not available - HuggingFace auth required.\n"
            f"To enable voice cloning:\n"
            f"  1. Accept terms at: https://huggingface.co/kyutai/pocket-tts\n"
            f"  2. Login: uvx huggingface-cli login\n"
            f"Available predefined voices (no auth needed): {', '.join(_voices)}"
        )
    
    print(f"[pocket_tts] Loading voice state for '{voice}'...")
    # truncate=True limits audio to 30s, prevents sequence length errors
    state = _model.get_state_for_audio_prompt(voice_input, truncate=is_custom_voice)
    _voice_states[voice] = state
    return state


def list_voices() -> list[str]:
    """Return available voices."""
    return _voices.copy()


def get_sample_rate() -> int:
    """Return sample rate (24kHz)."""
    return 24000


def stream(text: str, voice: str = "alba", **opts) -> Iterator[np.ndarray]:
    """
    Generate audio for text using streaming.
    Yields numpy float32 arrays.
    
    Opts:
        temp: float - sampling temperature (default 0.7, higher = more expressive)
        lsd_decode_steps: int - quality/speed tradeoff (default 1, higher = better quality)
        eos_threshold: float - end-of-sequence threshold (default -4.0)
        frames_after_eos: int - extra frames to generate after EOS detected (default auto)
    """
    _load()
    
    # Apply runtime opts to model
    if "temp" in opts:
        _model.temp = float(opts["temp"])
        print(f"[pocket_tts] Using temp={_model.temp}")
    if "lsd_decode_steps" in opts:
        _model.lsd_decode_steps = int(opts["lsd_decode_steps"])
        print(f"[pocket_tts] Using lsd_decode_steps={_model.lsd_decode_steps}")
    if "eos_threshold" in opts:
        _model.eos_threshold = float(opts["eos_threshold"])
        print(f"[pocket_tts] Using eos_threshold={_model.eos_threshold}")
    
    # Get per-call opts
    frames_after_eos = opts.get("frames_after_eos")
    if frames_after_eos is not None:
        frames_after_eos = int(frames_after_eos)
        print(f"[pocket_tts] Using frames_after_eos={frames_after_eos}")
    
    voice_state = _get_voice_state(voice)
    
    try:
        for chunk in _model.generate_audio_stream(voice_state, text, frames_after_eos=frames_after_eos):
            # chunk is a torch.Tensor, convert to numpy
            audio = chunk.detach().cpu().numpy().astype(np.float32)
            yield audio
            
    except Exception as e:
        print(f"[pocket_tts] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
