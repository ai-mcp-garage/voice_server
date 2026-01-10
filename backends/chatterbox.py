"""
Chatterbox Backend Adapter

Self-contained adapter for Chatterbox TTS (both Turbo and standard models).
Supports paralinguistic tags like [laugh], [chuckle], [cough], etc.
Standard model supports voice cloning on MPS; Turbo has issues with it.
"""
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml

# Load backend path from config
_config_path = Path(__file__).parent.parent / "backends.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
_backend_cfg = _config["backends"]["chatterbox"]
CHATTERBOX_DIR = (Path(__file__).parent.parent / _backend_cfg["path"]).resolve() / "src"
sys.path.insert(0, str(CHATTERBOX_DIR))

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 24000

# Voice samples directory for cloning
VOICE_SAMPLES_DIR = Path(__file__).parent.parent / "voice_samples"

# ============================================================================
# Backend State (lazy loaded)
# ============================================================================
_model = None
_device = None
_voices: dict[str, Path] = {}
_loaded = False
_use_turbo = True  # Default to Turbo


def _scan_voices():
    """Scan voice_samples directory for audio files."""
    global _voices
    _voices = {"default": None}
    
    if VOICE_SAMPLES_DIR.exists():
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            for f in VOICE_SAMPLES_DIR.glob(ext):
                _voices[f.stem] = f
    
    _voices = dict(sorted(_voices.items()))


def _get_device() -> str:
    """Auto-detect best device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Patch torch.load for MPS compatibility (from chatterbox example_for_mac.py)
_device_for_patch = None
_torch_load_original = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs and _device_for_patch:
        kwargs['map_location'] = torch.device(_device_for_patch)
    return _torch_load_original(*args, **kwargs)


def _load(turbo: bool = True):
    """Load Chatterbox model."""
    global _model, _device, _loaded, _use_turbo, _device_for_patch
    
    if _loaded and _use_turbo == turbo:
        return
    
    _use_turbo = turbo
    _device = _get_device()
    _device_for_patch = _device
    
    # Apply MPS patch
    if _device == "mps":
        torch.load = _patched_torch_load
    
    if turbo:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print(f"[chatterbox] Loading Turbo model on {_device}...")
        _model = ChatterboxTurboTTS.from_pretrained(device=_device)
    else:
        from chatterbox.tts import ChatterboxTTS
        print(f"[chatterbox] Loading Standard model on {_device}...")
        _model = ChatterboxTTS.from_pretrained(device=_device)
    
    _scan_voices()
    _loaded = True
    
    model_type = "Turbo" if turbo else "Standard"
    print(f"[chatterbox] Loaded {model_type} (sample_rate={_model.sr}, voices={len(_voices)})")


# ============================================================================
# Public API
# ============================================================================

def list_voices() -> list[str]:
    """
    Return available voices.
    
    'default' = no voice cloning (model's default voice).
    Other names = audio files in voice_samples/ folder for cloning.
    """
    _scan_voices()
    return list(_voices.keys())


def generate(
    text: str,
    voice: str = "default",
    *,
    turbo: bool = True,  # Use Turbo model (faster) or Standard (better voice cloning on MPS)
    audio_prompt: str = "",  # Override: direct path to reference audio
    exaggeration: float = 0.5,  # Expressiveness (Standard model only)
    cfg_weight: float = 0.5,  # CFG weight (Standard model only)
    **kwargs,
) -> np.ndarray:
    """
    Generate audio from text.
    
    Args:
        text: Text with optional paralinguistic tags: [laugh], [chuckle], [cough], etc.
        voice: Voice name from list_voices() - maps to audio file in voice_samples/
        turbo: Use Turbo model (fast) or Standard (better voice cloning on MPS)
        audio_prompt: Override - direct path to reference audio
        exaggeration: Expressiveness control, 0-1 (Standard model only)
        cfg_weight: CFG weight for pacing, 0-1 (Standard model only)
    
    Returns:
        Audio as float32 numpy array, mono, 24kHz
    """
    _load(turbo=turbo)
    
    # Build generate kwargs
    gen_kwargs = {}
    
    # Priority: audio_prompt override > voice name lookup
    if audio_prompt and Path(audio_prompt).exists():
        gen_kwargs["audio_prompt_path"] = audio_prompt
        print(f"[chatterbox] Using voice from: {audio_prompt}")
    elif voice and voice != "default" and voice in _voices and _voices[voice]:
        gen_kwargs["audio_prompt_path"] = str(_voices[voice])
        print(f"[chatterbox] Using voice: {voice}")
    
    # Standard model supports exaggeration and cfg_weight
    if not turbo and gen_kwargs.get("audio_prompt_path"):
        gen_kwargs["exaggeration"] = exaggeration
        gen_kwargs["cfg_weight"] = cfg_weight
    
    wav = _model.generate(text, **gen_kwargs)
    audio = wav.squeeze().cpu().numpy()
    
    return audio.astype(np.float32)


def stream(
    text: str,
    voice: str = "default",
    **kwargs,
) -> Iterator[np.ndarray]:
    """Generate audio (non-streaming - Chatterbox doesn't support streaming)."""
    audio = generate(text, voice, **kwargs)
    yield audio


def get_sample_rate() -> int:
    """Return the sample rate."""
    _load()
    return _model.sr


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    print("Voices:", list_voices())
    audio = generate("Hello, this is Chatterbox! [chuckle]", turbo=True)
    print(f"Generated {len(audio) / get_sample_rate():.2f}s of audio")
