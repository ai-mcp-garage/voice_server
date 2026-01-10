"""
Kokoro Backend Adapter

Self-contained adapter that wraps the Kokoro TTS model.
Kokoro is a lightweight 82M param TTS model supporting multiple languages.
"""
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

# Load backend path from config
_config_path = Path(__file__).parent.parent / "backends.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
_backend_cfg = _config["backends"]["kokoro"]
KOKORO_DIR = (Path(__file__).parent.parent / _backend_cfg["path"]).resolve()
sys.path.insert(0, str(KOKORO_DIR))

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 24000

# Available voices - voice name prefix determines language:
# af_* = American Female, am_* = American Male
# bf_* = British Female, bm_* = British Male
# See: https://huggingface.co/hexgrad/Kokoro-82M
VOICES = [
    # American English
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    # British English
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

# Language code mapping (first letter of voice name)
LANG_CODES = {
    "a": "a",  # American English
    "b": "b",  # British English
}

# ============================================================================
# Backend State (lazy loaded)
# ============================================================================
_pipeline = None
_current_lang = None
_loaded = False


def _get_device() -> str:
    """Auto-detect best device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_lang_code(voice: str) -> str:
    """Get language code from voice name."""
    if voice:
        first_char = voice[0].lower()
        return LANG_CODES.get(first_char, "a")
    return "a"


def _load(lang_code: str = "a"):
    """Load Kokoro pipeline for specified language."""
    global _pipeline, _current_lang, _loaded
    
    if _loaded and _current_lang == lang_code:
        return
    
    # Set MPS fallback env var for Mac
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    from kokoro import KPipeline
    
    device = _get_device()
    print(f"[kokoro] Loading pipeline (lang={lang_code}, device={device})...")
    
    _pipeline = KPipeline(lang_code=lang_code)
    _current_lang = lang_code
    _loaded = True
    
    print(f"[kokoro] Loaded with {len(VOICES)} voices")


# ============================================================================
# Public API
# ============================================================================

def list_voices() -> list[str]:
    """Return available voice names."""
    return VOICES.copy()


def generate(
    text: str,
    voice: str = "af_heart",
    *,
    speed: float = 1.0,
    cfg_scale: float = 1.0,  # Ignored, for API compatibility
) -> np.ndarray:
    """
    Generate audio from text.
    
    Args:
        text: Text to synthesize
        voice: Voice name (e.g., 'af_heart', 'am_adam')
        speed: Speech speed multiplier (default 1.0)
    
    Returns:
        Audio as float32 numpy array, mono, 24kHz
    """
    voice = voice or "af_heart"
    lang_code = _get_lang_code(voice)
    _load(lang_code)
    
    # Generate all chunks and concatenate
    generator = _pipeline(text, voice=voice, speed=speed)
    
    audio_chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)
    
    if not audio_chunks:
        return np.array([], dtype=np.float32)
    
    return np.concatenate(audio_chunks).astype(np.float32)


def stream(
    text: str,
    voice: str = "af_heart",
    *,
    speed: float = 1.0,
) -> Iterator[np.ndarray]:
    """
    Generate audio in streaming chunks.
    
    Args:
        text: Text to synthesize
        voice: Voice name (e.g., 'af_heart', 'am_adam')
        speed: Speech speed multiplier
    
    Yields:
        Audio chunks as float32 numpy arrays
    """
    voice = voice or "af_heart"
    lang_code = _get_lang_code(voice)
    _load(lang_code)
    
    generator = _pipeline(text, voice=voice, speed=speed)
    
    for gs, ps, audio in generator:
        if audio is not None and len(audio) > 0:
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            yield audio.astype(np.float32)


def get_sample_rate() -> int:
    """Return the sample rate (24000 Hz)."""
    return SAMPLE_RATE


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    print("Voices:", list_voices()[:5], "...")
    audio = generate("Hello, this is a test of Kokoro.", "af_heart")
    print(f"Generated {len(audio) / SAMPLE_RATE:.2f}s of audio")
