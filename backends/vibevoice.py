"""
VibeVoice Backend Adapter

Self-contained adapter that wraps the VibeVoice TTS model.
All configuration is internal - just import and use.
"""
import copy
import sys
import threading
from pathlib import Path
from typing import Iterator, Any, Optional

import numpy as np
import torch
import yaml

# Load backend path from config
_config_path = Path(__file__).parent.parent / "backends.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
_backend_cfg = _config["backends"]["vibevoice"]
VIBEVOICE_DIR = (Path(__file__).parent.parent / _backend_cfg["path"]).resolve()
sys.path.insert(0, str(VIBEVOICE_DIR))

# ============================================================================
# Configuration - edit these as needed
# ============================================================================
MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
VOICES_DIR = VIBEVOICE_DIR / "demo" / "voices" / "streaming_model"
SAMPLE_RATE = 24000

# ============================================================================
# Backend State (lazy loaded)
# ============================================================================
_processor = None
_model = None
_device = None
_voices: dict[str, Path] = {}
_voice_cache: dict[str, Any] = {}
_lock = threading.Lock()
_loaded = False


def _get_device() -> str:
    """Auto-detect best device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load():
    """Load model and voices (called automatically on first use)."""
    global _processor, _model, _device, _voices, _loaded
    
    if _loaded:
        return
    
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    
    _device = _get_device()
    print(f"[vibevoice] Loading on {_device}...")
    
    _processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)
    
    # Device-specific settings
    if _device == "mps":
        dtype, attn, dmap = torch.float32, "sdpa", None
    elif _device == "cuda":
        dtype, attn, dmap = torch.bfloat16, "flash_attention_2", "cuda"
    else:
        dtype, attn, dmap = torch.float32, "sdpa", "cpu"
    
    try:
        _model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, device_map=dmap, attn_implementation=attn
        )
        if _device == "mps":
            _model.to("mps")
    except Exception:
        _model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            MODEL_PATH, torch_dtype=dtype, device_map=dmap, attn_implementation="sdpa"
        )
        if _device == "mps":
            _model.to("mps")
    
    _model.eval()
    _model.set_ddpm_inference_steps(num_steps=5)
    
    # Scan voices
    if VOICES_DIR.exists():
        for pt in VOICES_DIR.glob("*.pt"):
            _voices[pt.stem] = pt
    _voices = dict(sorted(_voices.items()))
    
    print(f"[vibevoice] Loaded {len(_voices)} voices")
    _loaded = True


def _get_voice(name: str) -> tuple[str, Any]:
    """Get voice by name with caching."""
    _load()
    
    # Strip .pt extension if present
    name = name.replace(".pt", "")
    name_lower = name.lower()
    
    for k, p in _voices.items():
        if k.lower() == name_lower or name_lower in k.lower():
            if k not in _voice_cache:
                print(f"[vibevoice] Loading voice preset: {k}")
                _voice_cache[k] = torch.load(p, map_location=_device, weights_only=False)
            return k, _voice_cache[k]
    
    # Fallback to first voice
    k = list(_voices.keys())[0]
    print(f"[vibevoice] Voice '{name}' not found, using: {k}")
    if k not in _voice_cache:
        _voice_cache[k] = torch.load(_voices[k], map_location=_device, weights_only=False)
    return k, _voice_cache[k]


# ============================================================================
# Public API
# ============================================================================

def list_voices() -> list[str]:
    """Return available voice names."""
    _load()
    return list(_voices.keys())


def generate(
    text: str,
    voice: str = "",
    *,
    cfg_scale: float = 1.5,
    do_sample: bool = False,
) -> np.ndarray:
    """
    Generate audio from text.
    
    Args:
        text: Text to synthesize
        voice: Voice name (empty = first available)
        cfg_scale: CFG guidance scale (default 1.5)
        do_sample: Enable sampling (default False)
    
    Returns:
        Audio as float32 numpy array, mono, 24kHz
    """
    _load()
    
    text = text.replace("'", "'").replace('"', '"').replace('"', '"')
    voice_key, prefilled = _get_voice(voice or list(_voices.keys())[0])
    
    inputs = _processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=prefilled,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(_device)
    
    with _lock:
        outputs = _model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=_processor.tokenizer,
            generation_config={"do_sample": do_sample},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(prefilled),
        )
    
    audio = outputs.speech_outputs[0]
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    return audio.reshape(-1).astype(np.float32)


def stream(
    text: str,
    voice: str = "",
    *,
    cfg_scale: float = 1.5,
) -> Iterator[np.ndarray]:
    """
    Generate audio in streaming chunks.
    
    Args:
        text: Text to synthesize
        voice: Voice name (empty = first available)
        cfg_scale: CFG guidance scale
    
    Yields:
        Audio chunks as float32 numpy arrays
    """
    _load()
    
    from vibevoice.modular.streamer import AudioStreamer
    
    text = text.replace("'", "'").replace('"', '"').replace('"', '"')
    voice_key, prefilled = _get_voice(voice or list(_voices.keys())[0])
    
    inputs = _processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=prefilled,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(_device)
    
    audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
    stop_event = threading.Event()
    errors: list = []
    
    def run():
        try:
            with _lock:
                _model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=_processor.tokenizer,
                    generation_config={"do_sample": False},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled),
                )
        except Exception as e:
            errors.append(e)
            audio_streamer.end()
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
    try:
        for chunk in audio_streamer.get_stream(0):
            if torch.is_tensor(chunk):
                chunk = chunk.detach().cpu().to(torch.float32).numpy()
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            peak = np.max(np.abs(chunk)) if chunk.size else 0.0
            if peak > 1.0:
                chunk = chunk / peak
            yield chunk
    finally:
        stop_event.set()
        audio_streamer.end()
        thread.join()
        if errors:
            raise errors[0]


def get_sample_rate() -> int:
    """Return the sample rate (24000 Hz)."""
    return SAMPLE_RATE


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    print("Voices:", list_voices())
    audio = generate("Hello, this is a test.", "en-Carter_man")
    print(f"Generated {len(audio) / SAMPLE_RATE:.2f}s of audio")