"""
GLM-TTS Backend Adapter

Adapts GLM-TTS inference for the Voice Server.
"""
import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Iterator, Optional, Any
import copy

# Load backend path from config
_config_path = Path(__file__).parent.parent / "backends.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)
_backend_cfg = _config["backends"]["glm_tts"]
GLM_TTS_DIR = (Path(__file__).parent.parent / _backend_cfg["path"]).resolve()

# CRITICAL: Switch CWD to backend directory so it finds its utils/ckpts
# This is safe because this runs in a subprocess
os.chdir(GLM_TTS_DIR)
sys.path.insert(0, str(GLM_TTS_DIR))

# ============================================================================
# Backend State
# ============================================================================
_models: dict = {}
_loaded = False
_device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    _device = "mps"

# Define available voices (hardcoded from example for now)
# Format: name -> (prompt_text, prompt_speech_path)
_voices = {
    "jiayan": (
        "他当时还跟线下其他的站姐吵架，然后，打架进局子了。",
        "examples/prompt/jiayan_zh.wav"
    )
}

def _load():
    global _loaded, _models
    if _loaded:
        return

    print(f"[glm_tts] Loading on {_device}...")
    
    # Import here after path setup
    from glmtts_inference import load_models
    
    # Load models
    # Note: GLM-TTS currently only supports 24kHz
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=False,  # default from script
        sample_rate=24000
    )
    
    _models["frontend"] = frontend
    _models["text_frontend"] = text_frontend
    _models["speech_tokenizer"] = speech_tokenizer
    _models["llm"] = llm
    _models["flow"] = flow
    
    _loaded = True
    print("[glm_tts] Models loaded")

def list_voices() -> list[str]:
    return list(_voices.keys())

def get_sample_rate() -> int:
    return 24000

def stream(text: str, voice: str = "jiayan", **opts) -> Iterator[np.ndarray]:
    """
    Generate audio for text.
    Note: GLM-TTS doesn't support true streaming yet in the inference script,
    so we generate full audio and yield it as one chunk for compatibility.
    """
    _load()
    
    if voice not in _voices:
        print(f"[glm_tts] Voice '{voice}' not found, using 'jiayan'")
        voice = "jiayan"
        
    prompt_text, prompt_speech_path = _voices[voice]
    
    # Resolve absolute path for prompt speech to be safe
    prompt_speech_abs = str(GLM_TTS_DIR / prompt_speech_path)
    
    from glmtts_inference import generate_long
    from utils import seed_util
    import torchaudio

    # Prepare inputs similar to jsonl_generate in script
    frontend = _models["frontend"]
    text_frontend = _models["text_frontend"]
    llm = _models["llm"]
    flow = _models["flow"]
    
    # Seed
    seed_util.set_seed(0)
    
    # Clean text
    # The script does normalization inside generate_long, but we can do a pass
    prompt_text = text_frontend.text_normalize(prompt_text)
    syn_text = text_frontend.text_normalize(text)
    
    # Extract features
    prompt_text_token = frontend._extract_text_token(prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([prompt_speech_abs])
    speech_feat = frontend._extract_speech_feat(prompt_speech_abs, sample_rate=24000)
    embedding = frontend._extract_spk_embedding(prompt_speech_abs)
    
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(
        cache_speech_token, dtype=torch.int32
    ).to(_device)

    # Initialize Cache
    cache = {
        "cache_text": [prompt_text],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token,
        "use_cache": True,
    }
    
    # Generate
    try:
        tts_speech, _, _, _ = generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=["stream_req", syn_text],
            cache=cache,
            embedding=embedding,
            seed=0,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            device=_device,
            use_phoneme=False,
        )
        
        # Output is tensor on device
        audio = tts_speech.detach().cpu().numpy().flatten().astype(np.float32)
        yield audio
        
    except Exception as e:
        print(f"[glm_tts] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

