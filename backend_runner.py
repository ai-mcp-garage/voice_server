#!/usr/bin/env python3
"""
Backend Runner - Subprocess host for TTS backends.

Runs in backend's own venv, communicates via stdin/stdout msgpack.
Stays running between requests so model stays loaded.
"""
import os
import sys
import struct
import importlib
from pathlib import Path

# CRITICAL: Save real stdout before any imports that might print
# Then redirect stdout to stderr so library prints don't corrupt protocol
_real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
_real_stdin = os.fdopen(os.dup(sys.stdin.fileno()), 'rb', buffering=0)
sys.stdout = sys.stderr  # All prints go to stderr now

import msgpack
import numpy as np


def read_msg():
    """Read length-prefixed msgpack message from real stdin."""
    header = _real_stdin.read(4)
    if not header or len(header) < 4:
        return None
    length = struct.unpack('>I', header)[0]
    data = _real_stdin.read(length)
    return msgpack.unpackb(data, raw=False)


def write_msg(obj):
    """Write length-prefixed msgpack message to real stdout."""
    data = msgpack.packb(obj, use_bin_type=True)
    _real_stdout.write(struct.pack('>I', len(data)))
    _real_stdout.write(data)


def main():
    if len(sys.argv) < 3:
        print("Usage: backend_runner.py <backend_name> <backend_path>", file=sys.stderr)
        sys.exit(1)
    
    backend_name = sys.argv[1]
    backend_path = Path(sys.argv[2])
    
    # Add backend path and adapters dir to path
    sys.path.insert(0, str(backend_path))
    adapters_dir = Path(__file__).parent / "backends"
    sys.path.insert(0, str(adapters_dir))
    
    # Load backend adapter
    try:
        backend = importlib.import_module(f"backends.{backend_name}")
    except ImportError as e:
        write_msg({"type": "error", "msg": f"Backend load failed: {e}"})
        sys.exit(1)
    
    # Preload model
    voices = backend.list_voices()
    sample_rate = backend.get_sample_rate()
    
    # Signal ready
    write_msg({"type": "ready", "voices": voices, "sample_rate": sample_rate})
    
    # Command loop
    while True:
        msg = read_msg()
        if msg is None:
            break
        
        cmd = msg.get("cmd")
        
        if cmd == "quit":
            break
        
        elif cmd == "voices":
            write_msg({
                "type": "voices",
                "voices": backend.list_voices(),
                "sample_rate": backend.get_sample_rate()
            })
        
        elif cmd == "stream":
            text = msg.get("text", "")
            voice = msg.get("voice", "")
            opts = msg.get("opts", {})
            
            try:
                for chunk in backend.stream(text, voice, **opts):
                    # Convert to bytes
                    if hasattr(chunk, 'numpy'):
                        chunk = chunk.numpy()
                    chunk = np.asarray(chunk, dtype=np.float32)
                    write_msg({"type": "chunk", "data": chunk.tobytes()})
                write_msg({"type": "done"})
            except Exception as e:
                write_msg({"type": "error", "msg": str(e)})
        
        else:
            write_msg({"type": "error", "msg": f"Unknown command: {cmd}"})


if __name__ == "__main__":
    main()
