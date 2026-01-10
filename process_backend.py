"""
Subprocess Backend - Client for running backends in isolated subprocesses.
"""
import struct
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Any

import msgpack
import numpy as np


class SubprocessBackend:
    """Backend that runs in its own subprocess with isolated venv."""
    
    def __init__(self, name: str):
        self.name = name
        self._proc = None
        self._sample_rate = 24000
        self._voices = []
        
        # Load backend config
        import yaml
        config_path = Path(__file__).parent / "backends.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if name not in config["backends"]:
            raise RuntimeError(f"Backend '{name}' not found in backends.yaml")
        
        backend_cfg = config["backends"][name]
        backend_path = (Path(__file__).parent / backend_cfg["path"]).resolve()
        venv_python = backend_path / ".venv" / "bin" / "python"
        
        if not venv_python.exists():
            raise RuntimeError(
                f"Backend venv not found: {venv_python}\n"
                f"Run: cd {backend_path} && uv venv && uv pip install --python .venv/bin/python -e . msgpack pip"
            )
        
        runner = Path(__file__).parent / "backend_runner.py"
        
        # Start subprocess, passing backend path as argument
        self._proc = subprocess.Popen(
            [str(venv_python), str(runner), name, str(backend_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent,
        )
        
        # Wait for ready signal
        msg = self._read()
        if msg and msg.get("type") == "ready":
            self._voices = msg.get("voices", [])
            self._sample_rate = msg.get("sample_rate", 24000)
        elif msg and msg.get("type") == "error":
            raise RuntimeError(msg.get("msg"))
        else:
             raise RuntimeError("Backend failed to start (no ready signal)")

    
    def _check_alive(self):
        """Check if subprocess is still running, raise with stderr if not."""
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            raise RuntimeError(f"Backend subprocess died (exit={self._proc.returncode}): {stderr}")
    
    def _write(self, obj):
        self._check_alive()
        data = msgpack.packb(obj, use_bin_type=True)
        self._proc.stdin.write(struct.pack('>I', len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()
    
    def _read(self):
        self._check_alive()
        header = self._proc.stdout.read(4)
        if not header or len(header) < 4:
            self._check_alive()  # Check again for better error message
            return None
        length = struct.unpack('>I', header)[0]
        data = self._proc.stdout.read(length)
        return msgpack.unpackb(data, raw=False)
    
    @property
    def __name__(self):
        return f"backends.{self.name}"
    
    def list_voices(self):
        return self._voices.copy()
    
    def get_sample_rate(self):
        return self._sample_rate
    
    def stream(self, text: str, voice: str, **opts) -> Iterator[np.ndarray]:
        self._write({"cmd": "stream", "text": text, "voice": voice, "opts": opts})
        
        while True:
            msg = self._read()
            if msg is None:
                break
            
            msg_type = msg.get("type")
            if msg_type == "chunk":
                yield np.frombuffer(msg["data"], dtype=np.float32)
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise RuntimeError(msg.get("msg"))
            
    def generate(self, text: str, voice: str, **opts) -> np.ndarray:
        """Helper to generate full audio at once using stream."""
        chunks = list(self.stream(text, voice, **opts))
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    def close(self):
        if self._proc:
            try:
                self._write({"cmd": "quit"})
                self._proc.wait(timeout=5)
            except Exception:
                # If process is already dead or stuck, just kill it
                if self._proc:
                    self._proc.kill()
            finally:
                self._proc = None
    
    def __del__(self):
        self.close()
