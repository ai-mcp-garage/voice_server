#!/usr/bin/env python3
"""
Voice Server - Minimal TTS HTTP Server

Uses backend adapters from the backends/ folder via a worker pool.
Each backend runs in one or more isolated subprocesses.
"""
import argparse
import asyncio
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

from backend_pool import BackendPool


class TTSRequest(BaseModel):
    voice: str = ""
    text: str
    play_local: bool = False
    # Provider-specific options can be added here
    cfg_scale: float = 1.5





def create_app(backend_name: str = "vibevoice", pool_size: int = 1) -> FastAPI:
    """Create FastAPI app with specified backend pool."""
    app = FastAPI(title="Voice Server")
    
    @app.on_event("startup")
    async def startup():
        print(f"[startup] Starting backend pool: {backend_name} (x{pool_size})")
        pool = BackendPool(backend_name, pool_size)
        pool.start()
        app.state.backend = pool
        
        voices = pool.list_voices()
        print(f"[startup] Ready with {len(voices)} voices")

    @app.on_event("shutdown")
    async def shutdown():
        if hasattr(app.state, "backend"):
            print("[shutdown] Closing backend pool...")
            app.state.backend.close()
    
    @app.get("/")
    def root():
        backend = app.state.backend
        return {
            "backend": backend.__name__.split(".")[-1],
            "voices": len(backend.list_voices()),
            "sample_rate": backend.get_sample_rate(),
        }
    
    @app.get("/voices")
    def get_voices():
        return {"voices": app.state.backend.list_voices()}
    
    @app.post("/tts")
    async def generate_tts(req: TTSRequest):
        backend = app.state.backend
        
        if not req.text.strip():
            raise HTTPException(400, "Text cannot be empty")
        
        voice = req.voice or backend.list_voices()[0]
        audio = await asyncio.to_thread(
            backend.generate, req.text, voice, cfg_scale=req.cfg_scale
        )
        sr = backend.get_sample_rate()
        duration = len(audio) / sr
        
        if req.play_local:
            try:
                import sounddevice as sd
                await asyncio.to_thread(sd.play, audio, sr)
                await asyncio.to_thread(sd.wait)
                return {"status": "played", "duration": duration, "voice": voice}
            except Exception as e:
                raise HTTPException(500, f"Playback failed: {e}")
        else:
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            buf.seek(0)
            return Response(
                content=buf.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"},
            )
    
    @app.websocket("/stream")
    async def stream_tts(ws: WebSocket):
        await ws.accept()
        
        text = ws.query_params.get("text", "")
        voice = ws.query_params.get("voice", "")
        
        if not text.strip():
            await ws.close(code=1008, reason="Text required")
            return
        
        backend = app.state.backend
        voice = voice or backend.list_voices()[0]
        
        try:
            iterator = backend.stream(text, voice)
            sentinel = object()
            
            while ws.client_state == WebSocketState.CONNECTED:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
                await ws.send_bytes(pcm.tobytes())
            
            await ws.send_text(json.dumps({"done": True}))
        except WebSocketDisconnect:
            pass
        finally:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
    
    @app.websocket("/stream-tokens")
    async def stream_tokens(ws: WebSocket):
        """
        Token streaming endpoint for real-time LLM → TTS.
        
        Client sends:
            {"token": "Hello"}      - Accumulate token
            {"done": true}          - Flush buffer, close
            {"voice": "name"}       - Optional: set voice (first message)
            {"opts": {...}}         - Optional: backend options (first message)
        
        Server sends:
            bytes                   - PCM audio chunks (int16, mono, backend sample rate)
            {"type": "generating", "sentence": "...", "queue_depth": N}
            {"type": "done"}        - All audio complete
            {"type": "error", "msg": "..."}
        """
        from token_streamer import TokenStreamManager, SentenceQueue
        
        await ws.accept()
        
        backend = app.state.backend
        voice = ws.query_params.get("voice", "") or backend.list_voices()[0]
        
        # Initialize token manager and sentence queue
        manager = TokenStreamManager()
        queue = SentenceQueue(voice=voice, opts={})
        
        async def send_audio(data: bytes):
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_bytes(data)
        
        async def send_status(msg: dict):
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(json.dumps(msg))
        
        # Start generation loop in background
        gen_task = asyncio.create_task(
            queue.generate_loop(backend, send_audio, send_status)
        )
        
        try:
            while ws.client_state == WebSocketState.CONNECTED:
                data = await ws.receive_text()
                msg = json.loads(data)
                
                # Handle voice/opts setting (usually first message)
                if "voice" in msg:
                    queue.voice = msg["voice"]
                if "opts" in msg:
                    queue.opts = msg["opts"]
                
                # Handle token
                if "token" in msg:
                    sentences = manager.add_token(msg["token"])
                    for sentence in sentences:
                        await queue.enqueue(sentence)
                
                # Handle done signal
                if msg.get("done"):
                    remaining = manager.flush()
                    if remaining:
                        await queue.enqueue(remaining)
                    await queue.mark_done()
                    break
        
        except WebSocketDisconnect:
            await queue.mark_done()
        except Exception as e:
            await send_status({"type": "error", "msg": str(e)})
            await queue.mark_done()
        
        # Wait for generation to complete
        await gen_task
        
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
    
    return app


def main():
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Voice Server")
    parser.add_argument("--backend", type=str, default="vibevoice", help="Backend name")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--pool-size", type=int, default=1, help="Number of backend workers")
    args = parser.parse_args()
    
    app = create_app(args.backend, args.pool_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
