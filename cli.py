#!/usr/bin/env python3
"""Voice Server CLI - Test backends from command line."""
import argparse
import struct
import subprocess
import time
from pathlib import Path
import time
import asyncio
import numpy as np
from process_backend import SubprocessBackend
from backend_pool import BackendPool


def repl_mode(backend, voice: str, opts: dict):
    """Interactive streaming TTS mode."""
    import sounddevice as sd
    
    sr = backend.get_sample_rate()
    backend_name = backend.__name__.split('.')[-1]
    
    # Preload model before entering loop
    print(f"[REPL] Warming up {backend_name}...")
    _ = list(backend.stream("warmup", voice, **opts))
    
    print(f"\n[REPL] Voice: {voice} | Backend: {backend_name}")
    print("[REPL] Type text and press Enter to speak. Ctrl+C to exit.\n")
    
    while True:
        try:
            text = input("> ").strip()
            if not text:
                continue
            
            start_time = time.perf_counter()
            first_chunk = True
            
            with sd.OutputStream(samplerate=sr, channels=1, dtype='float32') as stream:
                for chunk in backend.stream(text, voice, **opts):
                    if first_chunk:
                        latency = (time.perf_counter() - start_time) * 1000
                        print(f"[REPL] First audio: {latency:.0f}ms")
                        first_chunk = False
                    stream.write(chunk.reshape(-1, 1))
            
            total = time.perf_counter() - start_time
            print(f"[REPL] Total: {total:.2f}s\n")
            
        except KeyboardInterrupt:
            print("\n[REPL] Bye.")
            break
        except EOFError:
            print("\n[REPL] Bye.")
            break


def stream_tokens_mode(backend, voice: str, opts: dict, delay: float = 0.05):
    """
    Token streaming test mode.
    
    Uses TokenStreamManager and SentenceQueue to parallelize generation
    if backend is a pool.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from token_streamer import TokenStreamManager, SentenceQueue
    import sounddevice as sd
    
    sr = backend.get_sample_rate()
    backend_name = backend.__name__.split('.')[-1]
    
    # Preload model
    print(f"[STREAM] Warming up {backend_name}...")
    # Trigger a warmup on the pool (warmup all workers? or just one?)
    # Just simple warmup
    if hasattr(backend, "list_voices"):
        _ = backend.list_voices()
    
    print(f"\n[STREAM] Voice: {voice} | Backend: {backend_name}")
    print(f"[STREAM] Token delay: {delay*1000:.0f}ms")
    print("[STREAM] Enter text to stream (simulates LLM tokens). Ctrl+C to exit.\n")
    
    async def _run_stream(text):
        manager = TokenStreamManager()
        queue = SentenceQueue(voice=voice, opts=opts)
        
        # Audio output stream
        output_stream = sd.OutputStream(samplerate=sr, channels=1, dtype='float32')
        output_stream.start()
        
        async def send_audio(data: bytes):
            # Convert bytes back to numpy for sounddevice
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
            output_stream.write(pcm.reshape(-1, 1))

        async def send_status(msg: dict):
            if msg["type"] == "generating":
                print(f"[STREAM] → Generating: {msg['sentence']} (Queue: {msg['queue_depth']})")
            elif msg["type"] == "error":
                print(f"[STREAM] ✖ Error: {msg['msg']}")
            elif msg["type"] == "done":
                pass

        # Start producer
        gen_task = asyncio.create_task(queue.generate_loop(backend, send_audio, send_status))
        
        # Tokenize and feed
        tokens = []
        for word in text.split():
            tokens.append(word + " ")
            
        print(f"[STREAM] Streaming {len(tokens)} tokens...")
        start_time = time.perf_counter()
        
        for i, token in enumerate(tokens):
            if delay > 0:
                await asyncio.sleep(delay)
            
            print(f"  [{i+1}/{len(tokens)}] '{token.strip()}'")
            sentences = manager.add_token(token)
            for s in sentences:
                await queue.enqueue(s)
        
        # Flush
        remaining = manager.flush()
        if remaining:
             await queue.enqueue(remaining)
        
        await queue.mark_done()
        
        # Wait for finish
        await gen_task
        
        total = time.perf_counter() - start_time
        print(f"[STREAM] Done in {total:.2f}s\n")
        
        output_stream.stop()
        output_stream.close()

    while True:
        try:
            text = input("> ").strip()
            if not text:
                continue
            asyncio.run(_run_stream(text))
            
        except KeyboardInterrupt:
            print("\n[STREAM] Bye.")
            break
        except EOFError:
            print("\n[STREAM] Bye.")
            break


def parse_opt(opt: str) -> tuple[str, any]:
    """Parse key=value option, auto-converting types."""
    if "=" not in opt:
        return opt, True  # Flag style
    key, val = opt.split("=", 1)
    # Try to parse as number
    try:
        if "." in val:
            return key, float(val)
        return key, int(val)
    except ValueError:
        # Bool strings
        if val.lower() in ("true", "yes", "1"):
            return key, True
        if val.lower() in ("false", "no", "0"):
            return key, False
        return key, val


def main():
    parser = argparse.ArgumentParser(description="Voice Server CLI")
    parser.add_argument("--backend", type=str, default="vibevoice", help="Backend name")
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--voice", type=str, default="")
    parser.add_argument("--text", type=str)
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--repl", action="store_true", help="Interactive REPL mode with streaming")
    parser.add_argument("--stream-tokens", action="store_true", help="Token streaming test mode (simulates LLM)")
    parser.add_argument("--token-delay", type=float, default=0.05, help="Delay between tokens in stream mode (seconds)")
    # Backend-specific options as key=value pairs
    parser.add_argument("--opt", action="append", default=[],
                        help="Backend option: --opt speed=0.8 --opt cfg_scale=1.5")
    parser.add_argument("--pool-size", type=int, default=1, help="Number of backend workers (for stream-tokens)")
    args = parser.parse_args()
    
    print(f"Loading backend: {args.backend} (pool_size={args.pool_size})")
    if args.pool_size > 1 or args.stream_tokens:
        # Always use pool for streaming mode to test pool logic (size 1 is fine)
        pool = BackendPool(args.backend, args.pool_size)
        pool.start()
        backend = pool
    else:
        backend = SubprocessBackend(args.backend)
    
    if args.list_voices:
        voices = backend.list_voices()
        print(f"\nVoices ({len(voices)}):")
        for v in voices:
            print(f"  - {v}")
        return
    
    # Parse backend options
    opts = {}
    for opt in args.opt:
        k, v = parse_opt(opt)
        opts[k] = v
    
    voice = args.voice or backend.list_voices()[0]
    
    if args.repl:
        repl_mode(backend, voice, opts)
        return
    
    if args.stream_tokens:
        stream_tokens_mode(backend, voice, opts, delay=args.token_delay)
        return
    
    if not args.text:
        parser.error("--text required")
    
    print(f"Voice: {voice}")
    print(f"Text: '{args.text[:50]}...'")
    if opts:
        print(f"Options: {opts}")
    
    audio = backend.generate(args.text, voice, **opts)
    sr = backend.get_sample_rate()
    print(f"Generated {len(audio) / sr:.2f}s")
    
    import soundfile as sf
    sf.write(args.output, audio, sr)
    print(f"Saved: {args.output}")
    
    if args.play:
        import sounddevice as sd
        print("Playing...")
        sd.play(audio, sr)
        sd.wait()


if __name__ == "__main__":
    main()
