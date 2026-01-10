#!/usr/bin/env python3
"""
Voice Server Client Example.

Demonstrates how to connect to the /stream-tokens endpoint,
simulate an LLM streaming tokens, and play back the received audio.

Dependencies:
    pip install websockets sounddevice numpy msgpack
"""
import asyncio
import argparse
import sys
import json
import time
import numpy as np
import websockets
import sounddevice as sd

# Simulated LLM output
DEMO_TEXT = """
Hello! This is a demonstration of the Vibe Voice TTS server.
I am streaming these tokens one by one, simulating a Large Language Model.
The server accumulates them into sentences.
Then, it generates audio in parallel using a worker pool.
This ensures there are no pauses between sentences!
Let's see if it can handle a longer stream of thought without stuttering.
"""

async def stream_audio(uri, text, voice="af_heart", speed=1.0):
    async with websockets.connect(uri) as websocket:
        print(f"[client] Connected to {uri}")
        
        # 1. Send configuration
        await websocket.send(json.dumps({
            "voice": voice,
            "opts": {"speed": speed}
        }))
        
        # 2. Setup audio playback
        # Default Kokoro sample rate is 24000
        output_stream = sd.OutputStream(channels=1, samplerate=24000, dtype='float32')
        output_stream.start()
        
        # 3. Concurrent tasks: Sending tokens vs Receiving audio
        
        async def send_tokens():
            """Simulates LLM token streaming."""
            tokens = text.split(" ")
            print(f"[client] Streaming {len(tokens)} tokens...")
            
            for i, token in enumerate(tokens):
                # Send token (with trailing space usually)
                msg = {"token": token + " "}
                await websocket.send(json.dumps(msg))
                
                # Simulate LLM generation time
                await asyncio.sleep(0.05) 
                print(f"  -> Sent: {token}", end="\r")
            
            # Send done signal
            await websocket.send(json.dumps({"done": True}))
            print("\n[client] Finished sending tokens.")

        async def receive_audio():
            """Receives audio chunks and plays them."""
            print("[client] Listening for audio...")
            first_packet = True
            start_time = None
            
            try:
                while True:
                    message = await websocket.recv()
                    
                    if isinstance(message, bytes):
                        # Binary audio data (int16 PCM)
                        if first_packet:
                            start_time = time.perf_counter()
                            print(f"\n[client] 🔊 First audio received!")
                            first_packet = False
                        
                        # Convert int16 bytes -> float32 for sounddevice
                        pcm_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32767.0
                        output_stream.write(pcm_data.reshape(-1, 1))
                        
                    else:
                        # JSON status message
                        data = json.loads(message)
                        if data.get("type") == "done":
                            print(f"[client] Server signaled complete.")
                            break
                        elif data.get("type") == "error":
                            print(f"[client] ✖ Error: {data.get('msg')}")
                            break
                        elif data.get("type") == "generating":
                            print(f"[client] ⚙️ Generating: {data.get('sentence')}")
                            
            except websockets.exceptions.ConnectionClosed:
                print("[client] Connection closed.")

        # Run send and receive concurrently
        await asyncio.gather(send_tokens(), receive_audio())
        
        # Cleanup
        output_stream.stop()
        output_stream.close()

def main():
    parser = argparse.ArgumentParser(description="Voice Server Client")
    parser.add_argument("--url", default="ws://localhost:8000/stream-tokens", help="Server WebSocket URL")
    parser.add_argument("--text", default=DEMO_TEXT, help="Text to stream")
    parser.add_argument("--voice", default="af_heart", help="Voice ID")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    args = parser.parse_args()

    try:
        asyncio.run(stream_audio(args.url, args.text, args.voice, args.speed))
    except KeyboardInterrupt:
        print("\n[client] Interrupted.")
    except Exception as e:
        print(f"[client] Error: {e}")

if __name__ == "__main__":
    main()
