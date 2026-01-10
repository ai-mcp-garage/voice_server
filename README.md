# Voice Server

A high-performance, modular Text-to-Speech (TTS) server designed for low-latency real-time applications. It supports token-based streaming (perfect for LLMs) and parallel audio generation using a worker pool.

## Features

- 🚀 **Real-time Token Streaming**: Start generating audio mid-sentence as tokens arrive from an LLM.
- ⚡ **Parallel Generation**: Worker pool architecture generates sentence N+1 while sentence N plays, minimizing latency.
- 🔌 **Modular Backends**: Run multiple TTS backends (Kokoro, VibeVoice, Chatterbox) in isolated environments.
- 🛠️ **CLI & Client**: Includes a robust CLI for testing and a Python client example.

## Advanced Features

### 🗣️ Voice Cloning
You can clone any voice by simply dropping an audio file (`.wav`, `.mp3`, `.flac`) into the `voice_server/voice_samples/` directory. 

- **Supported Backends:** `chatterbox`
- **Usage:** The filename becomes the voice ID.
  - Example: Drop `obama.wav` into `voice_samples/`.
  - Run: `uv run cli.py --backend chatterbox --voice obama --text "Hello there."`

### 🎭 Paralinguistic Support
The **Chatterbox** backend supports rich emotional expression tags within the text:

- `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, `[throat]`, `[sneeze]`, `[sniff]`
- **Example:** `"I can't believe it! [laugh] That's hilarious."`

### ⚡ Worker Pool (Parallel Generation)
This server implements a **backend worker pool** to solve the "sentence latency" problem. 
- While Sentence 1 is playing, Sentence 2 is already generating in a background process.
- This ensures **zero-gap playback** even for heavy models.
- Configurable pool size via `--pool-size N`.

## Installation

This project is managed with `uv`.

```bash
cd voice_server
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

Run the server with a specific backend and pool size:

```bash
# Start with Kokoro backend and 2 workers
uv run server.py --backend kokoro --pool-size 2
```

### CLI Testing

Test the backend directly from the command line:

```bash
# Interactive REPL
uv run cli.py --backend kokoro --repl

# Token Streaming Test (Simulates LLM)
uv run cli.py --backend kokoro --stream-tokens --pool-size 2
```

### Python Client

Run the example client to connect to the WebSocket server:

```bash
uv run client.py --url ws://localhost:8000/stream-tokens
```

## API

### WebSocket `/stream-tokens`

Connect to `ws://localhost:8000/stream-tokens`.

**Protocol:**

1. **Send Config:**
   ```json
   {"voice": "af_heart", "opts": {"speed": 1.0}}
   ```

2. **Send Tokens:**
   ```json
   {"token": "Hello "}
   {"token": "world. "}
   ```

3. **Receive Updates:**
   - **Status:** `{"type": "generating", "sentence": "Hello world.", "queue_depth": 0}`
   - **Audio:** Binary messages containing PCM audio data (int16, 24kHz).

4. **Finish:**
   Send `{"done": true}` to close the stream.

## Backends

- **VibeVoice**: Default backend.
- **Kokoro**: High-quality boolean TTS (requires setup in `backends/kokoro`).
- **Chatterbox**: Experimental backend.

Add new backends by creating an adapter in `backends/` and updating `backends.yaml`.
