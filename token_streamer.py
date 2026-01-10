#!/usr/bin/env python3
"""
Token Streaming - Real-time token accumulation and sentence-based TTS pipeline.

Enables streaming TTS from LLM token output by:
1. Accumulating tokens into a buffer
2. Detecting sentence boundaries
3. Queueing complete sentences for audio generation
4. Streaming audio back as it's generated
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Protocol, Any
import collections
import time

import numpy as np


class TokenStreamManager:
    """
    Accumulates tokens and emits complete sentences.
    
    Sentence boundaries are detected by configurable delimiters.
    Handles edge cases like abbreviations (Mr., Dr., etc.) and ellipsis.
    """
    
    # Common abbreviations that shouldn't trigger sentence breaks
    ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
        "inc", "ltd", "corp", "st", "ave", "blvd", "apt", "no",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    }
    
    def __init__(
        self,
        delimiters: str = ".!?",
        min_length: int = 10,
        max_length: int = 500,
    ):
        """
        Args:
            delimiters: Characters that mark sentence boundaries
            min_length: Minimum sentence length before allowing break
            max_length: Force break at this length even without delimiter
        """
        self.buffer = ""
        self.delimiters = delimiters
        self.min_length = min_length
        self.max_length = max_length
    
    def add_token(self, token: str) -> list[str]:
        """
        Add token to buffer, return list of complete sentences (if any).
        
        Args:
            token: Text token to accumulate
            
        Returns:
            List of complete sentences extracted from buffer (may be empty)
        """
        self.buffer += token
        return self._extract_sentences()
    
    def _extract_sentences(self) -> list[str]:
        """Extract complete sentences from buffer."""
        sentences = []
        
        while True:
            # Find potential sentence boundary
            boundary_idx = self._find_boundary()
            
            if boundary_idx is None:
                # No boundary found, check max length
                if len(self.buffer) >= self.max_length:
                    # Force break at last space
                    last_space = self.buffer.rfind(" ", 0, self.max_length)
                    if last_space > self.min_length:
                        sentence = self.buffer[:last_space].strip()
                        self.buffer = self.buffer[last_space:].lstrip()
                        if sentence:
                            sentences.append(sentence)
                        continue
                break
            
            # Extract sentence
            sentence = self.buffer[:boundary_idx + 1].strip()
            self.buffer = self.buffer[boundary_idx + 1:].lstrip()
            
            if sentence and len(sentence) >= self.min_length:
                sentences.append(sentence)
            elif sentence:
                # Too short, put back in buffer
                self.buffer = sentence + " " + self.buffer
                break
        
        return sentences
    
    def _find_boundary(self) -> int | None:
        """Find the index of the next sentence boundary, or None."""
        for i, char in enumerate(self.buffer):
            if char not in self.delimiters:
                continue
            
            # Check if this looks like a real sentence boundary
            if self._is_sentence_boundary(i):
                return i
        
        return None
    
    def _is_sentence_boundary(self, idx: int) -> bool:
        """Check if delimiter at idx is a real sentence boundary."""
        # Must have enough content before
        if idx < self.min_length - 1:
            return False
        
        # Check for abbreviation (word before period)
        if self.buffer[idx] == ".":
            # Find the word before the period
            word_start = idx - 1
            while word_start >= 0 and self.buffer[word_start].isalpha():
                word_start -= 1
            word = self.buffer[word_start + 1:idx].lower()
            
            if word in self.ABBREVIATIONS:
                return False
            
            # Check for ellipsis (...)
            if idx + 2 < len(self.buffer) and self.buffer[idx:idx+3] == "...":
                return False
            
            # Check for decimal number (1.5)
            if word_start >= 0 and self.buffer[word_start].isdigit():
                if idx + 1 < len(self.buffer) and self.buffer[idx + 1].isdigit():
                    return False
        
        # Should have space or end of buffer after
        if idx + 1 < len(self.buffer):
            next_char = self.buffer[idx + 1]
            # Allow quotes, parens after punctuation
            if next_char not in " \t\n\"')]}'":
                return False
        
        return True
    
    def flush(self) -> str | None:
        """
        Return remaining buffer content and clear it.
        
        Called when client signals end of input to get any trailing text.
        
        Returns:
            Remaining text or None if empty
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = ""


@dataclass
class SentenceQueue:
    """
    Async queue managing sentence -> audio pipeline.
    
    Handles concurrent sentence generation with proper ordering.
    """
    
    voice: str = ""
    opts: dict = field(default_factory=dict)
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _done: bool = False
    
    async def enqueue(self, sentence: str):
        """Add sentence to generation queue."""
        await self._queue.put(sentence)
    
    async def mark_done(self):
        """Signal that no more sentences will be added."""
        self._done = True
        await self._queue.put(None)  # Sentinel
    
    def queue_depth(self) -> int:
        """Return current queue depth."""
        return self._queue.qsize()
    
    async def generate_loop(
        self,
        backend: Any,  # BackendPool or SubprocessBackend
        send_audio: Callable[[bytes], None],
        send_status: Callable[[dict], None],
    ) -> None:
        """
        Background task: pull sentences, dispatch to workers, ensure ordered output.
        """
        # Queue to hold running generation tasks in order: asyncio.Queue[asyncio.Queue[bytes]]
        ordered_results = asyncio.Queue()
        
        async def _worker_task(worker_instance, text, voice, opts, result_queue, should_release=False):
            """Runs blocking generator in thread and pushes chunks to result_queue."""
            try:
                # We need to iterate the generator in a thread to avoid blocking the loop
                # AND to ensure we drain the pipe concurrently.
                # Since passing a generator across threads is tricky, we define the iterator wrapper here.
                def _run_gen():
                    for chunk in worker_instance.stream(text, voice, **opts):
                        # Convert to bytes immediately to send over queue
                        if hasattr(chunk, 'numpy'):
                            chunk = chunk.numpy()
                        chunk = np.asarray(chunk, dtype=np.float32)
                        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
                        # We use call_soon_threadsafe to put into queue from thread
                        asyncio.run_coroutine_threadsafe(result_queue.put(pcm.tobytes()), loop)
                
                loop = asyncio.get_running_loop()
                await asyncio.to_thread(_run_gen)
                await result_queue.put(None)  # Sentinel for this sentence
            except Exception as e:
                # Signal error in queue
                await result_queue.put(e)
            finally:
                if should_release and hasattr(backend, 'release'):
                    await backend.release(worker_instance)

        async def _dispatcher():
            """Pulls sentences and spawns worker tasks."""
            while True:
                sentence = await self._queue.get()
                if sentence is None:
                    await ordered_results.put(None) # End of stream
                    break
                
                # Get worker (handle pool logic)
                should_release = False
                if hasattr(backend, 'acquire'):
                    worker = await backend.acquire()
                    should_release = True
                elif hasattr(backend, 'get_next_worker'):
                    worker = backend.get_next_worker() # Fallback for non-async pools (deprecated)
                else:
                    worker = backend
                    
                # Create a result queue for this specific sentence
                res_q = asyncio.Queue(maxsize=100) 
                
                # Notify generating
                await send_status({
                    "type": "generating",
                    "sentence": sentence[:50] + "..." if len(sentence) > 50 else sentence,
                    "queue_depth": self.queue_depth(),
                })
                
                # Spawn worker task
                asyncio.create_task(_worker_task(worker, sentence, self.voice, self.opts, res_q, should_release))
                
                # Queue the result channel
                await ordered_results.put(res_q)
                self._queue.task_done()
        
        # Start dispatcher
        dispatch_task = asyncio.create_task(_dispatcher())
        
        # Main Output Loop - Serializes the streams
        try:
            while True:
                sentence_res_q = await ordered_results.get()
                if sentence_res_q is None:
                    break
                
                # Drain chunks for this sentence
                while True:
                    chunk = await sentence_res_q.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, Exception):
                        await send_status({"type": "error", "msg": str(chunk)})
                        break
                    
                    await send_audio(chunk)
                
                ordered_results.task_done()
                
        except Exception as e:
            await send_status({"type": "error", "msg": str(e)})
        finally:
            dispatch_task.cancel()
            
        await send_status({"type": "done"})
