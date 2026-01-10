"""
Backend Pool - Manages multiple backend processes for parallel generation.
"""
from typing import Iterator
import asyncio
import itertools
import numpy as np
from process_backend import SubprocessBackend

class BackendPool:
    """
    Pool of SubprocessBackend instances.
    Generates audio by dispatching requests to workers in round-robin fashion.
    """
    
    def __init__(self, backend_name: str, pool_size: int = 1):
        self.backend_name = backend_name
        self.pool_size = max(1, pool_size)
        self.workers = []
        self.pool_size = max(1, pool_size)
        self.workers = []
        self._worker_cycle = None
        self._queue = None  # asyncio.Queue for worker acquisition
        self._sample_rate = 24000
    
    def start(self):
        """Start all workers."""
        print(f"[pool] Starting {self.pool_size} workers for {self.backend_name}...")
        for i in range(self.pool_size):
            try:
                worker = SubprocessBackend(self.backend_name)
                self.workers.append(worker)
                print(f"[pool] Worker {i+1}/{self.pool_size} ready")
            except Exception as e:
                print(f"[pool] Worker {i+1} failed to start: {e}")
                raise
        
        if not self.workers:
            raise RuntimeError("No workers started successfully")
            
        self._sample_rate = self.workers[0].get_sample_rate()
        self._worker_cycle = itertools.cycle(self.workers)
        
    async def acquire(self) -> SubprocessBackend:
        """Acquire an available worker (async, blocks if none available)."""
        if not self.workers:
            raise RuntimeError("Pool not started")
            
        if self._queue is None:
            # Initialize queue on first use (must be in loop)
            self._queue = asyncio.Queue()
            for w in self.workers:
                self._queue.put_nowait(w)
        
        return await self._queue.get()

    async def release(self, worker: SubprocessBackend):
        """Return a worker to the pool."""
        if self._queue:
            await self._queue.put(worker)

    def get_next_worker(self) -> SubprocessBackend:
        """
        Deprecated: Unsafe round-robin (for sync compatibility only).
        Use acquire() instead.
        """
        if not self.workers:
            raise RuntimeError("Pool not started")
        return next(self._worker_cycle)
    
    def list_voices(self):
        """Get voices from first worker."""
        if not self.workers:
            return []
        return self.workers[0].list_voices()
    
    def get_sample_rate(self):
        return self._sample_rate
    
    def generate(self, text: str, voice: str, **opts):
        """Generate audio using next available worker."""
        return self.get_next_worker().generate(text, voice, **opts)

    def stream(self, text: str, voice: str, **opts):
        """Stream audio using next available worker."""
        return self.get_next_worker().stream(text, voice, **opts)
        
    @property
    def __name__(self):
        return f"pool.{self.backend_name}"

    def close(self):
        """Stop all workers."""
        for w in self.workers:
            w.close()
        self.workers = []

    def __del__(self):
        self.close()
