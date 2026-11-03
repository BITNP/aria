"""Microphone capture with minimal dependencies.

Uses sounddevice (PortAudio) for a blocking raw PCM16 mono 16 kHz capture loop of 20 ms
blocks, matching the frame contract in voice_input.vad.  A reader callable is injected
so tests never touch a real device.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from voice_input.vad import FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE_HZ

if TYPE_CHECKING:
    from sounddevice import RawInputStream

LOGGER = logging.getLogger(__name__)

RawBlock = bytes
RawBlockReader = Callable[[], Awaitable[RawBlock | None]]

class MicReader(Protocol):
    """Anything that yields 20 ms PCM16 blocks until None."""

    def read_block(self) -> Awaitable[RawBlock | None]: ...


class PortAudioMic(MicReader):
    """Async wrapper around a blocking sounddevice RawInputStream.

    open() starts a dedicated capture task that fills an asyncio queue from the
    blocking PortAudio stream; read_block() pops from the queue.  This keeps the
    audio callback path off the event loop without device-specific thread wiring.
    """

    def __init__(self, device: str | None = None, reader: RawBlockReader | None = None) -> None:
        self._device = device
        self._injected_reader = reader is not None
        self._reader: RawBlockReader = reader if reader is not None else self._sounddevice_read
        self._queue: asyncio.Queue[RawBlock | None] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stream: RawInputStream | None = None
        self._closed = False

    async def open(self) -> None:
        """Open the device and start reading. No-op when an injected reader is used."""
        if self._injected():
            return
        import sounddevice  # lazy import keeps tests/CI device-free

        self._queue = asyncio.Queue(maxsize=64)
        self._stream = sounddevice.RawInputStream(
            samplerate=int(SAMPLE_RATE_HZ),
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=self._device,
        )
        self._stream.start()
        self._task = asyncio.create_task(self._capture_loop())

    def _injected(self) -> bool:
        # Bound methods compare by value but never by identity (each attribute
        # access creates a fresh object), so `self._reader is not
        # self._sounddevice_read` was always True: open() silently no-opped on
        # the real-device path. The constructor records injection explicitly.
        return self._injected_reader

    async def _sounddevice_read(self) -> RawBlock | None:
        assert self._queue is not None
        assert self._stream is not None
        return await self._queue.get()

    async def _capture_loop(self) -> None:
        stream = self._stream
        queue = self._queue
        assert stream is not None
        assert queue is not None
        try:
            while True:
                payload, overflowed = stream.read(FRAME_SAMPLES)
                if overflowed:
                    LOGGER.warning("audio input overflowed (device too slow)")
                block = bytes(payload)
                if len(block) != FRAME_BYTES:
                    LOGGER.warning("captured block length %d; dropping", len(block))
                    continue
                await queue.put(block)
        finally:
            try:
                queue.put_nowait(None)  # signal end of stream; never block in shutdown
            except asyncio.QueueFull:
                pass

    async def read_block(self) -> RawBlock | None:
        """Return the next 20 ms block, or None when capture is over."""
        return await self._reader()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


__all__ = ["PortAudioMic", "MicReader", "RawBlock", "RawBlockReader"]
