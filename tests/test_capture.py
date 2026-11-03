"""Capture seam: injected reader drives the mic without a real device."""
from __future__ import annotations

import asyncio

import pytest

from voice_input.capture import PortAudioMic
from voice_input.vad import FRAME_BYTES


def test_injected_reader_roundtrip():
    blocks = [bytes(FRAME_BYTES), bytes(FRAME_BYTES), None]

    async def reader():
        return blocks.pop(0)

    async def scenario():
        mic = PortAudioMic(reader=reader)
        got = []
        while True:
            b = await mic.read_block()
            if b is None:
                break
            got.append(b)
        await mic.aclose()
        return got

    assert asyncio.run(scenario()) == [bytes(FRAME_BYTES), bytes(FRAME_BYTES)]


def test_open_noop_with_injected_reader():
    async def scenario():
        mic = PortAudioMic(reader=lambda: asyncio.sleep(0) or None)
        await mic.open()  # must not import sounddevice
        await mic.aclose()

    asyncio.run(scenario())


def test_open_uses_stream_without_injected_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: bound-method identity made _injected() always True, so
    open() skipped stream creation and the first read_block asserted."""

    class FakeStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def read(self, frames: int) -> tuple[bytes, bool]:
            return b"\x00" * (frames * 2), False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    streams: list[FakeStream] = []

    def factory(**kwargs: object) -> FakeStream:
        stream = FakeStream(**kwargs)
        streams.append(stream)
        return stream

    monkeypatch.setattr("sounddevice.RawInputStream", factory)

    async def scenario() -> None:
        mic = PortAudioMic(device="pipewire")
        await mic.open()
        assert streams and streams[0].started
        block = await asyncio.wait_for(mic.read_block(), timeout=2.0)
        assert block is not None and len(block) == FRAME_BYTES
        await mic.aclose()
        assert streams[0].closed

    asyncio.run(scenario())
