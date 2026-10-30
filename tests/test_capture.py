"""Capture seam: injected reader drives the mic without a real device."""
from __future__ import annotations

import asyncio

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
