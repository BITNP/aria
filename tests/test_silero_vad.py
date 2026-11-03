"""Silero endpointer: task-route contract, endpointing, trimming, degradation.

The audio.cpp service is stubbed with an injected httpx transport (test-only,
mirroring tests/test_asr.py); the polling and endpointing logic is exercised
for real.
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import wave
from io import BytesIO

import httpx

from voice_input.config import load_settings
from voice_input.main import build_endpointer
from voice_input.silero_vad import SileroEndpointer, SileroVadClient
from voice_input.vad import EndpointDetector, EnergyEndpointer

FRAME = b"\x00" * 640  # one silent 20 ms frame
LOUD_FRAME = struct.pack("<320h", *([1000] * 320))


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request):
        return await self._handler(request)


def _posted_samples(request: httpx.Request) -> int:
    body = json.loads(request.content)
    wav = base64.b64decode(body["request"]["audio_base64"])
    with wave.open(BytesIO(wav), "rb") as reader:
        return reader.getnframes()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_FakeTransport(handler))


def _run(coro):
    return asyncio.run(coro)


def _endpointer(handler, **kwargs) -> SileroEndpointer:
    fallback = EnergyEndpointer(
        EndpointDetector(threshold=300, trailing_silence_frames=2, max_frames=100, pre_roll_frames=1)
    )
    client = SileroVadClient("http://vad.test:8080", "silero-vad", client=_client(handler))
    return SileroEndpointer(client, fallback=fallback, poll_s=0.0, **kwargs)


def _feed_all(endpointer: SileroEndpointer, frames: list[bytes]) -> list:
    async def scenario():
        emitted = []
        for index, frame in enumerate(frames):
            segment = await endpointer.feed(frame, index * 320)
            if segment is not None:
                emitted.append(segment)
        return emitted

    return _run(scenario())


def test_client_posts_task_run_with_512_aligned_wav():
    seen: dict[str, object] = {}

    async def handler(request):
        seen["path"] = request.url.path
        body = json.loads(request.content)
        seen["model"] = body["model"]
        seen["request_keys"] = sorted(body["request"])
        seen["samples"] = _posted_samples(request)
        return httpx.Response(200, json={"segments": [{"start_sample": 512, "end_sample": 1024}]})

    async def scenario():
        client = SileroVadClient("http://vad.test:8080", "silero-vad", client=_client(handler))
        try:
            return await client.segments(b"\x00" * 1000)  # 500 samples -> padded to 512
        finally:
            await client.aclose()

    segments = _run(scenario())
    assert segments == [(512, 1024)]
    assert seen["path"] == "/v1/tasks/run"
    assert seen["model"] == "silero-vad"
    # Streaming sessions bake their VAD config at model load and ignore
    # per-request options (ADR-0004): sending any would be a silent no-op.
    assert seen["request_keys"] == ["audio_base64"]
    assert seen["samples"] == 512  # zero-padded to a whole streaming window


def test_client_raises_on_service_error():
    async def handler(request):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async def scenario():
        client = SileroVadClient("http://vad.test:8080", "silero-vad", client=_client(handler))
        try:
            await client.segments(b"\x00" * 640)
        finally:
            await client.aclose()

    try:
        _run(scenario())
        raise AssertionError("expected the HTTP error to surface")
    except httpx.HTTPStatusError:
        pass


def test_client_raises_on_malformed_response():
    async def handler(request):
        return httpx.Response(200, json={"segments": "not-a-list"})

    async def scenario():
        client = SileroVadClient("http://vad.test:8080", "silero-vad", client=_client(handler))
        try:
            await client.segments(b"\x00" * 640)
        finally:
            await client.aclose()

    try:
        _run(scenario())
        raise AssertionError("expected the malformed response to raise")
    except ValueError:
        pass


def test_endpoint_fires_after_trailing_silence_past_last_segment():
    # Speech stays fixed at [1000, 5000]; the endpoint must fire once the
    # buffer tail past 5000 reaches trailing (2 frames) + blind window (3200).
    async def handler(request):
        return httpx.Response(200, json={"segments": [
            {"start_sample": 1_000, "end_sample": 5_000},
        ]})

    endpointer = _endpointer(handler, trailing_silence_frames=2, min_speech_ms=250)
    emitted = _feed_all(endpointer, [FRAME] * 32)
    # Fires on the 31st feed: buffer 9920 samples, tail 4920 ≥ 640 + 4160,
    # cut = 5000 + 1600 tail pad.
    assert len(emitted) == 1
    assert emitted[0].duration_ms == 6_600 // 16
    assert len(emitted[0].pcm16le) == 6_600 * 2


def test_second_utterance_endpoints_after_reset():
    async def handler(request):
        return httpx.Response(200, json={"segments": [
            {"start_sample": 1_000, "end_sample": 5_000},
        ]})

    endpointer = _endpointer(handler, trailing_silence_frames=2, min_speech_ms=250)
    emitted = _feed_all(endpointer, [FRAME] * 70)
    assert len(emitted) >= 2  # buffer recycles and keeps segmenting new speech


def test_silence_only_buffer_stays_bounded():
    async def handler(request):
        return httpx.Response(200, json={"segments": []})

    endpointer = _endpointer(handler)
    _feed_all(endpointer, [FRAME] * 400)  # 128_000 samples of silence
    assert len(endpointer._buffer) <= 8_000 * 2 + 640  # kept pre-roll, not the whole run


def test_short_blips_are_discarded_as_noise():
    async def handler(request):
        samples = _posted_samples(request)
        segments = [] if samples < 2_000 else [{"start_sample": 1_000, "end_sample": 2_000}]
        return httpx.Response(200, json={"segments": segments})

    endpointer = _endpointer(handler, trailing_silence_frames=2, min_speech_ms=250)
    emitted = _feed_all(endpointer, [FRAME] * 40)
    assert emitted == []


def test_repeated_poll_failures_degrade_to_energy_vad():
    async def handler(request):
        return httpx.Response(500, json={"error": {"message": "down"}})

    endpointer = _endpointer(handler)
    frames = [FRAME] * 3 + [LOUD_FRAME] * 30 + [FRAME] * 5
    emitted = _feed_all(endpointer, frames)
    assert endpointer._degraded is True
    assert len(emitted) == 1  # the loud utterance is still endpointed, by energy


def test_build_endpointer_prefers_silero_only_when_url_set():
    base = {
        "WAYCAST_HUB_URL": "ws://hub.test:1",
        "VOICE_IN_ASR_ENDPOINT": "https://asr.test/v1",
        "VOICE_IN_ASR_MODEL": "asr",
    }

    async def scenario():
        energy = build_endpointer(load_settings(base))
        silero = build_endpointer(load_settings({**base, "VOICE_IN_VAD_URL": "http://127.0.0.1:8080"}))
        names = (energy.__class__.__name__, silero.__class__.__name__)
        await energy.aclose()
        await silero.aclose()
        return names

    assert _run(scenario()) == ("EnergyEndpointer", "SileroEndpointer")
