"""ASR client: request building, parsing, failure handling (no real network)."""
from __future__ import annotations

import asyncio
import wave
from io import BytesIO

import httpx

from voice_input.asr import OpenAICompatibleAsr, _wav_bytes
from voice_input.vad import SAMPLE_RATE_HZ


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request):
        return await self._handler(request)


def _client_text(text, status=200):
    async def handler(request):
        assert request.url.path.endswith("/audio/transcriptions")
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={"text": text})

    return httpx.AsyncClient(transport=_FakeTransport(handler))


def _run(coro):
    return asyncio.run(coro)


def test_transcribe_ok():
    client = _client_text("你好")

    async def scenario():
        asr = OpenAICompatibleAsr("https://asr.example/v1", "whisper", client=client)
        try:
            return await asr.transcribe(b"\x00" * 640)
        finally:
            await client.aclose()

    rec = _run(scenario())
    assert rec.text == "你好"
    assert rec.confidence is None


def test_transcribe_with_confidence():
    async def handler(request):
        return httpx.Response(200, json={"text": "ok", "confidence": 0.9})

    client = httpx.AsyncClient(transport=_FakeTransport(handler))

    async def scenario():
        asr = OpenAICompatibleAsr("https://asr.example/v1", "whisper", client=client)
        try:
            return await asr.transcribe(b"\x00" * 640)
        finally:
            await client.aclose()

    rec = _run(scenario())
    assert rec.text == "ok" and rec.confidence == 0.9


def test_transcribe_http_error_returns_empty():
    client = _client_text("", status=500)

    async def scenario():
        asr = OpenAICompatibleAsr("https://asr.example/v1", "whisper", client=client)
        try:
            return await asr.transcribe(b"\x00" * 640)
        finally:
            await client.aclose()

    rec = _run(scenario())
    assert rec.text == ""


def test_transcribe_bad_json_returns_empty():
    async def handler(request):
        return httpx.Response(200, text="not json")

    client = httpx.AsyncClient(transport=_FakeTransport(handler))

    async def scenario():
        asr = OpenAICompatibleAsr("https://asr.example/v1", "whisper", client=client)
        try:
            return await asr.transcribe(b"\x00" * 640)
        finally:
            await client.aclose()

    rec = _run(scenario())
    assert rec.text == ""


def test_wav_bytes_is_valid_wav():
    pcm = b"\x00" * 32000  # 1 second
    raw = _wav_bytes(pcm)
    with wave.open(BytesIO(raw), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE_HZ
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 16000
