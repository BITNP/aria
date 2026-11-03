"""OpenAI-compatible speech recognition client.

The ASR service is reached with the OpenAI-compatible /audio/transcriptions multipart
contract: a wav file plus the model name.  The module keeps raw audio in memory only
until the single ASR request completes and never writes utterance audio to disk.

An empty Recognition (text == "") is produced on any transport, HTTP, or decode failure
so the pipeline can drop the utterance without raising; the caller decides whether an
empty result is worth forwarding (it never is).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import cast

import httpx

from voice_input.vad import wav_bytes as _wav_bytes

LOGGER = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class Recognition:
    text: str
    confidence: float | None = None


class OpenAICompatibleAsr:
    """Small OpenAI-compatible transcription client.

    ``endpoint`` is the provider base URL (e.g. https://asr.example.test/v1); the path
    /audio/transcriptions is appended by this adapter.  Pass an injected httpx.AsyncClient
    in tests to avoid real network I/O; when one is injected it is not closed here.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = endpoint.rstrip('/') + '/audio/transcriptions'
        self._model = model
        self._api_key = api_key
        self._client = client if client is not None else _default_client()
        self._owns_client = client is None

    async def transcribe(self, pcm16le: bytes) -> Recognition:
        """Transcribe one utterance, returning an empty Recognition on any failure."""
        wav = _wav_bytes(pcm16le)
        LOGGER.debug(
            "batch asr request: %d bytes audio -> %s model=%s", len(pcm16le), self._url, self._model
        )
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=headers,
                data={"model": self._model},
                files={"file": ("utterance.wav", wav, "audio/wav")},
            ) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        LOGGER.warning("batch asr response exceeds %d bytes", MAX_RESPONSE_BYTES)
                        return Recognition("")
        except (httpx.HTTPError, OSError) as exc:
            LOGGER.warning("batch asr request failed: %s: %s", type(exc).__name__, exc)
            return Recognition("")
        try:
            value = json.loads(bytes(body))
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            LOGGER.warning("batch asr response is not JSON: %s", exc)
            return Recognition("")
        return _parse_recognition(value)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        follow_redirects=False,
        trust_env=False,
    )


def _parse_recognition(value: object) -> Recognition:
    if not isinstance(value, dict):
        return Recognition("")
    data = cast("dict[str, object]", value)
    text = data.get("text")
    if not isinstance(text, str):
        return Recognition("")
    confidence = data.get("confidence")
    if confidence is None:
        return Recognition(text.strip())
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return Recognition(text.strip())
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return Recognition(text.strip())
    return Recognition(text.strip(), float(confidence))


__all__ = [
    "Recognition",
    "OpenAICompatibleAsr",
    "MAX_RESPONSE_BYTES",
]
