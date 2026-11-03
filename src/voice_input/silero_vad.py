"""Model-backed speech endpointing via an audio.cpp Silero VAD deployment.

The audio.cpp server exposes its streaming ``silero_vad`` model through the
generic task route (``POST /v1/tasks/run`` with a base64 WAV and the model id);
the response carries absolute speech segments as sample indices at 16 kHz.
The streaming model consumes fixed 512-sample windows, so the WAV payload is
zero-padded to a whole number of windows before posting (ADR-0003).

Endpointing polls the model on the growing utterance buffer: every
``poll_s`` the buffer is re-segmented and an endpoint fires once the audio
after the last segment end reaches the configured trailing silence (or the
hard max-length cap is hit).  Leading silence is trimmed as it settles and
too-short segments are discarded as noise.

If the service keeps failing, the endpointer degrades permanently to the
bundled energy detector so the pipeline survives an audio.cpp outage.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from typing import Final, cast

import httpx

from voice_input.vad import (
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    EnergyEndpointer,
    SpeechSegment,
    wav_bytes,
)

LOGGER = logging.getLogger(__name__)

WINDOW_SAMPLES: Final = 512  # the streaming model's fixed chunk (32 ms @ 16 kHz)
MAX_POLL_FAILURES: Final = 3
KEEP_SILENCE_SAMPLES: Final = 8_000  # bounded silence kept while nothing is detected
LEAD_PAD_SAMPLES: Final = 1_600  # audio kept before the first detected segment
TAIL_PAD_SAMPLES: Final = 1_600  # audio kept after the last detected segment
_SAMPLES_PER_MS: Final = SAMPLE_RATE_HZ // 1000


class SileroVadClient:
    """Client for one audio.cpp silero_vad deployment (generic task route).

    ``segments`` returns the speech segments of ``pcm16le`` as
    ``(start_sample, end_sample)`` pairs at 16 kHz, raising on any transport or
    protocol failure. Pass an injected httpx.AsyncClient in tests to avoid
    real network I/O; when one is injected it is not closed here.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/tasks/run"
        self._model = model
        self._client = client if client is not None else _default_client()
        self._owns_client = client is None

    async def segments(self, pcm16le: bytes) -> list[tuple[int, int]]:
        sample_count = len(pcm16le) // 2
        padded = pcm16le + b"\x00" * ((-sample_count) % WINDOW_SAMPLES) * 2
        response = await self._client.post(
            self._url,
            json={
                "model": self._model,
                "request": {"audio_base64": base64.b64encode(wav_bytes(padded)).decode()},
            },
        )
        response.raise_for_status()
        return _parse_segments(response.json())

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _parse_segments(value: object) -> list[tuple[int, int]]:
    """Validate the task-run response shape, raising ValueError when malformed."""
    if not isinstance(value, dict):
        raise ValueError("silero VAD response is not a JSON object")
    data = cast("dict[str, object]", value)
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("silero VAD segments field is not a list")
    segments: list[tuple[int, int]] = []
    for item in cast("list[object]", raw_segments):
        if not isinstance(item, dict):
            raise ValueError("silero VAD segment is not a JSON object")
        segment = cast("dict[str, object]", item)
        start, end = segment.get("start_sample"), segment.get("end_sample")
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("silero VAD segment bounds are not numbers")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            raise ValueError("silero VAD segment lacks sample bounds")
        segments.append((int(start), int(end)))
    return segments


class SileroEndpointer:
    """Endpointer over Silero segments (Endpointer surface, async feed).

    Every pushed frame grows the utterance buffer; the buffer is re-segmented
    at most once per ``poll_s``. An endpoint is emitted when the buffer tail
    past the last segment end reaches ``trailing_silence_frames`` of audio (or
    the ``max_frames`` cap is reached); emitted audio spans the detected
    segments plus the lead/tail pads.
    """

    def __init__(
        self,
        client: SileroVadClient,
        *,
        fallback: EnergyEndpointer,
        poll_s: float = 0.3,
        min_speech_ms: int = 250,
        trailing_silence_frames: int = 20,
        max_frames: int = 1_500,
    ) -> None:
        self._client = client
        self._fallback = fallback
        self._poll_s = poll_s
        self._min_speech_samples = min_speech_ms * _SAMPLES_PER_MS
        self._trailing_samples = trailing_silence_frames * FRAME_SAMPLES
        self._max_samples = max_frames * FRAME_SAMPLES
        self._buffer = bytearray()
        self._start_sample = 0
        self._last_poll = 0.0
        self._failures = 0
        self._degraded = False

    async def feed(self, frame: bytes, sample: int) -> SpeechSegment | None:
        if self._degraded:
            return await self._fallback.feed(frame, sample)
        if not self._buffer:
            self._start_sample = sample
        self._buffer.extend(frame)
        now = asyncio.get_running_loop().time()
        if now - self._last_poll < self._poll_s:
            return None
        self._last_poll = now
        try:
            segments = await self._client.segments(bytes(self._buffer))
        except Exception:  # noqa: BLE001 - any poll fault degrades, never drops the loop
            self._failures += 1
            LOGGER.debug("silero VAD poll failed (x%d)", self._failures, exc_info=True)
            if self._failures >= MAX_POLL_FAILURES:
                LOGGER.warning(
                    "silero VAD unreachable repeatedly; degrading to energy VAD"
                )
                self._degraded = True
            return None
        self._failures = 0
        return self._evaluate(segments)

    def _evaluate(self, segments: list[tuple[int, int]]) -> SpeechSegment | None:
        end_samples = len(self._buffer) // 2
        if not segments:
            # Nothing but silence (or undetected noise) so far: keep a bounded
            # pre-roll window so a late-detected onset still has lead-in audio.
            self._trim_front(end_samples - KEEP_SILENCE_SAMPLES)
            return None
        first_start = min(start for start, _ in segments)
        last_end = max(end for _, end in segments)
        trimmed = max(0, first_start - LEAD_PAD_SAMPLES)
        self._trim_front(trimmed)
        first_start -= trimmed
        last_end -= trimmed
        end_samples -= trimmed
        tail_silence = end_samples - last_end
        if tail_silence < self._trailing_samples and end_samples < self._max_samples:
            return None  # utterance still in progress
        if last_end - first_start < self._min_speech_samples:
            # Too short to be an utterance: drop it and keep listening.
            self._trim_front(last_end)
            return None
        cut = min(last_end + TAIL_PAD_SAMPLES, end_samples)
        segment = SpeechSegment(
            pcm16le=bytes(self._buffer[: cut * 2]),
            duration_ms=cut // _SAMPLES_PER_MS,
        )
        LOGGER.debug(
            "silero endpoint: %d ms from start_sample=%d (buffer %d samples)",
            segment.duration_ms,
            self._start_sample,
            end_samples,
        )
        self._trim_front(cut)
        return segment

    def _trim_front(self, samples: int) -> None:
        if samples <= 0:
            return
        del self._buffer[: samples * 2]
        self._start_sample = (self._start_sample + samples) % (1 << 32)

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._fallback.aclose()


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(5.0),
        follow_redirects=False,
        trust_env=False,
    )


__all__ = ["SileroVadClient", "SileroEndpointer", "WINDOW_SAMPLES", "MAX_POLL_FAILURES"]
