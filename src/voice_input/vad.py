"""Speech endpoint detection for fixed 20 ms PCM16 mono frames (16 kHz).

This is a pure energy-based fallback VAD with trailing-silence endpointing and a hard
frame cap.  It intentionally stays free of model dependencies so the pipeline can run
headless in tests; the model-backed endpointer (``silero_vad``) shares the async
``feed()`` surface defined here.

A segment starts at the first speech frame (plus any preroll) and ends after
trailing_silence_frames consecutive non-speech frames or when max_frames is reached.
No partial/final semantics exist here: the module decides when a turn is complete and
every emitted endpoint becomes one waycast input frame (ADR-0011).
"""

from __future__ import annotations

import wave
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import Final, Protocol

SAMPLE_RATE_HZ: Final = 16_000
FRAME_SAMPLES: Final = 320  # 20 ms at 16 kHz
FRAME_BYTES: Final = 640  # 320 int16 samples = 640 bytes


@dataclass(frozen=True, slots=True)
class SpeechEndpoint:
    """One detected utterance: raw PCM16 mono bytes plus its start sample timestamp."""

    pcm16le: bytes
    start_sample: int


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One completed utterance with its speech duration: what the capture loop queues."""

    pcm16le: bytes
    duration_ms: int


class Endpointer(Protocol):
    """Async endpointing surface used by the capture loop.

    ``feed`` takes one 20 ms frame plus its sample timestamp and returns the
    completed utterance when a turn ends, else None.

    ``utterance_open`` / ``utterance_audio`` expose the in-progress utterance so
    the live ASR path (live_asr) can keep a streaming session in lockstep with
    exactly the audio endpointing will emit: the flag tracks "utterance audio is
    being collected", and the snapshot is the utterance buffered so far.
    """

    async def feed(self, frame: bytes, sample: int) -> SpeechSegment | None: ...

    @property
    def utterance_open(self) -> bool:
        """True while a started utterance's audio is being collected."""
        ...

    def utterance_audio(self) -> bytes:
        """Snapshot of the open utterance's buffered audio (empty when none)."""
        ...

    async def aclose(self) -> None: ...


class EndpointDetector:
    """Aggregates 20 ms PCM16 frames into speech endpoints."""

    def __init__(
        self,
        *,
        threshold: int = 300,
        trailing_silence_frames: int = 20,
        max_frames: int = 1500,
        pre_roll_frames: int = 5,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if not 0 <= pre_roll_frames < max_frames:
            raise ValueError("pre_roll_frames must be below max_frames")
        if trailing_silence_frames < 1:
            raise ValueError("trailing_silence_frames must be positive")
        self._threshold = threshold
        self._trailing_silence_frames = trailing_silence_frames
        self._max_frames = max_frames
        self._pre_roll: deque[tuple[bytes, int]] = deque(maxlen=pre_roll_frames)
        self._frames: list[bytes] = []
        self._start_sample: int | None = None
        self._silence = 0

    def is_speech(self, frame: bytes) -> bool:
        """Return whether a single 20 ms frame looks like speech by mean absolute energy."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be exactly {FRAME_BYTES} PCM16 bytes")
        return _mean_energy(frame) >= self._threshold

    def push(
        self, frame: bytes, sample: int, *, speech: bool | None = None
    ) -> SpeechEndpoint | None:
        """Feed one frame; return a completed endpoint when a turn ends, else None."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be exactly {FRAME_BYTES} PCM16 bytes")
        is_speech = self.is_speech(frame) if speech is None else speech
        if is_speech and self._start_sample is None:
            self._start_sample = self._pre_roll[0][1] if self._pre_roll else sample
            self._frames.extend(frame for frame, _ in self._pre_roll)
            self._pre_roll.clear()
        if self._start_sample is None:
            self._pre_roll.append((frame, sample))
            return None
        self._frames.append(frame)
        self._silence = 0 if is_speech else self._silence + 1
        if (
            len(self._frames) >= self._max_frames
            or self._silence >= self._trailing_silence_frames
        ):
            return self.flush()
        return None

    def flush(self) -> SpeechEndpoint | None:
        """Force-end the current utterance; None if none is in progress."""
        if self._start_sample is None:
            return None
        endpoint = SpeechEndpoint(
            pcm16le=b"".join(self._frames), start_sample=self._start_sample
        )
        self._frames = []
        self._start_sample = None
        self._silence = 0
        return endpoint

    def reset(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._start_sample = None
        self._silence = 0

    @property
    def collecting(self) -> bool:
        """True while an utterance (started, not yet flushed) is being collected."""
        return self._start_sample is not None

    def utterance_audio(self) -> bytes:
        """Snapshot of the collected utterance frames (incl. preroll), else empty."""
        return b"".join(self._frames) if self._start_sample is not None else b""


def _mean_energy(frame: bytes) -> int:
    samples = memoryview(frame).cast("h")
    return sum(abs(sample) for sample in samples) // len(samples)


class EnergyEndpointer:
    """Async ``feed()`` adapter over the energy detector (Endpointer surface).

    Keeps counting frames flagged as speech so ``duration_ms`` matches the
    historical speech-only accounting of the capture loop (speaker
    identification gates on it).
    """

    def __init__(self, detector: EndpointDetector) -> None:
        self._detector = detector
        self._speech_ms = 0

    async def feed(self, frame: bytes, sample: int) -> SpeechSegment | None:
        speech = self._detector.is_speech(frame)
        if speech:
            self._speech_ms += FRAME_SAMPLES * 1000 // SAMPLE_RATE_HZ
        endpoint = self._detector.push(frame, sample, speech=speech)
        if endpoint is None:
            return None
        duration_ms, self._speech_ms = self._speech_ms, 0
        return SpeechSegment(pcm16le=endpoint.pcm16le, duration_ms=duration_ms)

    @property
    def utterance_open(self) -> bool:
        return self._detector.collecting

    def utterance_audio(self) -> bytes:
        return self._detector.utterance_audio()

    async def aclose(self) -> None:
        return None


def wav_bytes(pcm16le: bytes) -> bytes:
    """Wrap raw PCM16 mono bytes into an in-memory 16 kHz WAV container."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE_HZ)
        output.writeframes(pcm16le)
    return buffer.getvalue()


__all__ = [
    "FRAME_BYTES",
    "FRAME_SAMPLES",
    "SAMPLE_RATE_HZ",
    "EndpointDetector",
    "Endpointer",
    "EnergyEndpointer",
    "SpeechEndpoint",
    "SpeechSegment",
    "wav_bytes",
]
