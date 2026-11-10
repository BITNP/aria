"""Runtime configuration for the voice input module.

All values come from the environment (the .env file is loaded first by main).  The
module connects to the waycast hub (WAYCAST_HUB_URL / WAYCAST_HUB_TOKEN) and
transcribes microphone audio with an OpenAI-compatible ASR service
(VOICE_IN_ASR_ENDPOINT / VOICE_IN_ASR_MODEL / VOICE_IN_ASR_API_KEY).

Endpoint detection defaults to the in-process Silero VAD ONNX model (ADR-0007),
with the energy VAD as automatic fallback when the model cannot load and as the
forced choice via VOICE_IN_VAD=energy.
Transcription prefers the streaming live route on the same ASR server and falls
back to the batch route per utterance on failure (ADR-0005); VOICE_IN_ASR_LIVE=0
disables the live path outright.

In-process GTCRN speech enhancement runs between capture and endpointing by
default (ADR-0006); VOICE_IN_DENOISE=0 disables it and the pipeline forwards
raw microphone audio.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

WAYCAST_HUB_URL_KEY: Final = "WAYCAST_HUB_URL"
WAYCAST_HUB_TOKEN_KEY: Final = "WAYCAST_HUB_TOKEN"
WAYCAST_TLS_CA_KEY: Final = "WAYCAST_TLS_CA"
ASR_ENDPOINT_KEY: Final = "VOICE_IN_ASR_ENDPOINT"
ASR_MODEL_KEY: Final = "VOICE_IN_ASR_MODEL"
ASR_API_KEY_KEY: Final = "VOICE_IN_ASR_API_KEY"
DEVICE_KEY: Final = "VOICE_IN_DEVICE"
VAD_SPEECH_THRESHOLD_KEY: Final = "VOICE_IN_VAD_SPEECH_THRESHOLD"
VAD_TRAILING_SILENCE_FRAMES_KEY: Final = "VOICE_IN_VAD_TRAILING_SILENCE_FRAMES"
VAD_MAX_FRAMES_KEY: Final = "VOICE_IN_VAD_MAX_FRAMES"
VAD_PRE_ROLL_FRAMES_KEY: Final = "VOICE_IN_VAD_PRE_ROLL_FRAMES"
VAD_ENGINE_KEY: Final = "VOICE_IN_VAD"
VAD_THRESHOLD_KEY: Final = "VOICE_IN_VAD_THRESHOLD"
VAD_MIN_SPEECH_MS_KEY: Final = "VOICE_IN_VAD_MIN_SPEECH_MS"
ASR_LIVE_KEY: Final = "VOICE_IN_ASR_LIVE"
ASR_LIVE_TIMEOUT_MS_KEY: Final = "VOICE_IN_ASR_LIVE_TIMEOUT_MS"
ASR_LIVE_MAX_FAILURES_KEY: Final = "VOICE_IN_ASR_LIVE_MAX_FAILURES"
DENOISE_KEY: Final = "VOICE_IN_DENOISE"
SPEAKER_THRESHOLD_KEY: Final = "VOICE_IN_SPEAKER_THRESHOLD"

DEFAULT_SPEECH_THRESHOLD: Final = 300
DEFAULT_TRAILING_SILENCE_FRAMES: Final = 20
DEFAULT_MAX_FRAMES: Final = 1_500
DEFAULT_PRE_ROLL_FRAMES: Final = 5
DEFAULT_VAD_MIN_SPEECH_MS: Final = 250
DEFAULT_VAD_ENGINE: Final = "silero"
DEFAULT_VAD_THRESHOLD: Final = 0.5
DEFAULT_ASR_LIVE_TIMEOUT_MS: Final = 10_000
DEFAULT_ASR_LIVE_MAX_FAILURES: Final = 3
DEFAULT_SPEAKER_THRESHOLD: Final = 0.31

_TRUE_VALUES: Final = {"1", "true", "yes", "on"}
_VAD_ENGINES: Final = ("silero", "energy")


@dataclass(frozen=True, slots=True)
class Settings:
    hub_url: str
    hub_token: str
    asr_endpoint: str
    asr_model: str
    tls_ca_path: str | None = None
    asr_api_key: str | None = None
    device: str | None = None
    speech_threshold: int = DEFAULT_SPEECH_THRESHOLD
    trailing_silence_frames: int = DEFAULT_TRAILING_SILENCE_FRAMES
    max_frames: int = DEFAULT_MAX_FRAMES
    pre_roll_frames: int = DEFAULT_PRE_ROLL_FRAMES
    vad_engine: str = DEFAULT_VAD_ENGINE
    vad_threshold: float = DEFAULT_VAD_THRESHOLD
    vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS
    asr_live: bool = True
    asr_live_timeout_ms: int = DEFAULT_ASR_LIVE_TIMEOUT_MS
    asr_live_max_failures: int = DEFAULT_ASR_LIVE_MAX_FAILURES
    denoise: bool = True
    speaker_threshold: float = DEFAULT_SPEAKER_THRESHOLD


@dataclass(frozen=True, slots=True)
class ConfigError(RuntimeError):
    key: str
    reason: str

    def __str__(self) -> str:
        return f"{self.key}: {self.reason}"


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read every setting from the environment with validation.

    Pass an explicit mapping (e.g. os.environ or a fixture) instead of a live
    process environment when testing.
    """
    source = os.environ if env is None else env
    hub_url = _required(source, WAYCAST_HUB_URL_KEY)
    _require_ws_scheme(hub_url, WAYCAST_HUB_URL_KEY)
    return Settings(
        hub_url=hub_url,
        hub_token=source.get(WAYCAST_HUB_TOKEN_KEY, "").strip(),
        asr_endpoint=_required(source, ASR_ENDPOINT_KEY).rstrip("/"),
        asr_model=_required(source, ASR_MODEL_KEY),
        tls_ca_path=_optional_text(source.get(WAYCAST_TLS_CA_KEY)),
        asr_api_key=_optional_text(source.get(ASR_API_KEY_KEY)),
        device=_optional_text(source.get(DEVICE_KEY)),
        speech_threshold=_parse_int(
            source, VAD_SPEECH_THRESHOLD_KEY, DEFAULT_SPEECH_THRESHOLD, minimum=1
        ),
        trailing_silence_frames=_parse_int(
            source,
            VAD_TRAILING_SILENCE_FRAMES_KEY,
            DEFAULT_TRAILING_SILENCE_FRAMES,
            minimum=1,
        ),
        max_frames=_parse_int(
            source, VAD_MAX_FRAMES_KEY, DEFAULT_MAX_FRAMES, minimum=1
        ),
        pre_roll_frames=_parse_int(
            source, VAD_PRE_ROLL_FRAMES_KEY, DEFAULT_PRE_ROLL_FRAMES, minimum=0
        ),
        vad_engine=_parse_choice(
            source, VAD_ENGINE_KEY, DEFAULT_VAD_ENGINE, _VAD_ENGINES
        ),
        vad_threshold=_parse_float(source, VAD_THRESHOLD_KEY, DEFAULT_VAD_THRESHOLD),
        vad_min_speech_ms=_parse_int(
            source, VAD_MIN_SPEECH_MS_KEY, DEFAULT_VAD_MIN_SPEECH_MS, minimum=1
        ),
        asr_live=_parse_bool(source, ASR_LIVE_KEY, default=True),
        asr_live_timeout_ms=_parse_int(
            source, ASR_LIVE_TIMEOUT_MS_KEY, DEFAULT_ASR_LIVE_TIMEOUT_MS, minimum=100
        ),
        asr_live_max_failures=_parse_int(
            source, ASR_LIVE_MAX_FAILURES_KEY, DEFAULT_ASR_LIVE_MAX_FAILURES, minimum=1
        ),
        denoise=_parse_bool(source, DENOISE_KEY, default=True),
        speaker_threshold=_parse_float(
            source, SPEAKER_THRESHOLD_KEY, DEFAULT_SPEAKER_THRESHOLD
        ),
    )


def _parse_bool(source: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = source.get(key, "").strip()
    if raw == "":
        return default
    return raw.lower() in _TRUE_VALUES


def _parse_choice(
    source: Mapping[str, str], key: str, default: str, allowed: tuple[str, ...]
) -> str:
    raw = source.get(key, "").strip().lower()
    if raw == "":
        return default
    if raw not in allowed:
        raise ConfigError(key=key, reason=f"must be one of {' / '.join(allowed)}")
    return raw


def _parse_float(source: Mapping[str, str], key: str, default: float) -> float:
    raw = source.get(key, "").strip()
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(key=key, reason="must be a number") from exc
    if not 0.0 < value < 1.0:
        raise ConfigError(key=key, reason="must be between 0 and 1 (exclusive)")
    return value


def _required(source: Mapping[str, str], key: str) -> str:
    value = source.get(key, "").strip()
    if not value:
        raise ConfigError(key=key, reason="required setting is missing or empty")
    return value


def _optional_text(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _require_ws_scheme(value: str, key: str) -> None:
    if not value.startswith(("ws://", "wss://")):
        raise ConfigError(key=key, reason="must be a ws:// or wss:// URL")


def _parse_int(
    source: Mapping[str, str], key: str, default: int, *, minimum: int
) -> int:
    raw = source.get(key, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(key=key, reason="must be an integer") from exc
    if value < minimum:
        raise ConfigError(key=key, reason=f"must be >= {minimum}")
    return value


__all__ = ["ConfigError", "Settings", "load_settings"]
