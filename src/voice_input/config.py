"""Runtime configuration for the voice input module.

All values come from the environment (the .env file is loaded first by main).  The
module connects to the waycast hub (WAYCAST_HUB_URL / WAYCAST_HUB_TOKEN) and
transcribes microphone audio with an OpenAI-compatible ASR service
(VOICE_IN_ASR_ENDPOINT / VOICE_IN_ASR_MODEL / VOICE_IN_ASR_API_KEY).

Endpoint detection uses the energy VAD unless VOICE_IN_VAD_URL points at an
audio.cpp server whose silero_vad model then drives endpointing (ADR-0003).
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
VAD_URL_KEY: Final = "VOICE_IN_VAD_URL"
VAD_MODEL_KEY: Final = "VOICE_IN_VAD_MODEL"
VAD_POLL_MS_KEY: Final = "VOICE_IN_VAD_POLL_MS"
VAD_MIN_SPEECH_MS_KEY: Final = "VOICE_IN_VAD_MIN_SPEECH_MS"
ASR_LIVE_KEY: Final = "VOICE_IN_ASR_LIVE"
ASR_LIVE_TIMEOUT_MS_KEY: Final = "VOICE_IN_ASR_LIVE_TIMEOUT_MS"
ASR_LIVE_MAX_FAILURES_KEY: Final = "VOICE_IN_ASR_LIVE_MAX_FAILURES"
DENOISE_KEY: Final = "VOICE_IN_DENOISE"

DEFAULT_SPEECH_THRESHOLD: Final = 300
DEFAULT_TRAILING_SILENCE_FRAMES: Final = 20
DEFAULT_MAX_FRAMES: Final = 1_500
DEFAULT_PRE_ROLL_FRAMES: Final = 5
DEFAULT_VAD_MODEL: Final = "silero-vad"
DEFAULT_VAD_POLL_MS: Final = 300
DEFAULT_VAD_MIN_SPEECH_MS: Final = 250
DEFAULT_ASR_LIVE_TIMEOUT_MS: Final = 10_000
DEFAULT_ASR_LIVE_MAX_FAILURES: Final = 3

_TRUE_VALUES: Final = {"1", "true", "yes", "on"}


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
    vad_url: str | None = None
    vad_model: str = DEFAULT_VAD_MODEL
    vad_poll_ms: int = DEFAULT_VAD_POLL_MS
    vad_min_speech_ms: int = DEFAULT_VAD_MIN_SPEECH_MS
    asr_live: bool = True
    asr_live_timeout_ms: int = DEFAULT_ASR_LIVE_TIMEOUT_MS
    asr_live_max_failures: int = DEFAULT_ASR_LIVE_MAX_FAILURES
    denoise: bool = True


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
            source, VAD_TRAILING_SILENCE_FRAMES_KEY, DEFAULT_TRAILING_SILENCE_FRAMES, minimum=1
        ),
        max_frames=_parse_int(source, VAD_MAX_FRAMES_KEY, DEFAULT_MAX_FRAMES, minimum=1),
        pre_roll_frames=_parse_int(
            source, VAD_PRE_ROLL_FRAMES_KEY, DEFAULT_PRE_ROLL_FRAMES, minimum=0
        ),
        vad_url=_optional_url(source.get(VAD_URL_KEY)),
        vad_model=source.get(VAD_MODEL_KEY, "").strip() or DEFAULT_VAD_MODEL,
        vad_poll_ms=_parse_int(source, VAD_POLL_MS_KEY, DEFAULT_VAD_POLL_MS, minimum=50),
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
    )


def _parse_bool(source: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = source.get(key, "").strip()
    if raw == "":
        return default
    return raw.lower() in _TRUE_VALUES


def _optional_url(value: str | None) -> str | None:
    url = _optional_text(value)
    if url is not None and not url.startswith(("http://", "https://")):
        raise ConfigError(key=VAD_URL_KEY, reason="must be an http:// or https:// URL")
    return url


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


def _parse_int(source: Mapping[str, str], key: str, default: int, *, minimum: int) -> int:
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


__all__ = ["Settings", "ConfigError", "load_settings"]
