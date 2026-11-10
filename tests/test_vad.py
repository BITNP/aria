"""Tests for the voice input pipeline pieces. Headless: no device, network, or hub needed."""

from __future__ import annotations

import pytest

from voice_input.vad import FRAME_BYTES, FRAME_SAMPLES, EndpointDetector, SpeechEndpoint


def _silent() -> bytes:
    return bytes(FRAME_BYTES)


def _loud(level: int = 1000) -> bytes:
    return int(level).to_bytes(2, "little", signed=True) * FRAME_SAMPLES


def test_silence_never_emits():
    d = EndpointDetector()
    for i in range(100):
        assert d.push(_silent(), i * FRAME_SAMPLES) is None
    assert d.flush() is None


def test_utterance_ends_after_trailing_silence():
    # trailing_silence_frames=5: the endpoint closes on the 5th consecutive
    # non-speech frame, and that frame is included in the utterance.
    d = EndpointDetector(trailing_silence_frames=5, pre_roll_frames=0)
    for i in range(3):
        assert d.push(_loud(), i * FRAME_SAMPLES, speech=True) is None
    for i in range(3, 7):  # 4 silence frames: no close yet
        assert d.push(_silent(), i * FRAME_SAMPLES, speech=False) is None
    endpoint = d.push(_silent(), 7 * FRAME_SAMPLES, speech=False)  # 5th silence
    assert endpoint is not None and isinstance(endpoint, SpeechEndpoint)
    assert endpoint.pcm16le == _loud() * 3 + _silent() * 5
    assert d.flush() is None


def test_pre_roll_included():
    d = EndpointDetector(pre_roll_frames=2, trailing_silence_frames=2)
    assert d.push(_silent(), 0) is None
    assert d.push(_silent(), FRAME_SAMPLES) is None
    d.push(_loud(), 2 * FRAME_SAMPLES, speech=True)  # starts with preroll frames
    assert d.push(_silent(), 3 * FRAME_SAMPLES, speech=False) is None  # silence 1
    endpoint = d.push(_silent(), 4 * FRAME_SAMPLES, speech=False)  # silence 2 -> close
    assert endpoint is not None
    assert endpoint.pcm16le == _silent() * 2 + _loud() + _silent() * 2


def test_max_frames_forces_endpoint():
    d = EndpointDetector(max_frames=4, trailing_silence_frames=100, pre_roll_frames=0)
    for i in range(3):
        assert d.push(_loud(), i * FRAME_SAMPLES, speech=True) is None
    endpoint = d.push(_loud(), 3 * FRAME_SAMPLES, speech=True)
    assert endpoint is not None and len(endpoint.pcm16le) == 4 * FRAME_BYTES


def test_energy_threshold_split():
    d = EndpointDetector(threshold=300)
    assert d.is_speech(_loud(1000)) is True
    assert d.is_speech(_silent()) is False


def test_bad_frame_length_raises():
    d = EndpointDetector()
    with pytest.raises(ValueError):
        d.push(b"x" * 100, 0)
    with pytest.raises(ValueError):
        d.is_speech(b"x")


def test_reset_drops_state():
    d = EndpointDetector()
    d.push(_loud(), 0, speech=True)
    d.reset()
    assert d.flush() is None


def test_pre_roll_must_be_below_max_frames():
    with pytest.raises(ValueError):
        EndpointDetector(max_frames=3, pre_roll_frames=3)


def test_frame_math():
    assert FRAME_SAMPLES == 320
    assert FRAME_BYTES == 640
