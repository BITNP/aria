"""FBank frontend tests: frame layout, determinism, edge cases (offline)."""

from __future__ import annotations

import math

import numpy

from aria.fbank import (
    FRAME_LENGTH_SAMPLES,
    FRAME_SHIFT_SAMPLES,
    NUM_BINS,
    extract,
)
from aria.vad import SAMPLE_RATE_HZ


def _tone(frames: int, frequency: float = 440.0) -> bytes:
    samples = numpy.sin(2 * math.pi * frequency * numpy.arange(frames) / SAMPLE_RATE_HZ)
    return (samples * 12000).astype("<i2").tobytes()


def test_one_and_a_half_seconds_yields_148_frames():
    features = extract(_tone(24_000))
    assert features.shape == (148, NUM_BINS)
    assert features.dtype == numpy.float32


def test_frame_count_formula():
    frames = 20_000
    expected = 1 + (frames - FRAME_LENGTH_SAMPLES) // FRAME_SHIFT_SAMPLES
    assert extract(_tone(frames)).shape[0] == expected


def test_short_audio_returns_empty():
    assert extract(_tone(FRAME_LENGTH_SAMPLES - 1)).shape == (0, NUM_BINS)


def test_tone_features_are_finite_and_non_trivial():
    features = extract(_tone(24_000))
    assert numpy.isfinite(features).all()
    assert float(numpy.abs(features).max()) > 0.0


def test_deterministic():
    audio = _tone(24_000, frequency=737.0)
    assert numpy.array_equal(extract(audio), extract(audio))


def test_per_bin_mean_subtraction():
    features = extract(_tone(24_000))
    assert numpy.allclose(features.mean(axis=0), 0.0, atol=1e-3)


def test_silence_extracts_to_zeros():
    assert numpy.allclose(extract(b"\x00" * (24_000 * 2)), 0.0)


def test_frame_layout_constant():
    assert FRAME_LENGTH_SAMPLES == 400
    assert FRAME_SHIFT_SAMPLES == 160
    assert NUM_BINS == 80
