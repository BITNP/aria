"""FBank frontend for the bundled CAM++ ONNX model (ADR-0002).

The model was trained with the constrained 3D-Speaker feature pipeline, so an
utterance only produces a comparable embedding when it goes through the same
frontend: 16 kHz / 25 ms frames / 10 ms shift / 80 mel bins / power (not
magnitude) spectrum / log compression / per-bin mean subtraction, with the
prenormalization and window shaping used by that runtime.  This is a NumPy
implementation of that constrained spec, kept dependency-free apart from numpy.

The frontend operates on the 1.5 s identification window (or the whole longer
utterance) assembled by voice_input.speaker.
"""
from __future__ import annotations

from typing import Final

import numpy

SAMPLE_RATE_HZ: Final = 16_000
FRAME_LENGTH_SAMPLES: Final = 400  # 25 ms
FRAME_SHIFT_SAMPLES: Final = 160  # 10 ms
NUM_BINS: Final = 80
FFT_SIZE: Final = 512
PRE_EMPHASIS: Final = 0.97
WINDOW_POWER: Final = 0.85
MEL_LOW_HZ: Final = 20.0


def extract(pcm16le: bytes) -> numpy.ndarray:
    """Return the (frames, 80) float32 log-power FBank for raw PCM16 audio.

    An empty (0, 80) array is returned when there are fewer samples than one
    frame so the caller can treat it as "not enough audio" instead of failing.
    """
    samples = numpy.frombuffer(pcm16le, dtype="<i2").astype(numpy.float32) / 32768
    if samples.size < FRAME_LENGTH_SAMPLES:
        return numpy.empty((0, NUM_BINS), dtype=numpy.float32)
    count = 1 + (samples.size - FRAME_LENGTH_SAMPLES) // FRAME_SHIFT_SAMPLES
    frames = numpy.stack(
        [
            samples[index * FRAME_SHIFT_SAMPLES : index * FRAME_SHIFT_SAMPLES + FRAME_LENGTH_SAMPLES]
            for index in range(count)
        ]
    )
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= PRE_EMPHASIS * frames[:, :-1]
    frames[:, 0] -= PRE_EMPHASIS * frames[:, 0]
    frames *= _frontend_window()
    spectrum = numpy.fft.rfft(frames, n=FFT_SIZE, axis=1)[:, :-1]
    power = (spectrum.real**2 + spectrum.imag**2).astype(numpy.float32)
    features = power @ _mel_filters().T
    features = numpy.log(numpy.maximum(features, numpy.finfo(numpy.float32).eps))
    return (features - features.mean(axis=0, keepdims=True)).astype(numpy.float32)


def _frontend_window() -> numpy.ndarray:
    indices = numpy.arange(FRAME_LENGTH_SAMPLES, dtype=numpy.float32)
    return (0.5 - 0.5 * numpy.cos(2 * numpy.pi * indices / (FRAME_LENGTH_SAMPLES - 1))) ** WINDOW_POWER


def _mel_filters() -> numpy.ndarray:
    """(80, 256) triangular mel filters over the 512-point power spectrum bins."""
    frequencies = numpy.arange(FFT_SIZE // 2, dtype=numpy.float32) * (SAMPLE_RATE_HZ / FFT_SIZE)
    mel_all = 1127.0 * numpy.log(1.0 + frequencies / 700.0)
    points = numpy.linspace(
        _hz_to_mel(MEL_LOW_HZ), _hz_to_mel(SAMPLE_RATE_HZ / 2), NUM_BINS + 2
    )
    filters = numpy.zeros((NUM_BINS, frequencies.size), dtype=numpy.float32)
    for index in range(NUM_BINS):
        left, center, right = points[index : index + 3]
        filters[index] = numpy.maximum(
            0.0,
            numpy.minimum(
                (mel_all - left) / (center - left),
                (right - mel_all) / (right - center),
            ),
        )
    return filters


def _hz_to_mel(value: float) -> float:
    return float(1127.0 * numpy.log(1.0 + value / 700.0))


__all__ = [
    "SAMPLE_RATE_HZ",
    "NUM_BINS",
    "extract",
]
