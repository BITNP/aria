"""GTCRN enhancer tests (ADR-0006), layered like the speaker tests.

T1 (always, offline): frame contract, WOLA front-end reconstruction against an
identity ONNX stand-in (tests/ only, ADR-0002 testing rules), failure bypass
and strict graph validation.  The identity stand-in must present the real
graph's names/shapes/metadata because the production validator runs on it.

T2 (bundled model present, else skipped): noise-only attenuation and
speech-like preservation through the real model on deterministic synthetic
audio.  An official noisy wav can additionally be fed via VOICE_IN_GTCRN_WAV.
"""

from __future__ import annotations

import asyncio
import math
import os
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy
import pytest

from voice_input.denoise import (
    MODEL_REVISION,
    GtcrnEnhancer,
    build_enhancer,
    bundled_model_path,
)

META = {
    "n_fft": "512",
    "hop_length": "256",
    "window_length": "512",
    "window_type": "hann_sqrt",
    "sample_rate": "16000",
}
SHAPES = {
    "mix": [1, 257, 1, 2],
    "enh": [1, 257, 1, 2],
    "conv_cache": [2, 1, 16, 16, 33],
    "tra_cache": [2, 3, 1, 1, 16],
    "inter_cache": [2, 1, 33, 16],
    "conv_cache_out": [2, 1, 16, 16, 33],
    "tra_cache_out": [2, 3, 1, 1, 16],
    "inter_cache_out": [2, 1, 33, 16],
}
INPUT_NAMES = ("mix", "conv_cache", "tra_cache", "inter_cache")
OUTPUT_NAMES = ("enh", "conv_cache_out", "tra_cache_out", "inter_cache_out")


class _Node:
    def __init__(self, name: str, shape: list[int] | None = None) -> None:
        self.name = name
        self.shape = shape if shape is not None else SHAPES[name]
        self.type = "tensor(float)"


class _Meta:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.custom_metadata_map = metadata


class IdentitySession:
    """Test-only stand-in: enh = mix, caches pass through unchanged."""

    def __init__(
        self,
        *,
        metadata: dict[str, str] | None = None,
        inputs: list[_Node] | None = None,
        outputs: list[_Node] | None = None,
        fail_hops: int = 0,
    ) -> None:
        self._meta = _Meta(dict(META) if metadata is None else metadata)
        self._inputs = inputs or [_Node(name) for name in INPUT_NAMES]
        self._outputs = outputs or [_Node(name) for name in OUTPUT_NAMES]
        self._fail_hops = fail_hops
        self.hop_calls = 0

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, object],
        run_options: object | None = None,
    ) -> list[Any]:
        self.hop_calls += 1
        if self.hop_calls <= self._fail_hops:
            raise RuntimeError("injected hop failure")
        feed = cast("dict[str, Any]", input_feed)
        return [feed["mix"].copy()] + [feed[name].copy() for name in INPUT_NAMES[1:]]

    def get_inputs(self) -> list[_Node]:
        return self._inputs

    def get_outputs(self) -> list[_Node]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_modelmeta(self) -> _Meta:
        return self._meta


async def _collect(enhancer: GtcrnEnhancer, pcm: numpy.ndarray) -> bytes:
    out = bytearray()
    for start in range(0, len(pcm) - len(pcm) % 320, 320):
        block = pcm[start : start + 320].tobytes()
        out.extend(b"".join(await enhancer.push_async(block)))
    return bytes(out)


def _run(enhancer: GtcrnEnhancer, pcm: numpy.ndarray) -> numpy.ndarray:
    async def scenario() -> bytes:
        return await _collect(enhancer, pcm)

    return numpy.frombuffer(asyncio.run(scenario()), dtype=numpy.int16)


def _rms(samples: numpy.ndarray) -> float:
    return math.sqrt(float(numpy.mean(samples.astype(numpy.float64) ** 2)))


# ---- T1: frame contract and front-end reconstruction -------------------------


def test_identity_session_reconstructs_input_within_wola_ripple():
    # Symmetric sqrt-Hann WOLA (the window the model itself was exported for)
    # reconstructs exactly except for a bounded ripple at the quantization level.
    rng = numpy.random.default_rng(7)
    pcm = rng.integers(-6000, 6000, size=320 * 30, dtype=numpy.int16)
    out = _run(GtcrnEnhancer(session=IdentitySession()), pcm)
    assert len(out) == 8960  # (hops - priming hop) * 256 drained to 320-frames
    diff = numpy.abs(out.astype(numpy.int64) - pcm[: len(out)].astype(numpy.int64))
    assert int(diff.max()) <= 64


def test_every_returned_frame_is_exactly_20ms():
    rng = numpy.random.default_rng(1)
    pcm = rng.integers(-4000, 4000, size=320 * 17, dtype=numpy.int16)

    async def scenario() -> list[bytes]:
        enhancer = GtcrnEnhancer(session=IdentitySession())
        frames: list[bytes] = []
        for start in range(0, len(pcm), 320):
            frames.extend(await enhancer.push_async(pcm[start : start + 320].tobytes()))
        return frames

    frames = asyncio.run(scenario())
    assert frames
    assert all(len(frame) == 640 for frame in frames)


def test_sample_conservation_matches_priming_and_buffering():
    # emitted = (hops - 1 priming) * 256, drained down to whole 320 frames
    rng = numpy.random.default_rng(2)
    pcm = rng.integers(-4000, 4000, size=320 * 21, dtype=numpy.int16)
    out = _run(GtcrnEnhancer(session=IdentitySession()), pcm)
    hops = len(pcm) // 256
    emitted = (hops - 1) * 256
    assert len(out) == emitted - emitted % 320


# ---- T1: failure bypass -------------------------------------------------------


def test_repeated_hop_failures_permanently_bypass_the_model():
    constant = numpy.full(320, 1234, dtype=numpy.int16)
    enhancer = GtcrnEnhancer(session=IdentitySession(fail_hops=10_000))

    async def scenario() -> list[bytes]:
        frames: list[bytes] = []
        for _ in range(12):
            frames.extend(await enhancer.push_async(constant.tobytes()))
        return frames

    frames = asyncio.run(scenario())
    assert enhancer.bypassed
    assert enhancer.model_active is False
    # everything after the bypass is raw audio: a constant signal makes every
    # whole frame identical regardless of the internal buffering offset
    assert set(frames[-5:]) == {constant.tobytes()}


def test_single_hop_failure_passes_raw_and_recovers():
    session = IdentitySession(fail_hops=1)
    enhancer = GtcrnEnhancer(session=session)
    rng = numpy.random.default_rng(3)
    pcm = rng.integers(-5000, 5000, size=320 * 20, dtype=numpy.int16)
    out = _run(enhancer, pcm)
    assert not enhancer.bypassed
    assert len(out) > 0


# ---- T1: strict validation and degraded build --------------------------------


def test_wrong_metadata_is_rejected():
    bad = dict(META)
    bad["hop_length"] = "128"
    with pytest.raises(ValueError, match="hop_length"):
        GtcrnEnhancer(session=IdentitySession(metadata=bad))


def test_wrong_graph_io_is_rejected():
    broken = [
        _Node("mix", shape=[1, 256, 1, 2]),
        *[_Node(name) for name in INPUT_NAMES[1:]],
    ]
    with pytest.raises(ValueError, match="mix"):
        GtcrnEnhancer(session=IdentitySession(inputs=broken))


def test_dynamic_cache_shapes_are_rejected():
    nodes = [
        _Node("mix"),
        _Node("conv_cache", shape=["batch", 1, 16, 16, 33]),
        _Node("tra_cache"),
        _Node("inter_cache"),
    ]
    outputs = [
        _Node("enh"),
        _Node("conv_cache_out", shape=["batch", 1, 16, 16, 33]),
        _Node("tra_cache_out"),
        _Node("inter_cache_out"),
    ]
    with pytest.raises(ValueError, match="static"):
        GtcrnEnhancer(session=IdentitySession(inputs=nodes, outputs=outputs))


class _Settings:
    def __init__(self, denoise: bool) -> None:
        self.denoise = denoise


def test_build_enhancer_disabled_returns_none(tmp_path: Path):
    assert build_enhancer(_Settings(False), model_path=bundled_model_path()) is None


def test_build_enhancer_missing_model_degrades_to_none(tmp_path: Path):
    assert build_enhancer(_Settings(True), model_path=tmp_path / "absent.onnx") is None


def test_build_enhancer_enabled_uses_bundled_model():
    enhancer = build_enhancer(_Settings(True))
    assert (enhancer is not None) == bundled_model_path().is_file()
    if enhancer is not None:
        assert enhancer.model_revision == MODEL_REVISION


# ---- T2: real bundled model ----------------------------------------------------

_HAS_MODEL = pytest.mark.skipif(
    not bundled_model_path().is_file(), reason="bundled GTCRN model not present"
)


def _speech_like(samples: int = 16_000 * 3, seed: int = 3) -> numpy.ndarray:
    """Harmonic stack (120 Hz F0) with syllabic AM: passes as speech-like."""
    rng = numpy.random.default_rng(seed)
    t = numpy.arange(samples)
    signal = numpy.zeros(samples)
    for harmonic, amplitude in enumerate(
        [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08], start=1
    ):
        signal += amplitude * numpy.sin(
            2 * numpy.pi * 120 * harmonic * t / 16_000 + rng.uniform(0, 6.28)
        )
    signal *= 0.6 + 0.4 * numpy.sin(2 * numpy.pi * 3.5 * t / 16_000)
    return (signal / numpy.abs(signal).max() * 9000).astype(numpy.int16)


@_HAS_MODEL
def test_real_model_loads_with_expected_revision():
    enhancer = GtcrnEnhancer(bundled_model_path())
    assert enhancer.model_revision == MODEL_REVISION
    assert enhancer.onnx_provider == "CPUExecutionProvider"


@_HAS_MODEL
def test_real_model_attenuates_stationary_noise():
    rng = numpy.random.default_rng(11)
    noise = rng.normal(0, 1200, size=16_000 * 3).astype(numpy.int16)
    out = _run(GtcrnEnhancer(bundled_model_path()), noise)
    tail = out[len(out) - 16_000 :]
    assert numpy.isfinite(tail).all()
    assert _rms(tail) < _rms(noise[-16_000:]) * 0.15


@_HAS_MODEL
def test_real_model_preserves_speech_like_signal():
    speech = _speech_like()
    out = _run(GtcrnEnhancer(bundled_model_path()), speech)
    tail = out[len(out) - 16_000 :]
    assert _rms(tail) > _rms(speech[-16_000:]) * 0.6


@_HAS_MODEL
def test_real_model_on_official_noisy_wav_is_stable():
    path = os.environ.get("VOICE_IN_GTCRN_WAV", "")
    if not path or not Path(path).is_file():
        pytest.skip(
            "set VOICE_IN_GTCRN_WAV to a 16 kHz mono WAV for the official fixture"
        )
    with wave.open(path) as source:
        assert source.getframerate() == 16_000 and source.getnchannels() == 1
        pcm = numpy.frombuffer(
            source.readframes(source.getnframes()), dtype=numpy.int16
        )
    out = _run(GtcrnEnhancer(bundled_model_path()), pcm)
    assert numpy.isfinite(out).all()
    fed = len(pcm) - len(pcm) % 320  # _collect feeds whole 20 ms blocks only
    hops = fed // 256
    emitted = (hops - 1) * 256
    assert len(out) == emitted - emitted % 320
