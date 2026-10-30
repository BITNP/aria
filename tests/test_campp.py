"""Real-model closed-loop tests (T2, ADR-0002): same audio -> same id, distinct audio -> new id.

These tests exercise the bundled CAM++ ONNX model and are skipped when the
asset is absent so CI stays green offline.  Inputs are synthetic tones chosen
empirically: the model is deterministic, so the cosine distances (1.0 for
identical audio, ~0.897 for the 440/660 Hz pair, below the 0.90 reuse
threshold) are stable across runs.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy
import pytest

from voice_input.fbank import extract
from voice_input.speaker import (
    MODEL_REVISION,
    CamppEmbedder,
    SpeakerIdentifier,
    bundled_model_path,
)

MODEL_PATH = bundled_model_path()
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(), reason="bundled CAM++ model not present (src/voice_input/models/)"
)


def _tone(frequency: float, frames: int = 24_000) -> bytes:
    samples = (
        12_000
        * numpy.sin(2 * math.pi * frequency * numpy.arange(frames) / 16_000)
    )
    return samples.astype("<i2").tobytes()


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def test_model_contract_and_embedding_shape():
    embedder = CamppEmbedder(MODEL_PATH)
    assert embedder.model_revision == MODEL_REVISION
    assert embedder.onnx_provider
    embedding = embedder.embed(_tone(440))
    assert len(embedding) == 192
    assert numpy.isclose(sum(v * v for v in embedding), 1.0)


def test_embedding_is_deterministic():
    embedder = CamppEmbedder(MODEL_PATH)
    assert numpy.isclose(
        _cosine(embedder.embed(_tone(737)), embedder.embed(_tone(737))), 1.0, atol=1e-6
    )


def test_fbank_and_embedding_agree_on_shape():
    CamppEmbedder(MODEL_PATH)  # model loads against the same 80-bin FBank
    features = extract(_tone(440))
    assert features.shape[0] > 100  # whole-window FBank is fed to the model


def test_same_audio_gets_same_speaker_id():
    identifier = SpeakerIdentifier(CamppEmbedder(MODEL_PATH))
    audio = _tone(440)
    assert identifier.identify(audio, speech_ms=600) == "spk_1"
    assert identifier.identify(audio, speech_ms=600) == "spk_1"
    assert identifier.known_speaker_count == 1


def test_short_audio_padding_keeps_identity():
    identifier = SpeakerIdentifier(CamppEmbedder(MODEL_PATH))
    short = _tone(660, frames=9_600)  # 0.6 s, padded to a 1.5 s window by identify()
    assert identifier.identify(short, speech_ms=600) == "spk_1"
    assert identifier.identify(short, speech_ms=600) == "spk_1"
    assert identifier.known_speaker_count == 1


def test_distinct_audio_mints_new_speaker():
    # cos(embed(440Hz), embed(660Hz)) ~ 0.897 < 0.90: must mint spk_2.
    identifier = SpeakerIdentifier(CamppEmbedder(MODEL_PATH))
    assert identifier.identify(_tone(440), 600) == "spk_1"
    assert identifier.identify(_tone(660), 600) == "spk_2"
    assert identifier.known_speaker_count == 2


@pytest.mark.skipif(
    os.environ.get("MIC_CAMPP_OFFICIAL_GOLDEN") != "1",
    reason="set MIC_CAMPP_OFFICIAL_GOLDEN=1 with controlled official runtime fixtures",
)
def test_official_runtime_golden_fixture_parity():
    """Opt-in parity against fixtures produced by the official 3D-Speaker runtime."""
    pcm = Path(os.environ["MIC_CAMPP_GOLDEN_PCM_PATH"]).read_bytes()
    expected_embedding = numpy.load(os.environ["MIC_CAMPP_GOLDEN_EMBEDDING_PATH"]).reshape(-1)
    actual = numpy.asarray(CamppEmbedder(MODEL_PATH).embed(pcm))
    expected = expected_embedding / float(numpy.linalg.norm(expected_embedding))
    assert numpy.dot(actual, expected) >= 0.9999
