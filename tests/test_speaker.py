"""Speaker identification unit tests (T1): fake embedder, offline clustering.

The fake embedder is test-only (placed in tests/ per workspace rule); the real
CAM++ model is exercised separately in test_campp.py (T2, skipped when the
bundled model is absent).
"""
from __future__ import annotations

import math
from collections import deque

import pytest

from voice_input.speaker import (
    ID_WINDOW_SAMPLES,
    MEAN_MAX_CONTRIBUTIONS,
    MIN_SPEECH_MS,
    SpeakerIdentifier,
    build_speaker_identifier,
    disabled_status,
)


class _FakeEmbedder:
    """Returns queued unit vectors; records the pcm lengths it saw."""

    def __init__(self, vectors: list[tuple[float, ...]]) -> None:
        self._vectors = list(vectors)
        self.model_revision = "fake"
        self.onnx_provider = "fake"
        self.pcm_lengths: list[int] = []

    def embed(self, pcm16le: bytes) -> tuple[float, ...]:
        self.pcm_lengths.append(len(pcm16le))
        return self._vectors.pop(0)


def _unit(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _identifier(*vectors: tuple[float, float]) -> SpeakerIdentifier:
    return SpeakerIdentifier(_FakeEmbedder(list(vectors)))


def test_short_utterance_is_unlabeled_and_not_enrolled():
    identifier = _identifier(_unit(0.0))
    assert identifier.identify(b"\x00" * 64_000, speech_ms=MIN_SPEECH_MS - 1) is None
    assert identifier.known_speaker_count == 0
    assert identifier._embedder.pcm_lengths == []  # embedder never invoked


def test_unknown_speaker_mints_monotonic_ids():
    identifier = _identifier(_unit(0.0), _unit(1.2), _unit(2.0))
    assert identifier.identify(b"a", speech_ms=600) == "spk_1"
    assert identifier.identify(b"b", speech_ms=600) == "spk_2"
    assert identifier.identify(b"c", speech_ms=600) == "spk_3"
    assert identifier.known_speaker_count == 3


def test_same_speaker_reuses_id():
    identifier = _identifier(_unit(0.0), _unit(0.0), _unit(0.05))
    assert identifier.identify(b"a", 600) == "spk_1"
    assert identifier.identify(b"b", 600) == "spk_1"
    assert identifier.identify(b"c", 600) == "spk_1"
    assert identifier.known_speaker_count == 1


def test_distinct_speaker_is_not_reused_below_threshold():
    # cos(1.2 rad) < 0.90 -> must mint a new id rather than reuse spk_1.
    identifier = _identifier(_unit(0.0), _unit(1.2))
    assert identifier.identify(b"a", 600) == "spk_1"
    assert identifier.identify(b"b", 600) == "spk_2"
    assert identifier.known_speaker_count == 2


def test_boundary_cosine_just_above_threshold_reuses():
    # cos(0.40 rad) ~ 0.921 >= 0.90 with no second candidate -> reuse spk_1.
    identifier = _identifier(_unit(0.0), _unit(0.40))
    assert identifier.identify(b"a", 600) == "spk_1"
    assert identifier.identify(b"b", 600) == "spk_1"
    assert identifier.known_speaker_count == 1


def test_short_endpoint_is_padded_to_window():
    identifier = _identifier(_unit(0.0))
    short = b"\x00" * 16_000  # 0.5 s
    identifier.identify(short, speech_ms=600)
    assert identifier._embedder.pcm_lengths == [ID_WINDOW_SAMPLES * 2]


def test_short_endpoint_restored_to_full_window():
    # An endpoint shorter than the window is embedded as a padded 1.5 s one, so
    # the same short audio keeps its identity across utterances.
    identifier = _identifier(_unit(0.0), _unit(0.0))
    short = bytes(8_000)
    assert identifier.identify(short, 600) == "spk_1"
    assert identifier.identify(short, 600) == "spk_1"


def test_padding_is_zeros_not_silence_reuse():
    # Padded zeros yield the same embedding as the same audio re-embedded.
    identifier = _identifier(_unit(0.0), _unit(0.0))
    audio = b"\x01\x02" * 6_000
    assert identifier.identify(audio, 600) == "spk_1"
    assert identifier.identify(audio, 600) == "spk_1"


def test_ambiguous_match_is_not_labeled_or_minted():
    # Two seeded speakers A=(1,0) and B=(cos 0.24, sin 0.24); a query midway
    # (cos 0.12) is >= threshold to both and within AMBIGUITY_GAP of both.
    identifier = _identifier(_unit(0.0))
    identifier._vectors["spk_a"] = deque([_unit(0.0)])
    identifier._vectors["spk_b"] = deque([_unit(0.24)])
    count_before = identifier.known_speaker_count
    assert identifier.identify(b"x", 600) is None
    assert identifier.known_speaker_count == count_before
    # Neither seeded speaker gained a contribution.
    assert len(identifier._vectors["spk_a"]) == 1
    assert len(identifier._vectors["spk_b"]) == 1


def test_unambiguous_reuse_when_second_is_far():
    # Seeded mean A=(1,0); a second speaker far away means the reuse decision
    # has room for the gap: best ~0.98, second ~0.66 > 0.05 apart.
    identifier = _identifier(_unit(0.0))
    identifier._vectors["spk_a"] = deque([_unit(0.0)])
    identifier._vectors["spk_b"] = deque([_unit(1.0)])
    assert identifier.identify(b"x", 600) == "spk_a"
    assert len(identifier._vectors["spk_a"]) == 2


def test_anti_pollution_guard_blocks_unrelated_merge():
    identifier = _identifier(_unit(0.0))
    identifier._vectors["spk_a"] = deque([_unit(0.0)])
    identifier._enroll("spk_a", _unit(1.4))  # cos(1.4) < threshold -> rejected
    assert len(identifier._vectors["spk_a"]) == 1


def test_running_mean_window_is_capped():
    identifier = _identifier(_unit(0.0))
    window = deque(maxlen=MEAN_MAX_CONTRIBUTIONS)
    identifier._vectors["spk_a"] = window
    for index in range(MEAN_MAX_CONTRIBUTIONS + 10):
        window.append(_unit(index / 100))
    assert len(window) == MEAN_MAX_CONTRIBUTIONS


def test_status_reports_facts():
    identifier = _identifier(_unit(0.0))
    identifier.identify(b"a", 600)
    status = identifier.status()
    assert status["speaker_enabled"] is True
    assert status["model_revision"] == "fake"
    assert status["onnx_provider"] == "fake"
    assert status["known_speaker_count"] == 1


def test_zero_norm_embedding_raises():
    identifier = SpeakerIdentifier(_FakeEmbedder([(0.0, 0.0)]))
    with pytest.raises(ValueError):
        identifier.identify(b"x" * 64, 600)


def test_build_returns_none_for_missing_model(tmp_path):
    assert build_speaker_identifier(tmp_path / "missing.onnx") is None


def test_build_returns_none_for_corrupt_model(tmp_path):
    bogus = tmp_path / "bogus.onnx"
    bogus.write_bytes(b"not an onnx model")
    assert build_speaker_identifier(bogus) is None


def test_disabled_status_payload():
    status = disabled_status()
    assert status["speaker_enabled"] is False
    assert status["model_revision"] is None
    assert status["onnx_provider"] is None
    assert status["known_speaker_count"] == 0
