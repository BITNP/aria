"""Silero endpointer tests (ADR-0007), layered like the denoise tests.

T1 (always, offline): the hysteresis state machine, audio slicing/padding,
live-ASR lockstep, max-length cap, failure degradation and strict graph
validation, all against a scripted probability model (tests/ only, ADR-0002
testing rules).  The graph validator runs against fake onnxruntime sessions
presenting the real graph's names/shapes/types.

T2 (bundled model present, else skipped): the real model sees deterministic
noise and produces no endpoints; an official speech wav can additionally be fed
via VOICE_IN_SILERO_WAV to assert real segmentation.
"""
from __future__ import annotations

import asyncio
import math
import os
import struct
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from voice_input.config import ConfigError, load_settings
from voice_input.main import build_endpointer
from voice_input.silero_vad import (
    MODEL_REVISION,
    SileroEndpointer,
    SileroVadModel,
    build_vad_model,
    bundled_model_path,
)
from voice_input.vad import EndpointDetector, EnergyEndpointer, SpeechSegment

SILENT_FRAME = b"\x00" * 640
LOUD_FRAME = struct.pack("<320h", *([1000] * 320))


class ScriptedModel:
    """Test-only VadModel stand-in: returns scripted probabilities per chunk."""

    model_revision = "scripted"

    def __init__(self, probs: Sequence[float] | None = None, fail_calls: int = 0) -> None:
        self._probs = list(probs or [])
        self._fail_calls = fail_calls
        self.calls = 0
        self.chunk_lengths: list[int] = []

    def run_chunks(self, chunks: Sequence[Any]) -> list[float]:
        self.calls += 1
        assert len(chunks) == 1
        self.chunk_lengths.append(len(chunks[0]))
        if self.calls <= self._fail_calls:
            raise RuntimeError("injected chunk failure")
        if not self._probs:
            return [0.01]
        return [self._probs.pop(0)]


def _endpointer(
    model: ScriptedModel,
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    max_frames: int = 1_500,
) -> SileroEndpointer:
    fallback = EnergyEndpointer(
        EndpointDetector(threshold=300, trailing_silence_frames=2, max_frames=100, pre_roll_frames=1)
    )
    return SileroEndpointer(
        model, fallback=fallback, threshold=threshold, min_speech_ms=min_speech_ms, max_frames=max_frames
    )


def _feed(
    endpointer: SileroEndpointer,
    frames: list[bytes],
    *,
    capture_open: bool = False,
) -> tuple[list[SpeechSegment], list[bool], bytes | None]:
    """Feed frames; return (segments, utterance_open samples, first-open snapshot)."""

    async def scenario() -> tuple[list[SpeechSegment], list[bool], bytes | None]:
        segments: list[SpeechSegment] = []
        opens: list[bool] = []
        snapshot: bytes | None = None
        for index, frame in enumerate(frames):
            segment = await endpointer.feed(frame, index * 320)
            opens.append(endpointer.utterance_open)
            if capture_open and snapshot is None and endpointer.utterance_open:
                snapshot = endpointer.utterance_audio()
            if segment is not None:
                segments.append(segment)
        return segments, opens, snapshot

    return asyncio.run(scenario())


def _frames(count: int) -> list[bytes]:
    return [SILENT_FRAME] * count


# ---- T1: hysteresis endpointing ------------------------------------------------


def test_endpoint_after_speech_then_silence():
    # 10 speech chunks (start at 0), then sub-negative probabilities: the
    # utterance closes once temp_end (chunk 10) has >= 100 ms of silence.
    model = ScriptedModel([0.9] * 10 + [0.05] * 5)
    endpointer = _endpointer(model)
    segments, _, _ = _feed(endpointer, _frames(24))  # 15 chunks fit in 24 frames
    assert len(segments) == 1
    assert segments[0].duration_ms == 5_120 // 16  # speech span only, pads excluded
    assert len(segments[0].pcm16le) == 5_120 * 2  # start pad clamps at buffer head
    assert endpointer.utterance_open is False


def test_hysteresis_band_holds_the_utterance_open():
    # probabilities between neg (0.35) and threshold (0.5) neither extend nor
    # cancel the silence timer: the 0.4 stretch stays part of the utterance.
    model = ScriptedModel([0.9] * 6 + [0.4] * 4 + [0.9] * 2 + [0.05] * 5)
    endpointer = _endpointer(model)
    segments, _, _ = _feed(endpointer, _frames(28))
    assert len(segments) == 1
    assert segments[0].duration_ms == 12 * 512 // 16  # closes at the 12th chunk start


def test_short_dip_reopens_silence_timer():
    # a sub-negative dip shorter than min_silence is cancelled by returning
    # speech: one segment whose span contains the dip, not two (or none).
    model = ScriptedModel([0.9] * 4 + [0.05] * 2 + [0.9] * 2 + [0.05] * 6)
    endpointer = _endpointer(model)
    segments, _, _ = _feed(endpointer, _frames(21))
    assert len(segments) == 1
    assert segments[0].duration_ms == 4_096 // 16  # 256 ms span across the dip


def test_blip_below_min_speech_is_discarded():
    model = ScriptedModel([0.9] * 3 + [0.05] * 6)  # 1536-sample span <= 4000
    endpointer = _endpointer(model)
    segments, opens, _ = _feed(endpointer, _frames(13))
    assert segments == []
    assert any(opens)  # the blip did open the utterance (live ASR lockstep)...
    assert opens[-1] is False  # ...and it was closed again as noise


def test_max_frames_caps_a_runaway_utterance():
    model = ScriptedModel([0.9] * 16)
    endpointer = _endpointer(model, max_frames=20)  # 6400-sample cap
    segments, _, _ = _feed(endpointer, _frames(25))
    assert len(segments) == 1
    assert segments[0].duration_ms == 6_656 // 16  # hard cut at the cap boundary


def test_threshold_is_configurable():
    model = ScriptedModel([0.6] * 10 + [0.5] * 5)
    endpointer = _endpointer(model, threshold=0.7)  # 0.6 no longer speech
    segments, _, _ = _feed(endpointer, _frames(24))
    assert segments == []


# ---- T1: live-ASR lockstep ------------------------------------------------------


def test_open_snapshot_is_a_prefix_of_the_emitted_audio():
    # live_asr opens its session with utterance_audio() at first open and then
    # streams frames: the emitted endpoint audio must extend that snapshot.
    model = ScriptedModel([0.9] * 10 + [0.05] * 5)
    endpointer = _endpointer(model)
    segments, _, snapshot = _feed(endpointer, _frames(24), capture_open=True)
    assert snapshot is not None and len(snapshot) > 0
    assert len(segments) == 1
    assert segments[0].pcm16le.startswith(snapshot)


def test_utterance_audio_empty_while_idle():
    model = ScriptedModel()
    endpointer = _endpointer(model)
    _feed(endpointer, _frames(8))
    assert endpointer.utterance_audio() == b""


# ---- T1: failure degradation -----------------------------------------------------


def test_repeated_chunk_failures_degrade_to_energy_vad():
    model = ScriptedModel(fail_calls=10**6)
    endpointer = _endpointer(model)
    frames = _frames(3) + [LOUD_FRAME] * 30 + _frames(5)
    segments, _, _ = _feed(endpointer, frames)
    assert endpointer.degraded is True
    assert endpointer.vad_engine == "energy"
    assert len(segments) == 1  # the loud utterance is still endpointed, by energy


def test_single_chunk_failure_retries_without_degrading():
    model = ScriptedModel([0.9] * 10 + [0.05] * 5, fail_calls=1)
    endpointer = _endpointer(model)
    segments, _, _ = _feed(endpointer, _frames(24))
    assert endpointer.degraded is False
    assert len(segments) == 1  # the failed chunk is retried, nothing shifts


def test_feed_rejects_wrong_frame_size():
    endpointer = _endpointer(ScriptedModel())

    async def scenario() -> None:
        await endpointer.feed(b"\x00" * 638, 0)

    with pytest.raises(ValueError):
        asyncio.run(scenario())


def test_idle_buffer_stays_bounded():
    model = ScriptedModel()  # all silence
    endpointer = _endpointer(model)
    _feed(endpointer, _frames(400))
    assert len(endpointer._pcm) <= (1_600 + 640) * 2


# ---- T1: strict graph validation ---------------------------------------------------


class _Node:
    def __init__(self, name: str, shape: list[Any], type_name: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_name


class ScriptedSession:
    """Minimal InferenceSession stand-in shaped like the real Silero graph."""

    def __init__(
        self,
        *,
        inputs: list[_Node] | None = None,
        outputs: list[_Node] | None = None,
    ) -> None:
        self._inputs = inputs or [
            _Node("input", [None, None]),
            _Node("state", [2, None, 128]),
            _Node("sr", [], "tensor(int64)"),
        ]
        self._outputs = outputs or [
            _Node("output", [None, 1]),
            _Node("stateN", [None, None, None]),
        ]

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, object],
        run_options: object | None = None,
    ) -> list[Any]:
        import numpy

        feed = cast("dict[str, Any]", input_feed)
        return [numpy.array([[0.25]], dtype=numpy.float32), feed["state"]]  # type: ignore[call-overload]

    def get_inputs(self) -> list[_Node]:
        return self._inputs

    def get_outputs(self) -> list[_Node]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def test_model_runs_over_valid_graph_shape():
    model = SileroVadModel(session=ScriptedSession())  # type: ignore[arg-type]
    import numpy

    probs = model.run_chunks([numpy.zeros(512, dtype=numpy.float32)])
    assert probs == [0.25]


def test_missing_or_extra_inputs_are_rejected():
    broken = [_Node("input", [None, None]), _Node("state", [2, None, 128])]
    with pytest.raises(ValueError, match="input/state/sr"):
        SileroVadModel(session=ScriptedSession(inputs=broken))  # type: ignore[arg-type]


def test_wrong_state_rank_is_rejected():
    broken = [
        _Node("input", [None, None]),
        _Node("state", [2, 128]),
        _Node("sr", [], "tensor(int64)"),
    ]
    with pytest.raises(ValueError, match="3-dimensional"):
        SileroVadModel(session=ScriptedSession(inputs=broken))  # type: ignore[arg-type]


def test_non_scalar_sr_is_rejected():
    broken = [
        _Node("input", [None, None]),
        _Node("state", [2, None, 128]),
        _Node("sr", [1], "tensor(int64)"),
    ]
    with pytest.raises(ValueError, match="sr"):
        SileroVadModel(session=ScriptedSession(inputs=broken))  # type: ignore[arg-type]


def test_malformed_probability_output_is_rejected():
    outputs = [_Node("output", [None, 2]), _Node("stateN", [None, None, None])]
    with pytest.raises(ValueError, match="probability output"):
        SileroVadModel(session=ScriptedSession(outputs=outputs))  # type: ignore[arg-type]


def test_runtime_state_shape_guard_rejects_mismatched_state():
    import numpy

    class BadStateSession(ScriptedSession):
        def run(self, output_names, input_feed, run_options=None):  # type: ignore[no-untyped-def]
            return [numpy.array([[0.2]], dtype=numpy.float32), numpy.zeros((2, 1, 64), numpy.float32)]

    model = SileroVadModel(session=BadStateSession())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state shape"):
        model.run_chunks([numpy.zeros(512, dtype=numpy.float32)])


# ---- T1: build wiring -------------------------------------------------------------

BASE_ENV = {
    "WAYCAST_HUB_URL": "ws://hub.test:1",
    "VOICE_IN_ASR_ENDPOINT": "https://asr.test/v1",
    "VOICE_IN_ASR_MODEL": "asr",
}


def test_build_endpointer_defaults_to_silero_when_model_present():
    endpointer = build_endpointer(load_settings(BASE_ENV))
    asyncio.run(endpointer.aclose())
    expected = "SileroEndpointer" if bundled_model_path().is_file() else "EnergyEndpointer"
    assert endpointer.__class__.__name__ == expected


def test_build_endpointer_forces_energy():
    endpointer = build_endpointer(load_settings({**BASE_ENV, "VOICE_IN_VAD": "energy"}))
    asyncio.run(endpointer.aclose())
    assert endpointer.__class__.__name__ == "EnergyEndpointer"


def test_unknown_vad_engine_is_a_config_error():
    with pytest.raises(ConfigError):
        load_settings({**BASE_ENV, "VOICE_IN_VAD": "audio-cpp"})


def test_build_vad_model_missing_file_degrades_to_none(tmp_path: Path):
    assert build_vad_model(tmp_path / "absent.onnx") is None


# ---- T2: real bundled model ---------------------------------------------------------

_HAS_MODEL = pytest.mark.skipif(
    not bundled_model_path().is_file(), reason="bundled Silero VAD model not present"
)


@_HAS_MODEL
def test_real_model_loads_with_expected_revision():
    model = build_vad_model()
    assert model is not None
    assert model.model_revision == MODEL_REVISION
    assert model.onnx_provider == "CPUExecutionProvider"


@_HAS_MODEL
def test_real_model_keeps_quiet_noise_endpoint_free():
    import numpy

    rng = numpy.random.default_rng(11)
    noise = (rng.normal(0, 0.02, size=16_000 * 8) * 32768).astype(numpy.int16)
    endpointer = _endpointer(cast(Any, build_vad_model()))
    frames = [
        noise[start : start + 320].tobytes() for start in range(0, len(noise) - 319, 320)
    ]
    segments, _, _ = _feed(endpointer, frames)
    assert segments == []
    assert endpointer.degraded is False


@_HAS_MODEL
def test_real_model_segments_official_speech_wav():
    path = os.environ.get("VOICE_IN_SILERO_WAV", "")
    if not path or not Path(path).is_file():
        pytest.skip("set VOICE_IN_SILERO_WAV to a 16 kHz mono WAV for the speech fixture")
    import numpy

    with wave.open(path) as source:
        assert source.getframerate() == 16_000 and source.getnchannels() == 1
        pcm = numpy.frombuffer(source.readframes(source.getnframes()), dtype=numpy.int16)
    endpointer = _endpointer(cast(Any, build_vad_model()))
    frames = [pcm[start : start + 320].tobytes() for start in range(0, len(pcm) - 319, 320)]
    segments, _, _ = _feed(endpointer, frames)
    assert segments, "real speech must produce at least one endpoint"
    speech_ms = sum(segment.duration_ms for segment in segments)
    assert speech_ms > len(pcm) / 16 * 0.3  # most of the fixture recognized as speech
    assert all(len(s.pcm16le) >= s.duration_ms * 32 for s in segments)  # pads included
    assert all(math.isfinite(s.duration_ms) and s.duration_ms > 0 for s in segments)
