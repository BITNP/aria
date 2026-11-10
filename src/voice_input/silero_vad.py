"""Streaming endpoint detection with the bundled Silero VAD ONNX model (ADR-0007).

The audio.cpp HTTP task route is gone: the official streaming export
(github.com/snakers4/silero-vad, MIT) runs in-process on onnxruntime, mirroring
the OnnxWrapper semantics exactly -- one 512-sample chunk (32 ms @ 16 kHz) at a
time, a 64-sample context from the previous chunk prepended on the host side,
and the ``state [2, 1, 128]`` tensor fed back on every call.  Because state
persists across the whole connection, the polling artifacts of the old route
(ADR-0003/0004: full-buffer re-segmentation, per-request resets, the 4160-sample
blind window) no longer exist and endpoint latency is set by the detector itself.

Endpointing follows the official post-processing (``utils_vad`` hysteresis):
speech triggers at ``probability >= threshold`` (default 0.5), only probabilities
below ``threshold - 0.15`` start a silence timer, and silence of
``MIN_SILENCE_SAMPLES`` (100 ms, official default) closes the utterance.  Gaps
between the two thresholds neither extend nor cancel the timer -- faithful
translation, max-segment-length branches are dropped because this module caps
length via ``max_frames`` instead.  A closed span shorter than
``min_speech_ms`` is discarded as noise (the live ASR session, if any, is
aborted by the tracker when ``utterance_open`` goes back to False).

Frame contract: ``feed`` takes one 640-byte mic frame; chunk boundaries fall
inside frames (LCM(320, 512) = 2560), so samples buffer across feeds.  Emitted
utterance audio is ``[speech_start - 150 ms, end + 150 ms]`` clipped to the
buffer (ADR-0010: the lead pad covers the model's attack latency -- on real
speech ``prob >= threshold`` was measured to lag the true onset by 56..88 ms,
so the official 30 ms pad still cut 26..58 ms into the first syllable);
``utterance_audio`` reports ``[speech_start - pad, fed)`` so the live ASR
session streams exactly the bytes the endpoint will emit (the frames that
trigger a close may carry up to one chunk of trailing silence the live stream
never saw -- silence only, which costs the transcript nothing).

Failure policy mirrors the other model paths: an inference fault skips that one
chunk (state survives), ``MAX_VAD_FAILURES`` consecutive faults permanently
degrade this connection to the energy endpointer (one warning; reconnect or
restart to recover), and a model that cannot be loaded at startup never builds
-- ``build_vad_model`` returns None and the pipeline runs on energy VAD.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, Protocol

import numpy
import numpy.typing as npt

from voice_input.vad import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE_HZ,
    Endpointer,
    SpeechSegment,
)

LOGGER = logging.getLogger(__name__)

WINDOW_SAMPLES: Final = 512  # the model's fixed chunk (32 ms @ 16 kHz)
CONTEXT_SAMPLES: Final = 64  # previous-chunk tail prepended to every feed
STATE_SHAPE: Final = (2, 1, 128)
MIN_SILENCE_SAMPLES: Final = 1_600  # official min_silence_duration_ms = 100
SPEECH_PAD_SAMPLES: Final = 2_400  # 150 ms, 5x the official speech_pad_ms=30 (ADR-0010)
NEG_THRESHOLD_OFFSET: Final = 0.15  # official neg = threshold - 0.15
RING_SAMPLES: Final = 4_800  # idle audio kept for the next onset's lead pad (300 ms)
DEFAULT_THRESHOLD: Final = 0.5
MAX_VAD_FAILURES: Final = 3
MODEL_REVISION: Final = "silero-vad-v5-onnx"

_SAMPLES_PER_MS: Final = SAMPLE_RATE_HZ // 1000
Array = npt.NDArray[Any]


class VadModel(Protocol):
    """Per-chunk probability surface used by the endpointer.

    ``run_chunks`` consumes 512-sample float32 chunks in order and returns one
    speech probability each; implementations keep the streaming state (context
    + recurrent state) between calls.
    """

    model_revision: str

    def run_chunks(self, chunks: Sequence[numpy.ndarray]) -> list[float]: ...


class _NodeMeta(Protocol):
    name: str
    shape: Sequence[int | str | None]
    type: str


class _VadSession(Protocol):
    """The subset of onnxruntime.InferenceSession this module calls.

    Tests inject session stand-ins (tests/ only, ADR-0002 testing rules); the
    validation below runs against whatever object is used, so a stand-in must
    present the real graph's names, shapes and types.
    """

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, object],
        run_options: object | None = None,
    ) -> Sequence[Array]: ...

    def get_inputs(self) -> Sequence[_NodeMeta]: ...

    def get_outputs(self) -> Sequence[_NodeMeta]: ...

    def get_providers(self) -> list[str]: ...


class SileroVadModel:
    """Stateful streaming runner for the bundled ``silero_vad.onnx`` graph.

    Validation is strict because the model is fixed: float ``input`` with
    ``[2, batch, 128]`` float state and an int64 ``sr`` scalar, returning the
    ``[batch, 1]`` probability plus a state of the input shape.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        session: _VadSession | None = None,
    ) -> None:
        if SAMPLE_RATE_HZ != 16_000:  # the fixed 512-sample window is 16 kHz-only
            raise ValueError("the bundled Silero VAD model requires a 16 kHz pipeline")
        if session is None:
            import onnxruntime  # local import keeps this module importable without ORT

            if model_path is None:
                raise ValueError("either model_path or session must be given")
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"bundled Silero VAD model not found: {model_path}"
                )
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = (
                1  # official OnnxWrapper settings: one thread
            )
            options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
        self._session = session
        _validate_session(session)
        self.model_revision = MODEL_REVISION
        self.onnx_provider = (session.get_providers() or ["unknown"])[0]
        self._state = numpy.zeros(STATE_SHAPE, dtype=numpy.float32)
        self._context = numpy.zeros(CONTEXT_SAMPLES, dtype=numpy.float32)

    def reset(self) -> None:
        self._state = numpy.zeros(STATE_SHAPE, dtype=numpy.float32)
        self._context = numpy.zeros(CONTEXT_SAMPLES, dtype=numpy.float32)

    def run_chunks(self, chunks: Sequence[numpy.ndarray]) -> list[float]:
        """Feed 512-sample float32 chunks (state carried across calls); one prob each."""
        probs: list[float] = []
        for chunk in chunks:
            fed = numpy.concatenate((self._context, chunk))
            outputs = self._session.run(
                None,
                {
                    "input": numpy.ascontiguousarray(
                        fed.reshape(1, -1), dtype=numpy.float32
                    ),
                    "state": self._state,
                    "sr": numpy.array(SAMPLE_RATE_HZ, dtype=numpy.int64),
                },
            )
            prob = numpy.asarray(outputs[0], dtype=numpy.float32).reshape(-1)
            state = numpy.asarray(outputs[1], dtype=numpy.float32)
            if (
                prob.size != 1
                or not numpy.isfinite(prob).all()
                or not numpy.isfinite(state).all()
            ):
                raise ValueError("Silero VAD returned a non-finite chunk result")
            if state.shape != STATE_SHAPE:
                raise ValueError(
                    f"Silero VAD state shape {state.shape} != {STATE_SHAPE}"
                )
            self._state = state
            self._context = fed[-CONTEXT_SAMPLES:].astype(numpy.float32)
            probs.append(float(prob[0]))
        return probs


class SileroEndpointer:
    """Endpointer over in-process Silero probabilities (Endpointer surface).

    The hysteresis state machine (see module docstring) runs per 512-sample
    chunk; inference happens off the event loop via ``asyncio.to_thread`` so a
    slow core never stalls the capture pump.  Sample coordinates are absolute
    per endpointer (per hub connection), which mirrors the endpointer's own
    lifetime.
    """

    def __init__(
        self,
        model: VadModel,
        *,
        fallback: Endpointer,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_ms: int = 250,
        max_frames: int = 1_500,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("silero VAD threshold must be between 0 and 1")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        self._model = model
        self._fallback = fallback
        self._threshold = threshold
        self._neg_threshold = max(threshold - NEG_THRESHOLD_OFFSET, 0.01)
        self._min_speech_samples = min_speech_ms * _SAMPLES_PER_MS
        self._max_samples = max_frames * FRAME_SAMPLES
        self.model_revision = getattr(model, "model_revision", "unknown")

        self._pcm = bytearray()  # audio held for the current/idle utterance
        self._pcm_start = 0  # absolute sample index of _pcm[0]
        self._fed = 0  # samples handed in since construction
        self._chunk_pos = 0  # samples consumed by the model (chunk-aligned)
        self._triggered = False
        self._speech_start = 0
        self._temp_end = 0  # 0 = unset; only ever set while triggered (then > 0)
        self._failures = 0
        self.degraded = False

    @property
    def vad_engine(self) -> str:
        return "energy" if self.degraded else "silero"

    async def feed(self, frame: bytes, sample: int) -> SpeechSegment | None:
        if self.degraded:
            return await self._fallback.feed(frame, sample)
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"frame must be exactly {FRAME_BYTES} PCM16 bytes")
        self._pcm.extend(frame)
        self._fed += FRAME_SAMPLES
        while self._chunk_pos + WINDOW_SAMPLES <= self._fed:
            chunk = self._take_chunk()
            try:
                prob = (await asyncio.to_thread(self._model.run_chunks, [chunk]))[0]
            except Exception as exc:  # noqa: BLE001 - any model fault degrades, never drops the pump
                self._failures += 1
                LOGGER.debug("silero VAD chunk failed (x%d): %s", self._failures, exc)
                if self._failures >= MAX_VAD_FAILURES:
                    LOGGER.warning(
                        "silero VAD failed %d consecutive chunks; degrading to energy VAD",
                        self._failures,
                    )
                    self.degraded = True
                    self._reset_utterance()
                    return await self._fallback.feed(frame, sample)
                continue
            self._failures = 0
            segment = self._on_probability(prob)
            if segment is not None:
                return segment
        self._trim_idle_ring()
        return None

    @property
    def utterance_open(self) -> bool:
        if self.degraded:
            return self._fallback.utterance_open
        return self._triggered

    def utterance_audio(self) -> bytes:
        if self.degraded:
            return self._fallback.utterance_audio()
        if not self._triggered:
            return b""
        return self._slice(
            max(self._speech_start - SPEECH_PAD_SAMPLES, self._pcm_start), self._fed
        )

    async def aclose(self) -> None:
        await self._fallback.aclose()

    # ---- hysteresis state machine (faithful to official get_speech_timestamps) --

    def _on_probability(self, prob: float) -> SpeechSegment | None:
        cur = self._chunk_pos  # absolute sample index of this chunk's first sample
        self._chunk_pos += WINDOW_SAMPLES
        if prob >= self._threshold:
            self._temp_end = 0
            if not self._triggered:
                self._triggered = True
                self._speech_start = cur
                LOGGER.debug(
                    "silero speech start at %d ms (prob %.3f)",
                    cur // _SAMPLES_PER_MS,
                    prob,
                )
            elif cur - self._speech_start >= self._max_samples:
                # official layout: a high probability while already triggered falls
                # through to the max-length check (min-speech filter still applies)
                return self._close(cur, hard_cut=True)  # hard cap: cut here
            return None
        if self._triggered and cur - self._speech_start >= self._max_samples:
            return self._close(cur, hard_cut=True)
        if self._triggered and prob < self._neg_threshold:
            if self._temp_end == 0:
                self._temp_end = cur
            if cur - self._temp_end < MIN_SILENCE_SAMPLES:
                return None
            return self._close(self._temp_end)
        return None  # between thresholds: hold the utterance open, timer untouched

    def _close(self, end: int, *, hard_cut: bool = False) -> SpeechSegment | None:
        start = self._speech_start
        segment: SpeechSegment | None = None
        if end - start > self._min_speech_samples:
            segment = SpeechSegment(
                pcm16le=self._slice(
                    max(start - SPEECH_PAD_SAMPLES, self._pcm_start),
                    min(end + SPEECH_PAD_SAMPLES, self._fed),
                ),
                duration_ms=(end - start) // _SAMPLES_PER_MS,
            )
            LOGGER.debug(
                "silero endpoint: %d ms speech from %d ms",
                segment.duration_ms,
                start // _SAMPLES_PER_MS,
            )
        self._reset_utterance()
        if segment is not None:
            # ADR-0010 boundary policy: a silence close keeps [end, end + pad) in
            # the idle ring so the next onset's lead pad can reach back there (a
            # resume right after min_silence never eats its speech); a mid-speech
            # hard cap discards exactly the audio it emitted (end + pad), keeping
            # consecutive segments contiguous without duplicating speech.
            self._discard_before(end + SPEECH_PAD_SAMPLES if hard_cut else end)
        return segment

    def _reset_utterance(self) -> None:
        self._triggered = False
        self._speech_start = 0
        self._temp_end = 0

    # ---- audio bookkeeping ------------------------------------------------------

    def _take_chunk(self) -> numpy.ndarray:
        first = self._chunk_pos - self._pcm_start
        chunk = numpy.frombuffer(self._pcm, dtype=numpy.int16)[
            first : first + WINDOW_SAMPLES
        ]
        return chunk.astype(numpy.float32) / 32768.0

    def _slice(self, start: int, end: int) -> bytes:
        first = (start - self._pcm_start) * 2
        return bytes(self._pcm[first : (end - self._pcm_start) * 2])

    def _discard_before(self, sample: int) -> None:
        count = min(max(sample - self._pcm_start, 0), len(self._pcm) // 2)
        if count:
            del self._pcm[: count * 2]
            self._pcm_start += count

    def _trim_idle_ring(self) -> None:
        if self._triggered:
            return
        # keep the idle ring, but never past the lead-pad reach of the next
        # possible onset (chunk start - pad): that audio is emission territory
        self._discard_before(
            max(self._fed - RING_SAMPLES, self._chunk_pos - SPEECH_PAD_SAMPLES)
        )


def _validate_session(session: _VadSession) -> None:
    """Reject anything that is not the official streaming Silero VAD graph."""
    inputs = {node.name: node for node in session.get_inputs()}
    if set(inputs) != {"input", "state", "sr"}:
        raise ValueError(
            f"Silero VAD graph must expose input/state/sr, got {sorted(inputs)}"
        )
    if (
        inputs["input"].type != "tensor(float)"
        or inputs["state"].type != "tensor(float)"
    ):
        raise ValueError("Silero VAD input/state must be float tensors")
    if inputs["sr"].type != "tensor(int64)":
        raise ValueError("Silero VAD sr must be an int64 tensor")
    if len(inputs["sr"].shape) != 0:
        raise ValueError("Silero VAD sr must be a scalar tensor")
    state_shape = list(inputs["state"].shape)
    if len(state_shape) != 3:
        raise ValueError(f"Silero VAD state shape {state_shape} must be 3-dimensional")
    # the state's [2, batch, 128] dims are dynamic in the official export, so
    # the exact shape is enforced at runtime in run_chunks instead
    outputs = list(session.get_outputs())
    if len(outputs) != 2:
        raise ValueError(
            "Silero VAD graph must return exactly the probability and the next state"
        )
    out_shape = list(outputs[0].shape)
    if (
        outputs[0].type != "tensor(float)"
        or out_shape[-1:] != [1]
        or len(out_shape) != 2
    ):
        raise ValueError(
            f"Silero VAD probability output shape {out_shape} must be [batch, 1]"
        )


def bundled_model_path() -> Path:
    """Resolve the bundled Silero VAD model file (ADR-0007: never from the environment)."""
    return Path(str(files("voice_input.models").joinpath("silero_vad.onnx")))


def build_vad_model(model_path: Path | None = None) -> SileroVadModel | None:
    """Load the bundled model; None keeps the caller on the energy VAD.

    ``model_path`` is injectable for tests; production always uses the bundled
    asset via ``bundled_model_path``.  Any load failure degrades to energy
    endpointing (the pipeline behaves exactly as with VOICE_IN_VAD=energy).
    """
    path = model_path if model_path is not None else bundled_model_path()
    try:
        model = SileroVadModel(path)
    except Exception as exc:  # noqa: BLE001 - any load fault degrades to energy VAD
        LOGGER.warning("silero VAD model load failed (%s); using energy VAD", exc)
        return None
    LOGGER.info(
        "silero VAD enabled in-process (revision=%s onnx_provider=%s)",
        model.model_revision,
        model.onnx_provider,
    )
    return model


__all__ = [
    "CONTEXT_SAMPLES",
    "DEFAULT_THRESHOLD",
    "MAX_VAD_FAILURES",
    "MIN_SILENCE_SAMPLES",
    "MODEL_REVISION",
    "SPEECH_PAD_SAMPLES",
    "WINDOW_SAMPLES",
    "SileroEndpointer",
    "SileroVadModel",
    "VadModel",
    "build_vad_model",
    "bundled_model_path",
]
