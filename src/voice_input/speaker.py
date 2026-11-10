"""Dynamic speaker identification with the bundled CAM++ ONNX model (ADR-0002).

The module re-introduces CAM++ speaker recognition as an in-process, fully
in-memory 1:N dynamic identification: no registration, no consent, no
encryption, no persistence.  One VAD endpoint produces one embedding and one
match; the identity travels into the Agent as a content prefix
(``【说话人spk_N】ASR文本``) while ``stable_prefix`` stays pure ASR text.

Labeling rules (ADR-0002):

- utterances with less than ``MIN_SPEECH_MS`` of speech are sent unlabeled
  and are never enrolled;
- shorter utterances are zero-padded to the ``ID_WINDOW_MS`` window before
  embedding; longer ones are embedded whole;
- a speaker id is reused only when the best cosine match is >= the
  identifier's ``reuse_threshold`` (default ``DEFAULT_REUSE_THRESHOLD``, ADR-0009)
  and the gap to the second-best is > ``AMBIGUITY_GAP``;
- a match with all similarities below the threshold (or no known speakers yet)
  mints a new ``spk_N`` id immediately (no probation);
- an ambiguous match (best-second gap <= ``AMBIGUITY_GAP``) is neither labeled
  nor enrolled and no new id is minted;
- before merging a window into a speaker's running mean, its cosine to the
  mean must be >= the reuse threshold (anti-pollution); the mean keeps
  at most ``MEAN_MAX_CONTRIBUTIONS`` recent windows.

The identity table resets on process restart (speakers are renumbered from
``spk_1``), matching global memory (waycast ADR-0005): identity is a
session-scoped concept for personalization only, never an authority.

If the bundled model cannot be loaded, ``build_speaker_identifier`` returns
None and the pipeline degrades to no-ID mode with a warning.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Final, Protocol

import numpy

from voice_input.fbank import extract as extract_fbank
from voice_input.vad import SAMPLE_RATE_HZ

LOGGER = logging.getLogger(__name__)

MIN_SPEECH_MS: Final = 500
ID_WINDOW_MS: Final = 1_500
ID_WINDOW_SAMPLES: Final = SAMPLE_RATE_HZ * ID_WINDOW_MS // 1_000
# Real-speech operating point for CAM++ embeddings (ADR-0009): the official
# 3D-Speaker CAM++ model card ships thr=0.31 (accessed 2026-11-06), and the
# bundled model measures same-speaker cross-segment cosines of 0.47–0.84
# (measured on ref_audio.wav) — the previous 0.90 default was unreachable and
# minted a fresh speaker for every utterance.
DEFAULT_REUSE_THRESHOLD: Final = 0.31
AMBIGUITY_GAP: Final = 0.05
MEAN_MAX_CONTRIBUTIONS: Final = 64
MODEL_REVISION: Final = "camplusplus-onnx-v1"
_EMBEDDING_DIMENSIONS: Final = frozenset({192, 512})

Embedding = tuple[float, ...]


class Embedder(Protocol):
    """Anything that turns 1.5 s+ PCM16 mono audio into a unit-norm embedding."""

    model_revision: str
    onnx_provider: str

    def embed(self, pcm16le: bytes) -> Embedding: ...


class CamppEmbedder:
    """Runs the bundled ``feature -> embedding`` CAM++ ONNX model.

    Validation is strict because the model is fixed: the graph must expose a
    ``[batch, frames, 80]`` float32 ``feature`` input with a dynamic frame axis
    and a ``[batch, 192|512]`` float32 ``embedding`` output.
    """

    def __init__(
        self, model_path: Path, *, providers: Sequence[str] | None = None
    ) -> None:
        import onnxruntime  # local import keeps this module importable without ORT

        if not model_path.is_file():
            raise FileNotFoundError(f"bundled CAM++ model not found: {model_path}")
        try:
            session = onnxruntime.InferenceSession(
                str(model_path), providers=_preferred_providers(onnxruntime, providers)
            )
        except Exception as exc:
            raise ValueError(f"cannot load CAM++ ONNX model: {exc}") from exc
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if (
            len(inputs) != 1
            or inputs[0].name != "feature"
            or len(inputs[0].shape) != 3
            or inputs[0].shape[-1] != 80
            or inputs[0].type != "tensor(float)"
            or isinstance(inputs[0].shape[1], int)
        ):
            raise ValueError(
                "bundled CAM++ model must expose a dynamic [batch, frames, 80] "
                "float32 'feature' input"
            )
        if (
            len(outputs) != 1
            or outputs[0].name != "embedding"
            or len(outputs[0].shape) != 2
            or outputs[0].shape[-1] not in _EMBEDDING_DIMENSIONS
            or outputs[0].type != "tensor(float)"
        ):
            raise ValueError(
                "bundled CAM++ model must expose a [batch, 192|512] "
                "float32 'embedding' output"
            )
        self.model_revision = MODEL_REVISION
        self._session = session
        self._input_name = inputs[0].name
        self._dimension = int(outputs[0].shape[-1])
        self.onnx_provider = session.get_providers()[0]

    def embed(self, pcm16le: bytes) -> Embedding:
        """Embed one 1.5 s identification window into a unit-norm vector."""
        features = extract_fbank(pcm16le)
        if features.size == 0:
            raise ValueError("insufficient audio for CAM++ FBank embedding")
        raw = self._session.run(
            None, {self._input_name: features[numpy.newaxis, :, :]}
        )[0]
        output = numpy.asarray(raw).reshape(-1).astype(numpy.float64)
        if output.size != self._dimension or not numpy.isfinite(output).all():
            raise ValueError(
                f"CAM++ embedding output must be {self._dimension}D and finite"
            )
        norm = float(numpy.linalg.norm(output))
        if norm == 0:
            raise ValueError("CAM++ embedding has zero norm")
        output /= norm
        return tuple(float(value) for value in output)


class SpeakerIdentifier:
    """In-memory 1:N dynamic speaker identification (ADR-0002)."""

    def __init__(
        self, embedder: Embedder, *, reuse_threshold: float = DEFAULT_REUSE_THRESHOLD
    ) -> None:
        if not 0.0 < reuse_threshold < 1.0:
            raise ValueError("reuse_threshold must be between 0 and 1 (exclusive)")
        self._embedder = embedder
        self._reuse_threshold = reuse_threshold
        self._vectors: dict[str, deque[Embedding]] = {}
        self._next_number = 1

    @property
    def known_speaker_count(self) -> int:
        return len(self._vectors)

    def identify(self, pcm16le: bytes, speech_ms: int) -> str | None:
        """Return the speaker id for one utterance, or None when no label applies."""
        if speech_ms < MIN_SPEECH_MS:
            return None
        window = _pad_to_window(pcm16le)
        vector = _normalize(self._embedder.embed(window))
        return self._decide(vector)

    def _decide(self, vector: Embedding) -> str | None:
        if not self._vectors:
            decision = self._mint(vector)
            LOGGER.debug("speaker mint %s (no known speakers yet)", decision)
            return decision
        matches = sorted(
            (
                (spk, _cosine(vector, _mean(contributions)))
                for spk, contributions in self._vectors.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        best_id, best = matches[0]
        second: float | None = matches[1][1] if len(matches) > 1 else None
        if best < self._reuse_threshold:
            decision = self._mint(vector)
            LOGGER.debug(
                "speaker mint %s (best %s=%.3f < %.3f, %d known)",
                decision,
                best_id,
                best,
                self._reuse_threshold,
                len(matches),
            )
            return decision
        if second is not None and best - second <= AMBIGUITY_GAP:
            LOGGER.debug(
                "speaker ambiguous (best %s=%.3f, second=%.3f, gap <= %.2f)",
                best_id,
                best,
                second,
                AMBIGUITY_GAP,
            )
            return None
        self._enroll(best_id, vector)
        LOGGER.debug(
            "speaker reuse %s (best=%.3f, second=%s, threshold=%.3f)",
            best_id,
            best,
            f"{second:.3f}" if second is not None else "none",
            self._reuse_threshold,
        )
        return best_id

    def _mint(self, vector: Embedding) -> str:
        spk = f"spk_{self._next_number}"
        self._next_number += 1
        self._enroll(spk, vector)
        return spk

    def _enroll(self, spk: str, vector: Embedding) -> None:
        window = self._vectors.setdefault(spk, deque(maxlen=MEAN_MAX_CONTRIBUTIONS))
        if window and _cosine(vector, _mean(window)) < self._reuse_threshold:
            LOGGER.debug("anti-pollution: not merging window into %s", spk)
            return
        window.append(vector)

    def status(self) -> dict[str, object]:
        return {
            "speaker_enabled": True,
            "model_revision": self._embedder.model_revision,
            "known_speaker_count": self.known_speaker_count,
            "onnx_provider": self._embedder.onnx_provider,
        }


def bundled_model_path() -> Path:
    """Resolve the bundled CAM++ model file (ADR-0002: never from the environment)."""
    return Path(str(files("voice_input.models").joinpath("campp.onnx")))


def build_speaker_identifier(
    model_path: Path | None = None,
    *,
    providers: Sequence[str] | None = None,
    reuse_threshold: float = DEFAULT_REUSE_THRESHOLD,
) -> SpeakerIdentifier | None:
    """Load the bundled model; return None (degraded no-ID mode) on any failure.

    ``model_path`` is injectable for tests; production always uses the bundled
    asset via ``bundled_model_path``.
    """
    path = model_path if model_path is not None else bundled_model_path()
    try:
        embedder = CamppEmbedder(path, providers=providers)
    except Exception as exc:
        LOGGER.warning(
            "speaker identification load failed (%s); degraded to no-ID mode", exc
        )
        return None
    LOGGER.info(
        "speaker identification enabled (revision=%s onnx_provider=%s reuse_threshold=%.2f)",
        embedder.model_revision,
        embedder.onnx_provider,
        reuse_threshold,
    )
    return SpeakerIdentifier(embedder, reuse_threshold=reuse_threshold)


def disabled_status() -> dict[str, object]:
    """Status payload when the module runs in degraded no-ID mode."""
    return {
        "speaker_enabled": False,
        "model_revision": None,
        "known_speaker_count": 0,
        "onnx_provider": None,
    }


def _pad_to_window(pcm16le: bytes) -> bytes:
    target = ID_WINDOW_SAMPLES * 2
    if len(pcm16le) >= target:
        return pcm16le
    return pcm16le + bytes(target - len(pcm16le))


def _normalize(vector: Embedding) -> Embedding:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero-norm embedding")
    return tuple(value / norm for value in vector)


def _mean(vectors: Sequence[Embedding]) -> Embedding:
    dimension = len(vectors[0])
    totals = [0.0] * dimension
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    return tuple(total / len(vectors) for total in totals)


def _cosine(left: Embedding, right: Embedding) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _preferred_providers(
    onnxruntime: object, providers: Sequence[str] | None
) -> list[str]:
    """CUDA first when available, CPU otherwise; honours an explicit override."""
    if providers:
        return list(providers)
    available = set(onnxruntime.get_available_providers())  # type: ignore[attr-defined]
    preferred = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    return preferred or ["CPUExecutionProvider"]


__all__ = [
    "MIN_SPEECH_MS",
    "ID_WINDOW_MS",
    "ID_WINDOW_SAMPLES",
    "DEFAULT_REUSE_THRESHOLD",
    "AMBIGUITY_GAP",
    "MEAN_MAX_CONTRIBUTIONS",
    "MODEL_REVISION",
    "Embedding",
    "Embedder",
    "CamppEmbedder",
    "SpeakerIdentifier",
    "bundled_model_path",
    "build_speaker_identifier",
    "disabled_status",
]
