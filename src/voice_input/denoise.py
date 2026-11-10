"""Streaming speech enhancement (降噪) with the bundled GTCRN model (ADR-0006).

Enhancement runs in-process between microphone capture and endpoint detection:
every 640-byte mic block is re-emitted as complete GTCRN-enhanced 20 ms frames,
so the Silero/energy VAD, the live and batch ASR paths and speaker
identification all consume the cleaned signal (the enhancer hook in
``main._collect``).

The bundled model (``models/gtcrn_simple.onnx``) is the official streaming
export of GTCRN (ICASSP 2024, MIT, github.com/Xiaobin-Rong/gtcrn), re-hosted by
sherpa-onnx with metadata (sha256 pinned in DEVELOPMENT.md). The graph consumes one
STFT frame at a time -- ``mix [1, 257, 1, 2]`` float32 real/imag pairs from a
512-point, 256-hop sqrt-Hann STFT -- plus three state caches that are fed back
on every call; the STFT/ISTFT and overlap-add live here, mirroring sherpa-onnx's
``OnlineSpeechDenoiserStftImpl`` semantics exactly (the first hop primes the
analysis buffer and emits nothing, so the enhancer holds audio back by 512
samples = 32 ms plus its frame-alignment buffering; ADR-0006 has the budget).

Frame contract: ``push_async`` takes one 640-byte mic block and returns whole
640-byte frames only.  Hop 256 and frame 320 meet at LCM 1280, so steady state
passes four blocks through as four frames per 80 ms with a bounded internal
buffer; whatever is buffered when the process exits (< 40 ms) is dropped.

Failure policy mirrors the live ASR path (ADR-0005): a hop whose inference
raises passes its samples through raw and counts one failure; after
``MAX_ENHANCE_FAILURES`` consecutive failures the enhancer permanently bypasses
the model (raw samples flow through the frame buffer; one warning, restart to
re-enable).  A model that cannot be loaded at startup degrades to no
enhancement at all -- ``build_enhancer`` returns None with a warning.
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

from voice_input.vad import FRAME_BYTES, SAMPLE_RATE_HZ

LOGGER = logging.getLogger(__name__)

Array = npt.NDArray[Any]

N_FFT: Final = 512
HOP_SAMPLES: Final = 256
NUM_BINS: Final = N_FFT // 2 + 1
MODEL_REVISION: Final = "gtcrn-stream-onnx-v1"
MAX_ENHANCE_FAILURES: Final = 3

_EXPECTED_METADATA: Final = {
    "n_fft": "512",
    "hop_length": "256",
    "window_length": "512",
    "window_type": "hann_sqrt",
    "sample_rate": "16000",
}
_CACHE_PAIRS: Final = (("conv_cache", "conv_cache_out"), ("tra_cache", "tra_cache_out"), ("inter_cache", "inter_cache_out"))

_MIX_NAME: Final = "mix"
_ENH_NAME: Final = "enh"


class Enhancer(Protocol):
    """Async enhancer surface used by the capture loop (main._collect).

    ``push_async`` takes one 20 ms block and returns zero or more whole 20 ms
    blocks; it must never raise into the pump.
    """

    async def push_async(self, block: bytes) -> tuple[bytes, ...]: ...

    async def aclose(self) -> None: ...


class _NodeMeta(Protocol):
    """The subset of onnxruntime's IO metadata that validation reads."""

    name: str
    shape: Sequence[int | str | None]
    type: str


class _ModelMeta(Protocol):
    custom_metadata_map: dict[str, str]


class _DenoiseSession(Protocol):
    """The subset of onnxruntime.InferenceSession this module calls.

    Tests inject a session stand-in (tests/ only, ADR-0002 testing rules); the
    validation below runs against whatever object is used, so the stand-in must
    present the real graph's names, shapes and metadata.
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

    def get_modelmeta(self) -> _ModelMeta: ...


class GtcrnEnhancer:
    """Frame-aligned streaming enhancement over one stateful GTCRN ONNX session.

    Validation is strict because the model is fixed: the graph must expose the
    ``mix/enh [1, 257, 1, 2]`` float32 complex-pair I/O, the three
    ``conv/tra/inter`` cache pairs, and the sherpa-onnx metadata the STFT
    front-end here assumes (512/256/sqrt-Hann/16 kHz).
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        session: _DenoiseSession | None = None,
    ) -> None:
        if session is None:
            import onnxruntime  # local import keeps this module importable without ORT

            if model_path is None:
                raise ValueError("either model_path or session must be given")
            if not model_path.is_file():
                raise FileNotFoundError(f"bundled GTCRN model not found: {model_path}")
            session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        if SAMPLE_RATE_HZ != 16_000:  # the bundled graph is fixed at 16 kHz
            raise ValueError("the bundled GTCRN model requires a 16 kHz pipeline")
        self._session = session
        cache_shapes = _validate_session(session)
        self.model_revision = MODEL_REVISION
        self.onnx_provider = (session.get_providers() or ["unknown"])[0]

        self._window = numpy.sqrt(numpy.hanning(N_FFT)).astype(numpy.float32)
        self._caches: dict[str, Array] = {
            name: numpy.zeros(tuple(shape), dtype=numpy.float32)
            for name, shape in cache_shapes.items()
        }
        self._analysis = numpy.zeros(N_FFT, dtype=numpy.float32)
        self._overlap = numpy.zeros(N_FFT, dtype=numpy.float32)
        self._pending = numpy.zeros(0, dtype=numpy.float32)
        self._started = False
        self._out = bytearray()
        self._failures = 0
        self.bypassed = False

    @property
    def model_active(self) -> bool:
        """True while frames actually go through the model (not primed-out/bypassed)."""
        return not self.bypassed

    async def push_async(self, block: bytes) -> tuple[bytes, ...]:
        """Enhance one 20 ms block; return the whole 20 ms frames now complete.

        Inference runs off the event loop so a slow core never stalls the
        capture pump; only buffer bookkeeping happens on the loop.
        """
        if len(block) % 2:
            raise ValueError("block must contain whole int16 samples")
        return await asyncio.to_thread(self._process, block)

    async def aclose(self) -> None:
        return None

    # ---- processing (worker thread) -----------------------------------------

    def _process(self, block: bytes) -> tuple[bytes, ...]:
        samples = numpy.frombuffer(block, dtype=numpy.int16).astype(numpy.float32)
        samples /= 32768.0
        if self.bypassed:
            self._emit(samples)
            return self._drain_frames()
        self._pending = numpy.concatenate((self._pending, samples))
        offset = 0
        while len(self._pending) - offset >= HOP_SAMPLES:
            self._process_hop(self._pending[offset : offset + HOP_SAMPLES])
            offset += HOP_SAMPLES
        self._pending = self._pending[offset:]
        return self._drain_frames()

    def _process_hop(self, hop: Array) -> None:
        self._analysis[:-HOP_SAMPLES] = self._analysis[HOP_SAMPLES:]
        self._analysis[HOP_SAMPLES:] = hop
        spec = numpy.fft.rfft(self._analysis * self._window)
        mix = numpy.stack((spec.real, spec.imag), axis=-1).reshape(1, NUM_BINS, 1, 2)
        try:
            outputs = self._session.run(
                [_ENH_NAME] + [out for _, out in _CACHE_PAIRS],
                {_MIX_NAME: numpy.ascontiguousarray(mix, dtype=numpy.float32), **self._caches},
            )
            enh = numpy.asarray(outputs[0], dtype=numpy.float32)
            caches = {
                input_name: numpy.asarray(value, dtype=numpy.float32)
                for (input_name, _), value in zip(_CACHE_PAIRS, outputs[1:])
            }
        except Exception as exc:  # noqa: BLE001 - any model fault degrades, never drops the loop
            self._handle_hop_failure(exc)
            return
        if (
            enh.shape != (1, NUM_BINS, 1, 2)
            or not numpy.isfinite(enh).all()
            or any(
                caches[input_name].shape != self._caches[input_name].shape
                for input_name, _ in _CACHE_PAIRS
            )
        ):
            self._handle_hop_failure(ValueError("malformed GTCRN hop output"))
            return
        self._caches = caches
        self._failures = 0
        enhanced = numpy.fft.irfft(
            enh.reshape(NUM_BINS, 2)[:, 0] + 1j * enh.reshape(NUM_BINS, 2)[:, 1], n=N_FFT
        ).astype(numpy.float32)
        self._overlap[:-HOP_SAMPLES] = self._overlap[HOP_SAMPLES:]
        self._overlap[HOP_SAMPLES:] = 0.0
        self._overlap += enhanced * self._window
        if not self._started:
            self._started = True  # priming hop: the analysis buffer was still half zeros
            return
        self._emit(self._overlap[:HOP_SAMPLES])

    def _handle_hop_failure(self, exc: Exception) -> None:
        self._failures += 1
        if self._started:
            self._emit(self._analysis[:HOP_SAMPLES])  # raw pass-through, delay aligned
        else:
            self._started = True
        if self._failures >= MAX_ENHANCE_FAILURES and not self.bypassed:
            self.bypassed = True
            LOGGER.warning(
                "GTCRN enhancer failed %d consecutive hops (last: %s); "
                "permanently bypassed for this connection",
                self._failures,
                exc,
            )
        else:
            LOGGER.debug("GTCRN hop failed, raw samples passed through (x%d)", self._failures)

    def _emit(self, samples: Array) -> None:
        # round-half-even then clamp: float 1.0 must become 32767, never wrap
        scaled = numpy.clip(numpy.rint(samples * 32768.0), -32768.0, 32767.0)
        self._out.extend(scaled.astype(numpy.int16).tobytes())

    def _drain_frames(self) -> tuple[bytes, ...]:
        count = len(self._out) // FRAME_BYTES
        frames = tuple(
            bytes(self._out[offset * FRAME_BYTES : (offset + 1) * FRAME_BYTES])
            for offset in range(count)
        )
        del self._out[: count * FRAME_BYTES]
        return frames


def _validate_session(session: _DenoiseSession) -> dict[str, list[int]]:
    """Reject anything that is not the bundled streaming GTCRN graph.

    Returns the (static) cache input shapes so the caller can zero-init state.
    """
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    for key, expected in _EXPECTED_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"GTCRN model metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    inputs = {node.name: node for node in session.get_inputs()}
    outputs = {node.name: node for node in session.get_outputs()}
    complex_shape = [1, NUM_BINS, 1, 2]
    if (
        inputs.get(_MIX_NAME) is None
        or list(inputs[_MIX_NAME].shape) != complex_shape
        or inputs[_MIX_NAME].type != "tensor(float)"
    ):
        raise ValueError("GTCRN model must expose a [1, 257, 1, 2] float32 'mix' input")
    if outputs.get(_ENH_NAME) is None or list(outputs[_ENH_NAME].shape) != complex_shape:
        raise ValueError("GTCRN model must expose a [1, 257, 1, 2] float32 'enh' output")
    for input_name, output_name in _CACHE_PAIRS:
        if inputs.get(input_name) is None or outputs.get(output_name) is None:
            raise ValueError(f"GTCRN model must expose cache '{input_name}' / '{output_name}'")
        if list(inputs[input_name].shape) != list(outputs[output_name].shape):
            raise ValueError(f"GTCRN cache '{input_name}' input/output shapes differ")
    shapes: dict[str, list[int]] = {}
    for name, _ in _CACHE_PAIRS:
        shape: list[int] = []
        for dim in inputs[name].shape:
            if dim is None or isinstance(dim, str):
                raise ValueError(f"GTCRN cache '{name}' must have static shapes")
            shape.append(dim)
        shapes[name] = shape
    return shapes


def bundled_model_path() -> Path:
    """Resolve the bundled GTCRN model file (ADR-0006: never from the environment)."""
    return Path(str(files("voice_input.models").joinpath("gtcrn_simple.onnx")))


def build_enhancer(
    settings: Any, *, model_path: Path | None = None
) -> GtcrnEnhancer | None:
    """Load the bundled model when VOICE_IN_DENOISE is on; None keeps raw audio.

    ``model_path`` is injectable for tests; production always uses the bundled
    asset via ``bundled_model_path``.  Any load failure degrades to no
    enhancement (the pipeline runs exactly as before, with a warning).
    """
    if not settings.denoise:
        return None
    path = model_path if model_path is not None else bundled_model_path()
    try:
        enhancer = GtcrnEnhancer(path)
    except Exception as exc:  # noqa: BLE001 - any load fault degrades to raw audio
        LOGGER.warning("speech enhancement load failed (%s); running without denoise", exc)
        return None
    LOGGER.info(
        "speech enhancement enabled (revision=%s onnx_provider=%s)",
        enhancer.model_revision,
        enhancer.onnx_provider,
    )
    return enhancer


__all__ = [
    "N_FFT",
    "HOP_SAMPLES",
    "MODEL_REVISION",
    "MAX_ENHANCE_FAILURES",
    "Enhancer",
    "GtcrnEnhancer",
    "bundled_model_path",
    "build_enhancer",
    "SAMPLE_RATE_HZ",
]
