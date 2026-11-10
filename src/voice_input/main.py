"""Voice input module entry point.

Pipeline: microphone capture -> speech enhancement (GTCRN, ADR-0006) -> endpoint
detection (VAD) -> ASR transcription -> speaker identification (ADR-0002) ->
waycast input frame.  Each detected utterance becomes exactly one input frame
with no partial/final marking and no turn-taking involvement (ADR-0011): the
module decides only when an utterance is complete, the core decides everything
else.

Identified speakers travel into the Agent as a content prefix
(``【说话人spk_N】文本``) while ``stable_prefix`` stays pure ASR text; when the
bundled speaker model cannot be loaded the module degrades to no-ID mode and
text is forwarded unlabeled.

The capture and ASR loops are decoupled by a bounded queue so a slow recognizer
never backs up the microphone path; when the queue is full a new endpoint is
dropped and logged, matching the backpressure contract of the hub (ADR-0006
for inputs).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from voice_input.asr import OpenAICompatibleAsr
from voice_input.capture import PortAudioMic
from voice_input.config import load_settings
from voice_input.denoise import Enhancer, build_enhancer
from voice_input.hub_client import MODULE_NAME, HubConnectionError, InputClient
from voice_input.live_asr import LiveAsrTracker, LiveTranscriptStream, live_url_for
from voice_input.silero_vad import SileroEndpointer, build_vad_model
from voice_input.speaker import (
    SpeakerIdentifier,
    build_speaker_identifier,
    disabled_status,
)
from voice_input.vad import (
    SAMPLE_RATE_HZ,
    EndpointDetector,
    Endpointer,
    EnergyEndpointer,
)

LOGGER = logging.getLogger(__name__)

ENDPOINT_QUEUE_SIZE = 4
MAX_TEXT_LENGTH = 8192
HUB_RECONNECT_DELAY_SECONDS = 3.0

LOG_LEVEL_KEY = "VOICE_IN_LOG_LEVEL"
DEPS_LOG_LEVEL_KEY = "VOICE_IN_DEPS_LOG_LEVEL"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# 第三方噪音源（WS 帧、HTTP 请求、事件循环、ONNX 运行时）默认钉在 WARNING，
# 本模块自己的调试日志（端点/转写结果等 DEBUG）不受影响；需排查依赖时
# 用 VOICE_IN_DEPS_LOG_LEVEL 单独放开。
# httpx2/httpcore2 是 openai SDK 内置 vendored 客户端的日志器名。
_THIRD_PARTY_LOGGERS = (
    "websockets",
    "websockets.client",
    "websockets.server",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "asyncio",
    "onnxruntime",
)


def configure_logging() -> None:
    """本模块日志级别走 VOICE_IN_LOG_LEVEL（默认 INFO），依赖库走
    VOICE_IN_DEPS_LOG_LEVEL（默认 WARNING）。"""
    level = os.environ.get(LOG_LEVEL_KEY, "INFO").strip().upper() or "INFO"
    deps_level = (
        os.environ.get(DEPS_LOG_LEVEL_KEY, "WARNING").strip().upper() or "WARNING"
    )
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(deps_level)


async def _send_one(
    ws: InputClient,
    speaker: SpeakerIdentifier | None,
    pcm: bytes,
    speech_ms: int,
    *,
    text: str,
) -> None:
    """Forward one utterance as an input frame, prefixing the speaker id when labeled.

    A raising embedder (e.g. a transient model fault) must never stop the
    pipeline: the utterance is still delivered, just without a label.
    """
    content = text
    if speaker is not None:
        try:
            spk = speaker.identify(pcm, speech_ms)
        except Exception:
            LOGGER.warning(
                "speaker identification failed; sending unlabeled", exc_info=True
            )
        else:
            if spk is not None:
                content = f"【说话人{spk}】{text}"
                LOGGER.debug("labeled utterance speaker=%s", spk)
    metadata = {"speech_duration_ms": speech_ms, "source": "asr", "stable_prefix": text}
    await ws.send_input(content, source="asr", metadata=metadata)


async def _recognize_and_send(
    ws: InputClient | None,
    asr: OpenAICompatibleAsr,
    speaker: SpeakerIdentifier | None,
    queue: asyncio.Queue[tuple[bytes, int, LiveTranscriptStream | None]],
    live: LiveAsrTracker | None,
) -> None:
    """Consume endpoints from the queue: transcribe and forward one input frame per utterance.

    A queued item carries the live streaming session when one was open (the
    normal case with live ASR enabled): its final text usually lands tens of ms
    after the endpoint, long before a batch upload of the same audio could
    start.  If the session failed or timed out, finish() has already closed the
    connection (freeing the server-side model lock) and we fall back to batch.
    ``ws=None`` (standalone debug mode, VOICE_IN_STANDALONE) keeps every other
    step identical but logs the transcript instead of sending a frame.
    """
    while True:
        pcm, speech_ms, stream = await queue.get()
        started_at = time.monotonic()
        try:
            text: str | None = None
            via = "live"
            if stream is not None and live is not None:
                raw = await live.finish(stream)
                if raw is not None:
                    text = raw.strip()
                    if not text:
                        # 空文本是模型对过短/不可辨音频的正常输出（协议成功、非失败）：
                        # 同一模型对同一段音频的批量结果必然也是空，回退批量只会与下
                        # 一句的 live 会话抢单实例模型锁（audio.cpp 串行独占），挂到
                        # 超时并拖垮端点队列。ADR-0005 的兜底只针对流式路径失败。
                        LOGGER.debug(
                            "empty live transcript for %d ms audio; dropping without batch fallback",
                            speech_ms,
                        )
                        continue
                else:
                    await live.discard(stream)  # 兜底：finish 已保证关闭，这里幂等
            elif stream is not None:
                await stream.abort()  # 无跟踪器持有：不留悬空连接，直接走批量
            if text is None:
                via = "batch"
                result = await asr.transcribe(pcm)
                text = result.text
            text = text.strip()
            elapsed = time.monotonic() - started_at
            if not text:
                LOGGER.debug(
                    "empty transcript via %s (%.2fs, %d ms audio); dropping utterance",
                    via,
                    elapsed,
                    speech_ms,
                )
                continue
            LOGGER.debug(
                "asr transcript via %s (%.2fs, %d ms audio): %r",
                via,
                elapsed,
                speech_ms,
                text,
            )
            if len(text) > MAX_TEXT_LENGTH:
                LOGGER.warning("transcript too long (%d chars); dropping", len(text))
                continue
            if ws is None:
                LOGGER.info("standalone: transcript not sent: %r", text)
                continue
            await _send_one(ws, speaker, pcm, speech_ms, text=text)
            LOGGER.info("sent voice input (%d chars)", len(text))
        finally:
            queue.task_done()


def build_endpointer(settings: Any) -> Endpointer:
    """In-process Silero VAD by default, energy VAD on demand or on load failure.

    ``VOICE_IN_VAD=energy`` skips the model entirely; a missing or unloadable
    bundled model degrades the same way (ADR-0007).  The Silero endpointer also
    degrades to the energy detector on its own if inference keeps failing.
    """
    energy = EnergyEndpointer(
        EndpointDetector(
            threshold=settings.speech_threshold,
            trailing_silence_frames=settings.trailing_silence_frames,
            max_frames=settings.max_frames,
            pre_roll_frames=settings.pre_roll_frames,
        )
    )
    if settings.vad_engine == "energy":
        return energy
    model = build_vad_model()
    if model is None:
        return energy
    return SileroEndpointer(
        model,
        fallback=energy,
        threshold=settings.vad_threshold,
        min_speech_ms=settings.vad_min_speech_ms,
        max_frames=settings.max_frames,
    )


def _write_endpoint_wav(directory: str, seq: int, speech_ms: int, pcm: bytes) -> None:
    """Write one endpoint's utterance bytes as a 16 kHz mono WAV (VOICE_IN_DUMP_WAV).

    The file is byte-identical to what the batch route uploads; the live route
    streams a strict prefix of it (``utterance_audio``, ADR-0007), so listening
    to a dump settles exactly what ASR received for that utterance -- including
    whether the leading pad carried the speech onset.
    """
    name = f"ep{seq:04d}-{time.strftime('%H%M%S')}-{speech_ms}ms.wav"
    path = Path(directory) / name
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm)
    LOGGER.debug("dumped endpoint audio (%d bytes) to %s", len(pcm), path)


async def _collect(
    mic: PortAudioMic,
    enhancer: Enhancer | None,
    vad: Endpointer,
    queue: asyncio.Queue[tuple[bytes, int, LiveTranscriptStream | None]],
    live: LiveAsrTracker | None = None,
    dump_wav_dir: str | None = None,
) -> None:
    """Read mic blocks, run endpoint detection, and push complete utterances to the queue.

    With an enhancer (GTCRN, ADR-0006) every block is replaced by the enhanced
    20 ms frames it completes before endpointing: VAD, live ASR and the queued
    endpoint audio all see the cleaned signal.
    With a live ASR tracker, every frame also feeds the per-utterance streaming
    session (started when the VAD opens an utterance); the endpoint's queue item
    carries that session so the worker can collect the transcript that has been
    streaming all along.
    With ``dump_wav_dir`` every fired endpoint is additionally written there as
    a WAV (debug aid; failures never break the capture loop).
    """
    timestamp = 0
    dump_seq = 0
    if dump_wav_dir is not None:
        Path(dump_wav_dir).mkdir(parents=True, exist_ok=True)
        LOGGER.info("endpoint audio dump enabled: %s", dump_wav_dir)
    await mic.open()
    try:
        while True:
            block = await mic.read_block()
            if block is None:
                break
            frames = (
                await enhancer.push_async(block) if enhancer is not None else (block,)
            )
            for frame in frames:
                segment = await vad.feed(frame, timestamp)
                timestamp = (timestamp + 320) % (1 << 32)
                if live is not None:
                    await live.update(frame, vad, endpoint_fired=segment is not None)
                if segment is not None:
                    if dump_wav_dir is not None:
                        dump_seq += 1
                        try:
                            await asyncio.to_thread(
                                _write_endpoint_wav,
                                dump_wav_dir,
                                dump_seq,
                                segment.duration_ms,
                                segment.pcm16le,
                            )
                        except OSError:
                            LOGGER.warning("endpoint dump failed", exc_info=True)
                    stream = live.take() if live is not None else None
                    LOGGER.debug(
                        "endpoint fired: %d ms audio, live_session=%s, queue_depth=%d/%d",
                        segment.duration_ms,
                        stream is not None,
                        queue.qsize(),
                        queue.maxsize,
                    )
                    try:
                        queue.put_nowait((segment.pcm16le, segment.duration_ms, stream))
                    except asyncio.QueueFull:
                        LOGGER.error("endpoint queue full; dropping utterance")
                        if live is not None:
                            await live.discard(stream)
    finally:
        await mic.aclose()


async def _run_with_hub(settings: Any) -> None:
    """One hub connection lifetime: admit, then run capture + ASR until the connection drops."""
    speaker = build_speaker_identifier(reuse_threshold=settings.speaker_threshold)
    enhancer = build_enhancer(settings)

    def status_provider() -> dict[str, object]:
        base = speaker.status() if speaker is not None else disabled_status()
        base["denoise_enabled"] = enhancer is not None and enhancer.model_active
        base["denoise_model_revision"] = (
            enhancer.model_revision if enhancer is not None else None
        )
        return base

    async with InputClient(
        hub_url=settings.hub_url,
        token=settings.hub_token,
        tls_ca_path=settings.tls_ca_path,
        name=MODULE_NAME,
        status_provider=status_provider,
    ) as ws:
        asr = OpenAICompatibleAsr(
            settings.asr_endpoint,
            settings.asr_model,
            settings.asr_api_key,
        )
        queue: asyncio.Queue[tuple[bytes, int, LiveTranscriptStream | None]] = (
            asyncio.Queue(maxsize=ENDPOINT_QUEUE_SIZE)
        )
        mic = PortAudioMic(device=settings.device)
        vad = build_endpointer(settings)
        live = build_live_asr(settings)
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(
                    _collect(mic, enhancer, vad, queue, live, settings.dump_wav_dir)
                )
                tasks.create_task(_recognize_and_send(ws, asr, speaker, queue, live))
        finally:
            if live is not None:
                await live.aclose()
            await asr.aclose()
            await vad.aclose()
            if enhancer is not None:
                await enhancer.aclose()


async def _run_standalone(settings: Any) -> None:
    """Standalone debug run (VOICE_IN_STANDALONE): the full pipeline, no hub.

    Capture, enhancement, endpointing, dumps (VOICE_IN_DUMP_WAV) and ASR all
    run exactly as in production; transcripts are logged instead of sent and no
    speaker labels are computed.  A pipeline failure ends the run (no hub
    means no reconnect loop -- a debug run should die loudly).
    """
    LOGGER.warning(
        "standalone mode: not connecting to the hub, transcripts logged only"
    )
    enhancer = build_enhancer(settings)
    asr = OpenAICompatibleAsr(
        settings.asr_endpoint,
        settings.asr_model,
        settings.asr_api_key,
    )
    queue: asyncio.Queue[tuple[bytes, int, LiveTranscriptStream | None]] = (
        asyncio.Queue(maxsize=ENDPOINT_QUEUE_SIZE)
    )
    mic = PortAudioMic(device=settings.device)
    vad = build_endpointer(settings)
    live = build_live_asr(settings)
    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(
                _collect(mic, enhancer, vad, queue, live, settings.dump_wav_dir)
            )
            tasks.create_task(_recognize_and_send(None, asr, None, queue, live))
    finally:
        if live is not None:
            await live.aclose()
        await asr.aclose()
        await vad.aclose()
        if enhancer is not None:
            await enhancer.aclose()


def build_live_asr(settings: Any) -> LiveAsrTracker | None:
    """Streaming ASR tracker when enabled; None keeps the pipeline batch-only.

    The live route URL is derived from the batch ASR endpoint (same server,
    /audio/transcriptions/live).  Disabling is automatic on repeated failure or
    on a server that rejects the model for streaming (live_asr.LiveAsrTracker).
    """
    if not settings.asr_live:
        return None
    LOGGER.info(
        "live ASR enabled: %s model=%s",
        live_url_for(settings.asr_endpoint),
        settings.asr_model,
    )
    return LiveAsrTracker(
        url=live_url_for(settings.asr_endpoint),
        model=settings.asr_model,
        sample_rate=SAMPLE_RATE_HZ,
        finish_timeout=settings.asr_live_timeout_ms / 1000,
        max_failures=settings.asr_live_max_failures,
    )


RECONNECTABLE_ERRORS: tuple[type[BaseException], ...] = (
    HubConnectionError,
    ConnectionError,
    OSError,
)


def _reconnectable(exc: BaseException) -> bool:
    """True when every exception (TaskGroup wraps child failures in groups) is reconnectable."""
    if isinstance(exc, BaseExceptionGroup):
        inner = cast("tuple[BaseException, ...]", exc.exceptions)
        return bool(inner) and all(_reconnectable(item) for item in inner)
    return isinstance(exc, RECONNECTABLE_ERRORS)


async def run() -> None:
    """Entry loop: connect to the hub with reconnection until interrupted."""
    settings = load_settings()
    if settings.standalone:
        await _run_standalone(settings)
        return
    while True:
        try:
            await _run_with_hub(settings)
        except BaseException as exc:
            # A dropped hub surfaces wrapped: TaskGroup raises the read/send
            # failures as an ExceptionGroup, which except-clauses on the
            # wrapped types never match.
            if not _reconnectable(exc):
                raise
            LOGGER.warning("hub connection failed: %s", exc)
        await asyncio.sleep(HUB_RECONNECT_DELAY_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="树莓娘语音输入模块：采集 + VAD + ASR -> waycast input 帧"
    )
    parser.parse_args()
    load_dotenv()
    configure_logging()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


__all__ = ["MODULE_NAME", "main", "run"]
