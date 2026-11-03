"""Voice input module entry point.

Pipeline: microphone capture -> endpoint detection (VAD) -> ASR transcription ->
speaker identification (ADR-0002) -> waycast input frame.  Each detected
utterance becomes exactly one input frame with no partial/final marking and no
turn-taking involvement (ADR-0011): the module decides only when an utterance
is complete, the core decides everything else.

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
from typing import Any

from dotenv import load_dotenv

from voice_input.asr import OpenAICompatibleAsr
from voice_input.capture import PortAudioMic
from voice_input.config import load_settings
from voice_input.hub_client import MODULE_NAME, HubConnectionError, InputClient
from voice_input.silero_vad import SileroEndpointer, SileroVadClient
from voice_input.speaker import (
    SpeakerIdentifier,
    build_speaker_identifier,
    disabled_status,
)
from voice_input.vad import EndpointDetector, Endpointer, EnergyEndpointer

LOGGER = logging.getLogger(__name__)

ENDPOINT_QUEUE_SIZE = 4
MAX_TEXT_LENGTH = 8192
HUB_RECONNECT_DELAY_SECONDS = 3.0


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
            LOGGER.warning("speaker identification failed; sending unlabeled", exc_info=True)
        else:
            if spk is not None:
                content = f"【说话人{spk}】{text}"
                LOGGER.debug("labeled utterance speaker=%s", spk)
    metadata = {"speech_duration_ms": speech_ms, "source": "asr", "stable_prefix": text}
    await ws.send_input(content, source="asr", metadata=metadata)


async def _recognize_and_send(
    ws: InputClient,
    asr: OpenAICompatibleAsr,
    speaker: SpeakerIdentifier | None,
    queue: asyncio.Queue[tuple[bytes, int]],
) -> None:
    """Consume endpoints from the queue: transcribe and forward one input frame per utterance."""
    while True:
        pcm, speech_ms = await queue.get()
        try:
            result = await asr.transcribe(pcm)
            text = result.text.strip()
            if not text:
                LOGGER.debug("empty transcript; dropping utterance")
                continue
            if len(text) > MAX_TEXT_LENGTH:
                LOGGER.warning("transcript too long (%d chars); dropping", len(text))
                continue
            await _send_one(ws, speaker, pcm, speech_ms, text=text)
            LOGGER.info("sent voice input (%d chars)", len(text))
        finally:
            queue.task_done()


def build_endpointer(settings: Any) -> Endpointer:
    """Model-backed endpointer when VOICE_IN_VAD_URL is set, else the energy VAD.

    The Silero endpointer degrades to the energy detector internally if the
    audio.cpp service keeps failing (ADR-0003).
    """
    energy = EnergyEndpointer(
        EndpointDetector(
            threshold=settings.speech_threshold,
            trailing_silence_frames=settings.trailing_silence_frames,
            max_frames=settings.max_frames,
            pre_roll_frames=settings.pre_roll_frames,
        )
    )
    if settings.vad_url is None:
        return energy
    LOGGER.info("silero VAD enabled: %s model=%s", settings.vad_url, settings.vad_model)
    return SileroEndpointer(
        SileroVadClient(settings.vad_url, settings.vad_model),
        fallback=energy,
        poll_s=settings.vad_poll_ms / 1000,
        min_speech_ms=settings.vad_min_speech_ms,
        trailing_silence_frames=settings.trailing_silence_frames,
        max_frames=settings.max_frames,
    )


async def _collect(
    mic: PortAudioMic,
    enhancer: Any | None,
    vad: Endpointer,
    queue: asyncio.Queue[tuple[bytes, int]],
) -> None:
    """Read mic blocks, run endpoint detection, and push complete utterances to the queue."""
    timestamp = 0
    await mic.open()
    try:
        while True:
            block = await mic.read_block()
            if block is None:
                break
            frames = await enhancer.push_async(block) if enhancer is not None else (block,)
            for frame in frames:
                segment = await vad.feed(frame, timestamp)
                timestamp = (timestamp + 320) % (1 << 32)
                if segment is not None:
                    try:
                        queue.put_nowait((segment.pcm16le, segment.duration_ms))
                    except asyncio.QueueFull:
                        LOGGER.error("endpoint queue full; dropping utterance")
    finally:
        await mic.aclose()


async def _run_with_hub(settings: Any) -> None:
    """One hub connection lifetime: admit, then run capture + ASR until the connection drops."""
    speaker = build_speaker_identifier()

    def status_provider() -> dict[str, object]:
        return speaker.status() if speaker is not None else disabled_status()

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
        queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=ENDPOINT_QUEUE_SIZE)
        mic = PortAudioMic(device=settings.device)
        vad = build_endpointer(settings)
        enhancer = None  # optional zipenhancer wiring lands here (module-internal)
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(_collect(mic, enhancer, vad, queue))
                tasks.create_task(_recognize_and_send(ws, asr, speaker, queue))
        finally:
            await asr.aclose()
            await vad.aclose()


async def run() -> None:
    """Entry loop: connect to the hub with reconnection until interrupted."""
    settings = load_settings()
    while True:
        try:
            await _run_with_hub(settings)
        except (HubConnectionError, ConnectionError, OSError) as exc:
            LOGGER.warning("hub connection failed: %s", exc)
        await asyncio.sleep(HUB_RECONNECT_DELAY_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="树莓娘语音输入模块：采集 + VAD + ASR -> waycast input 帧"
    )
    parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


__all__ = ["main", "run", "MODULE_NAME"]
