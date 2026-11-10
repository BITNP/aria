"""Pipeline wiring tests: speaker prefixing, degraded no-ID mode, fault tolerance."""

from __future__ import annotations

import asyncio
import wave

from voice_input.main import (
    MAX_TEXT_LENGTH,
    _recognize_and_send,
    _send_one,
    _write_endpoint_wav,
)


class _RecorderWs:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []

    async def send_input(
        self, content: str, *, source: str, metadata: dict[str, object]
    ) -> None:
        self.sent.append((content, source, metadata))


class _LabelingSpeaker:
    """Test-only fake embedding speaker that always labels as spk_1."""

    def identify(self, pcm: bytes, speech_ms: int) -> str | None:
        self.last_pcm = pcm
        self.last_speech_ms = speech_ms
        return "spk_1"


class _BoomSpeaker:
    def identify(self, pcm: bytes, speech_ms: int) -> str:
        raise RuntimeError("embedder fault")


def test_labeled_utterance_prefixes_content_but_not_stable_prefix():
    async def scenario():
        ws = _RecorderWs()
        speaker = _LabelingSpeaker()
        await _send_one(ws, speaker, b"\x00" * 640, 600, text="你好")
        return ws

    ws = asyncio.run(scenario())
    assert len(ws.sent) == 1
    content, _source, metadata = ws.sent[0]
    assert content == "【说话人spk_1】你好"
    assert metadata["stable_prefix"] == "你好"
    assert metadata["speech_duration_ms"] == 600
    assert metadata["source"] == "asr"


def test_unlabeled_utterance_is_forwarded_plain():
    async def scenario():
        ws = _RecorderWs()
        await _send_one(ws, None, b"\x00" * 640, 400, text="太短不标")
        return ws

    ws = asyncio.run(scenario())
    assert ws.sent[0][0] == "太短不标"
    assert ws.sent[0][2]["stable_prefix"] == "太短不标"


def test_embedder_fault_does_not_stop_pipeline():
    async def scenario():
        ws = _RecorderWs()
        await _send_one(ws, _BoomSpeaker(), b"\x00" * 640, 600, text="容错文本")
        return ws

    ws = asyncio.run(scenario())
    assert ws.sent[0][0] == "容错文本"  # dropped label, kept delivery


class _FakeAsr:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def transcribe(self, pcm: bytes) -> object:
        from voice_input.asr import Recognition

        self.calls += 1
        return Recognition(self._text)


class _StubStream:
    """Test-only stand-in for a handed-off LiveTranscriptStream."""

    def __init__(self, final: str | None) -> None:
        self._final = final
        self.aborted = False

    async def finish(self, timeout: float) -> str:
        from voice_input.live_asr import LiveAsrError

        if self._final is None:
            raise LiveAsrError("stub failure")
        return self._final

    async def abort(self) -> None:
        self.aborted = True


class _StubTracker:
    def __init__(self, final: str | None) -> None:
        self.stream = _StubStream(final)
        self.discarded: list[_StubStream] = []

    async def finish(self, stream: _StubStream) -> str | None:
        try:
            return await stream.finish(timeout=1.0)
        except Exception:  # noqa: BLE001 - stub mirrors LiveAsrTracker.finish swallowing
            return None

    async def discard(self, stream: _StubStream | None) -> None:
        if stream is not None:
            self.discarded.append(stream)


async def _drain(
    worker: asyncio.Task[None], queue: asyncio.Queue[object], ws: _RecorderWs
) -> None:
    for _ in range(100):
        if queue.empty():
            break
        await asyncio.sleep(0.01)
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


def test_overlong_transcript_is_dropped_in_consumer():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        queue.put_nowait((b"\x00" * 640, 600, None))
        ws = _RecorderWs()
        asr = _FakeAsr("x" * (MAX_TEXT_LENGTH + 1))
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, None))
        await _drain(worker, queue, ws)
        assert ws.sent == []

    asyncio.run(scenario())


def test_empty_transcript_is_dropped():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        queue.put_nowait((b"\x00" * 640, 600, None))
        ws = _RecorderWs()
        asr = _FakeAsr("   ")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, None))
        await _drain(worker, queue, ws)
        assert ws.sent == []

    asyncio.run(scenario())


def test_live_session_result_bypasses_batch_asr():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        tracker = _StubTracker("你好")
        queue.put_nowait((b"\x00" * 640, 600, tracker.stream))
        ws = _RecorderWs()
        asr = _FakeAsr("不该被调用")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, tracker))
        await _drain(worker, queue, ws)
        return ws, asr

    ws, asr = asyncio.run(scenario())
    assert asr.calls == 0  # live 成功：批量 ASR 完全不参与
    assert ws.sent[0][0] == "你好"


def test_failed_live_session_falls_back_to_batch_asr():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        tracker = _StubTracker(None)
        queue.put_nowait((b"\x00" * 640, 600, tracker.stream))
        ws = _RecorderWs()
        asr = _FakeAsr("批量兜底")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, tracker))
        await _drain(worker, queue, ws)
        return ws, asr

    ws, asr = asyncio.run(scenario())
    assert asr.calls == 1
    assert ws.sent[0][0] == "批量兜底"


def test_empty_live_final_drops_without_batch_fallback():
    """live 协议成功但文本为空 = 模型对过短/不可辨音频的正常输出。

    同一模型对同一段音频的批量结果必然也是空：回退批量只会与下一句的
    live 会话抢单实例模型锁（audio.cpp 串行独占），挂到超时并拖垮端点队列。
    """

    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        tracker = _StubTracker("   ")  # 协议成功、文本为空
        queue.put_nowait((b"\x00" * 640, 600, tracker.stream))
        ws = _RecorderWs()
        asr = _FakeAsr("不该被调用")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, tracker))
        await _drain(worker, queue, ws)
        return ws, asr

    ws, asr = asyncio.run(scenario())
    assert asr.calls == 0  # 空文本不回退批量
    assert ws.sent == []  # 整句按空丢弃


def test_stream_without_tracker_is_aborted_and_batch_used():
    async def scenario():
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        stream = _StubStream("不该被用")
        queue.put_nowait((b"\x00" * 640, 600, stream))
        ws = _RecorderWs()
        asr = _FakeAsr("批量")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue, None))
        await _drain(worker, queue, ws)
        return ws, stream

    ws, stream = asyncio.run(scenario())
    assert stream.aborted
    assert ws.sent[0][0] == "批量"


def test_run_loop_reconnects_on_taskgroup_connection_failure(monkeypatch):
    """Hub drops surface as ExceptionGroup; the reconnect loop must survive them."""
    from voice_input import main as main_module

    attempts: list[int] = []

    async def flaky(settings: object) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise ExceptionGroup("dropped", [ConnectionError("1012 service restart")])
        raise ValueError("non-reconnectable failures still propagate")

    async def scenario():
        from voice_input.main import run

        await run()

    monkeypatch.setattr(main_module, "load_settings", lambda *a: object())
    monkeypatch.setattr(main_module, "_run_with_hub", flaky)
    monkeypatch.setattr(main_module, "HUB_RECONNECT_DELAY_SECONDS", 0.0)
    try:
        asyncio.run(scenario())
    except ValueError:
        pass
    else:
        raise AssertionError("the non-reconnectable failure must propagate")
    assert len(attempts) == 2  # reconnected after the wrapped connection error


def test_endpoint_dump_wav_carries_exact_bytes(tmp_path):
    """VOICE_IN_DUMP_WAV files are playable WAVs of the exact utterance bytes."""
    pcm = bytes(range(256)) * 4  # 1024 bytes = 32 ms
    _write_endpoint_wav(str(tmp_path), 7, 600, pcm)
    files = list(tmp_path.glob("ep0007-*-600ms.wav"))
    assert len(files) == 1
    with wave.open(str(files[0])) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(wav_file.getnframes()) == pcm
