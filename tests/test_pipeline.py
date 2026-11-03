"""Pipeline wiring tests: speaker prefixing, degraded no-ID mode, fault tolerance."""
from __future__ import annotations

import asyncio

from voice_input.main import MAX_TEXT_LENGTH, _recognize_and_send, _send_one


class _RecorderWs:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict]] = []

    async def send_input(self, content: str, *, source: str, metadata: dict[str, object]) -> None:
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
    content, source, metadata = ws.sent[0]
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

    async def transcribe(self, pcm: bytes) -> object:
        from voice_input.asr import Recognition

        return Recognition(self._text)


def test_overlong_transcript_is_dropped_in_consumer():
    async def scenario():
        queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=1)
        queue.put_nowait((b"\x00" * 640, 600))
        ws = _RecorderWs()
        asr = _FakeAsr("x" * (MAX_TEXT_LENGTH + 1))
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue))
        for _ in range(100):
            if queue.empty() or ws.sent:
                break
            await asyncio.sleep(0.01)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        assert ws.sent == []

    asyncio.run(scenario())


def test_empty_transcript_is_dropped():
    async def scenario():
        queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=1)
        queue.put_nowait((b"\x00" * 640, 600))
        ws = _RecorderWs()
        asr = _FakeAsr("   ")
        worker = asyncio.create_task(_recognize_and_send(ws, asr, None, queue))
        for _ in range(100):
            if queue.empty() or ws.sent:
                break
            await asyncio.sleep(0.01)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        assert ws.sent == []


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
