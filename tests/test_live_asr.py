"""Live streaming ASR client tests against a scripted local live-route server.

The fake server speaks the real audio.cpp live wire contract (chunked request
body, SSE response events, terminating chunk) so the client's HTTP framing and
SSE parsing are exercised end to end over a real socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Self

import pytest

from aria.live_asr import (
    LiveAsrError,
    LiveAsrTracker,
    LiveAsrUnsupported,
    LiveTranscriptStream,
    _ChunkedDecoder,
    live_url_for,
)

FRAME = b"\x01\x02" * 320  # one 20 ms 16 kHz mono frame


class Conn:
    """One accepted connection: parsed request side plus scripted send helpers."""

    def __init__(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.request_line = ""
        self.headers: dict[str, str] = {}
        self.body = bytearray()
        self.terminated = False
        self.eof_without_terminator = False
        self._data_event = asyncio.Event()
        self._done_reading = asyncio.Event()

    def notify(self) -> None:
        self._data_event.set()

    def finish_body(self, *, terminated: bool) -> None:
        self.terminated = terminated
        if not terminated:
            self.eof_without_terminator = True
        self._done_reading.set()

    async def wait_terminated(self) -> None:
        await self._done_reading.wait()

    async def wait_body(self, size: int) -> None:
        while True:
            if len(self.body) >= size:
                return
            self._data_event.clear()
            if len(self.body) >= size:
                return
            await self._data_event.wait()

    def send_head(self, status: int = 200) -> None:
        self.writer.write(
            f"HTTP/1.1 {status} OK\r\ncontent-type: text/event-stream\r\n"
            f"transfer-encoding: chunked\r\n\r\n".encode()
        )

    def send_event(self, payload: str) -> None:
        data = f"data: {payload}\n\n".encode()
        self.writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")

    def send_event_fragmented(self, payload: str, size: int) -> None:
        data = f"data: {payload}\n\n".encode()
        for start in range(0, len(data), size):  # tiny chunks: split across reads
            piece = data[start : start + size]
            self.writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")

    def send_done_marker(self) -> None:
        self.send_event("[DONE]")
        self.writer.write(b"0\r\n\r\n")

    def send_error_body(self, status: int, message: str) -> None:
        body = json.dumps(
            {"error": {"message": message, "type": "invalid_request_error"}}
        )
        self.writer.write(
            f"HTTP/1.1 {status} Bad Request\r\ncontent-type: application/json\r\n"
            f"content-length: {len(body)}\r\n\r\n{body}".encode()
        )


class ScriptedLiveServer:
    """Local stand-in for the audio.cpp live route driven by a per-test script."""

    def __init__(self, script) -> None:
        self._script = script
        self.conns: list[Conn] = []
        self.server: asyncio.Server | None = None

    @property
    def url(self) -> str:
        if self.server is None or not self.server.sockets:
            raise RuntimeError("server not started")
        port = self.server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1/audio/transcriptions/live"

    async def __aenter__(self) -> Self:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        conn = Conn(reader, writer)
        self.conns.append(conn)
        try:
            conn.request_line = (await reader.readline()).decode("latin-1").strip()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                conn.headers[name.strip().lower()] = value.strip()
            body_task = asyncio.create_task(self._read_body(conn))
            try:
                await self._script(self, conn)
            except (ConnectionError, OSError):
                pass  # 客户端中途断开：脚本侧的 send 失败属预期
            await body_task
        finally:
            with contextlib.suppress(Exception):  # cleanup is best effort
                writer.close()

    async def _read_body(self, conn: Conn) -> None:
        reader = conn.reader
        try:
            while True:
                size_line = await reader.readline()
                if not size_line:
                    conn.finish_body(terminated=False)
                    return
                size = int(size_line.strip(), 16)
                if size == 0:
                    await reader.readline()  # trailing CRLF
                    conn.finish_body(terminated=True)
                    return
                conn.body += await reader.readexactly(size)
                await reader.readexactly(2)
                conn.notify()
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            conn.finish_body(terminated=False)


async def _happy(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_head()
    await conn.wait_body(640)
    conn.send_event(json.dumps({"type": "transcript.text.delta", "delta": "你好"}))
    await conn.wait_terminated()
    conn.send_event(
        json.dumps({"type": "transcript.text.done", "text": "你好世界", "timing": {}})
    )
    conn.send_done_marker()


async def _echo(server: ScriptedLiveServer, conn: Conn) -> None:
    """Terminal-only responder: done text once the client terminates the body."""
    conn.send_head()
    await conn.wait_terminated()
    conn.send_event(json.dumps({"type": "transcript.text.done", "text": "你好世界"}))
    conn.send_done_marker()


async def _open(server: ScriptedLiveServer) -> LiveTranscriptStream:
    return await LiveTranscriptStream.open(
        url=server.url, model="nemotron-test", sample_rate=16000
    )


def test_open_streams_and_finishes_with_done_text() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_happy) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            text = await stream.finish(timeout=5.0)
            assert text == "你好世界"  # done.text 为准，不是 delta 拼接
            conn = server.conns[0]
            assert conn.terminated
            assert bytes(conn.body) == FRAME
            assert text[:2] == stream.text[:2]  # delta 已提前流出
            assert "/v1/audio/transcriptions/live" in conn.request_line
            assert "model=nemotron-test" in conn.request_line
            assert "sample_rate=16000" in conn.request_line
            assert "sample_format=s16le" in conn.request_line
            assert conn.headers["transfer-encoding"] == "chunked"
            assert conn.headers["accept"] == "text/event-stream"

    asyncio.run(scenario())


def test_deltas_arrive_while_audio_still_flowing() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_happy) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            for _ in range(100):
                if stream.text:
                    break
                await asyncio.sleep(0.01)
            assert stream.text == "你好"  # finish 之前增量已可见（真流式的意义）
            assert await stream.finish(timeout=5.0) == "你好世界"

    asyncio.run(scenario())


async def _fragmented(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_head()
    await conn.wait_terminated()
    conn.send_event_fragmented(
        json.dumps({"type": "transcript.text.done", "text": "拆包也收得齐"}), 7
    )
    conn.send_done_marker()


def test_sse_split_across_many_small_reads() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_fragmented) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            assert await stream.finish(timeout=5.0) == "拆包也收得齐"

    asyncio.run(scenario())


async def _error_event(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_head()
    await conn.wait_body(640)
    conn.send_event(
        json.dumps(
            {"type": "error", "error": {"message": "stalled: no audio received"}}
        )
    )


def test_error_event_fails_finish_and_closes() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_error_event) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            with pytest.raises(LiveAsrError, match="no audio received"):
                await stream.finish(timeout=5.0)
            assert stream.failed
            await server.conns[0].wait_terminated()  # 连接被彻底关闭，服务端不再等待

    asyncio.run(scenario())


async def _never_finishes(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_head()
    await conn.wait_terminated()
    await asyncio.sleep(0.5)  # 卡在 done 之前：客户端超时自行断开


def test_finish_timeout_raises_and_abandons_connection() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_never_finishes) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            with pytest.raises(LiveAsrError, match="timed out"):
                await stream.finish(timeout=0.2)

    asyncio.run(scenario())


async def _reject_model(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_error_body(
        400,
        "live transcription requires a model configured with mode=streaming: batch-asr",
    )
    await asyncio.sleep(0)


def test_non_streaming_model_rejection_is_unsupported() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_reject_model) as server:
            with pytest.raises(LiveAsrUnsupported, match="mode=streaming"):
                await _open(server)

    asyncio.run(scenario())


async def _busy_error_event(server: ScriptedLiveServer, conn: Conn) -> None:
    conn.send_head()
    conn.send_event(
        json.dumps(
            {
                "type": "error",
                "error": {"message": "model 'asr' is busy: timed out after 0 ms"},
            }
        )
    )
    await asyncio.sleep(0)


def test_busy_surfaces_as_failure_for_batch_fallback() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_busy_error_event) as server:
            stream = await _open(server)
            await stream.feed(FRAME)
            with pytest.raises(LiveAsrError, match="busy"):
                await stream.finish(timeout=5.0)

    asyncio.run(scenario())


class _FakeVad:
    """Minimal Endpointer stand-in: scripted open flag plus buffer."""

    def __init__(self) -> None:
        self.open_flag = False
        self.buffer = bytearray()

    @property
    def utterance_open(self) -> bool:
        return self.open_flag

    def utterance_audio(self) -> bytes:
        return bytes(self.buffer) if self.open_flag else b""

    async def aclose(self) -> None:
        return None


def _tracker(url: str, **kwargs: object) -> LiveAsrTracker:
    return LiveAsrTracker(url=url, model="nemotron-test", sample_rate=16000, **kwargs)  # type: ignore[arg-type]


def test_tracker_opens_once_per_utterance_and_hands_off_on_endpoint() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_echo) as server:
            tracker = _tracker(server.url)
            vad = _FakeVad()
            vad.open_flag, vad.buffer = True, bytearray(b"AA")
            await tracker.update(b"AA", vad)
            first = tracker._active
            assert first is not None
            await tracker.update(b"BB", vad)
            await tracker.update(b"CC", vad)
            assert tracker._active is first  # 一条语音区间一条会话
            stream = tracker.take()
            assert stream is first and tracker._active is None
            assert await tracker.finish(stream) == "你好世界"
            assert bytes(server.conns[0].body) == b"AABBCC"  # 回填 + 增量，无重复
            assert len(server.conns) == 1

    asyncio.run(scenario())


def test_tracker_reopens_for_next_utterance() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_echo) as server:
            tracker = _tracker(server.url)
            vad = _FakeVad()
            vad.open_flag, vad.buffer = True, bytearray(b"A")
            await tracker.update(b"A", vad)
            first = tracker.take()
            assert first is not None
            await first.abort()  # worker 侧收尾
            vad.open_flag = False  # 终点后 vad 关闭区间
            await tracker.update(b"Z", vad)  # 无活动会话，也不误开
            vad.open_flag, vad.buffer = True, bytearray(b"B")
            await tracker.update(b"B", vad)
            assert tracker._active is not None
            assert len(server.conns) == 2
            await tracker.aclose()  # 第二条会话随会话结束关闭

    asyncio.run(scenario())


def test_tracker_aborts_stream_when_vad_discards_noise() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_echo) as server:
            tracker = _tracker(server.url)
            vad = _FakeVad()
            vad.open_flag, vad.buffer = True, bytearray(b"A")
            await tracker.update(b"A", vad)
            vad.open_flag = False  # 噪音丢弃：区间关闭但没有 endpoint
            await tracker.update(b"Z", vad)
            assert tracker._active is None
            await server.conns[0].wait_terminated()
            assert server.conns[0].eof_without_terminator  # 连接被直接掐断，模型释放

    asyncio.run(scenario())


async def _immediate_error(server: ScriptedLiveServer, conn: Conn) -> None:
    """Error before any audio matters (e.g. busy timeout): tracker must fall back."""
    conn.send_head()
    conn.send_event(
        json.dumps(
            {"type": "error", "error": {"message": "stalled: no audio received"}}
        )
    )
    await asyncio.sleep(0)


def test_tracker_disables_after_consecutive_failures() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_immediate_error) as server:
            tracker = _tracker(server.url, finish_timeout=1.0, max_failures=2)
            vad = _FakeVad()
            for _ in range(2):
                vad.open_flag, vad.buffer = True, bytearray(b"A")
                await tracker.update(b"A" * 700, vad)
                stream = tracker.take()
                assert stream is not None
                assert await tracker.finish(stream) is None
            assert not tracker.enabled
            vad.open_flag, vad.buffer = True, bytearray(b"A")
            await tracker.update(b"A", vad)
            assert tracker._active is None  # 关闭后不再尝试

    asyncio.run(scenario())


def test_tracker_busy_does_not_count_as_failure() -> None:
    async def scenario() -> None:
        async with ScriptedLiveServer(_busy_error_event) as server:
            tracker = _tracker(server.url, finish_timeout=1.0, max_failures=1)
            vad = _FakeVad()
            vad.open_flag, vad.buffer = True, bytearray(b"A")
            await tracker.update(b"A" * 700, vad)
            stream = tracker.take()
            assert stream is not None
            assert await tracker.finish(stream) is None
            assert tracker.enabled  # 占用是瞬时状态，下次语音照常走流式

    asyncio.run(scenario())


def test_tracker_disabled_by_config_stays_silent() -> None:
    async def scenario() -> None:
        vad = _FakeVad()
        vad.open_flag, vad.buffer = True, bytearray(b"A")
        tracker = _tracker("http://127.0.0.1:1/live", enabled=False)
        await tracker.update(b"A", vad)
        assert tracker.take() is None

    asyncio.run(scenario())


def test_tracker_opens_at_most_once_per_utterance() -> None:
    """开流失败（连接被拒）时不得逐帧重试：一段语音只试一次，失败计一次。"""

    async def scenario() -> None:
        tracker = _tracker("http://127.0.0.1:1/live")
        vad = _FakeVad()
        vad.open_flag, vad.buffer = True, bytearray(b"A")
        for _ in range(5):
            await tracker.update(b"A", vad)
        assert tracker._active is None
        assert tracker._failures == 1  # 五帧只产生一次开流尝试
        assert tracker.enabled

    asyncio.run(scenario())


def test_chunked_decoder_handles_split_framing() -> None:
    decoder = _ChunkedDecoder()
    assert decoder.feed(b"5\r\nhe") == b""
    assert decoder.feed(b"llo\r\n6\r\nworld") == b"hello"
    assert decoder.feed(b"!\r\n0\r\n") == b"world!"
    assert decoder.feed(b"\r\n") == b""
    assert decoder.terminated


def test_live_url_derivation() -> None:
    assert live_url_for("http://127.0.0.1:8080/v1") == (
        "http://127.0.0.1:8080/v1/audio/transcriptions/live"
    )
    assert live_url_for("http://host:8080/v1/") == (
        "http://host:8080/v1/audio/transcriptions/live"
    )
    assert live_url_for("https://api.example.com/v1/audio/transcriptions") == (
        "https://api.example.com/v1/audio/transcriptions/live"
    )
