"""真流式 ASR 路径：语音在途时边喂边出增量文本，出结果早于话说完。

协议是 audio.cpp 的 live 路由（api.md 的 live 一节与 runtime.cpp 的
handle_transcription_live 已核对）：请求体为裸 PCM 分块上传
（Transfer-Encoding: chunked），参数走查询串；响应为 SSE 增量事件
（transcript.text.delta / transcript.text.done / error）。发送终止块（0 长度）
表示说完，服务端冲完解码再发 done 与 [DONE]；实测 done 在终止块后约 40ms 到达，
而增量文本在语音进行中持续流出（批量路由只能等整段 WAV 上传完才开始解码）。
请求存续期间服务端一直持有模型占用、断开即释放，因此回退批量转写之前必须先
关闭本会话连接，否则批量请求会撞上「model is busy」。

httpx 无法在同一条 HTTP/1.1 连接上边发边收（发完请求体才开始读响应），所以这里是
裸 asyncio 的 HTTP 分块实现：写侧按 chunked 框定推音频，读侧增量解 chunk 并解析
SSE。写侧在独立任务里进行，喂音频（feed）只入缓冲，绝不因网络背压卡住音频泵。

换代验证（2026-11，httpx2 2.16.0）：httpx2 虽原生支持 SSE（client.sse()）与
异步生成器请求体，但其请求路径同样是「请求体全部传完才开始读响应」——环回实测
（服务端每收一个请求块即时回一个 SSE 事件）中 client.sse() 与
send(request, stream=True) 的响应均挤在上传结束后一次性到达，换过来会让增量文本
退化成立等完整上传，功能倒退。故本实现保留手写，勿再尝试用 httpx/httpx2 承接
本路由的全双工流式上传。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast
from urllib.parse import urlsplit

from aria.vad import Endpointer

LOGGER = logging.getLogger(__name__)

__all__ = [
    "LiveAsrError",
    "LiveAsrTracker",
    "LiveAsrUnsupported",
    "LiveTranscriptStream",
    "live_url_for",
]

# (host, port) -> (StreamReader, StreamWriter)。默认 asyncio.open_connection；
# 测试注入假 socket 对，免去真网络 I/O（与 asr 的注入 transport 同一手法）。
Connector = Callable[[str, int], Awaitable[tuple[Any, Any]]]

_CRLF = b"\r\n"


class LiveAsrError(RuntimeError):
    """一次流式会话失败（连接中断、服务端 error 事件、超时等）。"""


class LiveAsrUnsupported(LiveAsrError):
    """路由或模型不支持流式（4xx）：本路径永久关闭，全部走批量。"""


def live_url_for(endpoint: str) -> str:
    """从批量 ASR 端点（如 http://host:8080/v1）推出 live 路由 URL。"""
    base = endpoint.rstrip("/")
    if not base.endswith("/audio/transcriptions"):
        base += "/audio/transcriptions"
    return base + "/live"


class _SseLineParser:
    """按行解析 SSE：只认 data: 行，事件体是单行 JSON（audio.cpp 用 \\n 分行）。"""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, data: bytes) -> list[str]:
        self._buffer += data
        payloads: list[str] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("data:"):
                payloads.append(text[len("data:") :].strip())
        return payloads


class _ChunkedDecoder:
    """增量解 Transfer-Encoding: chunked，返回负载字节（不完整部分留待下次）。"""

    def __init__(self) -> None:
        self._buffer = b""
        self.terminated = False

    def feed(self, data: bytes) -> bytes:
        self._buffer += data
        out = bytearray()
        while not self.terminated:
            index = self._buffer.find(_CRLF)
            if index < 0:
                break
            try:
                size = int(self._buffer[:index].split(b";")[0], 16)
            except ValueError as exc:
                raise LiveAsrError(
                    f"malformed chunked framing: {self._buffer[:index]!r}"
                ) from exc
            if len(self._buffer) < index + 2 + size + 2:
                break  # 负载还没收满
            out += self._buffer[index + 2 : index + 2 + size]
            self._buffer = self._buffer[index + 2 + size + 2 :]
            if size == 0:
                self.terminated = True
        return bytes(out)


class LiveTranscriptStream:
    """一条话说完前不关闭的双工转写会话：请求侧喂 PCM，响应侧出增量文本。

    生命周期：open() -> feed()* -> finish()（发终止块并等最终文本）或 abort()。
    feed 只入缓冲、从不抛出（失败记在 _failure，finish 时统一暴露），音频泵不受
    网络状况影响。finish/abort 之后连接必定关闭，服务端随即释放模型占用。
    """

    def __init__(self, reader: Any, writer: Any, *, label: str) -> None:
        self._reader = reader
        self._writer = writer
        self._label = label
        self._pending = bytearray()
        self._wake = asyncio.Event()
        self._done = asyncio.Event()
        self._eof_notified = False
        self._deltas: list[str] = []
        self._final: str | None = None
        self._failure: str | None = None
        self._closed = False
        self._tasks = (
            asyncio.create_task(self._write_loop(), name=f"{label}-live-write"),
            asyncio.create_task(self._read_loop(), name=f"{label}-live-read"),
        )

    # ---- 对外接口 ---------------------------------------------------------

    @classmethod
    async def open(
        cls,
        *,
        url: str,
        model: str,
        sample_rate: int,
        channels: int = 1,
        connector: Connector | None = None,
    ) -> LiveTranscriptStream:
        """建立会话并等待 2xx；模型加载完成的信号就是响应头到达。"""
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise LiveAsrUnsupported(f"live url must be an http(s) URL: {url!r}")
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        query = f"model={model}&sample_rate={sample_rate}&channels={channels}&sample_format=s16le"
        path = f"{parts.path}?{query}"
        connect: Connector = (
            connector if connector is not None else asyncio.open_connection
        )
        reader, writer = await connect(host, port)
        try:
            request_head = (
                f"POST {path} HTTP/1.1\r\nHost: {parts.netloc}\r\n"
                f"Transfer-Encoding: chunked\r\nAccept: text/event-stream\r\n\r\n"
            ).encode()
            writer.write(request_head)
            await writer.drain()
            status_line = await asyncio.wait_for(reader.readline(), timeout=120.0)
            fields = status_line.decode("latin-1").split()
            status = int(fields[1]) if len(fields) >= 2 else 0
            while True:  # 头部余下行读到空行为止
                header = await reader.readline()
                if header in (b"\r\n", b"\n", b""):
                    break
            if status != 200:
                body = await reader.read(2048)
                message = body.decode("utf-8", errors="replace")[:300]
                try:
                    parsed = json.loads(message)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    parsed_fields = cast("dict[str, Any]", parsed)
                    error_field = parsed_fields.get("error", parsed_fields)
                    if isinstance(error_field, dict):
                        inner = cast("dict[str, Any]", error_field).get("message")
                        message = (
                            str(inner)
                            if inner is not None
                            else json.dumps(error_field, ensure_ascii=False)
                        )
                    elif error_field is not None:
                        message = str(error_field)
                writer.close()
                if status in (400, 404) and "streaming" in message:
                    raise LiveAsrUnsupported(f"live route rejects model: {message}")
                if "busy" in message:
                    raise LiveAsrError(
                        f"asr model busy: {message}"
                    )  # 并发占用：瞬时，不计失败
                raise LiveAsrError(f"live open failed: HTTP {status}: {message}")
        except BaseException:
            with contextlib.suppress(Exception):  # 清理尽力而为
                writer.close()
            raise
        return cls(reader, writer, label=f"asr:{model}")

    @property
    def text(self) -> str:
        """截至当前的增量拼接文本（最终文本以 finish() 返回为准）。"""
        return "".join(self._deltas)

    @property
    def failed(self) -> bool:
        return self._failure is not None

    async def feed(self, pcm: bytes) -> None:
        """喂入音频字节。永不抛出：失败在 finish()/aborted 时统一暴露。"""
        if self._closed or self._failure is not None or not pcm:
            return
        self._pending += pcm
        self._wake.set()

    async def finish(self, timeout: float) -> str:
        """发终止块并等待最终文本；失败抛 LiveAsrError。无论成败连接都关闭。"""
        try:
            self._eof_notified = True
            self._wake.set()
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
            if self._failure is not None:
                raise LiveAsrError(self._failure)
            if self._final is None:
                raise LiveAsrError("live stream ended without a done event")
            return self._final
        except TimeoutError as exc:
            raise LiveAsrError(f"live finish timed out after {timeout}s") from exc
        finally:
            await self._shutdown()

    async def abort(self) -> None:
        """放弃本会话（语音被 VAD 丢弃、排队失败等），关闭连接。"""
        await self._shutdown()

    # ---- 内部：写侧 / 读侧 -------------------------------------------------

    async def _write_loop(self) -> None:
        try:
            while True:
                if not self._pending and not self._eof_notified:
                    self._wake.clear()
                    await self._wake.wait()
                chunk, self._pending = bytes(self._pending), bytearray()
                if chunk:
                    self._writer.write(
                        f"{len(chunk):x}\r\n".encode("ascii") + chunk + _CRLF
                    )
                    await self._writer.drain()
                if self._eof_notified and not self._pending:
                    self._writer.write(b"0\r\n\r\n")
                    await self._writer.drain()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 写失败记录，读侧/finish 统一处理
            self._remember_failure(f"live upload failed: {exc}")

    async def _read_loop(self) -> None:
        decoder = _ChunkedDecoder()
        sse = _SseLineParser()
        try:
            while not decoder.terminated:
                data = await self._reader.read(65536)
                if not data:
                    break
                payload = decoder.feed(data)
                if not payload:
                    continue
                for body in sse.feed(payload):
                    self._handle_event(body)
            if self._failure is None and self._final is None:
                self._remember_failure("live connection closed before done")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._remember_failure(f"live stream failed: {exc}")

    def _handle_event(self, body: str) -> None:
        if body == "[DONE]":
            self._done.set()
            return
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return
        if not isinstance(parsed, dict):
            return
        event = cast("dict[str, Any]", parsed)
        kind = event.get("type")
        if kind == "transcript.text.delta":
            self._deltas.append(str(event.get("delta", "")))
        elif kind == "transcript.text.done":
            self._final = str(event.get("text", ""))
            self._done.set()
        elif kind == "error":
            error_field = event.get("error")
            if isinstance(error_field, dict):
                message = cast("dict[str, Any]", error_field).get("message")
                detail = (
                    message
                    if isinstance(message, str)
                    else json.dumps(error_field, ensure_ascii=False)
                )
            else:
                detail = str(error_field if error_field is not None else body[:200])
            self._remember_failure(f"live server error: {detail}")

    def _remember_failure(self, message: str) -> None:
        if self._failure is None:
            self._failure = message
        self._done.set()  # 唤醒等待中的 finish

    async def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        with contextlib.suppress(Exception):  # 关闭尽力而为
            self._writer.close()


class LiveAsrTracker:
    """按 VAD 语音区间开合流式会话的协调器（跑在音频泵里，出结果在 worker 里）。

    - 语音开始：开一条会话并回填 VAD 缓冲的语音起点音频（与批量 PCM 完全同源，
      含 pre-roll），此后每帧增量喂入；
    - 终点产生：take() 摘下会话交给 worker，worker finish() 拿最终文本；
      失败返回 None，调用方回退批量转写（回退前连接已关闭，不会撞模型占用）；
    - 连续 max_failures 次失败或收到不支持信号：本路径永久关闭；
      「模型被占用」属并发瞬时状态，只降级当次，不计失败。
    """

    def __init__(
        self,
        *,
        url: str,
        model: str,
        sample_rate: int,
        enabled: bool = True,
        finish_timeout: float = 10.0,
        max_failures: int = 3,
        connector: Connector | None = None,
    ) -> None:
        self._url = url
        self._model = model
        self._sample_rate = sample_rate
        self._finish_timeout = finish_timeout
        self._max_failures = max_failures
        self._connector = connector
        self._disabled_reason: str | None = None if enabled else "disabled by config"
        self._failures = 0
        self._active: LiveTranscriptStream | None = None
        self._open_attempted = False

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def active_text(self) -> str:
        return self._active.text if self._active is not None else ""

    async def update(
        self, frame: bytes, vad: Endpointer, *, endpoint_fired: bool = False
    ) -> None:
        """音频泵每帧调用：维护当前语音区间的流式会话。绝不抛出。"""
        if not self.enabled:
            return
        if vad.utterance_open:
            if self._active is None:
                if not self._open_attempted:
                    self._open_attempted = (
                        True  # 一条语音区间只开一次，失败即整段走批量
                    )
                    self._active = await self._open_stream(vad.utterance_audio())
            else:
                await self._active.feed(frame)
            return
        # 语音区间已闭合（endpoint 帧除外，所有权由 take() 移交 worker）。
        if not endpoint_fired:
            if self._active is not None:
                # VAD 把这段语音当噪音丢弃（未入队）：会话没有消费者，立刻关闭。
                await self._active.abort()
                self._active = None
            self._open_attempted = False

    def take(self) -> LiveTranscriptStream | None:
        """终点产生时摘下当前会话（所有权移交给 worker），并允许下一段重新开会话。"""
        self._open_attempted = False
        stream, self._active = self._active, None
        return stream

    async def discard(self, stream: LiveTranscriptStream | None) -> None:
        """排队失败等场景：丢弃已摘下的会话，不计失败。"""
        if stream is not None:
            await stream.abort()

    async def finish(self, stream: LiveTranscriptStream) -> str | None:
        """等最终文本；失败返回 None（调用方回退批量），并做失败计数/关闭判断。

        返回前连接已关闭，服务端已释放模型占用，回退的批量请求不会被 busy 挡住；
        留一小段缓冲让服务端完成会话清理。
        """
        try:
            text = await stream.finish(timeout=self._finish_timeout)
        except LiveAsrUnsupported as exc:
            self._disable(f"server says live is unsupported: {exc}")
            return None
        except LiveAsrError as exc:
            if "busy" in str(exc):
                await asyncio.sleep(0.15)  # 与占用方错峰，回退批量照走
                return None
            self._failures += 1
            if self._failures >= self._max_failures:
                self._disable(
                    f"{self._failures} consecutive live failures, last: {exc}"
                )
            else:
                LOGGER.debug("live asr session failed (fallback batch): %s", exc)
            await asyncio.sleep(0.15)
            return None
        except OSError as exc:
            self._failures += 1
            if self._failures >= self._max_failures:
                self._disable(
                    f"{self._failures} consecutive live failures, last: {exc}"
                )
            LOGGER.debug("live asr session errored (fallback batch): %s", exc)
            await asyncio.sleep(0.15)
            return None
        self._failures = 0
        return text

    async def aclose(self) -> None:
        if self._active is not None:
            await self._active.abort()
            self._active = None

    # ---- 内部 ---------------------------------------------------------------

    async def _open_stream(self, initial_audio: bytes) -> LiveTranscriptStream | None:
        try:
            stream = await LiveTranscriptStream.open(
                url=self._url,
                model=self._model,
                sample_rate=self._sample_rate,
                connector=self._connector,
            )
        except LiveAsrUnsupported as exc:
            self._disable(f"server says live is unsupported: {exc}")
            return None
        except LiveAsrError as exc:
            if "busy" not in str(exc):
                self._failures += 1
                if self._failures >= self._max_failures:
                    self._disable(
                        f"{self._failures} consecutive live failures, last: {exc}"
                    )
            LOGGER.debug(
                "live asr open failed (utterance falls back to batch): %s", exc
            )
            return None
        except OSError as exc:
            self._failures += 1
            if self._failures >= self._max_failures:
                self._disable(
                    f"{self._failures} consecutive live connect failures, last: {exc}"
                )
            LOGGER.debug(
                "live asr connect failed (utterance falls back to batch): %s", exc
            )
            return None
        await stream.feed(initial_audio)
        return stream

    def _disable(self, reason: str) -> None:
        LOGGER.warning("live asr path disabled, batch only: %s", reason)
        self._disabled_reason = reason
