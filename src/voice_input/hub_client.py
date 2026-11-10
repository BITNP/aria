"""Waycast hub connection for an input module.

A module connects to the hub /ws/modules endpoint, advertises its capability as
the first frame, and after the hub acknowledges it may send input frames over
the same full-duplex connection.  The wire contract (ADR-0008) is implemented
locally in voice_input.wire as plain JSON; the module and the core are separate
microservices that talk only over the network and share no Python package.

This module is input-only: it never receives output_unit frames and holds no core
logic; turn-taking decisions belong to the core Agent (ADR-0011).
"""
from __future__ import annotations

import asyncio
import logging
import ssl
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx2
from httpx2.websockets import (
    WebSocketDisconnect,
    WebSocketNetworkError,
    WebSocketUpgradeError,
)

from voice_input.wire import dumps, make_frame, parse_frame

LOGGER = logging.getLogger(__name__)

MODULE_NAME = "voice_input"
CAPABILITY_INPUT_DESCRIPTION = (
    "unified voice transcriptions from the microphone: captured via VAD endpoint "
    "detection and transcribed by an OpenAI-compatible ASR service"
)
CAPABILITY_STATUS_DESCRIPTION = (
    "dynamic speaker identification state: speaker_enabled, model_revision, "
    "known_speaker_count, onnx_provider"
)


class HubConnectionError(RuntimeError):
    """Raised when the hub rejects the module connection."""


class InputClient:
    """Async context manager providing connection admission and send_input().

    Usage:

        async with InputClient(hub_url, token) as client:
            await client.send_input("你好", source="asr", metadata={...})

    The client keeps the socket open until the context exits; a reconnect loop is the
    caller responsibility (main.run() wraps this in a retry loop).
    """

    def __init__(
        self,
        *,
        hub_url: str,
        token: str,
        tls_ca_path: str | None = None,
        name: str = MODULE_NAME,
        status_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """``status_provider`` answers status_query frames when not None.

        Passing a provider also advertises the optional status capability
        (waycast A15/A19); the provider is called on every query and its dict
        becomes the status_result payload.
        """
        self._hub_url = hub_url
        self._token = token
        self._tls_ca = tls_ca_path
        self._name = name
        self._status_provider = status_provider
        self._ws: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._reader: asyncio.Task[None] | None = None
        # reader 观察到断线时记录异常：此后 send_input 立即以同一异常失败，
        # 让重连循环在下一帧输入上马上发现断线（对齐旧 websockets 里
        # reader 主动 close 后 send 抛错的语义；ConnectionClosed 曾天然属于
        # ConnectionError/OSError，httpx2 的断线异常族不是，必须显式转达）。
        self._dropped: Exception | None = None

    @property
    def name(self) -> str:
        return self._name

    def _ws_uri(self) -> str:
        parts = urlparse(self._hub_url.rstrip("/") + "/ws/modules")
        query = urlencode({"token": self._token}) if self._token else parts.query
        return urlunparse(parts._replace(query=query))

    def _client_kwargs(self) -> dict[str, Any]:
        # 与旧 websockets.connect 的裸连接语义一致：10s 握手时限、绝不读取
        # 环境代理变量（HTTP_PROXY 等不得劫持到 hub 的直连）。
        kwargs: dict[str, Any] = {
            "timeout": httpx2.Timeout(10.0),
            "trust_env": False,
        }
        if self._tls_ca:
            kwargs["verify"] = ssl.create_default_context(cafile=self._tls_ca)
        return kwargs

    async def _capability(self) -> dict[str, Any]:
        capability: dict[str, Any] = {
            "name": self._name,
            "kind": "input",
            "input": {"description": CAPABILITY_INPUT_DESCRIPTION},
        }
        if self._status_provider is not None:
            capability["status"] = {
                "enabled": True,
                "description": CAPABILITY_STATUS_DESCRIPTION,
            }
        return capability

    async def _read_loop(self) -> None:
        """Answer status_query frames and surface remote connection closure.

        The hub otherwise sends the module nothing, but a closed socket would
        otherwise go unnoticed until the next send; closing from here lets the
        producer see the error and reconnect.
        """
        try:
            while True:
                frame = parse_frame(await self._ws.receive_text())
                if frame["type"] == "status_query":
                    assert self._status_provider is not None
                    reply = make_frame(
                        "status_result",
                        self._name,
                        self._status_provider(),
                        request_id=frame["request_id"],
                    )
                    await self._send_text(dumps(reply))
        except (WebSocketDisconnect, WebSocketNetworkError, ConnectionError, OSError) as exc:
            self._dropped = exc
            LOGGER.debug("hub connection closed (status reader)")
        except Exception as exc:
            self._dropped = WebSocketNetworkError(f"status reader error: {exc}")
            LOGGER.warning("status reader error: %s", exc)
        finally:
            if self._dropped is None:  # 非异常退出也须让 send 侧发现会话已死
                self._dropped = WebSocketNetworkError("status reader exited")
            try:  # 主动回应对端关闭（对齐旧 reader finally close：否则对端的
                await self._ws.close()  # close 握手要空等 close_timeout）
            except Exception:  # noqa: BLE001 - 清理尽力而为
                pass

    async def _send_text(self, payload: str) -> None:
        """One text frame; low-level protocol errors mean dropped, unify them."""
        try:
            await self._ws.send_text(payload)
        except (WebSocketDisconnect, WebSocketNetworkError, ConnectionError, OSError):
            raise
        except Exception as exc:  # noqa: BLE001 - wsproto 底层协议错误（如对端已关仍发送）收敛为 httpx2 断线类型
            raise WebSocketNetworkError("hub connection dropped mid-send") from exc

    async def __aenter__(self) -> "InputClient":
        uri = self._ws_uri()
        client = httpx2.AsyncClient(**self._client_kwargs())
        self._exit_stack = AsyncExitStack()
        self._exit_stack.push_async_callback(client.aclose)
        try:
            self._ws = await self._exit_stack.enter_async_context(client.websocket(uri))
        except (WebSocketUpgradeError, WebSocketNetworkError, httpx2.HTTPError, OSError) as exc:
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise HubConnectionError(f"cannot reach hub at {uri}: {exc}") from exc
        try:
            await self._send_text(
                dumps(make_frame("capability", self._name, await self._capability()))
            )
            ack = parse_frame(await self._ws.receive_text())
            if ack["type"] != "capability" or not ack["payload"].get("accepted"):
                raise HubConnectionError(
                    f"hub did not accept capability: {dumps(ack)}"
                )
        except HubConnectionError:
            await self._shutdown_ws()
            raise
        except Exception as exc:
            await self._shutdown_ws()
            raise HubConnectionError("hub handshake failed") from exc
        if self._status_provider is not None:
            self._reader = asyncio.create_task(self._read_loop())
        LOGGER.info("%s admitted to waycast hub (kind=input)", self._name)
        return self

    async def _shutdown_ws(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._ws = None

    async def send_input(
        self,
        content: str,
        *,
        source: str = "voice_input",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send one unified input frame; content is the ASR transcript, metadata carries module facts."""
        if self._ws is None:
            raise RuntimeError("send_input before __aenter__")
        if self._dropped is not None:
            raise self._dropped
        payload = {
            "content": content,
            "source": source,
            "metadata": metadata or {},
        }
        await self._send_text(dumps(make_frame("input", self._name, payload)))

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        await self._shutdown_ws()


__all__ = ["InputClient", "HubConnectionError", "MODULE_NAME"]
