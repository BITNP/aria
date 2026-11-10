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
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import websockets

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
        self._reader: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self._name

    def _ws_uri(self) -> str:
        parts = urlparse(self._hub_url.rstrip("/") + "/ws/modules")
        query = urlencode({"token": self._token}) if self._token else parts.query
        return urlunparse(parts._replace(query=query))

    def _connect_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"open_timeout": 10.0}
        if self._tls_ca:
            import ssl

            kwargs["ssl"] = ssl.create_default_context(cafile=self._tls_ca)
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
                frame = parse_frame(await self._ws.recv())
                if frame["type"] == "status_query":
                    assert self._status_provider is not None
                    reply = make_frame(
                        "status_result",
                        self._name,
                        self._status_provider(),
                        request_id=frame["request_id"],
                    )
                    await self._ws.send(dumps(reply))
        except (websockets.ConnectionClosed, ConnectionError, OSError):
            LOGGER.debug("hub connection closed (status reader)")
        except Exception as exc:
            LOGGER.warning("status reader error: %s", exc)
        finally:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def __aenter__(self) -> "InputClient":
        uri = self._ws_uri()
        try:
            self._ws = await websockets.connect(uri, **self._connect_kwargs())
        except OSError as exc:
            raise HubConnectionError(f"cannot reach hub at {uri}: {exc}") from exc
        try:
            await self._ws.send(
                dumps(make_frame("capability", self._name, await self._capability()))
            )
            ack = parse_frame(await self._ws.recv())
            if ack["type"] != "capability" or not ack["payload"].get("accepted"):
                raise HubConnectionError(f"hub did not accept capability: {dumps(ack)}")
        except HubConnectionError:
            await self._ws.close()
            raise
        except Exception as exc:
            await self._ws.close()
            raise HubConnectionError("hub handshake failed") from exc
        if self._status_provider is not None:
            self._reader = asyncio.create_task(self._read_loop())
        LOGGER.info("%s admitted to waycast hub (kind=input)", self._name)
        return self

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
        payload = {
            "content": content,
            "source": source,
            "metadata": metadata or {},
        }
        await self._ws.send(dumps(make_frame("input", self._name, payload)))

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


__all__ = ["InputClient", "HubConnectionError", "MODULE_NAME"]
