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

import logging
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
    ) -> None:
        self._hub_url = hub_url
        self._token = token
        self._tls_ca = tls_ca_path
        self._name = name
        self._ws: Any = None

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
        return {
            "name": self._name,
            "kind": "input",
            "input": {"description": CAPABILITY_INPUT_DESCRIPTION},
        }

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
                raise HubConnectionError(
                    f"hub did not accept capability: {dumps(ack)}"
                )
        except HubConnectionError:
            await self._ws.close()
            raise
        except Exception as exc:
            await self._ws.close()
            raise HubConnectionError("hub handshake failed") from exc
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
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


__all__ = ["InputClient", "HubConnectionError", "MODULE_NAME"]
