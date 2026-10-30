"""Hub client frame contract: capability handshake and input frames match waycast wire (ADR-0008)."""
from __future__ import annotations

import asyncio

import pytest
import websockets

from voice_input.hub_client import HubConnectionError, InputClient
from voice_input.wire import dumps, make_frame, parse_frame


def _echo_server(scope):
    """A minimal hub that accepts a capability and records what it receives."""

    async def handler(ws):
        first = await ws.recv()
        frame = parse_frame(first)
        assert frame["type"] == "capability" and frame["payload"]["kind"] == "input"
        await ws.send(dumps(make_frame("capability", "voice_input", {"accepted": True})))
        while True:
            raw = await ws.recv()
            scope.append(parse_frame(raw))

    return websockets.serve(handler, "127.0.0.1", 0)


def test_handshake_and_send_input():
    scope = []

    async def scenario():
        server = await _echo_server(scope)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with InputClient(hub_url=url, token="t") as client:
            await client.send_input("你好", source="asr", metadata={"speech_duration_ms": 500})
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert len(scope) == 1
    inp = scope[0]
    assert inp["type"] == "input"
    assert inp["module"] == "voice_input"
    assert inp["payload"]["content"] == "你好"
    assert inp["payload"]["source"] == "asr"
    assert inp["payload"]["metadata"]["speech_duration_ms"] == 500


def test_hub_rejects_capability():
    async def handler(ws):
        await ws.recv()
        await ws.send(dumps(make_frame("capability", "voice_input", {"accepted": False})))

    async def scenario():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            with pytest.raises(HubConnectionError):
                async with InputClient(hub_url=url, token="t"):
                    pass
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())

