"""Hub client frame contract: capability handshake and input frames match waycast wire (ADR-0008)."""

from __future__ import annotations

import asyncio

import pytest
import websockets

from aria.hub_client import HubConnectionError, InputClient
from aria.wire import dumps, make_frame, parse_frame


def _echo_server(scope):
    """A minimal hub that accepts a capability and records what it receives."""

    async def handler(ws):
        first = await ws.recv()
        frame = parse_frame(first)
        assert frame["type"] == "capability" and frame["payload"]["kind"] == "input"
        await ws.send(dumps(make_frame("capability", "aria", {"accepted": True})))
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
            await client.send_input(
                "你好", source="asr", metadata={"speech_duration_ms": 500}
            )
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert len(scope) == 1
    inp = scope[0]
    assert inp["type"] == "input"
    assert inp["module"] == "aria"
    assert inp["payload"]["content"] == "你好"
    assert inp["payload"]["source"] == "asr"
    assert inp["payload"]["metadata"]["speech_duration_ms"] == 500


def test_hub_rejects_capability():
    async def handler(ws):
        await ws.recv()
        await ws.send(dumps(make_frame("capability", "aria", {"accepted": False})))

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


def test_status_capability_and_query_roundtrip():
    """A status_query frame is answered with a status_result carrying the provider payload."""
    received = []

    async def handler(ws):
        first = await ws.recv()
        frame = parse_frame(first)
        assert frame["payload"]["status"]["enabled"] is True
        await ws.send(dumps(make_frame("capability", "aria", {"accepted": True})))
        await ws.send(dumps(make_frame("status_query", "aria", {}, request_id="q-1")))
        reply = parse_frame(await ws.recv())
        received.append(reply)
        await ws.close()

    async def scenario():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"

        def provider() -> dict[str, object]:
            return {"speaker_enabled": True, "known_speaker_count": 2}

        async with InputClient(hub_url=url, token="t", status_provider=provider):
            await asyncio.sleep(0.2)  # let the reader task answer before teardown
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert len(received) == 1
    reply = received[0]
    assert reply["type"] == "status_result"
    assert reply["request_id"] == "q-1"
    assert reply["payload"]["speaker_enabled"] is True
    assert reply["payload"]["known_speaker_count"] == 2


def test_no_status_capability_without_provider():
    """Without a status_provider the capability carries no status key (ADR-0008 default)."""
    seen: dict = {}

    async def handler(ws):
        frame = parse_frame(await ws.recv())
        seen.update(frame["payload"])
        await ws.send(dumps(make_frame("capability", "aria", {"accepted": True})))
        # keep the connection open until the client exits
        try:
            await ws.recv()
        except websockets.ConnectionClosed:
            pass

    async def scenario():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with InputClient(hub_url=url, token="t"):
            pass
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert "status" not in seen
