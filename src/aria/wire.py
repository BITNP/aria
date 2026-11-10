"""Client-side wire protocol for the waycast module network contract (ADR-0008).

The module talks to the waycast hub only over the network: a WebSocket carries
JSON frames defined by the waycast wire protocol (authoritative contract in
waycast/src/waycast/wire.py).  This local implementation mirrors that JSON
envelope so the module has no code or package dependency on the core -- the
microservice boundary is the wire, not Python imports.  Frames here are plain
dicts serialized with json; both directions of one full-duplex connection use
the same envelope:

  module -> hub: capability advertisement (first frame), then input frames
  hub -> module: capability acknowledgement, later output/result frames
"""

from __future__ import annotations

import json
from typing import Any, cast

PROTOCOL_VERSION = 1


class WireError(ValueError):
    """Raised when a message is not a valid wire frame."""


def make_frame(
    frame_type: str,
    module: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a wire frame as a JSON-serializable dict."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": frame_type,
        "module": module,
        "request_id": request_id,
        "payload": payload or {},
    }


def dumps(frame: dict[str, Any]) -> str:
    """Serialize a frame to the JSON text sent over the socket."""
    return json.dumps(frame, ensure_ascii=True, separators=(",", ":"))


def parse_frame(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate an incoming wire frame, returning its dict."""
    data: dict[str, object]
    if isinstance(raw, (str, bytes)):
        try:
            parsed: object = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise WireError(f"malformed wire frame: {exc}") from exc
        if not isinstance(parsed, dict):
            raise WireError("wire frame must be a JSON object")
        data = cast("dict[str, object]", parsed)
    else:
        data = cast("dict[str, object]", raw)
    version = data.get("protocol_version")
    if version != PROTOCOL_VERSION:
        raise WireError(
            f"unsupported protocol_version {version!r}; expected {PROTOCOL_VERSION}"
        )
    frame_type = data.get("type")
    if not isinstance(frame_type, str) or not frame_type:
        raise WireError("wire frame missing type")
    module = data.get("module")
    if not isinstance(module, str) or not module:
        raise WireError("wire frame missing module")
    result: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "type": frame_type,
        "module": module,
        "request_id": data.get("request_id"),
        "payload": {},
    }
    payload = data.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise WireError("wire frame payload must be a JSON object")
    if isinstance(payload, dict):
        result["payload"] = cast("dict[str, Any]", payload)
    return result


__all__ = ["PROTOCOL_VERSION", "WireError", "dumps", "make_frame", "parse_frame"]
