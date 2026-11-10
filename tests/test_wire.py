"""Wire protocol: frame construction, serialization, and parsing (ADR-0008)."""

from __future__ import annotations

import json

import pytest

from voice_input.wire import PROTOCOL_VERSION, WireError, dumps, make_frame, parse_frame


def test_roundtrip_capability_frame():
    frame = make_frame(
        "capability", "voice_input", {"name": "voice_input", "kind": "input"}
    )
    raw = dumps(frame)
    parsed = parse_frame(raw)
    assert parsed == {
        "protocol_version": PROTOCOL_VERSION,
        "type": "capability",
        "module": "voice_input",
        "request_id": None,
        "payload": {"name": "voice_input", "kind": "input"},
    }


def test_roundtrip_input_frame_matches_unified_input():
    payload = {
        "content": "你好",
        "source": "asr",
        "metadata": {"speech_duration_ms": 500},
    }
    frame = make_frame("input", "voice_input", payload)
    parsed = parse_frame(dumps(frame))
    assert parsed["type"] == "input"
    assert parsed["module"] == "voice_input"
    assert parsed["payload"] == payload


def test_parse_frame_accepts_dict_and_bytes():
    frame = make_frame("input", "voice_input", {"content": "x"})
    assert parse_frame(frame) == frame
    assert parse_frame(dumps(frame).encode("utf-8")) == frame


def test_parse_frame_rejects_malformed_json():
    with pytest.raises(WireError, match="malformed"):
        parse_frame("{not json")


def test_parse_frame_rejects_wrong_protocol_version():
    bad = json.dumps({"protocol_version": 999, "type": "input", "module": "m"})
    with pytest.raises(WireError, match="protocol_version"):
        parse_frame(bad)


def test_parse_frame_rejects_missing_fields_and_bad_payload():
    with pytest.raises(WireError):
        parse_frame(json.dumps({"type": "input"}))
    with pytest.raises(WireError):
        parse_frame(
            json.dumps(
                {"protocol_version": 1, "type": "input", "module": "m", "payload": []}
            )
        )


def test_make_frame_defaults_payload_to_empty_dict():
    assert make_frame("input", "m") == {
        "protocol_version": 1,
        "type": "input",
        "module": "m",
        "request_id": None,
        "payload": {},
    }


def test_dumps_uses_compact_json():
    raw = dumps(make_frame("input", "m", {"a": 1}))
    assert json.loads(raw)["payload"]["a"] == 1
    assert "\n" not in raw
