"""Configuration loading and validation."""
from __future__ import annotations

import pytest

from voice_input.config import ConfigError, load_settings

VALID = {
    "WAYCAST_HUB_URL": "ws://127.0.0.1:8000",
    "WAYCAST_HUB_TOKEN": "tok",
    "VOICE_IN_ASR_ENDPOINT": "https://asr.example/v1",
    "VOICE_IN_ASR_MODEL": "whisper",
}


def test_loads_minimal():
    s = load_settings(dict(VALID))
    assert s.hub_url == "ws://127.0.0.1:8000"
    assert s.hub_token == "tok"
    assert s.asr_endpoint == "https://asr.example/v1"
    assert s.asr_model == "whisper"
    assert s.asr_api_key is None
    assert s.device is None


def test_requires_hub_url():
    env = dict(VALID)
    env.pop("WAYCAST_HUB_URL")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_hub_url_must_be_ws():
    env = dict(VALID)
    env["WAYCAST_HUB_URL"] = "http://127.0.0.1:8000"
    with pytest.raises(ConfigError):
        load_settings(env)


def test_requires_asr_fields():
    env = dict(VALID)
    env.pop("VOICE_IN_ASR_MODEL")
    with pytest.raises(ConfigError):
        load_settings(env)


def test_optional_fields():
    env = dict(VALID)
    env["VOICE_IN_API_KEY"] = "k"
    s = load_settings(env)
    assert s.asr_api_key is None  # wrong key is ignored

    env["VOICE_IN_ASR_API_KEY"] = "k"
    env["VOICE_IN_DEVICE"] = "mic0"
    s = load_settings(env)
    assert s.asr_api_key == "k"
    assert s.device == "mic0"
