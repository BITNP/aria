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


def test_vad_defaults_are_inprocess_silero():
    s = load_settings(dict(VALID))
    assert s.vad_engine == "silero"
    assert s.vad_threshold == 0.5
    assert s.vad_min_speech_ms == 250


def test_vad_overrides():
    env = dict(VALID)
    env["VOICE_IN_VAD"] = "ENERGY"
    env["VOICE_IN_VAD_THRESHOLD"] = "0.65"
    env["VOICE_IN_VAD_MIN_SPEECH_MS"] = "400"
    s = load_settings(env)
    assert s.vad_engine == "energy"
    assert s.vad_threshold == 0.65
    assert s.vad_min_speech_ms == 400


def test_vad_engine_must_be_known():
    env = dict(VALID)
    env["VOICE_IN_VAD"] = "audio-cpp"
    with pytest.raises(ConfigError):
        load_settings(env)


def test_vad_threshold_must_be_a_probability():
    for bad in ("0", "1", "1.5", "abc"):
        env = dict(VALID)
        env["VOICE_IN_VAD_THRESHOLD"] = bad
        with pytest.raises(ConfigError):
            load_settings(env)


def test_denoise_defaults_on_and_overridable():
    assert load_settings(dict(VALID)).denoise is True
    assert load_settings({**VALID, "VOICE_IN_DENOISE": "0"}).denoise is False
    assert load_settings({**VALID, "VOICE_IN_DENOISE": "on"}).denoise is True


def test_dump_wav_dir_optional():
    assert load_settings(dict(VALID)).dump_wav_dir is None
    env = dict(VALID)
    env["VOICE_IN_DUMP_WAV"] = " /tmp/voice-in-dump "
    assert load_settings(env).dump_wav_dir == "/tmp/voice-in-dump"


def test_standalone_defaults_off_and_overridable():
    assert load_settings(dict(VALID)).standalone is False
    assert load_settings({**VALID, "VOICE_IN_STANDALONE": "1"}).standalone is True
