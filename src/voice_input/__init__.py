"""Raspberry Girl voice input module.

Captures microphone audio, detects speech endpoints, transcribes them with an
OpenAI-compatible ASR service, optionally labels the speaker (dynamic CAM++
identification, ADR-0002), and sends the result to the waycast hub as an input
frame (ADR-0008).  This module carries no core logic: it is an input module of
the waycast core and does not participate in turn-taking decisions (ADR-0011).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
