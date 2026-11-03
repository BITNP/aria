"""Bundled ONNX model assets: CAM++ speaker ID (ADR-0002) + GTCRN denoiser (ADR-0006).

The CAM++ model is exported by the 网协 group and is not reproducible from
public channels; the GTCRN model is the official streaming export (MIT,
provenance and sha256 in the README).  Both ship inside this repository so the
module never reads a model path from the environment.  Resolve the files with
importlib.resources (``voice_input.speaker.bundled_model_path`` /
``voice_input.denoise.bundled_model_path``).
"""
