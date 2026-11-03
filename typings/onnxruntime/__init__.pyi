"""Minimal type stubs for the onnxruntime surface used by voice_input.

The installed onnxruntime wheel ships no inline types; this stub lets
basedpyright strict check our usage of InferenceSession (speaker + denoise)
without treating the ONNX runtime as a fully-unknown boundary.
"""
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy
import numpy.typing as npt


def get_available_providers() -> list[str]: ...


class NodeArg:
    name: str
    shape: Sequence[int | str | None]
    type: str


class ModelMetadata:
    custom_metadata_map: dict[str, str]


class SessionOptions:
    intra_op_num_threads: int
    inter_op_num_threads: int


class InferenceSession:
    def __init__(
        self,
        path: str | Path,
        *,
        providers: Sequence[str],
        sess_options: SessionOptions | None = None,
    ) -> None: ...
    def get_providers(self) -> list[str]: ...
    def get_inputs(self) -> Sequence[NodeArg]: ...
    def get_outputs(self) -> Sequence[NodeArg]: ...
    def get_modelmeta(self) -> ModelMetadata: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, object],
        run_options: object | None = None,
    ) -> Sequence[npt.NDArray[numpy.float32]]: ...
