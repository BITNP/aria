from types import TracebackType
from typing import Literal, Self


class RawInputStream:
    def __init__(
        self,
        *,
        device: int | str | None = None,
        samplerate: int,
        channels: int,
        dtype: Literal["int16"],
        blocksize: int,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def read(self, frames: int) -> tuple[bytes, bool]: ...

