"""pyvirtualcam wrapper that implements the Gate's FrameSink protocol.

On Windows, pyvirtualcam routes to the OBS Virtual Camera plugin (which must
be installed via OBS Studio). The virtual camera appears as a regular webcam
device to any consumer — including OBS itself, Zoom, Discord, etc.
"""

from __future__ import annotations

import numpy as np
import pyvirtualcam
import structlog


class VirtualCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        fmt: str = "BGR",
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._log = logger or structlog.get_logger("output.virtual_camera")

        pixfmt = {
            "BGR": pyvirtualcam.PixelFormat.BGR,
            "RGB": pyvirtualcam.PixelFormat.RGB,
        }.get(fmt.upper())
        if pixfmt is None:
            raise ValueError(f"unsupported pixel format: {fmt}")
        self._pixfmt = pixfmt
        self._cam: pyvirtualcam.Camera | None = None

    def start(self) -> None:
        try:
            self._cam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                fmt=self._pixfmt,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Could not start OBS Virtual Camera.\n"
                "Fix: open OBS Studio -> Tools -> Start Virtual Camera, then retry.\n"
                f"Original error: {exc}"
            ) from exc
        self._log.info(
            "virtual_camera_started",
            backend=self._cam.backend,
            device=self._cam.device,
            size=(self._width, self._height),
            fps=self._fps,
        )

    def send(self, frame: np.ndarray) -> None:
        if self._cam is None:
            raise RuntimeError("VirtualCamera.start() must be called before send()")
        self._cam.send(frame)
        self._cam.sleep_until_next_frame()

    def close(self) -> None:
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self) -> "VirtualCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
