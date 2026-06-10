"""Threaded webcam capture with bounded queue and drop-old policy.

The capture thread reads frames from OpenCV as fast as the camera delivers
them. When the downstream consumer is slow, we drop the oldest frame rather
than block — real-time behavior is preferred over completeness.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Full, Queue

import cv2
import numpy as np
import structlog

from src.config import CaptureConfig

# Try backends in this order. DSHOW is first because MSMF is slow to open and
# slow to grab on the ASUS/OBSBOT streaming laptop (measured ~5 fps via MSMF vs
# 30 fps via DSHOW). MSMF/ANY remain as fallbacks for other machines.
_BACKENDS: list[tuple[str, int]] = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]


class Webcam:
    def __init__(
        self,
        config: CaptureConfig,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self.cfg = config
        self._log = logger or structlog.get_logger("capture.webcam")
        self._cap: cv2.VideoCapture | None = None
        self._queue: Queue[np.ndarray] = Queue(maxsize=max(1, config.buffer_size))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._actual_size: tuple[int, int] | None = None  # (w, h)
        self._drops_total = 0

    @property
    def stats(self) -> dict[str, int]:
        return {"drops_total": self._drops_total, "queue_size": self._queue.qsize()}

    def _try_open(self, backend_name: str, backend: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.cfg.device_index, backend)
        if not cap.isOpened():
            return None

        # NOTE: we intentionally do NOT force MJPG fourcc. On DSHOW the OBSBOT/ASUS
        # webcam delivers YUY2/NV12 natively; forcing MJPG can fail or slow the
        # grab. Letting the backend pick its native format is faster here.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Sanity: read a real frame — some modes advertise but fail.
        ok, _ = cap.read()
        if not ok:
            self._log.warning("webcam_backend_first_read_failed", backend=backend_name)
            cap.release()
            return None

        # Warn (not error) if delivered resolution differs — we resize internally.
        if actual_w != self.cfg.width or actual_h != self.cfg.height:
            self._log.warning(
                "webcam_resolution_mismatch_will_resize",
                backend=backend_name,
                requested=(self.cfg.width, self.cfg.height),
                actual=(actual_w, actual_h),
                note="frames will be resized to requested dimensions",
            )

        self._actual_size = (actual_w, actual_h)
        self._log.info(
            "webcam_opened",
            backend=backend_name,
            device_index=self.cfg.device_index,
            requested=(self.cfg.width, self.cfg.height, self.cfg.fps),
            actual=(actual_w, actual_h, actual_fps),
        )
        return cap

    def start(self) -> None:
        last_error: str | None = None
        for backend_name, backend in _BACKENDS:
            try:
                cap = self._try_open(backend_name, backend)
            except Exception as exc:
                last_error = f"{backend_name}: {exc}"
                self._log.warning("webcam_backend_exception", backend=backend_name, exc=str(exc))
                continue
            if cap is not None:
                self._cap = cap
                break

        if self._cap is None:
            raise RuntimeError(
                f"cannot open webcam device {self.cfg.device_index} on any backend "
                f"(last error: {last_error}). "
                "Run scripts/list_cameras.py to check available devices."
            )

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._log.warning("capture_read_failed")
                # Avoid tight busy-loop when the camera disconnects briefly.
                time.sleep(0.005)
                continue

            # Resize to target shape so downstream always sees config dimensions.
            if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
                frame = cv2.resize(
                    frame, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA
                )

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._drops_total += 1
                except Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except Full:
                self._drops_total += 1

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Webcam":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
