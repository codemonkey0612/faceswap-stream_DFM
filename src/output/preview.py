"""Local OpenCV preview window — development only, never use in production."""

from __future__ import annotations

import numpy as np
import cv2


class PreviewSink:
    """FrameSink that shows frames in a local cv2 window instead of virtual camera."""

    def __init__(self, window_name: str = "faceswap-preview") -> None:
        self._name = window_name
        self._frame_count = 0

    def send(self, frame: np.ndarray) -> None:
        cv2.imshow(self._name, frame)
        # 1ms wait — enough to process GUI events without blocking.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt("q pressed in preview window")

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._name)
        except cv2.error:
            pass  # window was never shown (camera failed before first frame)
