"""Face detection via OpenCV YuNet (cv2.FaceDetectorYN).

YuNet gives bbox + confidence + 5 landmarks (right_eye, left_eye, nose_tip,
right_mouth, left_mouth) with no Cython/MSVC build step — just an ONNX model
loaded through OpenCV's built-in DNN backend.

YuNet output row format (15 values):
  [x, y, w, h,  x_re, y_re,  x_le, y_le,  x_nt, y_nt,
   x_rc, y_rc,  x_lc, y_lc,  score]
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

from src.failsafe.monitor import Detection


class FaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.70,
        nms_threshold: float = 0.30,
        top_k: int = 100,
        detect_scale: float = 1.0,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Run:  python models/download_models.py"
            )
        self._conf_thresh = confidence_threshold
        # Detection runs on a frame downscaled by this factor (<1.0), then the
        # bbox + landmarks are scaled back up. YuNet stays accurate at 0.5 and
        # it roughly quarters detection cost — the main detection speedup.
        self._scale = float(detect_scale) if 0.0 < detect_scale <= 1.0 else 1.0
        self._last_size: tuple[int, int] | None = None  # cache to avoid setInputSize churn
        self._log = logger or structlog.get_logger("detection.face_detector")

        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(640, 480),   # updated per frame in detect()
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._log.info("face_detector_ready", model=str(model_path), detect_scale=self._scale)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame. Returns detections sorted by confidence (desc)."""
        if self._scale < 1.0:
            small = cv2.resize(frame, None, fx=self._scale, fy=self._scale,
                               interpolation=cv2.INTER_AREA)
            inv = 1.0 / self._scale
        else:
            small = frame
            inv = 1.0

        sh, sw = small.shape[:2]
        if self._last_size != (sw, sh):
            self._detector.setInputSize((sw, sh))
            self._last_size = (sw, sh)

        try:
            _, faces = self._detector.detect(small)
        except cv2.error as exc:
            self._log.warning("yunet_detect_error", exc=str(exc))
            return []

        if faces is None or len(faces) == 0:
            return []

        result: list[Detection] = []
        for row in faces:
            x, y, fw, fh = float(row[0]) * inv, float(row[1]) * inv, float(row[2]) * inv, float(row[3]) * inv
            score = float(row[14])
            bbox = (x, y, x + fw, y + fh)
            landmarks = (
                (float(row[4])  * inv, float(row[5])  * inv),   # right eye
                (float(row[6])  * inv, float(row[7])  * inv),   # left eye
                (float(row[8])  * inv, float(row[9])  * inv),   # nose tip
                (float(row[10]) * inv, float(row[11]) * inv),   # right mouth
                (float(row[12]) * inv, float(row[13]) * inv),   # left mouth
            )
            result.append(Detection(bbox=bbox, confidence=score, landmarks=landmarks))

        result.sort(key=lambda d: d.confidence, reverse=True)
        return result
