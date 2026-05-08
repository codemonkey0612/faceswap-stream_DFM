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
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Run:  python models/download_models.py"
            )
        self._conf_thresh = confidence_threshold
        self._log = logger or structlog.get_logger("detection.face_detector")

        self._detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(640, 480),   # updated per frame in detect()
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
        )
        self._log.info("face_detector_ready", model=str(model_path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame. Returns detections sorted by confidence (desc)."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))

        try:
            _, faces = self._detector.detect(frame)
        except cv2.error as exc:
            self._log.warning("yunet_detect_error", exc=str(exc))
            return []

        if faces is None or len(faces) == 0:
            return []

        result: list[Detection] = []
        for row in faces:
            x, y, fw, fh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[14])
            bbox = (x, y, x + fw, y + fh)
            landmarks = (
                (float(row[4]),  float(row[5])),   # right eye
                (float(row[6]),  float(row[7])),   # left eye
                (float(row[8]),  float(row[9])),   # nose tip
                (float(row[10]), float(row[11])),  # right mouth
                (float(row[12]), float(row[13])),  # left mouth
            )
            result.append(Detection(bbox=bbox, confidence=score, landmarks=landmarks))

        result.sort(key=lambda d: d.confidence, reverse=True)
        return result
