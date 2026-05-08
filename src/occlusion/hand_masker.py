"""Hand occlusion mask generator — prevents swap from bleeding onto hands.

MediaPipe HandLandmarker (Tasks API, v0.10+) detects 21 landmarks per hand
in normalised [0, 1] coords. We project them to pixel space, compute the
convex hull, fill it, then apply a Gaussian blur for soft edges.

The resulting mask (1.0 = hand, 0.0 = background) is subtracted from the
swap mask in the pipeline so the user's real hands always show through,
even when they partially cover the face.

Architecture note
-----------------
`landmarks_to_mask` is a pure function (no MediaPipe dependency) and is
tested directly. `HandMasker` wraps it with the live MediaPipe backend.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import structlog


# ---------------------------------------------------------------------------
# Pure geometry — testable without MediaPipe
# ---------------------------------------------------------------------------

def landmarks_to_mask(
    h: int,
    w: int,
    landmarks_list: list[list[tuple[float, float]]],
    feather_radius: int = 15,
) -> np.ndarray:
    """Convert normalised hand landmarks to a float32 occlusion mask.

    Parameters
    ----------
    h, w            : frame dimensions in pixels.
    landmarks_list  : one entry per detected hand; each entry is a list of
                      (x, y) pairs in normalised [0, 1] coords.
    feather_radius  : Gaussian blur half-width (px). 0 → hard-edged mask.

    Returns
    -------
    mask : (h, w) float32 in [0, 1]. 1.0 = hand pixel.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    for lms in landmarks_list:
        pts = np.array(
            [(int(x * w), int(y * h)) for x, y in lms],
            dtype=np.int32,
        )
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1.0)

    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        np.clip(mask, 0.0, 1.0, out=mask)

    return mask


# ---------------------------------------------------------------------------
# HandMasker — MediaPipe Tasks API wrapper
# ---------------------------------------------------------------------------

class HandMasker:
    """Detects hands and returns a per-pixel float32 occlusion mask.

    Requires ``models/hand_landmarker.task`` (run models/download_models.py).

    Parameters
    ----------
    model_path           : path to hand_landmarker.task bundle.
    max_num_hands        : maximum hands to track simultaneously (1 or 2).
    min_detection_conf   : confidence for initial hand detection.
    min_presence_conf    : confidence that a detected hand is still present.
    min_tracking_conf    : confidence for landmark tracking.
    feather_radius       : Gaussian blur half-width (px) for soft mask edges.
    """

    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        max_num_hands: int = 2,
        min_detection_conf: float = 0.5,
        min_presence_conf: float = 0.5,
        min_tracking_conf: float = 0.5,
        feather_radius: int = 15,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self._feather = feather_radius
        self._log = logger or structlog.get_logger("occlusion.hand_masker")
        self._landmarker = None
        self._t0 = time.monotonic()

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
            opts = mp_vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_conf,
                min_hand_presence_confidence=min_presence_conf,
                min_tracking_confidence=min_tracking_conf,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            self._log.info(
                "hand_masker_ready",
                model=str(model_path),
                max_hands=max_num_hands,
            )
        except Exception as exc:
            self._log.warning(
                "hand_masker_unavailable",
                exc=str(exc),
                note="hand occlusion disabled — running without hand mask",
            )

    @property
    def enabled(self) -> bool:
        return self._landmarker is not None

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Generate hand occlusion mask from a BGR frame.

        Returns (H, W) float32 in [0, 1]. All-zeros if disabled or no hands.
        """
        h, w = frame_bgr.shape[:2]
        if self._landmarker is None:
            return np.zeros((h, w), dtype=np.float32)

        import mediapipe as mp

        # Tasks API requires RGB MediaPipe Image and a monotonic timestamp.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._t0) * 1000)

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            self._log.warning("hand_detection_error", exc=str(exc))
            return np.zeros((h, w), dtype=np.float32)

        if not result.hand_landmarks:
            return np.zeros((h, w), dtype=np.float32)

        landmarks_list = [
            [(lm.x, lm.y) for lm in hand]
            for hand in result.hand_landmarks
        ]
        return landmarks_to_mask(h, w, landmarks_list, self._feather)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
