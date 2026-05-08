"""Failsafe Monitor — runs pre-swap and post-composite validity checks.

The Monitor is pure: it reads inputs and returns TriggerResult. It never
writes to the virtual camera — that is exclusively Gate's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .triggers import (
    PostCompositeTrigger,
    PreSwapTrigger,
    TriggerResult,
)


@dataclass(frozen=True)
class Detection:
    """One face detection: bbox in pixel coords (x1, y1, x2, y2), confidence, and optional landmarks.

    landmarks: 5 keypoints as ((x0,y0), (x1,y1), ...) in pixel coords.
    Order: right_eye, left_eye, nose_tip, right_mouth, left_mouth.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: tuple[tuple[float, float], ...] = ()

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def inside(self, frame_w: int, frame_h: int, tolerance: float | None = None) -> bool:
        """True if the face centre is inside the frame and the bbox is not
        wildly out of bounds.

        Tolerance defaults to 15% of the shorter frame dimension, which
        handles the common case of a close-up face whose chin or forehead
        extends slightly past the frame edge after resolution upscaling.
        A fixed 4 px value was too tight when a 640×480 webcam feed is
        upscaled to 1920×1080.
        """
        if tolerance is None:
            tolerance = min(frame_w, frame_h) * 0.15
        x1, y1, x2, y2 = self.bbox
        # Centre must be inside the frame (no tolerance on centre).
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if not (0 <= cx <= frame_w and 0 <= cy <= frame_h):
            return False
        # Edges may overshoot by up to `tolerance` pixels.
        return (
            -tolerance <= x1 < x2 <= frame_w + tolerance
            and -tolerance <= y1 < y2 <= frame_h + tolerance
        )


@dataclass(frozen=True)
class MonitorConfig:
    min_confidence: float = 0.70
    min_face_area_ratio: float = 0.005
    # Max allowed distance between detected bbox center and swap bbox center,
    # expressed as a fraction of the detected bbox's longer edge.
    max_swap_bbox_divergence_ratio: float = 0.20


class Monitor:
    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.cfg = config or MonitorConfig()

    # --- Pre-swap ----------------------------------------------------------

    def check_pre_swap(
        self,
        frame_shape: tuple[int, int, int],
        detections: Iterable[Detection] | None,
        detector_error: BaseException | None = None,
    ) -> TriggerResult:
        """Runs after face detection, before DFM swap."""
        if detector_error is not None:
            return TriggerResult.block(
                PreSwapTrigger.DETECTOR_ERROR,
                exc_type=type(detector_error).__name__,
                exc_message=str(detector_error),
            )

        dets = list(detections) if detections is not None else []
        if len(dets) == 0:
            return TriggerResult.block(PreSwapTrigger.NO_FACE)

        best = max(dets, key=lambda d: d.confidence)

        if best.confidence < self.cfg.min_confidence:
            return TriggerResult.block(
                PreSwapTrigger.LOW_CONFIDENCE,
                confidence=float(best.confidence),
                threshold=self.cfg.min_confidence,
            )

        h, w = frame_shape[0], frame_shape[1]
        if h <= 0 or w <= 0:
            return TriggerResult.block(PreSwapTrigger.DETECTOR_ERROR, reason="invalid_frame_shape")

        area_ratio = best.area() / float(h * w)
        if area_ratio < self.cfg.min_face_area_ratio:
            return TriggerResult.block(
                PreSwapTrigger.TINY_FACE,
                area_ratio=float(area_ratio),
                threshold=self.cfg.min_face_area_ratio,
            )

        if not best.inside(w, h):
            return TriggerResult.block(
                PreSwapTrigger.BBOX_OUT_OF_FRAME,
                bbox=tuple(float(v) for v in best.bbox),
                frame_size=(w, h),
            )

        return TriggerResult.pass_through()

    # --- Post-composite ----------------------------------------------------

    def check_post_composite(
        self,
        composited: np.ndarray,
        mask: np.ndarray,
        detected_bbox: tuple[float, float, float, float],
        swap_bbox: tuple[float, float, float, float],
        expected_shape: tuple[int, int, int],
        compositor_error: BaseException | None = None,
    ) -> TriggerResult:
        """Runs on the composited frame just before virtual-camera write."""
        if compositor_error is not None:
            return TriggerResult.block(
                PostCompositeTrigger.COMPOSITOR_ERROR,
                exc_type=type(compositor_error).__name__,
                exc_message=str(compositor_error),
            )

        if composited.shape != expected_shape:
            return TriggerResult.block(
                PostCompositeTrigger.MASK_SHAPE_MISMATCH,
                expected=tuple(expected_shape),
                actual=tuple(composited.shape),
            )

        if mask.shape[:2] != expected_shape[:2]:
            return TriggerResult.block(
                PostCompositeTrigger.MASK_SHAPE_MISMATCH,
                expected=tuple(expected_shape[:2]),
                actual=tuple(mask.shape[:2]),
            )

        if not np.any(mask):
            return TriggerResult.block(PostCompositeTrigger.EMPTY_MASK)

        if not np.all(np.isfinite(composited)):
            return TriggerResult.block(PostCompositeTrigger.NAN_OR_INF)

        # Composited frame must be in valid color domain.
        # For uint8 this is automatic, but catch float pipelines.
        if composited.dtype.kind == "f":
            if composited.min() < 0.0 or composited.max() > 255.0:
                return TriggerResult.block(
                    PostCompositeTrigger.COLOR_DOMAIN_ERROR,
                    min=float(composited.min()),
                    max=float(composited.max()),
                )

        div = _bbox_center_distance(detected_bbox, swap_bbox)
        det_longer_edge = max(
            detected_bbox[2] - detected_bbox[0],
            detected_bbox[3] - detected_bbox[1],
        )
        if det_longer_edge <= 0:
            return TriggerResult.block(PostCompositeTrigger.SWAP_BBOX_DIVERGENCE, reason="invalid_detected_bbox")
        if div / det_longer_edge > self.cfg.max_swap_bbox_divergence_ratio:
            return TriggerResult.block(
                PostCompositeTrigger.SWAP_BBOX_DIVERGENCE,
                center_distance=float(div),
                detected_longer_edge=float(det_longer_edge),
                threshold_ratio=self.cfg.max_swap_bbox_divergence_ratio,
            )

        return TriggerResult.pass_through()


def _bbox_center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax, ay = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bx, by = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    return float(np.hypot(ax - bx, ay - by))
