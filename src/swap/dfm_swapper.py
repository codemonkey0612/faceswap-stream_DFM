"""DFM swap orchestrator: align → infer → paste back.

When no model is loaded, runs in identity mode: aligns the face to the
canonical crop, then pastes the unmodified crop back. This lets the full
geometry pipeline (alignment, warping, masking) be tested before a trained
.dfm is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import structlog

from src.failsafe.monitor import Detection
from src.swap.aligner import align_face, paste_back
from src.swap.dfm_loader import DFMLoader


@dataclass
class SwapResult:
    composited: np.ndarray          # (H, W, 3) uint8 — full frame with face swapped
    mask_full: np.ndarray           # (H, W) float32 — mask in frame coords
    swap_bbox: tuple[float, float, float, float]  # bbox of the swapped region
    aligned_crop: np.ndarray        # (S, S, 3) uint8 — face crop before swap (for XSeg)
    affine_M: np.ndarray            # (2, 3) float64 — affine transform crop→frame


class DFMSwapper:
    def __init__(
        self,
        loader: DFMLoader,
        output_size: int = 256,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self._loader = loader
        self._size = output_size
        self._log = logger or structlog.get_logger("swap.dfm_swapper")
        if not loader.loaded:
            self._log.info("dfm_swapper_identity_mode",
                           note="no model loaded, face geometry is tested without swap")

    def process(
        self,
        frame: np.ndarray,
        detection: Detection,
    ) -> SwapResult:
        """Align detected face, optionally swap, paste back into frame.

        Returns SwapResult with composited frame and mask.
        """
        if not detection.landmarks or len(detection.landmarks) < 5:
            # No landmarks: fall back to bbox-centre crop without alignment.
            return self._bbox_passthrough(frame, detection)

        # --- 1. Align face to canonical crop ---
        aligned, M = align_face(frame, detection.landmarks, self._size)

        # --- 2. Swap inference (or identity pass-through) ---
        if self._loader.loaded:
            try:
                swapped, mask_crop = self._loader.run(aligned)
            except Exception as exc:
                self._log.warning("dfm_inference_error", exc=str(exc))
                swapped, mask_crop = aligned, _soft_ellipse(self._size)
        else:
            swapped = aligned
            mask_crop = _soft_ellipse(self._size)

        # --- 3. Paste swapped crop back into the original frame ---
        composited = paste_back(frame, swapped, mask_crop, M)

        # --- 4. Compute swap bbox in frame coords (for post-composite check) ---
        swap_bbox = _warp_bbox_back(self._size, M, frame.shape)

        # --- 5. Build full-frame mask (for Monitor + skin smoother) ---
        h, w = frame.shape[:2]
        M_inv = cv2.invertAffineTransform(M)
        mask_full = cv2.warpAffine(
            mask_crop, M_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        return SwapResult(
            composited=composited,
            mask_full=mask_full,
            swap_bbox=swap_bbox,
            aligned_crop=aligned,
            affine_M=M,
        )

    def _bbox_passthrough(self, frame: np.ndarray, det: Detection) -> SwapResult:
        """No-landmarks fallback: return frame unmodified."""
        h, w = frame.shape[:2]
        dummy_crop = np.zeros((self._size, self._size, 3), dtype=np.uint8)
        dummy_M    = np.eye(2, 3, dtype=np.float64)
        return SwapResult(
            composited=frame.copy(),
            mask_full=np.zeros((h, w), dtype=np.float32),
            swap_bbox=det.bbox,
            aligned_crop=dummy_crop,
            affine_M=dummy_M,
        )


def _soft_ellipse(size: int) -> np.ndarray:
    """Soft-edged ellipse mask for identity/no-mask pass-through."""
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    cv2.ellipse(mask, (cx, cy), (size // 2 - 4, size // 2 - 4),
                angle=0, startAngle=0, endAngle=360,
                color=1.0, thickness=-1)
    return cv2.GaussianBlur(mask, (31, 31), 0)


def _warp_bbox_back(
    size: int,
    M: np.ndarray,
    frame_shape: tuple[int, ...],
) -> tuple[float, float, float, float]:
    """Map the crop corners through M_inv to get the swap bbox in frame coords."""
    corners = np.array(
        [[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32
    )
    M_inv = cv2.invertAffineTransform(M)
    # Affine transform of points: result = M_inv[:, :2] @ corner.T + M_inv[:, 2]
    transformed = (M_inv[:, :2] @ corners.T + M_inv[:, 2:]).T
    x1 = float(transformed[:, 0].min())
    y1 = float(transformed[:, 1].min())
    x2 = float(transformed[:, 0].max())
    y2 = float(transformed[:, 1].max())
    h, w = frame_shape[:2]
    return (
        max(0.0, x1), max(0.0, y1),
        min(float(w), x2), min(float(h), y2),
    )
