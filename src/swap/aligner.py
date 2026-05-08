"""5-point face alignment: landmark -> affine warp to canonical crop.

Reference landmarks are computed so that eyes, nose, and mouth fall at
fixed positions inside a square crop of size `output_size`. This is the
standard preprocessing step for face-swap models (DFM, InSwapper, etc.).

The inverse transform is stored alongside the forward warp so the swapped
crop can be pasted back into the original frame at the correct position.
"""

from __future__ import annotations

import cv2
import numpy as np


# Canonical landmark positions inside a 256×256 crop.
# Source: commonly used ArcFace / DFL reference template, scaled to 256px.
_REF_LANDMARKS_256 = np.array(
    [
        [85.82,  106.26],  # right eye center
        [170.18, 106.26],  # left eye center
        [128.00, 158.57],  # nose tip
        [92.00,  204.96],  # right mouth corner
        [164.00, 204.96],  # left mouth corner
    ],
    dtype=np.float32,
)


def get_reference_landmarks(size: int) -> np.ndarray:
    """Scale the 256-px reference template to `size`."""
    return _REF_LANDMARKS_256 * (size / 256.0)


def align_face(
    frame: np.ndarray,
    landmarks: tuple[tuple[float, float], ...],
    output_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp `frame` so that `landmarks` align to the canonical template.

    Returns
    -------
    aligned : (output_size, output_size, 3) uint8  —  the aligned face crop.
    M        : (2, 3) float32 affine matrix from frame → aligned space.
    """
    src_pts = np.array(landmarks, dtype=np.float32)          # (5, 2)
    dst_pts = get_reference_landmarks(output_size)            # (5, 2)

    M, _ = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.LMEDS,
        confidence=0.99,
    )
    if M is None:
        # Fallback: crude center-crop if alignment fails.
        h, w = frame.shape[:2]
        cx, cy = int(np.mean(src_pts[:, 0])), int(np.mean(src_pts[:, 1]))
        half = output_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        crop = frame[y1:y1 + output_size, x1:x1 + output_size]
        crop = cv2.resize(crop, (output_size, output_size))
        M = np.eye(2, 3, dtype=np.float32)
        return crop, M

    aligned = cv2.warpAffine(
        frame, M, (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, M.astype(np.float32)


def paste_back(
    frame: np.ndarray,
    swapped_crop: np.ndarray,
    mask_crop: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Paste `swapped_crop` back into `frame` using inverse of affine matrix `M`.

    Parameters
    ----------
    frame        : original full frame (H, W, 3) uint8
    swapped_crop : swapped face (crop_size, crop_size, 3) uint8
    mask_crop    : blend mask (crop_size, crop_size) float32 in [0, 1]
    M            : affine matrix used in align_face (frame → crop space)

    Returns
    -------
    composited : (H, W, 3) uint8 with face pasted back.
    """
    h, w = frame.shape[:2]
    crop_size = swapped_crop.shape[0]

    # Invert the affine transform: crop space → frame space.
    M_inv = cv2.invertAffineTransform(M)

    # Warp the swapped crop and its mask back to frame coordinates.
    swapped_full = cv2.warpAffine(
        swapped_crop, M_inv, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    mask_full = cv2.warpAffine(
        mask_crop, M_inv, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # Expand mask to 3 channels for broadcasting.
    if mask_full.ndim == 2:
        mask_full = mask_full[:, :, np.newaxis]

    # Alpha-blend: result = mask * swapped + (1 - mask) * original
    frame_f = frame.astype(np.float32)
    swap_f = swapped_full.astype(np.float32)
    composited = mask_full * swap_f + (1.0 - mask_full) * frame_f
    return np.clip(composited, 0, 255).astype(np.uint8)
