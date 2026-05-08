"""Skin smoother — applies bilateral filter to the user's own visible skin.

CRITICAL RULE: smoothing is applied ONLY to areas outside the swap mask
(the user's neck, jaw margin, forehead above the crop, etc.).
The swapped face region is NEVER filtered — it has its own quality from
the DFM model and must not be altered.

Algorithm
---------
1. Apply cv2.bilateralFilter to the entire composited frame.
2. Blend: result = swap_mask * composited + (1 - swap_mask) * smoothed
   → swap region: unchanged composited pixels
   → non-swap region: bilateral-filtered pixels (smoother skin)

This is fast enough for real-time at 640×480 (~3-5ms on CPU).
"""

from __future__ import annotations

import cv2
import numpy as np


class SkinSmoother:
    """Bilateral-filter skin smoother for real-time streaming.

    Parameters
    ----------
    d            : filter diameter (neighbourhood size). Larger = slower.
                   Use odd numbers. 5 or 7 is fast; 9 is max for real-time.
    sigma_color  : colour sigma — larger values mix more dissimilar colours.
    sigma_space  : spatial sigma — larger values blur more distant pixels.
    enabled      : False → detect() is a no-op (returns frame unchanged).
    """

    def __init__(
        self,
        d: int = 7,
        sigma_color: float = 25.0,
        sigma_space: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self._d = d
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space
        self.enabled = enabled

    def apply(
        self,
        composited: np.ndarray,
        swap_mask_full: np.ndarray,
    ) -> np.ndarray:
        """Apply skin smoothing outside the swap mask.

        Parameters
        ----------
        composited      : (H, W, 3) uint8 — frame after face swap and hand restore.
        swap_mask_full  : (H, W) float32 in [0, 1] — 1.0 = swap pixel (do NOT smooth).

        Returns
        -------
        (H, W, 3) uint8 with user's skin outside swap region softened.
        """
        if not self.enabled:
            return composited

        # Bilateral filter on the full frame (fast: operates on uint8 BGR).
        smoothed = cv2.bilateralFilter(
            composited, self._d, self._sigma_color, self._sigma_space
        )

        # Expand mask to 3 channels.
        alpha = swap_mask_full[:, :, np.newaxis]  # (H, W, 1)

        # Blend: keep swap pixels, replace non-swap with smoothed.
        out = alpha * composited.astype(np.float32) + (1.0 - alpha) * smoothed.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)
