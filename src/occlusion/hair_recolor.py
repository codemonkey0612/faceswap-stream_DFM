"""Real-time hair color recoloring via HSV channel transfer.

Algorithm per-frame (all vectorized numpy + two cv2 color conversions):
  1. Convert composited BGR frame to HSV.
  2. For pixels where hair_mask > threshold: replace H with target_H,
     blend S toward target_S (gated on local saturation for dark hair).
  3. V (brightness/value) is NEVER modified — preserves all lighting,
     specular highlights, and texture from the original hair.
  4. Convert back to BGR.
  5. Soft-blend recolored ↔ original composited using hair_mask as alpha.

Near-black hair handling:
  Dark/black hair has near-zero S in HSV, so the hue is unreliable.
  The S-blend is gated: pixels below `min_saturation` receive a proportionally
  reduced S shift (local_S / min_saturation * target_S_blend), which smoothly
  transitions from "pure dark" to "visible color" at the saturation boundary.

Performance (1080p, CPU):  ~4-6 ms per frame.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

_log = structlog.get_logger("occlusion.hair_recolor")


class HairRecolor:
    """Recolors the subject's hair to match a configured target color.

    Parameters
    ----------
    target_bgr     : (B, G, R) integer tuple — desired hair color.
    blend          : float [0, 1]. 1.0 = full target color; 0.0 = S unchanged.
    mask_threshold : float [0, 1]. Hair mask pixels below this are ignored.
    min_saturation : int [0, 255] OpenCV HSV. Below this, S-blend is gated
                     proportionally to handle dark/black hair gracefully.
    enabled        : False → apply() returns the frame unchanged (no-op).
    """

    def __init__(
        self,
        target_bgr: tuple[int, int, int] = (30, 30, 30),
        blend: float = 0.85,
        mask_threshold: float = 0.05,
        min_saturation: int = 20,
        enabled: bool = False,
    ) -> None:
        if len(target_bgr) != 3:
            raise ValueError(f"target_bgr must have exactly 3 elements, got {len(target_bgr)}")

        self.enabled = enabled
        self._blend = float(np.clip(blend, 0.0, 1.0))
        self._mask_threshold = float(mask_threshold)
        self._min_sat = int(min_saturation)

        # Convert target BGR to HSV once at init — zero per-frame cost.
        bgr_pixel = np.array([[list(target_bgr)]], dtype=np.uint8)  # (1, 1, 3)
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
        self._target_h = int(hsv_pixel[0, 0, 0])   # H in [0, 179] (OpenCV)
        self._target_s = int(hsv_pixel[0, 0, 1])   # S in [0, 255]

        _log.info(
            "hair_recolor_init",
            enabled=enabled,
            target_bgr=target_bgr,
            target_h=self._target_h,
            target_s=self._target_s,
            blend=self._blend,
        )

    def apply(
        self,
        composited: np.ndarray,
        hair_mask: np.ndarray,
    ) -> np.ndarray:
        """Recolor hair pixels in the composited frame.

        Parameters
        ----------
        composited : (H, W, 3) uint8 BGR — frame after BiSeNet hair restore.
        hair_mask  : (H, W) float32 [0, 1] — full-frame mask from FaceParser.

        Returns
        -------
        (H, W, 3) uint8 BGR with hair pixels recolored.
        Returns composited unchanged if disabled or mask is empty.
        """
        if not self.enabled:
            return composited
        if hair_mask.max() < self._mask_threshold:
            return composited

        # BGR → HSV
        hsv = cv2.cvtColor(composited, cv2.COLOR_BGR2HSV)  # uint8, H∈[0,179]

        # Boolean mask of active hair pixels
        active = hair_mask > self._mask_threshold  # (H, W) bool

        # --- H channel: full replacement for all active pixels ---
        h_channel = hsv[:, :, 0].copy()
        h_channel[active] = self._target_h

        # --- S channel: blend toward target, gated on local saturation ---
        s_channel = hsv[:, :, 1].copy().astype(np.float32)
        target_s_f = float(self._target_s)

        s_active = s_channel[active]

        if self._min_sat > 0:
            # Pixels with very low S (dark/black hair) get a proportionally
            # smaller S shift so near-black hair stays near-black.
            sat_scale = np.clip(s_active / float(self._min_sat), 0.0, 1.0)
            eff_blend = self._blend * sat_scale
        else:
            eff_blend = self._blend

        s_channel[active] = s_active + eff_blend * (target_s_f - s_active)
        s_channel = np.clip(s_channel, 0.0, 255.0)

        # Assemble modified HSV (V channel untouched — preserves all lighting)
        hsv_out = hsv.copy()
        hsv_out[:, :, 0] = h_channel
        hsv_out[:, :, 1] = s_channel.astype(np.uint8)

        recolored = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

        # Soft alpha blend using hair_mask — smooth edges at hair boundary
        alpha = hair_mask[:, :, np.newaxis]  # (H, W, 1)
        out = (
            alpha * recolored.astype(np.float32)
            + (1.0 - alpha) * composited.astype(np.float32)
        )
        return np.clip(out, 0, 255).astype(np.uint8)
