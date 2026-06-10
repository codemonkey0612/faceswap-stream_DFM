"""Hand reshaper — privacy warp for distinctive finger geometry (v1).

Purpose
-------
The streamer has brachydactyly (短指症): the little finger is markedly short.
To prevent identification from hand shape, this module lengthens the pinky
(and optionally other fingers) using the MediaPipe 21-point hand landmarks.

Approach (v1, geometric — no model)
-----------------------------------
MediaPipe gives 21 landmarks per hand. The pinky is points 17-20
(MCP, PIP, DIP, TIP). We push the outer pinky joints (18, 19, 20) outward
along the finger's axis to lengthen it, then apply a localized
displacement-field remap so the surrounding pixels follow smoothly.

This is a real-time, training-free first pass. It is NOT a full hand
replacement — it reduces the distinctive short-pinky signature on roughly
front-facing, open-hand poses. Fast motion / heavy occlusion will be weaker;
those need the model-based hand-swap (separate, future work).

MediaPipe hand landmark indices:
    0 wrist
    thumb  1-4    index 5-8    middle 9-12    ring 13-16    pinky 17-20
    (each finger: MCP, PIP, DIP, TIP)
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

_log = structlog.get_logger("occlusion.hand_reshaper")

# Finger -> (MCP, PIP, DIP, TIP) landmark indices
_PINKY = (17, 18, 19, 20)
_RING = (13, 14, 15, 16)


class HandReshaper:
    """Lengthens short fingers in the composited frame using hand landmarks.

    Parameters
    ----------
    enable        : master on/off.
    pinky_gain    : how much to extend the pinky along its axis. 0.0 = no change,
                    0.6 = move the tip out by ~60% of the finger's current length.
    ring_gain     : optional extension for the ring finger (often also short).
    influence_px  : radius (px) of the smooth displacement falloff around each
                    moved joint. Larger = smoother but blurrier.
    """

    def __init__(
        self,
        enable: bool = False,
        pinky_gain: float = 0.6,
        ring_gain: float = 0.0,
        influence_px: float = 45.0,
    ) -> None:
        self.enabled = enable
        self._pinky_gain = float(pinky_gain)
        self._ring_gain = float(ring_gain)
        self._influence = float(influence_px)
        _log.info(
            "hand_reshaper_init",
            enable=enable,
            pinky_gain=pinky_gain,
            ring_gain=ring_gain,
        )

    def apply(
        self,
        frame: np.ndarray,
        hands_px: list[np.ndarray],
    ) -> np.ndarray:
        """Reshape fingers in `frame` for each detected hand.

        frame    : (H, W, 3) uint8 BGR — composited frame.
        hands_px : list of (21, 2) float32 landmark arrays in pixel coords.

        Returns the reshaped frame (or the original if disabled / no hands).
        """
        if not self.enabled or not hands_px:
            return frame

        h, w = frame.shape[:2]
        # Accumulate a displacement field, then remap once for all hands.
        disp = np.zeros((h, w, 2), dtype=np.float32)
        touched = False

        for lm in hands_px:
            if lm.shape[0] < 21:
                continue
            if self._pinky_gain > 0:
                touched |= self._extend_finger(disp, lm, _PINKY, self._pinky_gain, (h, w))
            if self._ring_gain > 0:
                touched |= self._extend_finger(disp, lm, _RING, self._ring_gain, (h, w))

        if not touched:
            return frame

        # Build remap grids: sample source = dest - displacement.
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xs - disp[:, :, 0]
        map_y = ys - disp[:, :, 1]
        return cv2.remap(
            frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    # ------------------------------------------------------------------

    def _extend_finger(
        self,
        disp: np.ndarray,
        lm: np.ndarray,
        idx: tuple[int, int, int, int],
        gain: float,
        shape: tuple[int, int],
    ) -> bool:
        """Add a smooth outward displacement to the outer joints of one finger.

        Returns True if any displacement was added.
        """
        mcp, pip, dip, tip = (lm[i] for i in idx)
        axis = tip - mcp
        length = float(np.hypot(axis[0], axis[1]))
        if length < 1e-3:
            return False
        axis = axis / length  # unit vector along the finger

        h, w = shape
        added = False
        # Move DIP, TIP outward (and PIP a little) along the finger axis.
        # The TIP moves most → the finger appears longer.
        for joint, frac in ((tip, 1.0), (dip, 0.6), (pip, 0.25)):
            shift = axis * (gain * length * frac)
            if abs(shift[0]) + abs(shift[1]) < 0.5:
                continue
            self._splat(disp, joint, shift, h, w)
            added = True
        return added

    def _splat(
        self,
        disp: np.ndarray,
        center: np.ndarray,
        shift: np.ndarray,
        h: int,
        w: int,
    ) -> None:
        """Add a Gaussian-weighted displacement `shift` around `center`."""
        cx, cy = float(center[0]), float(center[1])
        r = self._influence
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        if x1 <= x0 or y1 <= y0:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        wgt = np.exp(-d2 / (2.0 * (r * 0.5) ** 2)).astype(np.float32)
        disp[y0:y1, x0:x1, 0] += wgt * shift[0]
        disp[y0:y1, x0:x1, 1] += wgt * shift[1]
