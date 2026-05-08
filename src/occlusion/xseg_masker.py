"""XSeg occlusion masker.

Runs a DFL-exported XSeg ONNX model on an aligned face crop and returns a
per-pixel mask indicating which parts of the crop are clean face (1.0) vs.
occluded by a foreign object (0.0 — microphone, headset, crossing hand, etc.).

Model source
------------
After Phase 3 DFL SAEHD training, train XSeg inside DeepFaceLab:
  DeepFaceLab -> XSeg -> Train XSeg
  DeepFaceLab -> XSeg -> Export XSeg to ONNX
  Copy the exported .onnx to:  models/xseg.onnx

XSeg ONNX format (DFL convention):
  Input  : (1, 3, H, W)  float32  BGR  normalised [0, 1]
  Output : (1, 1, H, W)  float32  sigmoid logits — squeeze + threshold at 0.5

If the file is absent the masker disables itself and returns all-ones masks
(no-op), so the rest of the pipeline is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

_log = structlog.get_logger("occlusion.xseg")


class XSegMasker:
    """Runs XSeg inference on aligned face crops.

    Thread-safety: single-threaded (one InferenceSession, no locking).
    The pipeline calls this once per frame from the main loop.
    """

    def __init__(
        self,
        model_path: str | Path = "models/xseg.onnx",
        feather_radius: int = 8,
    ) -> None:
        self._enabled      = False
        self._sess         = None
        self._input_name   = ""
        self._input_size   = 256
        self._nchw         = True   # True = (B,C,H,W), False = (B,H,W,C)
        self._feather_r    = feather_radius

        model_path = Path(model_path)
        if not model_path.exists():
            _log.info(
                "xseg_disabled",
                reason="model not found",
                path=str(model_path),
                hint="Train XSeg in DFL then export to models/xseg.onnx",
            )
            return

        try:
            import onnxruntime as ort
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if p in list(ort.get_all_providers())]
            self._sess = ort.InferenceSession(str(model_path), providers=providers)

            inp   = self._sess.get_inputs()[0]
            shape = inp.shape          # e.g. [1, 3, 256, 256] or [1, 256, 256, 3]
            self._input_name = inp.name

            # Detect layout: NCHW if dim[1] == 3, else NHWC
            if len(shape) == 4 and isinstance(shape[1], int) and shape[1] == 3:
                self._nchw       = True
                self._input_size = shape[2] if isinstance(shape[2], int) else 256
            else:
                self._nchw       = False
                self._input_size = shape[1] if isinstance(shape[1], int) else 256

            self._enabled = True
            _log.info(
                "xseg_ready",
                model=model_path.name,
                input_size=self._input_size,
                layout="NCHW" if self._nchw else "NHWC",
                providers=providers,
            )

        except Exception as exc:
            _log.warning("xseg_load_failed", exc=str(exc))

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_face_mask_crop(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Return a mask in crop-space: 1.0 = clean face, 0.0 = occluded.

        Parameters
        ----------
        aligned_crop : (H, W, 3) uint8 BGR — the face crop from DFMSwapper.

        Returns
        -------
        (H, W) float32 in [0, 1].  All-ones if XSeg is disabled.
        """
        if not self._enabled:
            h, w = aligned_crop.shape[:2]
            return np.ones((h, w), dtype=np.float32)

        h, w = aligned_crop.shape[:2]
        size = self._input_size

        # Resize to model input resolution
        resized = cv2.resize(aligned_crop, (size, size), interpolation=cv2.INTER_LINEAR)

        # Normalise BGR [0,255] → [0,1]
        norm = resized.astype(np.float32) / 255.0

        if self._nchw:
            # DFL convention: BGR channel order is kept (not converted to RGB)
            tensor = norm.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)
        else:
            tensor = norm[np.newaxis]                       # (1, H, W, 3)

        raw = self._sess.run(None, {self._input_name: tensor})[0]

        # Squeeze to (H, W) regardless of output shape
        mask = raw.squeeze()

        # Apply sigmoid if output is logits (outside [0, 1])
        if mask.max() > 1.0 or mask.min() < 0.0:
            mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float64)))
            mask = mask.astype(np.float32)

        # Resize back to crop dimensions
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Feather edges so the transition is smooth
        if self._feather_r > 0:
            ksize = self._feather_r * 2 + 1
            mask  = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def get_occluder_mask_fullframe(
        self,
        aligned_crop: np.ndarray,
        M: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Warp the XSeg occluder mask from crop space to full-frame space.

        Parameters
        ----------
        aligned_crop : (S, S, 3) uint8 BGR — from SwapResult.aligned_crop
        M            : (2, 3) float64     — affine from SwapResult.affine_M
        frame_shape  : (H, W, 3)          — original webcam frame shape

        Returns
        -------
        (H, W) float32 — 1.0 where something is occluding the face,
        0.0 where the face is clean.  All-zeros if XSeg is disabled.
        """
        h, w = frame_shape[:2]
        if not self._enabled:
            return np.zeros((h, w), dtype=np.float32)

        face_mask_crop = self.get_face_mask_crop(aligned_crop)   # 1 = face
        occluder_crop  = 1.0 - face_mask_crop                    # 1 = occluded

        M_inv = cv2.invertAffineTransform(M)
        occluder_full = cv2.warpAffine(
            occluder_crop, M_inv, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return np.clip(occluder_full, 0.0, 1.0).astype(np.float32)
