"""BiSeNet face parser — excludes hair (and optionally glasses/accessories)
from the swap region.

BiSeNet19 class map (standard face-parsing.PyTorch convention):
    0  background     1  skin           2  l-brow
    3  r-brow         4  l-eye          5  r-eye
    6  eyeglasses     7  l-ear          8  r-ear
    9  earring       10  nose          11  mouth-inside
   12  u-lip         13  l-lip         14  neck
   15  necklace      16  clothing      17  hair
   18  headwear

Default "exclude" classes (things that cross the face and should NOT be swapped):
    17  hair        — bangs / loose strands falling over the face
    18  headwear    — hats, headbands crossing the forehead

Optionally also:
     6  eyeglasses  — if the AI face has no glasses but you do (or vice versa)

Model source
------------
  scripts/download_face_parser.py   downloads the model automatically.
  Target path: models/face_parser.onnx

  Input  : (1, 3, 512, 512) float32, RGB, ImageNet-normalised
  Output : (1, 19, 512, 512) float32 class logits  (argmax → class index)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import structlog

_log = structlog.get_logger("occlusion.face_parser")

# ImageNet normalisation constants (RGB order)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default classes to exclude from the swap (= treat as occluders)
_DEFAULT_EXCLUDE = frozenset({17, 18})   # hair + headwear

# Minimum fraction of crop pixels that must be "excluded class" before
# bothering to apply the mask (avoids GPU round-trip for clean frames)
_MIN_COVERAGE = 0.005


class FaceParser:
    """Runs BiSeNet face parsing on aligned face crops.

    stride : run inference every N frames, cache the mask in between.
             Default 1 = every frame (GPU).  On CPU set to 3-6 to avoid
             killing throughput; hair moves slowly so a stale mask is fine.
    """

    def __init__(
        self,
        model_path: str | Path = "models/face_parser.onnx",
        exclude_classes: frozenset[int] = _DEFAULT_EXCLUDE,
        feather_radius: int = 6,
        stride: int = 1,
    ) -> None:
        self._enabled      = False
        self._sess         = None
        self._input_name   = ""
        self._input_size   = 512
        self._exclude      = exclude_classes
        self._feather_r    = feather_radius
        self._stride       = max(1, stride)
        self._frame_count  = 0
        self._cached_mask: np.ndarray | None = None  # last computed mask

        model_path = Path(model_path)
        if not model_path.exists():
            _log.info(
                "face_parser_disabled",
                reason="model not found",
                path=str(model_path),
                hint="run: .venv\\Scripts\\python.exe scripts\\download_face_parser.py",
            )
            return

        try:
            import onnxruntime as ort
            providers = [p for p in
                         ["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if p in list(ort.get_all_providers())]
            self._sess = ort.InferenceSession(str(model_path), providers=providers)

            inp = self._sess.get_inputs()[0]
            self._input_name = inp.name
            if isinstance(inp.shape[2], int):
                self._input_size = inp.shape[2]

            self._enabled = True
            _log.info(
                "face_parser_ready",
                model=model_path.name,
                input_size=self._input_size,
                exclude_classes=sorted(exclude_classes),
                providers=providers,
            )

        except Exception as exc:
            _log.warning("face_parser_load_failed", exc=str(exc))

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------

    def get_hair_mask_crop(self, aligned_crop: np.ndarray) -> np.ndarray:
        """Return a hair/accessory mask in crop-space.

        Parameters
        ----------
        aligned_crop : (H, W, 3) uint8 BGR

        Returns
        -------
        (H, W) float32 — 1.0 where excluded classes detected, 0.0 elsewhere.
        All-zeros if face parser is disabled (no-op).
        """
        if not self._enabled:
            h, w = aligned_crop.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        h, w = aligned_crop.shape[:2]
        self._frame_count += 1

        # Return cached mask on skipped frames (hair moves slowly).
        if self._stride > 1 and self._frame_count % self._stride != 0:
            if self._cached_mask is not None and self._cached_mask.shape == (h, w):
                return self._cached_mask
            return np.zeros((h, w), dtype=np.float32)

        size = self._input_size

        # Resize, BGR → RGB, normalise
        resized  = cv2.resize(aligned_crop, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb      = resized[:, :, ::-1].astype(np.float32) / 255.0
        norm     = (rgb - _MEAN) / _STD
        tensor   = norm.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)

        logits = self._sess.run(None, {self._input_name: tensor})[0]  # (1, 19, H, W)
        classes = logits[0].argmax(axis=0).astype(np.uint8)           # (H, W)

        # Build binary mask for excluded classes
        excl_mask = np.zeros_like(classes, dtype=np.float32)
        for cls in self._exclude:
            excl_mask[classes == cls] = 1.0

        # Skip if nearly nothing detected (saves warp + blend cost)
        if excl_mask.mean() < _MIN_COVERAGE:
            return np.zeros((h, w), dtype=np.float32)

        # Feather for smooth blending
        if self._feather_r > 0:
            ksize     = self._feather_r * 2 + 1
            excl_mask = cv2.GaussianBlur(excl_mask, (ksize, ksize), 0)

        # Resize back to crop resolution
        if excl_mask.shape != (h, w):
            excl_mask = cv2.resize(excl_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        result = np.clip(excl_mask, 0.0, 1.0).astype(np.float32)
        self._cached_mask = result
        return result

    def get_hair_mask_fullframe(
        self,
        aligned_crop: np.ndarray,
        M: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> np.ndarray:
        """Warp the hair/accessory mask to full-frame coordinates.

        Returns (H, W) float32 — all-zeros if disabled.
        """
        fh, fw = frame_shape[:2]
        if not self._enabled:
            return np.zeros((fh, fw), dtype=np.float32)

        hair_crop = self.get_hair_mask_crop(aligned_crop)
        if hair_crop.max() < _MIN_COVERAGE:
            return np.zeros((fh, fw), dtype=np.float32)

        M_inv      = cv2.invertAffineTransform(M)
        hair_full  = cv2.warpAffine(
            hair_crop, M_inv, (fw, fh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return np.clip(hair_full, 0.0, 1.0).astype(np.float32)
