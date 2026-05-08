"""DFM model loader — wraps an ONNX Runtime InferenceSession.

A .dfm file is an ONNX model (just renamed). DeepFaceLive models typically
have two outputs: the swapped face and a blend mask. Exact input/output names
vary by model version; we probe them at load time.

When no model path is provided (or the file doesn't exist), DFMLoader returns
None and the swapper falls back to an identity pass-through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import structlog


_PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


@dataclass(frozen=True)
class DFMModelInfo:
    input_name: str
    input_size: int          # square: input_size × input_size
    output_face_name: str    # swapped face output
    output_mask_name: str    # blend mask output  ('' if absent)


class DFMLoader:
    def __init__(
        self,
        model_path: str | Path | None,
        providers: list[str] | None = None,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self._log = logger or structlog.get_logger("swap.dfm_loader")
        self._session: ort.InferenceSession | None = None
        self._info: DFMModelInfo | None = None

        if model_path is None:
            self._log.info("dfm_loader_no_model", note="identity pass-through active")
            return

        path = Path(model_path)
        if not path.exists():
            self._log.warning("dfm_model_not_found", path=str(path),
                              note="identity pass-through active")
            return

        self._load(path, providers or _PREFERRED_PROVIDERS)

    def _load(self, path: Path, providers: list[str]) -> None:
        # Filter to providers actually available on this machine.
        available = list(ort.get_all_providers())
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]

        try:
            session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            self._log.error("dfm_load_failed", path=str(path), exc=str(exc))
            return

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        if not inputs:
            self._log.error("dfm_no_inputs", path=str(path))
            return

        input_name = inputs[0].name
        input_shape = inputs[0].shape   # e.g. [1, 3, 256, 256]
        input_size = int(input_shape[-1]) if len(input_shape) >= 2 else 256

        # Identify face and mask output names by convention.
        output_names = [o.name for o in outputs]
        face_name = output_names[0] if output_names else ""
        mask_name = output_names[1] if len(output_names) > 1 else ""

        self._session = session
        self._info = DFMModelInfo(
            input_name=input_name,
            input_size=input_size,
            output_face_name=face_name,
            output_mask_name=mask_name,
        )
        self._log.info(
            "dfm_loaded",
            path=str(path),
            input=(input_name, input_shape),
            outputs=output_names,
            providers=session.get_providers(),
        )

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def info(self) -> DFMModelInfo | None:
        return self._info

    def run(self, face_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on an aligned face crop.

        Parameters
        ----------
        face_crop : (H, W, 3) uint8 BGR aligned crop.

        Returns
        -------
        swapped   : (H, W, 3) uint8 BGR swapped face.
        mask      : (H, W) float32 blend mask in [0, 1].
        """
        assert self._session is not None and self._info is not None
        info = self._info

        h, w = face_crop.shape[:2]

        # Preprocess: BGR → RGB, HWC → NCHW, normalize to [0, 1].
        rgb = face_crop[:, :, ::-1].astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)

        outputs = self._session.run(None, {info.input_name: blob})

        # Decode swapped face: NCHW float [0,1] → HWC uint8 BGR.
        face_out = outputs[0][0]                     # (3, H, W)
        face_out = np.clip(face_out, 0.0, 1.0)
        face_bgr = (face_out.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

        # Decode mask: (1, 1, H, W) or (1, H, W) float → (H, W) float32.
        if len(outputs) > 1 and info.output_mask_name:
            mask_raw = outputs[1][0]
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[0]   # (1, H, W) → (H, W)
            mask = np.clip(mask_raw, 0.0, 1.0).astype(np.float32)
        else:
            # No mask output — use a solid elliptical mask.
            mask = _ellipse_mask(h, w)

        return face_bgr, mask


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Solid soft-edged ellipse mask for models without a mask output."""
    mask = np.zeros((h, w), dtype=np.float32)
    cv2_available = True
    try:
        import cv2
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (w // 2 - 4, h // 2 - 4),
                    angle=0, startAngle=0, endAngle=360,
                    color=1.0, thickness=-1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
    except Exception:
        mask[:] = 1.0
    return mask
