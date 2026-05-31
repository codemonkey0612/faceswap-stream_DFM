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
    nhwc: bool = True        # True = (N,H,W,C) layout (DeepFaceLive DFM); False = (N,C,H,W)


# DeepFaceLive DFM output names (exported by DFL's "export DFM"):
#   in_face:0  ->  out_face_mask:0, out_celeb_face:0, out_celeb_face_mask:0
# The swapped (celebrity) face is out_celeb_face; its blend mask is
# out_celeb_face_mask. Note these are NOT outputs[0]/[1] — select by name.
_DFL_FACE_OUTPUT = "out_celeb_face"
_DFL_MASK_OUTPUT = "out_celeb_face_mask"


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
        input_shape = inputs[0].shape   # DFL DFM: [N, 256, 256, 3] (NHWC)

        # Detect layout from the channel position. NHWC if the last dim is 3
        # (DeepFaceLive convention); NCHW if the second dim is 3.
        nhwc = not (len(input_shape) == 4 and input_shape[1] == 3)

        # Spatial size is H (square). NHWC -> shape[1]; NCHW -> shape[2].
        size_dim = input_shape[1] if nhwc else input_shape[2]
        input_size = int(size_dim) if isinstance(size_dim, int) and size_dim > 0 else 256

        # Select face/mask outputs BY NAME (DFL DFM order is mask-first, so
        # index-based selection picks the wrong tensor). Fall back to index.
        output_names = [o.name for o in outputs]
        face_name = next((n for n in output_names if _DFL_FACE_OUTPUT in n), "")
        mask_name = next((n for n in output_names if _DFL_MASK_OUTPUT in n), "")
        if not face_name:
            # Non-DFL model: assume first output is the face, second the mask.
            face_name = output_names[0] if output_names else ""
            mask_name = output_names[1] if len(output_names) > 1 else ""

        self._session = session
        self._info = DFMModelInfo(
            input_name=input_name,
            input_size=input_size,
            output_face_name=face_name,
            output_mask_name=mask_name,
            nhwc=nhwc,
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

        # DeepFaceLive DFM uses BGR [0,1] for BOTH input and output — no channel
        # flip. NHWC (1, H, W, 3) for DFL models; legacy NCHW gets the transpose.
        inp = face_crop.astype(np.float32) / 255.0
        if info.nhwc:
            blob = inp[np.newaxis]                  # (1, H, W, 3)
        else:
            blob = inp.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)

        # Request only the tensors we need, by name when available.
        out_names = [n for n in (info.output_face_name, info.output_mask_name) if n]
        results = self._session.run(out_names or None, {info.input_name: blob})
        outputs = dict(zip(out_names, results)) if out_names else None

        # --- Decode swapped face → HWC uint8 BGR ---
        face_out = outputs[info.output_face_name] if outputs else results[0]
        face_out = np.asarray(face_out)[0]          # drop batch dim
        if not info.nhwc:                           # (3, H, W) → (H, W, 3)
            face_out = face_out.transpose(1, 2, 0)
        face_out = np.clip(face_out, 0.0, 1.0)
        face_bgr = (face_out * 255).astype(np.uint8)   # already BGR

        # --- Decode blend mask → (H, W) float32, feathered ---
        if outputs is not None and info.output_mask_name:
            mask_raw = np.asarray(outputs[info.output_mask_name])[0]  # (H,W,1) or (1,H,W)
            mask = np.clip(np.squeeze(mask_raw), 0.0, 1.0).astype(np.float32)
            mask = _feather_mask(mask)   # soften the hard DFM edge to avoid a seam
        else:
            mask = _ellipse_mask(h, w)

        return face_bgr, mask


def _feather_mask(mask: np.ndarray, erode_frac: float = 0.04, blur_frac: float = 0.06) -> np.ndarray:
    """Erode then Gaussian-blur the DFM mask so paste-back has a soft edge.

    The raw out_celeb_face_mask has a hard, near-rectangular boundary that
    shows as a seam when composited. Eroding pulls the edge inward onto the
    face; blurring fades it. Fractions are relative to the crop height so this
    scales with input_size.
    """
    try:
        import cv2
        h = mask.shape[0]
        e = max(1, int(h * erode_frac))
        b = max(1, int(h * blur_frac))
        b = b + 1 if b % 2 == 0 else b          # Gaussian kernel must be odd
        m = cv2.erode(mask, np.ones((e, e), np.float32), iterations=1)
        m = cv2.GaussianBlur(m, (b, b), 0)
        return np.clip(m, 0.0, 1.0).astype(np.float32)
    except Exception:
        return mask


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    """Solid soft-edged ellipse mask for models without a mask output."""
    mask = np.zeros((h, w), dtype=np.float32)
    try:
        import cv2
        cx, cy = w // 2, h // 2
        cv2.ellipse(mask, (cx, cy), (w // 2 - 4, h // 2 - 4),
                    angle=0, startAngle=0, endAngle=360,
                    color=1.0, thickness=-1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
    except Exception as exc:
        structlog.get_logger("swap.dfm_loader").warning(
            "ellipse_mask_fallback", exc=str(exc), note="using solid mask"
        )
        mask[:] = 1.0
    return mask
