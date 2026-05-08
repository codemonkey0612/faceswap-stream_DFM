"""Application config loading — YAML + pydantic models.

Profile files are overlaid on top of config/default.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class CaptureConfig(BaseModel):
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 2


class VirtualCameraConfig(BaseModel):
    backend: str = "obs"
    fmt: str = "BGR"


class PreviewConfig(BaseModel):
    enable: bool = False


class OutputConfig(BaseModel):
    virtual_camera: VirtualCameraConfig = Field(default_factory=VirtualCameraConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)


class PreSwapThresholds(BaseModel):
    require_face: bool = True
    min_confidence: float = 0.70
    min_face_area_ratio: float = 0.005
    bbox_must_be_inside_frame: bool = True


class PostCompositeThresholds(BaseModel):
    require_nonzero_mask: bool = True
    require_swap_bbox_match: bool = True
    forbid_nan_pixels: bool = True


class FailsafeTriggerAction(BaseModel):
    action: str = "black_frame"


class FailsafeLogging(BaseModel):
    path: str = "logs/failsafe.log"


class FailsafeConfig(BaseModel):
    pre_swap: PreSwapThresholds = Field(default_factory=PreSwapThresholds)
    post_composite: PostCompositeThresholds = Field(default_factory=PostCompositeThresholds)
    on_trigger: FailsafeTriggerAction = Field(default_factory=FailsafeTriggerAction)
    logging: FailsafeLogging = Field(default_factory=FailsafeLogging)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class DetectionConfig(BaseModel):
    model: str = "models/face_detection_yunet_2023mar.onnx"
    confidence_threshold: float = 0.70
    nms_threshold: float = 0.30
    top_k: int = 100
    min_face_area_ratio: float = 0.005


class SwapConfig(BaseModel):
    dfm_path: str | None = None          # None → identity pass-through
    input_size: int = 256                 # crop size fed to the DFM model
    providers: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


class SmoothingConfig(BaseModel):
    method: str = "bilateral"
    d: int = 7                    # bilateral filter diameter
    sigma_color: float = 25.0
    sigma_space: float = 15.0


class BeautyConfig(BaseModel):
    enable: bool = True
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    swap: SwapConfig = Field(default_factory=SwapConfig)
    beauty: BeautyConfig = Field(default_factory=BeautyConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    failsafe: FailsafeConfig = Field(default_factory=FailsafeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Sections we don't model yet (detection, swap, occlusion, beauty, ...)
    # are permitted but not validated — they will be added as each module lands.
    model_config = {"extra": "allow"}


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(profile: str | None = None) -> AppConfig:
    default_path = CONFIG_DIR / "default.yaml"
    merged: dict[str, Any] = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}

    if profile:
        prof_path = CONFIG_DIR / "profiles" / f"{profile}.yaml"
        if not prof_path.exists():
            raise FileNotFoundError(f"profile not found: {prof_path}")
        overlay: dict[str, Any] = yaml.safe_load(prof_path.read_text(encoding="utf-8")) or {}
        merged = _deep_merge(merged, overlay)

    return AppConfig(**merged)
