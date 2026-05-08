"""Failsafe trigger definitions.

Every trigger here corresponds to a row in docs/failsafe.md. Adding a trigger
here is a contract change and requires a matching test in tests/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union


class PreSwapTrigger(str, Enum):
    """Checked immediately after face detection, before running the DFM."""

    NO_FACE = "no_face"
    LOW_CONFIDENCE = "low_confidence"
    TINY_FACE = "tiny_face"
    BBOX_OUT_OF_FRAME = "bbox_out_of_frame"
    DETECTOR_ERROR = "detector_error"


class PostCompositeTrigger(str, Enum):
    """Checked on the composited frame just before virtual-camera write."""

    EMPTY_MASK = "empty_mask"
    MASK_SHAPE_MISMATCH = "mask_shape_mismatch"
    SWAP_BBOX_DIVERGENCE = "swap_bbox_divergence"
    NAN_OR_INF = "nan_or_inf"
    COLOR_DOMAIN_ERROR = "color_domain_error"
    COMPOSITOR_ERROR = "compositor_error"


Trigger = Union[PreSwapTrigger, PostCompositeTrigger]


@dataclass(frozen=True)
class TriggerResult:
    """Outcome of a monitor check.

    `fired=False` means the frame is green-lit for this check.
    `fired=True` means the gate must emit a black frame.
    """

    fired: bool
    trigger: Trigger | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_through(cls) -> TriggerResult:
        return cls(fired=False)

    @classmethod
    def block(cls, trigger: Trigger, **details: Any) -> TriggerResult:
        return cls(fired=True, trigger=trigger, details=dict(details))
