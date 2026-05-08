"""Failsafe layer — enforces R-S1: the face mask must never drop.

Hard rule: the virtual camera is written from exactly one call site —
`Gate.write()` in `gate.py`. No other module may construct or call pyvirtualcam.
"""

from .gate import Gate
from .monitor import Detection, Monitor
from .triggers import (
    PostCompositeTrigger,
    PreSwapTrigger,
    Trigger,
    TriggerResult,
)

__all__ = [
    "Detection",
    "Gate",
    "Monitor",
    "PostCompositeTrigger",
    "PreSwapTrigger",
    "Trigger",
    "TriggerResult",
]
