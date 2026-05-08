"""Failsafe Gate — the ONLY code path that writes to the virtual camera.

Defense-in-depth: even if every monitor check passes, the gate still verifies
frame shape and dtype and blacks out on any inconsistency.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import structlog

from .triggers import (
    PostCompositeTrigger,
    TriggerResult,
)


class FrameSink(Protocol):
    """Minimal interface the Gate needs from a virtual-camera writer."""

    def send(self, frame: np.ndarray) -> None: ...


class Gate:
    """Single writer to the virtual camera. Enforces the black-frame failsafe."""

    def __init__(
        self,
        sink: FrameSink,
        frame_width: int,
        frame_height: int,
        logger: structlog.BoundLogger | None = None,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self._sink = sink
        self._w = frame_width
        self._h = frame_height
        self._log = logger or structlog.get_logger("failsafe.gate")
        self._black = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self._rate_limit_s = rate_limit_seconds
        self._last_log: dict[str, float] = {}
        self._frames_total = 0
        self._frames_blacked = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_total": self._frames_total,
            "frames_blacked": self._frames_blacked,
        }

    def write(
        self,
        candidate: np.ndarray | None,
        trigger_result: TriggerResult | None = None,
        frame_idx: int | None = None,
    ) -> None:
        """Emit exactly one frame to the sink.

        Order of checks:
          1. A fired upstream trigger -> BLACK.
          2. Candidate is None -> BLACK.
          3. Shape or dtype mismatch -> BLACK.
          4. Final NaN/Inf sanity scan -> BLACK.
          5. Otherwise -> candidate.
        """
        self._frames_total += 1

        if trigger_result is not None and trigger_result.fired:
            self._emit_black(trigger_result, frame_idx)
            return

        if candidate is None:
            self._emit_black(
                TriggerResult.block(
                    PostCompositeTrigger.COMPOSITOR_ERROR,
                    reason="candidate_is_none",
                ),
                frame_idx,
            )
            return

        if candidate.shape != (self._h, self._w, 3):
            self._emit_black(
                TriggerResult.block(
                    PostCompositeTrigger.MASK_SHAPE_MISMATCH,
                    expected=(self._h, self._w, 3),
                    actual=tuple(candidate.shape),
                ),
                frame_idx,
            )
            return

        if candidate.dtype != np.uint8:
            self._emit_black(
                TriggerResult.block(
                    PostCompositeTrigger.COLOR_DOMAIN_ERROR,
                    reason="dtype_not_uint8",
                    dtype=str(candidate.dtype),
                ),
                frame_idx,
            )
            return

        # Cheap NaN guard — uint8 cannot be NaN, but catch upstream bugs that
        # mutate dtype silently.
        if candidate.dtype.kind == "f" and not np.all(np.isfinite(candidate)):
            self._emit_black(
                TriggerResult.block(PostCompositeTrigger.NAN_OR_INF),
                frame_idx,
            )
            return

        self._sink.send(candidate)

    def emit_black(self, reason: TriggerResult | None = None, frame_idx: int | None = None) -> None:
        """Explicit black-frame emission — used by the pipeline when capture stalls."""
        self._frames_total += 1
        self._emit_black(
            reason
            or TriggerResult.block(
                PostCompositeTrigger.COMPOSITOR_ERROR, reason="explicit_black"
            ),
            frame_idx,
        )

    def _emit_black(self, trigger_result: TriggerResult, frame_idx: int | None) -> None:
        self._frames_blacked += 1
        self._sink.send(self._black)
        self._log_trigger(trigger_result, frame_idx)

    def _log_trigger(self, trigger_result: TriggerResult, frame_idx: int | None) -> None:
        if trigger_result.trigger is None:
            return
        key = trigger_result.trigger.value
        now = time.monotonic()
        last = self._last_log.get(key, 0.0)
        if now - last < self._rate_limit_s:
            return
        self._last_log[key] = now
        self._log.warning(
            "failsafe_fired",
            trigger=key,
            frame_idx=frame_idx,
            **trigger_result.details,
        )
