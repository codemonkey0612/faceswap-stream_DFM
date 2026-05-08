"""Tests for trigger enums and TriggerResult construction."""

from __future__ import annotations

from src.failsafe.triggers import (
    PostCompositeTrigger,
    PreSwapTrigger,
    TriggerResult,
)


def test_pass_through_does_not_fire():
    r = TriggerResult.pass_through()
    assert r.fired is False
    assert r.trigger is None
    assert r.details == {}


def test_block_carries_trigger_and_details():
    r = TriggerResult.block(PreSwapTrigger.LOW_CONFIDENCE, confidence=0.42, threshold=0.7)
    assert r.fired is True
    assert r.trigger is PreSwapTrigger.LOW_CONFIDENCE
    assert r.details == {"confidence": 0.42, "threshold": 0.7}


def test_trigger_enums_are_exhaustive():
    # Sanity: every trigger in the contract is present.
    pre = {t.value for t in PreSwapTrigger}
    post = {t.value for t in PostCompositeTrigger}
    assert pre == {
        "no_face",
        "low_confidence",
        "tiny_face",
        "bbox_out_of_frame",
        "detector_error",
    }
    assert post == {
        "empty_mask",
        "mask_shape_mismatch",
        "swap_bbox_divergence",
        "nan_or_inf",
        "color_domain_error",
        "compositor_error",
    }


def test_trigger_result_is_frozen():
    r = TriggerResult.pass_through()
    try:
        r.fired = True  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("TriggerResult should be immutable")
