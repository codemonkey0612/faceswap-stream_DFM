"""Tests for the Monitor's pre-swap and post-composite checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.failsafe.monitor import Detection, Monitor, MonitorConfig
from src.failsafe.triggers import PostCompositeTrigger, PreSwapTrigger


# ---------- pre-swap ----------------------------------------------------------


def test_pre_swap_no_detections_fires_no_face(frame_shape):
    m = Monitor()
    r = m.check_pre_swap(frame_shape, detections=[])
    assert r.fired and r.trigger is PreSwapTrigger.NO_FACE


def test_pre_swap_detector_exception_fires_detector_error(frame_shape):
    m = Monitor()
    r = m.check_pre_swap(frame_shape, detections=None, detector_error=RuntimeError("gpu oom"))
    assert r.fired and r.trigger is PreSwapTrigger.DETECTOR_ERROR
    assert r.details["exc_type"] == "RuntimeError"


def test_pre_swap_low_confidence_fires(frame_shape):
    m = Monitor(MonitorConfig(min_confidence=0.7))
    det = Detection(bbox=(10, 10, 100, 100), confidence=0.5)
    r = m.check_pre_swap(frame_shape, detections=[det])
    assert r.fired and r.trigger is PreSwapTrigger.LOW_CONFIDENCE


def test_pre_swap_tiny_face_fires(frame_shape):
    m = Monitor(MonitorConfig(min_confidence=0.7, min_face_area_ratio=0.05))
    h, w = frame_shape[0], frame_shape[1]
    # 2x2 face => area 4, ratio << 0.05
    det = Detection(bbox=(0, 0, 2, 2), confidence=0.9)
    r = m.check_pre_swap(frame_shape, detections=[det])
    assert r.fired and r.trigger is PreSwapTrigger.TINY_FACE


def test_pre_swap_bbox_out_of_frame_fires(frame_shape):
    m = Monitor()
    h, w = frame_shape[0], frame_shape[1]
    det = Detection(bbox=(-10, 10, 50, 50), confidence=0.9)
    r = m.check_pre_swap(frame_shape, detections=[det])
    assert r.fired and r.trigger is PreSwapTrigger.BBOX_OUT_OF_FRAME


def test_pre_swap_bbox_past_right_edge_fires(frame_shape):
    # Use w+10: clearly outside frame, not subpixel jitter (tolerance is 4px).
    m = Monitor()
    h, w = frame_shape[0], frame_shape[1]
    det = Detection(bbox=(10, 10, w + 10, 50), confidence=0.9)
    r = m.check_pre_swap(frame_shape, detections=[det])
    assert r.fired and r.trigger is PreSwapTrigger.BBOX_OUT_OF_FRAME


def test_pre_swap_happy_path_passes(frame_shape):
    m = Monitor()
    h, w = frame_shape[0], frame_shape[1]
    det = Detection(bbox=(20, 20, w - 20, h - 20), confidence=0.95)
    r = m.check_pre_swap(frame_shape, detections=[det])
    assert r.fired is False


def test_pre_swap_picks_best_confidence(frame_shape):
    m = Monitor()
    bad = Detection(bbox=(-5, 10, 50, 50), confidence=0.3)
    good = Detection(bbox=(20, 20, 100, 100), confidence=0.95)
    r = m.check_pre_swap(frame_shape, detections=[bad, good])
    assert r.fired is False


# ---------- post-composite ----------------------------------------------------


def _valid_inputs(frame_shape):
    h, w = frame_shape[0], frame_shape[1]
    composited = np.full(frame_shape, 128, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:100, 20:100] = 255
    det_bbox = (20.0, 20.0, 100.0, 100.0)
    swap_bbox = (20.0, 20.0, 100.0, 100.0)
    return composited, mask, det_bbox, swap_bbox


def test_post_composite_happy_path_passes(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    r = m.check_post_composite(composited, mask, det, swap, frame_shape)
    assert r.fired is False


def test_post_composite_compositor_error_fires(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    r = m.check_post_composite(
        composited, mask, det, swap, frame_shape,
        compositor_error=ValueError("blend failed"),
    )
    assert r.fired and r.trigger is PostCompositeTrigger.COMPOSITOR_ERROR


def test_post_composite_shape_mismatch_fires(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    wrong = composited[:-10]  # trimmed height
    r = m.check_post_composite(wrong, mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.MASK_SHAPE_MISMATCH


def test_post_composite_mask_shape_mismatch_fires(frame_shape):
    m = Monitor()
    composited, _, det, swap = _valid_inputs(frame_shape)
    bad_mask = np.zeros((10, 10), dtype=np.uint8)
    r = m.check_post_composite(composited, bad_mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.MASK_SHAPE_MISMATCH


def test_post_composite_empty_mask_fires(frame_shape):
    m = Monitor()
    composited, _, det, swap = _valid_inputs(frame_shape)
    empty_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    r = m.check_post_composite(composited, empty_mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.EMPTY_MASK


def test_post_composite_nan_fires(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    nan_frame = composited.astype(np.float32)
    nan_frame[0, 0, 0] = np.nan
    r = m.check_post_composite(nan_frame, mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.NAN_OR_INF


def test_post_composite_inf_fires(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    bad = composited.astype(np.float32)
    bad[5, 5, 0] = np.inf
    r = m.check_post_composite(bad, mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.NAN_OR_INF


def test_post_composite_color_domain_error_fires(frame_shape):
    m = Monitor()
    composited, mask, det, swap = _valid_inputs(frame_shape)
    bad = composited.astype(np.float32) * 2.0  # peaks at 256
    r = m.check_post_composite(bad, mask, det, swap, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.COLOR_DOMAIN_ERROR


def test_post_composite_swap_bbox_divergence_fires(frame_shape):
    m = Monitor(MonitorConfig(max_swap_bbox_divergence_ratio=0.2))
    composited, mask, det, _ = _valid_inputs(frame_shape)
    # det bbox 80x80; longer edge = 80; 20% -> 16. Push swap bbox center 40 px away.
    swap_far = (60.0, 60.0, 140.0, 140.0)
    r = m.check_post_composite(composited, mask, det, swap_far, frame_shape)
    assert r.fired and r.trigger is PostCompositeTrigger.SWAP_BBOX_DIVERGENCE
