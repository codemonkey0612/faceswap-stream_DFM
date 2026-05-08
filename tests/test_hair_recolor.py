"""Tests for HairRecolor HSV channel transfer."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.occlusion.hair_recolor import HairRecolor

H, W = 480, 640


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


def _mask(fill: float = 0.0) -> np.ndarray:
    return np.full((H, W), fill, dtype=np.float32)


def _solid_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = bgr
    return frame


# ---------------------------------------------------------------------------
# disabled / no-op
# ---------------------------------------------------------------------------

def test_disabled_returns_identical_array():
    r = HairRecolor(enabled=False)
    frame = _frame()
    result = r.apply(frame, _mask(1.0))
    assert np.array_equal(result, frame)


def test_zero_mask_no_op():
    r = HairRecolor(enabled=True)
    frame = _frame()
    result = r.apply(frame, _mask(0.0))
    assert np.array_equal(result, frame)


def test_below_threshold_mask_no_op():
    r = HairRecolor(enabled=True, mask_threshold=0.05)
    frame = _frame()
    result = r.apply(frame, _mask(0.04))
    assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# output invariants
# ---------------------------------------------------------------------------

def test_output_shape_preserved():
    r = HairRecolor(enabled=True)
    result = r.apply(_frame(), _mask(1.0))
    assert result.shape == (H, W, 3)


def test_output_dtype_uint8():
    r = HairRecolor(enabled=True)
    result = r.apply(_frame(), _mask(1.0))
    assert result.dtype == np.uint8


def test_output_values_in_range():
    r = HairRecolor(enabled=True)
    result = r.apply(_frame(), _mask(1.0))
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_identity_on_black_frame():
    r = HairRecolor(enabled=True, target_bgr=(0, 200, 50))
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    result = r.apply(frame, _mask(1.0))
    assert result.dtype == np.uint8
    assert np.all(np.isfinite(result.astype(np.float32)))


# ---------------------------------------------------------------------------
# correctness — hair region changes, non-hair unchanged
# ---------------------------------------------------------------------------

def test_only_masked_region_changes():
    """Right half has mask=0 → must be pixel-identical to input after recolor."""
    r = HairRecolor(enabled=True, target_bgr=(0, 200, 50), blend=1.0)
    frame = _frame(seed=7)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, : W // 2] = 1.0   # left half = hair

    result = r.apply(frame, mask)
    np.testing.assert_array_equal(result[:, W // 2 :], frame[:, W // 2 :])


def test_full_mask_produces_color_shift():
    """Bright saturated input with a very different target color must differ from input."""
    r = HairRecolor(enabled=True, target_bgr=(200, 0, 0), blend=1.0)  # blue target
    frame = _solid_frame((0, 0, 200))  # solid red input
    result = r.apply(frame, _mask(1.0))
    assert not np.array_equal(result, frame)


def test_value_channel_preserved():
    """V (brightness) must not change by more than 1 (uint8 round-trip) in active pixels."""
    r = HairRecolor(enabled=True, target_bgr=(0, 200, 50), blend=1.0)
    frame = _frame(seed=42)
    mask = _mask(1.0)

    result = r.apply(frame, mask)

    frame_hsv  = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    v_diff = np.abs(frame_hsv[:, :, 2].astype(np.int16) - result_hsv[:, :, 2].astype(np.int16))
    assert int(v_diff.max()) <= 1


# ---------------------------------------------------------------------------
# blend=1.0: saturation should reach target
# ---------------------------------------------------------------------------

def test_blend_one_sets_saturation_to_target():
    """With blend=1.0 and high-saturation input, S in active pixels ≈ target_S."""
    target_bgr = (0, 255, 0)   # pure green: S=255 in OpenCV HSV
    r = HairRecolor(enabled=True, target_bgr=target_bgr, blend=1.0, min_saturation=0)

    # Bright red frame — fully saturated, different hue
    frame = _solid_frame((0, 0, 255))
    result = r.apply(frame, _mask(1.0))

    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    s_vals = result_hsv[:, :, 1].astype(np.float32)

    # Compute expected target S
    px = np.array([[[*target_bgr]]], dtype=np.uint8)
    target_s = int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 1])

    # Tolerance: 2 for uint8 round-trip + alpha blend edge effects
    assert abs(int(s_vals.mean()) - target_s) <= 2


# ---------------------------------------------------------------------------
# dark hair / min_saturation gating
# ---------------------------------------------------------------------------

def test_dark_hair_saturation_gating():
    """Near-black input (S≈5) with bright target (S=255) and min_saturation=30
    should produce much less than 255 S in output due to gating."""
    r = HairRecolor(
        enabled=True,
        target_bgr=(0, 255, 0),  # green, S=255
        blend=1.0,
        min_saturation=30,
    )
    # Near-black frame: B=15, G=15, R=15 → S ≈ 0 in HSV
    frame = _solid_frame((15, 15, 15))
    result = r.apply(frame, _mask(1.0))

    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    s_mean = float(result_hsv[:, :, 1].mean())
    # Gating: eff_blend ≈ 1.0 * (5/30) ≈ 0.17 → S should be well below 255
    assert s_mean < 50


def test_min_saturation_zero_disables_gating():
    """min_saturation=0 disables dark-hair gating; S shifts fully toward target."""
    r = HairRecolor(
        enabled=True,
        target_bgr=(0, 255, 0),  # S=255
        blend=1.0,
        min_saturation=0,
    )
    # Bright saturated red input (S≈255)
    frame = _solid_frame((0, 0, 255))
    result = r.apply(frame, _mask(1.0))

    result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    s_mean = float(result_hsv[:, :, 1].mean())
    # With no gating and blend=1, S should hit target (~255), allowing 5 tolerance
    assert s_mean > 250


# ---------------------------------------------------------------------------
# target BGR → HSV conversion at init
# ---------------------------------------------------------------------------

def test_target_bgr_pure_red_stores_correct_hsv():
    """Pure red BGR=(0,0,255) → OpenCV HSV H=0, S=255."""
    r = HairRecolor(target_bgr=(0, 0, 255), enabled=False)
    assert r._target_h == 0
    assert r._target_s == 255


def test_target_bgr_pure_blue_stores_correct_hsv():
    """Pure blue BGR=(255,0,0) → OpenCV HSV H=120, S=255."""
    r = HairRecolor(target_bgr=(255, 0, 0), enabled=False)
    assert r._target_h == 120
    assert r._target_s == 255


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_invalid_target_bgr_length_raises():
    with pytest.raises(ValueError, match="exactly 3"):
        HairRecolor(target_bgr=(0, 0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parametric: shape/dtype hold for all blend values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blend", [0.0, 0.5, 1.0])
def test_parametrize_blend_shape_dtype(blend: float):
    r = HairRecolor(enabled=True, target_bgr=(50, 100, 150), blend=blend)
    result = r.apply(_frame(), _mask(1.0))
    assert result.shape == (H, W, 3)
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255
