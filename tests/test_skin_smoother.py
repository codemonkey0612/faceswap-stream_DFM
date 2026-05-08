"""Tests for the SkinSmoother bilateral filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.beauty.skin_smoother import SkinSmoother


H, W = 480, 640


def _frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


def _mask(fill: float = 0.0) -> np.ndarray:
    return np.full((H, W), fill, dtype=np.float32)


# ---------------------------------------------------------------------------
# disabled smoother
# ---------------------------------------------------------------------------

def test_disabled_returns_identical_array():
    s = SkinSmoother(enabled=False)
    frame = _frame()
    result = s.apply(frame, _mask(0.0))
    assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# output properties
# ---------------------------------------------------------------------------

def test_output_shape_preserved():
    s = SkinSmoother(enabled=True)
    result = s.apply(_frame(), _mask(0.0))
    assert result.shape == (H, W, 3)


def test_output_dtype_uint8():
    s = SkinSmoother(enabled=True)
    result = s.apply(_frame(), _mask(0.0))
    assert result.dtype == np.uint8


def test_output_values_in_range():
    s = SkinSmoother(enabled=True)
    result = s.apply(_frame(), _mask(0.0))
    assert result.min() >= 0
    assert result.max() <= 255


# ---------------------------------------------------------------------------
# swap-mask semantics
# ---------------------------------------------------------------------------

def test_full_swap_mask_returns_frame_unchanged():
    """When swap_mask is all-1 (100% swap), no pixels are smoothed → identical output."""
    s = SkinSmoother(enabled=True, d=7, sigma_color=25, sigma_space=15)
    frame = _frame()
    result = s.apply(frame, _mask(1.0))
    np.testing.assert_array_equal(result, frame)


def test_zero_mask_smooths_all_pixels():
    """When swap_mask is all-0 (nothing swapped), entire frame is bilaterally filtered."""
    s = SkinSmoother(enabled=True, d=7, sigma_color=25, sigma_space=15)
    frame = _frame()
    result = s.apply(frame, _mask(0.0))
    # Bilateral filter always changes at least some pixels of a random frame.
    assert not np.array_equal(result, frame)


def test_partial_mask_swap_region_unchanged():
    """Pixels inside the swap region must not be altered by the smoother."""
    s = SkinSmoother(enabled=True)
    frame = _frame(seed=99)

    # Left half = swap (mask=1), right half = non-swap (mask=0).
    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, : W // 2] = 1.0

    result = s.apply(frame, mask)
    np.testing.assert_array_equal(result[:, : W // 2], frame[:, : W // 2])


def test_partial_mask_non_swap_region_may_change():
    """Pixels outside the swap region should be smoothed (potentially changed)."""
    s = SkinSmoother(enabled=True)
    frame = _frame(seed=42)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[:, : W // 2] = 1.0  # right half = non-swap

    result = s.apply(frame, mask)
    # Right half should differ from original (bilateral filter applied).
    assert not np.array_equal(result[:, W // 2 :], frame[:, W // 2 :])
