"""Tests for hand occlusion mask geometry.

We test `landmarks_to_mask` directly — the pure function that converts
normalised landmark lists to a float32 mask. No MediaPipe needed.
HandMasker itself is tested with a lightweight smoke import check.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.occlusion.hand_masker import landmarks_to_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _square_hand(cx: float = 0.5, cy: float = 0.5, half: float = 0.1) -> list[tuple[float, float]]:
    """Return 21 fake landmarks arranged in a square (for convex hull testing)."""
    pts: list[tuple[float, float]] = []
    for dx in (-half, 0.0, half):
        for dy in (-half, 0.0, half):
            pts.append((cx + dx, cy + dy))
    while len(pts) < 21:
        pts.append((cx, cy))
    return pts


H, W = 480, 640


# ---------------------------------------------------------------------------
# No-hand / empty input
# ---------------------------------------------------------------------------

def test_empty_landmarks_returns_zero_mask():
    mask = landmarks_to_mask(H, W, [])
    assert mask.shape == (H, W)
    assert mask.dtype == np.float32
    assert mask.max() == 0.0



# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_mask_shape_matches_h_w():
    mask = landmarks_to_mask(360, 640, [])
    assert mask.shape == (360, 640)


def test_mask_dtype_float32():
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=0)
    assert mask.dtype == np.float32


# ---------------------------------------------------------------------------
# Single hand — geometry
# ---------------------------------------------------------------------------

def test_hand_centred_in_frame_produces_nonzero_mask():
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=0)
    assert mask.max() > 0.0


def test_mask_values_in_range_sharp():
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=0)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_mask_values_in_range_feathered():
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=15)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_hand_at_top_left_corner():
    """Hand near a frame corner should still produce a nonzero mask."""
    corner = _square_hand(cx=0.05, cy=0.05, half=0.04)
    mask = landmarks_to_mask(H, W, [corner], feather_radius=0)
    assert mask.max() > 0.0


# ---------------------------------------------------------------------------
# Two hands
# ---------------------------------------------------------------------------

def test_two_hands_produce_larger_mask_than_one():
    left = _square_hand(cx=0.2, cy=0.5, half=0.1)
    right = _square_hand(cx=0.8, cy=0.5, half=0.1)

    mask_one = landmarks_to_mask(H, W, [left], feather_radius=0)
    mask_two = landmarks_to_mask(H, W, [left, right], feather_radius=0)

    assert mask_two.sum() > mask_one.sum()


def test_two_separate_hands_nonzero_in_both_regions():
    left = _square_hand(cx=0.1, cy=0.5, half=0.08)
    right = _square_hand(cx=0.9, cy=0.5, half=0.08)
    mask = landmarks_to_mask(H, W, [left, right], feather_radius=0)

    left_region = mask[:, :W // 4]
    right_region = mask[:, 3 * W // 4:]
    assert left_region.max() > 0.0
    assert right_region.max() > 0.0


# ---------------------------------------------------------------------------
# Feathering
# ---------------------------------------------------------------------------

def test_feather_zero_gives_binary_mask():
    """Without feathering, mask values should be exactly 0 or 1."""
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=0)
    unique = np.unique(mask)
    assert set(unique).issubset({0.0, 1.0})


def test_feather_nonzero_produces_intermediate_values():
    """Blurred mask must contain values strictly between 0 and 1."""
    mask = landmarks_to_mask(H, W, [_square_hand()], feather_radius=15)
    assert mask.min() < mask.max()
    interior = mask[(mask > 0.01) & (mask < 0.99)]
    assert len(interior) > 0, "feathered mask should have gradient pixels"


def test_feather_spreads_mask_wider_than_sharp():
    """Feathered mask should cover more pixels than the sharp version."""
    hand = _square_hand()
    sharp = landmarks_to_mask(H, W, [hand], feather_radius=0)
    soft = landmarks_to_mask(H, W, [hand], feather_radius=15)
    # nonzero pixel count grows with blur spread
    assert (soft > 0.0).sum() > (sharp > 0.0).sum()


# ---------------------------------------------------------------------------
# HandMasker import smoke test
# ---------------------------------------------------------------------------

def test_hand_masker_importable():
    """HandMasker must import cleanly even when the .task model is absent."""
    from src.occlusion.hand_masker import HandMasker  # noqa: F401
