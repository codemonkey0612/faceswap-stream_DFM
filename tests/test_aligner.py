"""Tests for face alignment and paste-back geometry."""

from __future__ import annotations

import numpy as np
import pytest

from src.swap.aligner import align_face, get_reference_landmarks, paste_back


LMKS_640x480 = (
    (220.0, 180.0),  # right eye
    (420.0, 180.0),  # left eye
    (320.0, 260.0),  # nose
    (240.0, 340.0),  # right mouth
    (400.0, 340.0),  # left mouth
)


def _fake_frame(h=480, w=640) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_align_returns_correct_size():
    frame = _fake_frame()
    aligned, M = align_face(frame, LMKS_640x480, output_size=256)
    assert aligned.shape == (256, 256, 3)
    assert M.shape == (2, 3)


def test_align_output_is_uint8():
    frame = _fake_frame()
    aligned, _ = align_face(frame, LMKS_640x480, output_size=256)
    assert aligned.dtype == np.uint8


def test_align_different_sizes():
    frame = _fake_frame()
    for size in (128, 224, 256, 512):
        aligned, M = align_face(frame, LMKS_640x480, output_size=size)
        assert aligned.shape == (size, size, 3)


def test_reference_landmarks_scale():
    ref_256 = get_reference_landmarks(256)
    ref_128 = get_reference_landmarks(128)
    np.testing.assert_allclose(ref_256, ref_128 * 2, rtol=1e-5)


def test_paste_back_shape_preserved():
    frame = _fake_frame()
    aligned, M = align_face(frame, LMKS_640x480, output_size=256)
    mask = np.ones((256, 256), dtype=np.float32)
    result = paste_back(frame, aligned, mask, M)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_paste_back_zero_mask_returns_original():
    frame = _fake_frame()
    aligned, M = align_face(frame, LMKS_640x480, output_size=256)
    mask_zero = np.zeros((256, 256), dtype=np.float32)
    result = paste_back(frame, aligned, mask_zero, M)
    np.testing.assert_array_equal(result, frame)


def test_paste_back_full_mask_differs_from_original():
    """Full-mask paste-back should change at least some pixels."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    # Use a distinct colour for the "swapped" crop.
    swapped = np.full((256, 256, 3), 200, dtype=np.uint8)
    _, M = align_face(frame, LMKS_640x480, output_size=256)
    mask_full = np.ones((256, 256), dtype=np.float32)
    result = paste_back(frame, swapped, mask_full, M)
    assert not np.array_equal(result, frame), "paste-back with full mask must change pixels"


def test_align_minimal_landmarks():
    """Degenerate case: if alignment fails, fallback crop is returned."""
    frame = _fake_frame()
    bad_lmks = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.2, 1.5), (0.8, 1.5))
    aligned, M = align_face(frame, bad_lmks, output_size=256)
    assert aligned.shape == (256, 256, 3)
