"""Tests for the DST face augmentation functions in scripts/augment_dst_faces.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

# The script lives in scripts/, not a package — add it to sys.path for import.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from augment_dst_faces import (  # noqa: E402
    _brightness,
    _contrast,
    _flip,
    _gamma,
    _hue_shift,
    _jpeg_quality,
    _noise,
    _perspective,
    _rotate,
    _saturation,
    _scale_crop,
    augment,
)


def _img(h: int = 128, w: int = 96, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Individual augmentation transforms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    _brightness, _contrast, _saturation, _hue_shift, _gamma,
    _noise, _jpeg_quality, _rotate, _flip, _scale_crop, _perspective,
])
def test_transform_preserves_shape(fn):
    img = _img()
    out = fn(img, _rng())
    assert out.shape == img.shape, f"{fn.__name__} changed shape"


@pytest.mark.parametrize("fn", [
    _brightness, _contrast, _saturation, _hue_shift, _gamma,
    _noise, _jpeg_quality, _rotate, _flip, _scale_crop, _perspective,
])
def test_transform_preserves_dtype(fn):
    img = _img()
    out = fn(img, _rng())
    assert out.dtype == np.uint8, f"{fn.__name__} changed dtype to {out.dtype}"


@pytest.mark.parametrize("fn", [
    _brightness, _contrast, _saturation, _hue_shift, _gamma,
    _noise, _jpeg_quality, _rotate, _flip, _scale_crop, _perspective,
])
def test_transform_values_in_range(fn):
    img = _img()
    out = fn(img, _rng())
    assert out.min() >= 0 and out.max() <= 255


def test_flip_is_horizontal():
    """Flipped image columns should mirror the original."""
    img = _img()
    out = _flip(img, _rng())
    np.testing.assert_array_equal(out, img[:, ::-1])


def test_brightness_darkens_with_low_factor():
    """With a very low factor the image gets darker."""
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    rng = _rng()
    rng.random = lambda: 0.0  # won't be called; we use uniform
    # Override uniform to always return 0.65
    orig_uniform = rng.uniform
    rng.uniform = lambda a, b: 0.65
    out = _brightness(img, rng)
    rng.uniform = orig_uniform
    assert out.mean() < img.mean()


# ---------------------------------------------------------------------------
# Full augment pipeline
# ---------------------------------------------------------------------------

def test_augment_shape_preserved():
    img = _img(h=256, w=192)
    out = augment(img, _rng())
    assert out.shape == img.shape


def test_augment_dtype_uint8():
    img = _img()
    out = augment(img, _rng())
    assert out.dtype == np.uint8


def test_augment_different_seeds_produce_different_results():
    img = _img()
    out1 = augment(img, random.Random(1))
    out2 = augment(img, random.Random(999))
    assert not np.array_equal(out1, out2)


def test_augment_same_seed_is_reproducible():
    img = _img()
    out1 = augment(img, random.Random(42))
    out2 = augment(img, random.Random(42))
    np.testing.assert_array_equal(out1, out2)


def test_augment_modifies_image():
    """augment should produce a result different from the input."""
    img = _img(seed=7)
    # Run multiple seeds — at least one must differ from input.
    any_different = any(
        not np.array_equal(augment(img, random.Random(s)), img)
        for s in range(10)
    )
    assert any_different, "augment never changed the image"
