"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def frame_shape() -> tuple[int, int, int]:
    # Small frame for fast tests; pipeline uses 1080p.
    return (120, 160, 3)


@pytest.fixture
def blank_frame(frame_shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(frame_shape, dtype=np.uint8)


@pytest.fixture
def gray_frame(frame_shape: tuple[int, int, int]) -> np.ndarray:
    return np.full(frame_shape, 128, dtype=np.uint8)


@pytest.fixture
def full_mask(frame_shape: tuple[int, int, int]) -> np.ndarray:
    return np.full(frame_shape[:2], 255, dtype=np.uint8)


class FakeSink:
    """In-memory virtual-camera stand-in for Gate tests."""

    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []

    def send(self, frame: np.ndarray) -> None:
        self.sent.append(frame.copy())


@pytest.fixture
def fake_sink() -> FakeSink:
    return FakeSink()
