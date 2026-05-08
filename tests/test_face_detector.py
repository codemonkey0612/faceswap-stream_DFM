"""Tests for FaceDetector.

Real model-based tests require models/face_detection_yunet_2023mar.onnx.
They are skipped automatically when the file is absent (CI / fresh clone).
Unit tests mock cv2.FaceDetectorYN.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.failsafe.monitor import Detection

MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")
MODEL_AVAILABLE = MODEL_PATH.exists()

pytestmark_needs_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="YuNet model not downloaded"
)


# ---------- unit tests (no real model needed) --------------------------------


def _make_mock_detector(faces_array: np.ndarray | None):
    mock_det = MagicMock()
    mock_det.detect.return_value = (None, faces_array)
    return mock_det


def _make_yunet_row(x, y, w, h, score) -> np.ndarray:
    row = np.zeros(15, dtype=np.float32)
    row[0], row[1], row[2], row[3] = x, y, w, h
    row[4], row[5] = x + w * 0.3, y + h * 0.3   # right eye
    row[6], row[7] = x + w * 0.7, y + h * 0.3   # left eye
    row[8], row[9] = x + w * 0.5, y + h * 0.5   # nose
    row[10], row[11] = x + w * 0.35, y + h * 0.75  # r mouth
    row[12], row[13] = x + w * 0.65, y + h * 0.75  # l mouth
    row[14] = score
    return row


@pytest.fixture
def mock_detector_factory():
    """Returns a factory that patches cv2.FaceDetectorYN.create with a given faces array."""
    def _factory(faces_array):
        patcher = patch("src.detection.face_detector.cv2.FaceDetectorYN")
        mock_cls = patcher.start()
        mock_cls.create.return_value = _make_mock_detector(faces_array)
        return patcher
    return _factory


def test_no_faces_returns_empty_list(tmp_path, mock_detector_factory):
    dummy_model = tmp_path / "yunet.onnx"
    dummy_model.write_bytes(b"\x00" * 16)
    patcher = mock_detector_factory(None)
    try:
        from src.detection.face_detector import FaceDetector
        fd = FaceDetector(dummy_model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert fd.detect(frame) == []
    finally:
        patcher.stop()


def test_single_face_parsed_correctly(tmp_path, mock_detector_factory):
    dummy_model = tmp_path / "yunet.onnx"
    dummy_model.write_bytes(b"\x00" * 16)
    row = _make_yunet_row(50, 30, 200, 250, 0.92)
    faces = row[np.newaxis, :]   # shape (1, 15)
    patcher = mock_detector_factory(faces)
    try:
        from importlib import reload
        import src.detection.face_detector as mod
        reload(mod)  # pick up fresh patch
        from src.detection.face_detector import FaceDetector
        fd = FaceDetector(dummy_model)
        dets = fd.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 1
        d = dets[0]
        assert abs(d.confidence - 0.92) < 1e-4
        # bbox should be (x1, y1, x2, y2)
        assert d.bbox == (50.0, 30.0, 250.0, 280.0)
        assert len(d.landmarks) == 5
    finally:
        patcher.stop()


def test_detections_sorted_by_confidence_descending(tmp_path, mock_detector_factory):
    dummy_model = tmp_path / "yunet.onnx"
    dummy_model.write_bytes(b"\x00" * 16)
    rows = np.array([
        _make_yunet_row(10, 10, 100, 100, 0.75),
        _make_yunet_row(200, 10, 100, 100, 0.95),
        _make_yunet_row(10, 200, 100, 100, 0.60),
    ])
    patcher = mock_detector_factory(rows)
    try:
        from importlib import reload
        import src.detection.face_detector as mod
        reload(mod)
        from src.detection.face_detector import FaceDetector
        fd = FaceDetector(dummy_model)
        dets = fd.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        scores = [d.confidence for d in dets]
        assert scores == sorted(scores, reverse=True)
    finally:
        patcher.stop()


def test_missing_model_raises_file_not_found():
    from src.detection.face_detector import FaceDetector
    with pytest.raises(FileNotFoundError, match="YuNet model not found"):
        FaceDetector("models/nonexistent.onnx")


# ---------- integration tests (require downloaded model) ---------------------


@pytestmark_needs_model
def test_detector_loads_real_model():
    from src.detection.face_detector import FaceDetector
    fd = FaceDetector(MODEL_PATH)
    assert fd is not None


@pytestmark_needs_model
def test_blank_frame_returns_no_faces():
    from src.detection.face_detector import FaceDetector
    fd = FaceDetector(MODEL_PATH)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = fd.detect(blank)
    assert isinstance(dets, list)
    assert len(dets) == 0


@pytestmark_needs_model
def test_detection_result_type():
    from src.detection.face_detector import FaceDetector
    fd = FaceDetector(MODEL_PATH)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dets = fd.detect(frame)
    for d in dets:
        assert isinstance(d, Detection)
        assert len(d.bbox) == 4
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.landmarks) in (0, 5)
