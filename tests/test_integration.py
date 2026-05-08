"""Phase 1i integration tests — full stack with placeholder.dfm.

These tests load the real ONNX placeholder model from disk and exercise
the complete swap pipeline: DFMLoader → DFMSwapper → pipeline compositing.

All tests skip if models/placeholder.dfm is absent (CI without model files).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.failsafe.monitor import Detection

MODEL_PATH = Path("models/placeholder.dfm")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/placeholder.dfm not found — run models/make_placeholder_dfm.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _face_detection(h: int = 480, w: int = 640) -> Detection:
    """Centre-of-frame face detection with 5 plausible landmarks."""
    cx, cy = w // 2, h // 2
    r = min(h, w) // 5
    return Detection(
        bbox=(cx - r, cy - r, cx + r, cy + r),
        confidence=0.92,
        landmarks=(
            (cx - r // 2, cy - r // 4),   # right eye
            (cx + r // 2, cy - r // 4),   # left eye
            (cx,          cy),             # nose
            (cx - r // 3, cy + r // 3),   # right mouth
            (cx + r // 3, cy + r // 3),   # left mouth
        ),
    )


# ---------------------------------------------------------------------------
# DFMLoader
# ---------------------------------------------------------------------------

class TestDFMLoader:
    def test_loads_placeholder_model(self):
        from src.swap.dfm_loader import DFMLoader
        loader = DFMLoader(str(MODEL_PATH))
        assert loader.loaded
        assert loader.info is not None

    def test_model_info_input_name(self):
        from src.swap.dfm_loader import DFMLoader
        info = DFMLoader(str(MODEL_PATH)).info
        assert info.input_name == "input"

    def test_model_info_output_names(self):
        from src.swap.dfm_loader import DFMLoader
        info = DFMLoader(str(MODEL_PATH)).info
        assert info.output_face_name == "face"
        assert info.output_mask_name == "mask"

    def test_run_returns_correct_shapes(self):
        from src.swap.dfm_loader import DFMLoader
        loader = DFMLoader(str(MODEL_PATH))
        crop = _random_frame(256, 256)
        face_out, mask_out = loader.run(crop)
        assert face_out.shape == (256, 256, 3)
        assert mask_out.shape == (256, 256)

    def test_run_face_output_is_uint8(self):
        from src.swap.dfm_loader import DFMLoader
        face_out, _ = DFMLoader(str(MODEL_PATH)).run(_random_frame(256, 256))
        assert face_out.dtype == np.uint8

    def test_run_mask_is_float32_in_range(self):
        from src.swap.dfm_loader import DFMLoader
        _, mask = DFMLoader(str(MODEL_PATH)).run(_random_frame(256, 256))
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_run_identity_face_roundtrip(self):
        """Placeholder model is identity — swapped face should be ≈ normalised input."""
        from src.swap.dfm_loader import DFMLoader
        loader = DFMLoader(str(MODEL_PATH))
        crop = _random_frame(256, 256)
        face_out, _ = loader.run(crop)
        # Identity model: input is BGR→RGB normalised, output is RGB→BGR denormalised.
        # Round-trip precision loss ≤ 1 DN (uint8 quantisation).
        diff = np.abs(face_out.astype(np.int32) - crop.astype(np.int32))
        assert diff.max() <= 1, f"Identity roundtrip max error: {diff.max()}"


# ---------------------------------------------------------------------------
# DFMSwapper
# ---------------------------------------------------------------------------

class TestDFMSwapper:
    def _make_swapper(self):
        from src.swap.dfm_loader import DFMLoader
        from src.swap.dfm_swapper import DFMSwapper
        loader = DFMLoader(str(MODEL_PATH))
        return DFMSwapper(loader, output_size=256)

    def test_process_returns_swap_result(self):
        from src.swap.dfm_swapper import SwapResult
        swapper = self._make_swapper()
        frame = _random_frame()
        result = swapper.process(frame, _face_detection())
        assert isinstance(result, SwapResult)

    def test_composited_shape_matches_frame(self):
        frame = _random_frame()
        result = self._make_swapper().process(frame, _face_detection())
        assert result.composited.shape == frame.shape

    def test_composited_dtype_uint8(self):
        frame = _random_frame()
        result = self._make_swapper().process(frame, _face_detection())
        assert result.composited.dtype == np.uint8

    def test_mask_full_shape(self):
        frame = _random_frame()
        result = self._make_swapper().process(frame, _face_detection())
        assert result.mask_full.shape == (480, 640)
        assert result.mask_full.dtype == np.float32

    def test_mask_full_nonzero(self):
        """Placeholder model outputs all-ones mask — composited mask must be nonzero."""
        frame = _random_frame()
        result = self._make_swapper().process(frame, _face_detection())
        assert result.mask_full.max() > 0.0

    def test_swap_bbox_within_frame(self):
        frame = _random_frame()
        h, w = frame.shape[:2]
        result = self._make_swapper().process(frame, _face_detection())
        x1, y1, x2, y2 = result.swap_bbox
        assert x1 >= 0 and y1 >= 0
        assert x2 <= w and y2 <= h

    def test_composited_differs_from_original(self):
        """Identity model pastes back an aligned/unaligned version — some pixels change."""
        rng = np.random.default_rng(99)
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self._make_swapper().process(frame, _face_detection())
        # After alignment + paste-back the pixels in the face region change.
        assert not np.array_equal(result.composited, frame)


# ---------------------------------------------------------------------------
# Full pipeline stack (DFMLoader + DFMSwapper + HandMasker + SkinSmoother)
# ---------------------------------------------------------------------------

class TestFullStack:
    def test_pipeline_components_compose_without_error(self):
        """Smoke test: all Phase 1 components wire together without crashing."""
        from src.beauty.skin_smoother import SkinSmoother
        from src.occlusion.hand_masker import landmarks_to_mask
        from src.swap.dfm_loader import DFMLoader
        from src.swap.dfm_swapper import DFMSwapper

        loader = DFMLoader(str(MODEL_PATH))
        swapper = DFMSwapper(loader, output_size=256)
        smoother = SkinSmoother(enabled=True)

        frame = _random_frame(h=480, w=640)
        det = _face_detection()

        # Swap
        swap_result = swapper.process(frame, det)

        # Hand mask (no hands → zero mask)
        hand_mask = landmarks_to_mask(480, 640, [])
        assert hand_mask.max() == 0.0

        # Skin smooth
        composited = smoother.apply(swap_result.composited, swap_result.mask_full)
        assert composited.shape == frame.shape
        assert composited.dtype == np.uint8

    def test_output_values_always_valid_uint8(self):
        """All pixel values must be in [0, 255] after the full stack."""
        from src.beauty.skin_smoother import SkinSmoother
        from src.swap.dfm_loader import DFMLoader
        from src.swap.dfm_swapper import DFMSwapper

        loader = DFMLoader(str(MODEL_PATH))
        swapper = DFMSwapper(loader, output_size=256)
        smoother = SkinSmoother()

        frame = _random_frame()
        swap_result = swapper.process(frame, _face_detection())
        out = smoother.apply(swap_result.composited, swap_result.mask_full)

        assert out.min() >= 0
        assert out.max() <= 255

    @pytest.mark.parametrize("h,w", [(480, 640), (720, 1280), (1080, 1920)])
    def test_various_resolutions(self, h: int, w: int):
        """Swap + smooth must work for all three target resolutions."""
        from src.beauty.skin_smoother import SkinSmoother
        from src.swap.dfm_loader import DFMLoader
        from src.swap.dfm_swapper import DFMSwapper

        loader = DFMLoader(str(MODEL_PATH))
        swapper = DFMSwapper(loader, output_size=256)
        smoother = SkinSmoother()

        frame = _random_frame(h=h, w=w)
        det = _face_detection(h=h, w=w)
        swap_result = swapper.process(frame, det)
        out = smoother.apply(swap_result.composited, swap_result.mask_full)

        assert out.shape == (h, w, 3)
        assert out.dtype == np.uint8

    def test_post_composite_check_passes_on_valid_output(self):
        """Monitor's post-composite check must not fire on real swap output."""
        from src.beauty.skin_smoother import SkinSmoother
        from src.failsafe.monitor import Monitor
        from src.swap.dfm_loader import DFMLoader
        from src.swap.dfm_swapper import DFMSwapper

        loader = DFMLoader(str(MODEL_PATH))
        swapper = DFMSwapper(loader, output_size=256)
        smoother = SkinSmoother()
        monitor = Monitor()

        frame = _random_frame()
        det = _face_detection()
        swap_result = swapper.process(frame, det)
        composited = smoother.apply(swap_result.composited, swap_result.mask_full)

        result = monitor.check_post_composite(
            composited=composited,
            mask=swap_result.mask_full,
            detected_bbox=det.bbox,
            swap_bbox=swap_result.swap_bbox,
            expected_shape=frame.shape,
        )
        assert not result.fired, f"post-composite fired unexpectedly: {result}"
