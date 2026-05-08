"""Tests for the Gate — the single writer to the virtual camera.

These are the most important tests in the project: they prove that a bad
candidate frame cannot leak to the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.failsafe.gate import Gate
from src.failsafe.triggers import (
    PostCompositeTrigger,
    PreSwapTrigger,
    TriggerResult,
)


H, W = 120, 160


@pytest.fixture
def gate(fake_sink):
    return Gate(fake_sink, frame_width=W, frame_height=H, rate_limit_seconds=0.0)


@pytest.fixture
def valid_frame():
    return np.full((H, W, 3), 128, dtype=np.uint8)


def _is_black(frame: np.ndarray) -> bool:
    return frame.shape == (H, W, 3) and frame.dtype == np.uint8 and not frame.any()


# ---------- happy path --------------------------------------------------------


def test_green_lit_frame_passes_through(gate, fake_sink, valid_frame):
    gate.write(valid_frame, TriggerResult.pass_through())
    assert len(fake_sink.sent) == 1
    np.testing.assert_array_equal(fake_sink.sent[0], valid_frame)


# ---------- upstream trigger forces black ------------------------------------


def test_fired_pre_swap_trigger_forces_black(gate, fake_sink, valid_frame):
    gate.write(
        valid_frame,
        TriggerResult.block(PreSwapTrigger.NO_FACE),
    )
    assert len(fake_sink.sent) == 1
    assert _is_black(fake_sink.sent[0])


def test_fired_post_composite_trigger_forces_black(gate, fake_sink, valid_frame):
    gate.write(
        valid_frame,
        TriggerResult.block(PostCompositeTrigger.EMPTY_MASK),
    )
    assert _is_black(fake_sink.sent[0])


def test_every_pre_swap_trigger_blacks_the_frame(fake_sink, valid_frame):
    g = Gate(fake_sink, W, H, rate_limit_seconds=0.0)
    for t in PreSwapTrigger:
        fake_sink.sent.clear()
        g.write(valid_frame, TriggerResult.block(t))
        assert len(fake_sink.sent) == 1 and _is_black(fake_sink.sent[0]), (
            f"trigger {t} did not force a black frame"
        )


def test_every_post_composite_trigger_blacks_the_frame(fake_sink, valid_frame):
    g = Gate(fake_sink, W, H, rate_limit_seconds=0.0)
    for t in PostCompositeTrigger:
        fake_sink.sent.clear()
        g.write(valid_frame, TriggerResult.block(t))
        assert len(fake_sink.sent) == 1 and _is_black(fake_sink.sent[0]), (
            f"trigger {t} did not force a black frame"
        )


# ---------- defense-in-depth checks inside the gate --------------------------


def test_none_candidate_forces_black(gate, fake_sink):
    gate.write(None, TriggerResult.pass_through())
    assert _is_black(fake_sink.sent[0])


def test_wrong_shape_forces_black(gate, fake_sink):
    bad = np.full((H - 1, W, 3), 128, dtype=np.uint8)
    gate.write(bad, TriggerResult.pass_through())
    assert _is_black(fake_sink.sent[0])


def test_wrong_channel_count_forces_black(gate, fake_sink):
    bad = np.full((H, W, 4), 128, dtype=np.uint8)  # BGRA instead of BGR
    gate.write(bad, TriggerResult.pass_through())
    assert _is_black(fake_sink.sent[0])


def test_wrong_dtype_forces_black(gate, fake_sink):
    bad = np.full((H, W, 3), 0.5, dtype=np.float32)
    gate.write(bad, TriggerResult.pass_through())
    assert _is_black(fake_sink.sent[0])


def test_no_trigger_result_still_checks_shape(gate, fake_sink):
    # Even without a monitor result, gate validates shape/dtype.
    bad = np.full((H - 1, W, 3), 128, dtype=np.uint8)
    gate.write(bad, trigger_result=None)
    assert _is_black(fake_sink.sent[0])


# ---------- stats & observability --------------------------------------------


def test_stats_count_frames(gate, fake_sink, valid_frame):
    gate.write(valid_frame, TriggerResult.pass_through())
    gate.write(None, TriggerResult.pass_through())
    gate.write(valid_frame, TriggerResult.block(PreSwapTrigger.NO_FACE))
    assert gate.stats == {"frames_total": 3, "frames_blacked": 2}


def test_explicit_emit_black(gate, fake_sink):
    gate.emit_black()
    assert _is_black(fake_sink.sent[0])
    assert gate.stats["frames_blacked"] == 1


# ---------- adversarial: gate must never pass something bad -------------------


@pytest.mark.parametrize(
    "builder",
    [
        lambda: None,
        lambda: np.zeros((0, 0, 3), dtype=np.uint8),
        lambda: np.zeros((H, W), dtype=np.uint8),  # 2D
        lambda: np.zeros((H, W, 3), dtype=np.int16),
        lambda: np.full((H, W, 3), 256, dtype=np.int32),  # out-of-range ints
    ],
)
def test_pathological_inputs_all_black_out(fake_sink, builder):
    g = Gate(fake_sink, W, H, rate_limit_seconds=0.0)
    g.write(builder(), TriggerResult.pass_through())
    assert _is_black(fake_sink.sent[0])


# ---------- rate-limited logging does not affect emission --------------------


def test_rate_limit_does_not_skip_frame(fake_sink, valid_frame):
    g = Gate(fake_sink, W, H, rate_limit_seconds=10.0)
    for _ in range(5):
        g.write(valid_frame, TriggerResult.block(PreSwapTrigger.NO_FACE))
    # 5 black frames emitted even if only 1 log line written.
    assert len(fake_sink.sent) == 5
    assert all(_is_black(f) for f in fake_sink.sent)
