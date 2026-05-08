# Failsafe Contract

The failsafe layer is the most important module in this project. It enforces requirement **R-S1** (the face mask must never drop).

## Invariants

1. **Single writer**: the virtual camera is written from exactly one call site - `src/failsafe/gate.py :: Gate.write()`. No other module may construct or call `pyvirtualcam`.
2. **Default deny**: if any failsafe check is inconclusive (exception, missing input, unexpected shape), the gate outputs a black frame. There is no "probably OK" path.
3. **Occlusion is not a freeze trigger**: partial occlusion is handled upstream in the mask fuser. The gate trusts the fused mask.

## Pre-swap triggers (action = black frame)

Checked immediately after face detection, before running the DFM.

| Trigger | Condition |
|---|---|
| `no_face` | RetinaFace returned zero detections |
| `low_confidence` | best detection confidence < `detection.confidence_threshold` (default 0.70) |
| `tiny_face` | bbox area / frame area < `detection.min_face_area_ratio` (default 0.005) |
| `bbox_out_of_frame` | bbox extends outside frame boundaries |
| `detector_error` | any exception from the detector |

## Post-composite triggers (action = black frame)

Checked on the composited frame just before virtual-camera write.

| Trigger | Condition |
|---|---|
| `empty_mask` | final alpha mask has zero non-zero pixels |
| `mask_shape_mismatch` | mask shape != frame shape |
| `swap_bbox_divergence` | swap region center shifted > 20% from detected bbox center |
| `nan_or_inf` | any NaN/Inf pixel values anywhere in the composited frame |
| `color_domain_error` | pixel values outside [0, 255] |
| `compositor_error` | any exception in the composite stage |

## Testing requirements

Every trigger above MUST have at least one dedicated test in `tests/test_failsafe_gate.py`. The test suite must:

- Construct a synthetic scenario that fires the trigger
- Assert that `Gate.write()` produced a black frame (all zeros)
- Assert that the underlying virtual-camera call received a black frame (mocked)

## Logging

Every trigger fires a single log line to `logs/failsafe.log`:

```
2026-04-14T10:23:45.123Z  trigger=low_confidence  confidence=0.42  frame_idx=12847
```

Spam suppression: repeated identical triggers within the same streaming session are rate-limited to 1 log line per second.

## What the failsafe layer does NOT do

- It does not recover, retry, or attempt fallback processing.
- It does not reason about hands, occlusion, or scene content (mask fuser's job).
- It does not write anywhere except the virtual camera.
