# Architecture

## Frame flow

```
+----------------+
| Webcam 1080p30 |  cv2.VideoCapture in capture thread
+-------+--------+
        | raw BGR frame
        v
+------------------------+
| Face Detector          |  RetinaFace ONNX (CUDA)
| (RetinaFace)           |  returns: bbox, 5 landmarks, confidence
+-------+----------------+
        |
        v
+------------------------------------------+
| Failsafe Monitor (PRE-SWAP CHECK)        |
|  * confidence < threshold -> FREEZE      |
|  * bbox out of frame     -> FREEZE       |
|  * face area too small   -> FREEZE       |
+-------+----------------------------------+
        | green-lit frames only
        v
+------------------------+
| DFM Swapper            |  ONNX Runtime CUDA / TensorRT
| (placeholder.dfm)      |  output: swapped face crop + raw mask
+-------+----------------+
        |
        v
+------------------------+
| Mask Fuser             |  BiSeNet face parse + XSeg occlusion
|  * subtract hands      |  output: final alpha mask
|  * subtract objects    |
+-------+----------------+
        |
        v
+------------------------+
| Compositor (blender)   |  alpha-blend swap into original frame
+-------+----------------+  hands/objects in front stay visible
        |
        v
+------------------------+
| Skin Smoother          |  ONLY on (frame - swap region)
| (forehead/neck)        |
+-------+----------------+
        |
        v
+------------------------------------------+
| Failsafe Gate (POST-COMPOSITE CHECK)     |
|  * mask integrity OK?                    |
|  * swap region matches detected bbox?    |
|  * PASS  -> composited frame             |
|  * FAIL  -> BLACK frame                  |
+-------+----------------------------------+
        |
        v
+------------------------+
| Virtual Camera Output  |  pyvirtualcam -> OBS
+------------------------+
```

## Threading model

```
Thread A  (capture):    webcam -> frame queue (size 2, drop-old)
Thread B  (inference):  frame -> detect -> swap -> mask -> composite
                        -> output queue
Thread C  (output):     output queue -> failsafe gate -> virtual camera
```

- Bounded queues prevent memory growth.
- Drop-old policy favors real-time behavior over completeness.
- A single inference thread keeps GPU utilization predictable.

## Latency budget (target <= 100 ms)

| Stage | Budget |
|---|---|
| Capture -> CPU buffer | 5 ms |
| Face detection (RetinaFace) | 8 ms |
| DFM swap inference | 30 ms |
| Mask fusion (BiSeNet + XSeg) | 20 ms |
| Composite + skin smooth | 10 ms |
| Failsafe + virtual-cam write | 5 ms |
| **Total compute** | **~78 ms** |
| Buffer for queue waits | ~22 ms |
| **Total** | **~100 ms** |

## Module responsibilities

| Module | Responsibility |
|---|---|
| `src/capture/webcam.py` | Threaded webcam read, bounded frame queue |
| `src/detection/face_detector.py` | RetinaFace ONNX wrapper -> (bbox, landmarks, confidence) |
| `src/swap/dfm_loader.py` | Load `.dfm` -> ONNX InferenceSession with correct providers |
| `src/swap/dfm_swapper.py` | Prepare input tensor, run inference, return swapped crop + raw mask |
| `src/occlusion/face_parser.py` | BiSeNet segmentation -> per-region face mask |
| `src/occlusion/xseg_mask.py` | DFL XSeg -> occlusion-aware mask |
| `src/occlusion/mask_fuser.py` | Combine parser + xseg -> final alpha mask |
| `src/beauty/skin_smoother.py` | Bilateral/guided filter on (frame - swap region) |
| `src/compositing/blender.py` | Alpha-blend swap onto original frame |
| `src/failsafe/monitor.py` | Pre-swap and post-composite validity checks |
| `src/failsafe/gate.py` | The ONLY code path that writes to virtual camera |
| `src/failsafe/triggers.py` | Declarative failure conditions |
| `src/output/virtual_camera.py` | pyvirtualcam wrapper |
| `src/output/preview.py` | Local debug preview (development only) |
| `src/pipeline.py` | Thread orchestration, stage wiring |
| `src/main.py` | CLI entry point |

## Key design rules

1. **Only `failsafe/gate.py` writes to the virtual camera.** All other modules return data. This makes it structurally impossible to bypass the gate.
2. **Skin smoothing runs on (full frame) MINUS (swap region).** The swapped face is the DFM's output verbatim.
3. **Occlusion is never a freeze trigger.** The mask fuser handles it; gate trusts the mask.
4. **TensorRT is preferred**, CUDA fallback, CPU only for unit tests.
