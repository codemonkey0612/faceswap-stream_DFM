# Deployment Guide (Windows — ASUS ROG Zephyrus G16, RTX 5090 Laptop)

This guide sets up the trained face-swap pipeline on the streaming laptop and
connects it to OBS → Stripchat. It assumes the trained model
`models/momo_SAEHD_model.dfm` has been placed in the project's `models/` folder.

---

## 0. What you should already have

| Item | Location |
|---|---|
| Project source code | this repo |
| Trained model | `models/momo_SAEHD_model.dfm` (≈555 MB) |
| Config (already points at the model) | `config/default.yaml` → `swap.dfm_path` |

---

## 1. Prerequisites (install once)

- **Miniconda** — https://docs.conda.io/en/latest/miniconda.html
- **Git** — https://git-scm.com/download/win
- **NVIDIA driver** — latest Studio/Game Ready driver for the RTX 5090 Laptop
- **OBS Studio** (for the virtual camera) — https://obsproject.com/

> Note: the runtime environment is **separate** from the DeepFaceLab training
> environment. DFL needs OpenCV 4.6; the live pipeline needs OpenCV 4.8+ and
> `onnxruntime-gpu`. Do not reuse the training env here.

---

## 2. Create the runtime environment

```bat
:: From the project root, in Anaconda Prompt:
conda create -n faceswap python=3.10 -y
conda activate faceswap
pip install -r requirements.txt
```

If `onnxruntime-gpu` cannot see CUDA at runtime, it falls back to CPU
automatically — but the RTX 5090 should be used. Verify GPU is visible:

```bat
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Expected to include `CUDAExecutionProvider`.

---

## 3. Download the auxiliary models

```bat
conda activate faceswap

:: YuNet face detector + MediaPipe hand model
python models\download_models.py

:: BiSeNet hair/headwear parser (~50 MB) — restores real hair over the swap
python scripts\download_face_parser.py
```

After this, `models/` should contain:
- `face_detection_yunet_2023mar.onnx`
- `hand_landmarker.task`
- `face_parser.onnx`
- `momo_SAEHD_model.dfm`  ← the trained model

---

## 4. Find the webcam index

```bat
python scripts\list_cameras.py
```
Note the index of the streaming webcam and, if it is not `0`, set it in
`config/default.yaml` → `capture.device_index`.

---

## 5. First run — local preview (no streaming yet)

```bat
conda activate faceswap
python -m src.main --no-vcam
```
A preview window opens showing the swapped feed. Confirm:
- The AI character face tracks your movement and expressions
- Hands passing over the face show **real** pixels (not swapped)
- **Failsafe:** cover your face / leave frame → output goes **BLACK**
  (this is the privacy safety net — it must black out, never show the real face)

Press `q` (or `Ctrl+C`) to stop.

---

## 6. Live run — OBS virtual camera

```bat
conda activate faceswap
python -m src.main --profile live
```

This writes the swapped feed to the **OBS Virtual Camera** device.

### Connect to Stripchat

**Option A — Browser broadcast (simplest):**
1. Run the pipeline (command above).
2. Log into Stripchat → *Broadcast Yourself / Start Broadcasting*.
3. In the camera dropdown choose **"OBS Virtual Camera"** (NOT the real webcam).
4. Choose the voice-changer output as the microphone.
5. Go live.

**Option B — OBS + RTMP (more control):**
1. In OBS add a **Video Capture Device** → select **"OBS Virtual Camera"**.
2. Stripchat → external/OBS broadcast mode → copy the server URL + stream key.
3. OBS → *Settings → Stream → Custom* → paste URL + key → *Start Streaming*.

---

## 7. Laptop performance (long sessions)

Laptops thermal-throttle. For multi-hour streams:
- Use the cooling pad
- Keep it on **AC power** (battery mode caps GPU performance)
- Set Windows / Armoury Crate to **Turbo / Performance** mode

---

## 8. Stop

Press `Ctrl+C` in the terminal. Stop the Virtual Camera in OBS if used.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDAExecutionProvider` missing | Update NVIDIA driver; reinstall `onnxruntime-gpu` |
| Output is black constantly | No face detected — improve lighting; check webcam index |
| `bbox_out_of_frame` firing | Move camera back / lower it slightly |
| Swap looks misaligned | Confirm `swap.input_size: 256` in `config/default.yaml` |
| Low FPS | Ensure GPU is used (step 2 check); AC power + performance mode |
| Virtual camera not appearing | Open OBS once → Tools → Start Virtual Camera |

---

## Safety rules (do not change)

- The Gate (`src/failsafe/gate.py`) is the **only** writer to the virtual camera;
  any error produces a solid black frame — never a frozen or real frame.
- Hands always show real pixels (MediaPipe mask).
- Hair/occlusions excluded from the swap (BiSeNet).
