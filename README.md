# faceswap-stream_DFM

Real-time face-swap streaming pipeline for solo streamer **Momo**.  
Webcam → YuNet detect → DFM ONNX swap → hair/hand masking → OBS Virtual Camera.  
Target: 1920×1080 @ 30 fps, < 100 ms latency.

---

## Hardware

| Machine | GPU | Role |
|---|---|---|
| Dev PC | GTX 1050 4 GB | Code, augmentation, testing |
| Streaming PC | RTX 5080 32 GB | DFL training, live streaming |

---

## Project Status

| Phase | What | Status |
|---|---|---|
| 1 | Runtime pipeline (detect → swap → mask → failsafe → vcam) | **Complete** — 135 tests pass |
| 2a | DST augmentation (6 stills → ~6,000 frames via classical + LivePortrait) | **Complete** |
| 2b | SRC recording (Momo's real face → ≥ 5,000 frames) | **Waiting — record the video** |
| 3 | DFL SAEHD training on streaming PC → export `.dfm` | Pending Phase 2b |
| 4 | Deploy trained model, go live | Pending Phase 3 |

---

## Occlusion Handling

| What covers the face | Tool | Model file | Status |
|---|---|---|---|
| Hair / bangs | BiSeNet FaceParser | `models/face_parser.onnx` | Active |
| Microphone / headset / objects | DFL XSeg | `models/xseg.onnx` | Active after Phase 3 |
| Hands | MediaPipe HandLandmarker | `models/hand_landmarker.task` | Active |

---

## One-Time Setup (Dev PC)

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Git](https://git-scm.com/download/win)
- NVIDIA drivers

### Install
```bat
:: Open Anaconda Prompt or run: conda init cmd.exe first
scripts\setup_conda_env.bat
conda activate faceswap
```

### Download models
```bat
:: Face detector (YuNet)
python models\download_models.py

:: BiSeNet hair parser (~50 MB)
python scripts\download_face_parser.py
```

Models already in `models\`:
- `face_detection_yunet_2023mar.onnx` — YuNet face detector
- `hand_landmarker.task` — MediaPipe hand detector
- `placeholder.dfm` — identity ONNX (used until trained model is ready)
- `face_parser.onnx` — BiSeNet19 hair/headwear masker

---

## Running the Pipeline

```bat
conda activate faceswap

# Dev PC — local preview window, no OBS required
python -m src.main --no-vcam

# Dev PC — debug profile (720p, CPU, preview window)
python -m src.main --profile debug --no-vcam

# Streaming PC — live profile (1080p, TensorRT, OBS virtual camera)
python -m src.main --profile live
```

Stop: press `Ctrl+C` in the terminal.

---

## Phase 2b — Record Momo's Face (SRC dataset)

**Record on your webcam — same one you stream with.**

| Requirement | Value |
|---|---|
| Duration | 5–10 minutes |
| Content | Talk naturally; look left/right/up/down; smile; open mouth wide |
| Lighting | Same as stream setup |
| Resolution | Any — 1080p preferred |
| Format | `.mp4`, `.mov`, `.avi` |

Save to: `training_data\src_video\` (any filename).

Then extract frames:
```bat
conda activate faceswap
python scripts\extract_src_frames.py
```

Target: ≥ 5,000 frames in `training_data\src_frames\`.

---

## Phase 3 — DFL Training (Streaming PC)

### Setup
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on the streaming PC
2. Copy this project folder to the streaming PC (same path)
3. Run the setup scripts:

```bat
scripts\setup_conda_env.bat
scripts\setup_liveportrait.bat   :: if you want more LP frames
```

4. Download [DeepFaceLab NVIDIA RTX release](https://github.com/iperov/DeepFaceLab/releases)  
   → Extract to `F:\DeepFaceLab\`

5. Copy training data to DFL workspace:
```bat
scripts\setup_dfl_training.bat
```

### Train SAEHD
Open DeepFaceLab and run **`4) train SAEHD.bat`** with these settings:

| Setting | Value |
|---|---|
| Resolution | 192 (or 256 if VRAM allows) |
| Batch size | 8+ |
| Eyes priority | ON |
| Mouth priority | ON |
| Stop when | Loss G < 0.03 |

Training takes 12–48 hours on RTX 5080 depending on resolution.

### Train XSeg (microphone / headset masking)
After SAEHD converges:  
DFL → **XSeg → Train XSeg** (2–4 hours)  
DFL → **XSeg → Export XSeg to ONNX**  
Copy to: `models\xseg.onnx`

### Export the model
DFL → **`6) export DFM.bat`**  
Copy exported `.dfm` to: `models\momo.dfm`

---

## Phase 4 — Deploy

```bat
conda activate faceswap

# Validates model, patches config, smoke-tests the swap pipeline
python scripts\deploy_model.py --dfm models\momo.dfm

# Launch
python -m src.main --profile live
```

### OBS Setup
1. **Tools → Start Virtual Camera** in OBS
2. Add source: **Video Capture Device** → your webcam (input)
3. Add source: **Video Capture Device** → OBS Virtual Camera (swap output)
4. The swap output appears on the Virtual Camera source in real time

---

## Config

Main config: `config\default.yaml`  
Profiles overlay on top of it:

| Profile | File | Use case |
|---|---|---|
| *(none)* | `config\default.yaml` | Default / dev |
| `live` | `config\profiles\live.yaml` | Streaming PC, TensorRT |
| `debug` | `config\profiles\debug.yaml` | Dev PC, CPU, preview |

After deploying the trained model, `deploy_model.py` automatically updates:
```yaml
swap:
  dfm_path: models/momo.dfm
  input_size: 256   # auto-detected from model
```

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts\setup_conda_env.bat` | Create `faceswap` conda env + install deps |
| `scripts\setup_liveportrait.bat` | Create `liveportrait` conda env + clone + download weights |
| `scripts\augment_dst_faces.py` | Generate classical augmentation frames from 6 stills |
| `scripts\animate_dst_liveportrait.py` | Run LivePortrait on all 12 bundled drivers × 6 stills |
| `scripts\extract_src_frames.py` | Extract face-verified frames from Momo's recording |
| `scripts\setup_dfl_training.bat` | Copy DST + SRC frames into DFL workspace |
| `scripts\deploy_model.py` | Validate + deploy trained `.dfm`, patch config |
| `scripts\download_face_parser.py` | Download BiSeNet19 ONNX from HuggingFace |
| `scripts\list_cameras.py` | List available webcam device indices |

---

## Troubleshooting

**`bbox_out_of_frame` firing constantly**  
Face is too close to camera / chin goes below frame. Move camera back slightly or lower it.

**`no_face` firing**  
Improve lighting. Face must occupy ≥ 0.5% of frame area (tiny face too far from camera).

**cuDNN / TensorRT errors on dev PC**  
Expected — dev PC uses CPU fallback. These errors are harmless during testing.

**BiSeNet face parser slow on dev PC**  
Runs every 4th frame automatically when CUDA is not available. FPS will be 5–15 fps on dev PC; RTX 5080 will hit 30+ fps.

**LivePortrait `conda run` fails**  
Run `conda init cmd.exe`, reopen terminal, then re-run `setup_liveportrait.bat`.

**DFM input size mismatch error**  
Update `config\default.yaml` → `swap.input_size` to match your trained model (192 or 256). `deploy_model.py` does this automatically.

---

## Safety Rules (Hard Requirements)

- **Black frame on any failure** — the Gate is the only writer to virtual camera; all errors produce solid black, never a frozen or real frame
- **Hands always show real pixels** — MediaPipe hand mask restores original webcam pixels over the swap region
- **Hair / occlusions excluded** — BiSeNet + XSeg prevent swap bleeding onto non-face objects
- Never modify `src/failsafe/gate.py` to pass through on error
