# Runtime Environment

Verified operating environment for the delivered face-swap pipeline.
(Values marked TBD are confirmed on the setup day, June 10, on the actual PC.)

## Hardware

| Component | Value |
|---|---|
| Machine | ASUS ROG Zephyrus G16 (2025) |
| CPU | Intel Core Ultra 9 285H |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| RAM | 64 GB |
| Storage | 2 TB SSD |
| Display | 16" OLED 240 Hz |

## Software

| Component | Version |
|---|---|
| OS | Windows 11 Pro |
| Python | 3.10 (conda env `faceswap`) |
| NVIDIA driver | TBD (record on setup day) |
| CUDA (runtime, via onnxruntime-gpu) | 12.x |
| onnxruntime-gpu | TBD (record `pip show onnxruntime-gpu`) |
| OpenCV | TBD (record `python -c "import cv2;print(cv2.__version__)"`) |
| OBS Studio | TBD |

## GPU usage

- Face detection (YuNet): GPU via OpenCV DNN / onnxruntime
- Face swap (DFM): GPU via onnxruntime-gpu (`CUDAExecutionProvider`)
- Hair parser (BiSeNet): GPU when available, else every-Nth-frame on CPU
- Hand detection (MediaPipe): CPU (lightweight)

## Models delivered

| File | Purpose | Size |
|---|---|---|
| `models/momo_SAEHD_model.dfm` | Trained face-swap model (SAEHD + GAN + XSeg) | ≈555 MB |
| `models/face_detection_yunet_2023mar.onnx` | Face detector | ~0.2 MB |
| `models/face_parser.onnx` | Hair/headwear masker (BiSeNet) | ~50 MB |
| `models/hand_landmarker.task` | Hand detector (MediaPipe) | ~7.8 MB |

## Verify the environment (run on setup day)

```bat
conda activate faceswap
python -c "import sys; print('Python', sys.version)"
python -c "import onnxruntime as ort; print('ORT', ort.__version__, ort.get_available_providers())"
python -c "import cv2; print('OpenCV', cv2.__version__)"
nvidia-smi --query-gpu=name,driver_version --format=csv
```
Paste the outputs into the tables above to finalize this document.
