"""Download all required model weights into models/.

Run once before first launch:
    .venv\\Scripts\\python.exe models\\download_models.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent

MODELS: list[dict] = [
    {
        "name": "YuNet face detector",
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,
    },
    {
        "name": "MediaPipe hand landmarker",
        "filename": "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "sha256": None,
    },
]


def _download(url: str, dest: Path, name: str) -> None:
    print(f"  Downloading {name} ...", end="", flush=True)
    try:
        headers = {"User-Agent": "faceswap-stream-DFM/0.1"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  Downloading {name} ... {pct}%", end="", flush=True)
        print(f"\r  Downloaded  {name} ({downloaded / 1024 / 1024:.1f} MB)")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {name}: {exc}") from exc


def main() -> None:
    print(f"Model directory: {MODELS_DIR}\n")
    errors: list[str] = []

    for spec in MODELS:
        dest = MODELS_DIR / spec["filename"]
        if dest.exists():
            print(f"  [OK] {spec['filename']} already present")
            continue
        try:
            _download(spec["url"], dest, spec["name"])
        except RuntimeError as exc:
            print(f"\n  [ERROR] {exc}")
            errors.append(spec["filename"])

    print()
    if errors:
        print(f"Failed to download: {', '.join(errors)}")
        print("If the URL is unreachable, download manually and place the file in models/")
        sys.exit(1)
    else:
        print("All models ready.")


if __name__ == "__main__":
    main()
