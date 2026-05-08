"""Phase 2b — SRC frame extraction pipeline.

Takes Momo's source-face recording video(s), extracts frames where a face
is detected with high confidence, and writes them to training_data/src_frames/
for DeepFaceLab SAEHD SRC training.

Recording guidelines for best training results
-----------------------------------------------
• Record at 1920×1080 or higher (the OBSBOT Meet 2 4K is ideal)
• Keep face well-lit and unobstructed for most of the clip
• Capture variety: look left, right, up, down, smile, neutral, talking
• 5-15 minutes of footage → ~5 000–10 000 usable frames (DFL needs ~5 000+)
• Avoid extreme motion blur (record at 30fps+)
• Save as MP4 (H.264) or MOV

Usage
-----
  # Single video
  .venv\\Scripts\\python.exe scripts\\extract_src_frames.py \\
      --video training_data/src_video/momo.mp4

  # All videos in a directory
  .venv\\Scripts\\python.exe scripts\\extract_src_frames.py \\
      --video-dir training_data/src_video

  # Tune face confidence threshold (default 0.6 — accepts partial occlusion)
  .venv\\Scripts\\python.exe scripts\\extract_src_frames.py \\
      --video training_data/src_video/momo.mp4 --conf 0.7

Output
------
  training_data/src_frames/<video_stem>_NNNNNN.jpg   (quality 95 JPEG)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_FACE_MODEL   = Path("models/face_detection_yunet_2023mar.onnx")
_DEFAULT_VID  = "training_data/src_video"
_DEFAULT_OUT  = "training_data/src_frames"
_CONF_THRESH  = 0.60      # lower than pipeline — ok for training data
_NMS_THRESH   = 0.30
_STRIDE       = 1         # extract every N-th frame (1 = all frames, 2 = every other, …)
_JPEG_QUAL    = 95
_VIDEO_EXTS   = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(model_path: Path, conf: float) -> cv2.FaceDetectorYN | None:
    if not model_path.exists():
        print(f"[WARN] Face model not found at {model_path}")
        print("       Run: .venv\\Scripts\\python.exe models\\download_models.py")
        return None
    det = cv2.FaceDetectorYN.create(
        str(model_path), "", (0, 0),
        score_threshold=conf,
        nms_threshold=_NMS_THRESH,
        top_k=5,
    )
    return det


def best_face(
    detector: cv2.FaceDetectorYN | None,
    frame: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Return (x1, y1, x2, y2) of highest-confidence face, or None."""
    if detector is None:
        return (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    # YuNet row: [x, y, w, h, …, score]
    best = max(faces, key=lambda r: r[14])
    x, y, bw, bh = best[0], best[1], best[2], best[3]
    return float(x), float(y), float(x + bw), float(y + bh)


def face_area_ratio(bbox: tuple, frame_shape: tuple) -> float:
    x1, y1, x2, y2 = bbox
    fh, fw = frame_shape[:2]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (fw * fh)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_video(
    video_path: Path,
    out_dir: Path,
    detector: cv2.FaceDetectorYN | None,
    stride: int,
    min_area_ratio: float = 0.003,
) -> tuple[int, int]:
    """Extract face-containing frames from one video.

    Returns (frames_extracted, frames_skipped).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n  Video : {video_path.name}")
    print(f"  Size  : {width}x{height} @ {fps:.1f}fps  "
          f"({total_frames} frames, ~{total_frames / fps:.0f}s)")

    stem = video_path.stem
    extracted = 0
    skipped   = 0
    frame_idx = 0
    t0 = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if (frame_idx % stride) != 0:
            continue

        bbox = best_face(detector, frame)
        if bbox is None:
            skipped += 1
            continue
        if face_area_ratio(bbox, frame.shape) < min_area_ratio:
            skipped += 1
            continue

        out_path = out_dir / f"{stem}_{extracted:06d}.jpg"
        cv2.imwrite(str(out_path), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUAL])
        extracted += 1

        if extracted % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = extracted / elapsed if elapsed > 0 else 0
            pct  = frame_idx / max(total_frames, 1) * 100
            print(f"  {pct:5.1f}%  extracted={extracted}  skipped={skipped}"
                  f"  ({rate:.0f} fr/s)", end="\r")

    cap.release()
    elapsed = time.monotonic() - t0
    print(f"  Done: {extracted} extracted, {skipped} skipped "
          f"({elapsed:.0f}s, {extracted / elapsed:.0f} fr/s)" + " " * 30)
    return extracted, skipped


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2b: Extract face frames from Momo's source video(s)"
    )
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--video",     type=Path,
                     help="Path to a single video file")
    grp.add_argument("--video-dir", type=Path, default=Path(_DEFAULT_VID),
                     help=f"Directory containing video files (default: {_DEFAULT_VID})")
    p.add_argument("--out-dir",     type=Path, default=Path(_DEFAULT_OUT),
                   help=f"Output directory for extracted frames (default: {_DEFAULT_OUT})")
    p.add_argument("--conf",        type=float, default=_CONF_THRESH,
                   help=f"Face detection confidence threshold (default: {_CONF_THRESH})")
    p.add_argument("--stride",      type=int, default=_STRIDE,
                   help="Extract every N-th frame (default: 1 = all frames)")
    p.add_argument("--no-face-check", action="store_true",
                   help="Disable face detection (extract all frames)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect video paths
    if args.video:
        videos = [args.video]
    else:
        vid_dir = args.video_dir
        videos = [p for p in sorted(vid_dir.iterdir())
                  if p.suffix.lower() in _VIDEO_EXTS]
        if not videos:
            print(f"No video files found in {vid_dir}")
            print(f"Supported formats: {', '.join(sorted(_VIDEO_EXTS))}")
            print("Place Momo's recording there and re-run.")
            sys.exit(0)

    detector = None if args.no_face_check else make_detector(_FACE_MODEL, args.conf)

    print(f"Videos to process  : {len(videos)}")
    print(f"Stride             : every {args.stride} frame(s)")
    print(f"Face conf threshold: {args.conf}")
    print(f"Output directory   : {out_dir}")

    total_ext = 0
    total_skp = 0
    for vid in videos:
        e, s = extract_video(vid, out_dir, detector, args.stride)
        total_ext += e
        total_skp += s

    print(f"\n{'='*60}")
    print(f"Total extracted : {total_ext} frames")
    print(f"Total skipped   : {total_skp} frames (no/low-quality face)")
    print(f"Output          : {out_dir}/")
    if total_ext < 5000:
        print(f"\n[WARN] DFL SAEHD typically needs ≥ 5 000 SRC frames for good")
        print(f"       results. Current count: {total_ext}.")
        print(f"       Record more footage or reduce --conf to include more frames.")
    else:
        print(f"\nFrame count looks good for DFL training.")
    print(f"\nNext step (Phase 3):")
    print(f"  Open DeepFaceLab and create a new workspace.")
    print(f"  DST frames: training_data/dst_augmented/")
    print(f"  SRC frames: training_data/src_frames/")
    print(f"  Train SAEHD until loss < 0.03, then run 'Export to DFM'.")


if __name__ == "__main__":
    main()
