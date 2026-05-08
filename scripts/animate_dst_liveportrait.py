"""Phase 2a (enhanced) — LivePortrait DST face animation.

Animates each of the 6 AI-beauty-face stills using LivePortrait and a
driving video, producing thousands of realistic frames with natural head
motion, speech, and expression variation.

The output frames are merged directly into training_data/dst_augmented/
alongside the classical-augmentation frames already there.

Architecture
------------
LivePortrait runs in its own conda environment 'liveportrait'
(set up by scripts/setup_liveportrait.bat). This script calls LP via
  conda run -n liveportrait python inference.py ...
so there are no PyTorch / onnxruntime import conflicts with the main env.

Usage
-----
  # Activate main env first, then:
  conda activate faceswap

  # Minimal (use LP sample driver)
  python scripts\\animate_dst_liveportrait.py

  # Custom driving video
  python scripts\\animate_dst_liveportrait.py \\
      --driving training_data\\driving\\driving.mp4

  # More frames per still (default 800)
  python scripts\\animate_dst_liveportrait.py \\
      --driving training_data\\driving\\driving.mp4 --max-frames 1500

Driving video tips
------------------
  * ANY frontal talking-head video works (identity does not transfer).
  * 30 seconds - 5 minutes is enough.
  * Good lighting + stable camera = better motion quality.
  * Record yourself, or use the bundled LP sample (d0.mp4 in LP repo).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR  = Path(__file__).resolve().parent.parent
LP_BASE      = PROJECT_DIR.parent / "LivePortrait_env"
LP_REPO      = LP_BASE / "repo"
LP_CONDA_ENV = "liveportrait"   # conda env created by setup_liveportrait.bat

SRC_STILLS   = sorted(PROJECT_DIR.glob("[123456].png"))
DST_OUT      = PROJECT_DIR / "training_data" / "dst_augmented"
DRIVING_DIR  = PROJECT_DIR / "training_data" / "driving"
FACE_MODEL   = PROJECT_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# Frame stride when extracting from LP output video (1 = every frame).
# At 30fps driving video, stride=2 gives ~15fps equivalent diversity.
EXTRACT_STRIDE = 2
FACE_CONF      = 0.45


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conda_env_exists(name: str) -> bool:
    """Return True if a conda environment with the given name is installed."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True, text=True, timeout=30,
        )
        return any(
            part == name
            for line in result.stdout.splitlines()
            for part in line.split()
        )
    except Exception:
        return False


def _check_environment() -> None:
    """Abort early with clear instructions if LP is not set up."""
    errors: list[str] = []
    if not LP_REPO.exists():
        errors.append(f"LivePortrait repo not found at:\n    {LP_REPO}")
    if not _conda_env_exists(LP_CONDA_ENV):
        errors.append(
            f"Conda env '{LP_CONDA_ENV}' not found.\n"
            f"    Run: scripts\\setup_liveportrait.bat"
        )
    if not SRC_STILLS:
        errors.append("No source stills (1.png-6.png) found in project root.")
    if errors:
        print("SETUP INCOMPLETE - fix these issues before running:\n")
        for e in errors:
            print(f"  * {e}")
        print(f"\nRun setup first:\n  {PROJECT_DIR / 'scripts' / 'setup_liveportrait.bat'}")
        sys.exit(1)


def _find_inference_script() -> Path:
    """Return path to LivePortrait's inference.py (handles repo layout changes)."""
    candidates = [
        LP_REPO / "inference.py",
        LP_REPO / "src" / "inference.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find inference.py under {LP_REPO}.\n"
        "The LivePortrait repo layout may have changed — check their README."
    )


def _find_sample_drivers() -> list[Path]:
    """Return all LP bundled sample driving videos, sorted by name."""
    driving_dir = LP_REPO / "assets" / "examples" / "driving"
    if not driving_dir.exists():
        return []
    return sorted(driving_dir.glob("*.mp4"))


def _make_face_detector() -> cv2.FaceDetectorYN | None:
    if not FACE_MODEL.exists():
        return None
    return cv2.FaceDetectorYN.create(
        str(FACE_MODEL), "", (0, 0),
        score_threshold=FACE_CONF, nms_threshold=0.3, top_k=5,
    )


def _has_face(detector: cv2.FaceDetectorYN | None, frame: np.ndarray) -> bool:
    if detector is None:
        return True
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return faces is not None and len(faces) > 0


def _extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    stem: str,
    start_idx: int,
    max_frames: int,
    stride: int,
    detector: cv2.FaceDetectorYN | None,
) -> int:
    """Extract face-verified JPEG frames from an animated output video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [WARN] Cannot open output video: {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx % stride) != 0:
            continue
        if not _has_face(detector, frame):
            continue
        out_path = out_dir / f"lp_{stem}_{start_idx + saved:05d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1

    cap.release()
    return saved


def _ensure_ffmpeg_for_lp() -> None:
    """Copy imageio-ffmpeg binary into LP_REPO/ffmpeg/ffmpeg.exe.

    LivePortrait's inference.py checks for an 'ffmpeg/' subdirectory in its
    CWD (the repo root) and auto-adds it to PATH — lines 41-43 of inference.py:
        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    Placing ffmpeg.exe there means no system-wide FFmpeg install is needed.
    The copy is skipped on subsequent runs if the file already exists.
    """
    ffmpeg_dest = LP_REPO / "ffmpeg" / "ffmpeg.exe"
    if ffmpeg_dest.exists():
        return
    try:
        import shutil
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        ffmpeg_dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, ffmpeg_dest)
        print(f"[setup] Copied FFmpeg binary → {ffmpeg_dest}")
    except Exception as exc:
        print(f"[WARN] Could not install bundled FFmpeg: {exc}")
        print("       Install FFmpeg manually: winget install Gyan.FFmpeg")


def _run_liveportrait(
    inference_py: Path,
    source: Path,
    driving: Path,
    output_dir: Path,
) -> Path | None:
    """Call LivePortrait inference in its own venv, return output video path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", LP_CONDA_ENV,
        "python", str(inference_py),
        "--source", str(source),
        "--driving", str(driving),
        "--output_dir", str(output_dir),
        "--flag_relative_motion",          # preserve source expression range
        "--flag_pasteback",                # composite back onto source frame
    ]

    print(f"    Running LivePortrait: {source.name} × {driving.name}")
    t0 = time.monotonic()
    # Build a clean environment for the LP subprocess:
    #   PYTHONIOENCODING=utf-8  — prevents CP1252 UnicodeEncodeError on emoji (Rich)
    #   NO_COLOR=1              — tells Rich to skip fancy rendering / emoji entirely
    #   PYTHONUTF8=1            — Python 3.7+ UTF-8 mode (belt-and-suspenders)
    lp_env = os.environ.copy()
    lp_env["PYTHONIOENCODING"] = "utf-8"
    lp_env["PYTHONUTF8"] = "1"
    lp_env["NO_COLOR"] = "1"

    result = subprocess.run(
        cmd,
        cwd=str(LP_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=lp_env,
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"    [ERROR] LivePortrait failed (exit {result.returncode}):")
        # Print last 20 lines of stderr for diagnosis.
        for line in result.stderr.strip().splitlines()[-20:]:
            print(f"      {line}")
        return None

    # LivePortrait writes  <output_dir>/<source_stem>--<driving_stem>.mp4
    # (naming varies by LP version — find the newest .mp4 in output_dir)
    mp4s = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        print(f"    [WARN] No .mp4 found in {output_dir} after inference.")
        return None

    out_video = mp4s[-1]
    duration  = cv2.VideoCapture(str(out_video)).get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"    Generated: {out_video.name}  ({int(duration)} frames, {elapsed:.0f}s)")
    return out_video


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Animate DST stills with LivePortrait for training diversity"
    )
    p.add_argument(
        "--driving",
        type=Path,
        default=None,
        help="Path to driving video. Defaults to LP bundled sample (d0.mp4).",
    )
    p.add_argument(
        "--src-dir",
        type=Path,
        default=PROJECT_DIR,
        help="Directory containing source stills (default: project root)",
    )
    p.add_argument(
        "--src-glob",
        default="[123456].png",
        help="Glob for source stills (default: [123456].png)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DST_OUT,
        help=f"Output directory (default: {DST_OUT})",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=800,
        help="Max frames to extract per still (default: 800)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=EXTRACT_STRIDE,
        help=f"Extract every N-th frame from LP output (default: {EXTRACT_STRIDE})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _check_environment()
    _ensure_ffmpeg_for_lp()

    inference_py = _find_inference_script()
    print(f"LivePortrait inference: {inference_py}")

    # Resolve driving video list.
    if args.driving is not None:
        if not args.driving.exists():
            print(f"Driving video not found: {args.driving}")
            sys.exit(1)
        drivers = [args.driving]
        print(f"Driving video: {args.driving}")
    else:
        drivers = _find_sample_drivers()
        if not drivers:
            print(
                "No driving videos found in LP repo and none specified.\n"
                f"Place a video at {DRIVING_DIR / 'driving.mp4'} and re-run with:\n"
                "  --driving training_data\\driving\\driving.mp4"
            )
            sys.exit(1)
        print(f"Using {len(drivers)} bundled sample driver(s):")
        for d in drivers:
            print(f"  {d.name}")

    stills = sorted(args.src_dir.glob(args.src_glob))
    if not stills:
        print(f"No source stills matched {args.src_dir}/{args.src_glob}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = _make_face_detector()

    # Frames-per-combination cap: spread max_frames evenly across all drivers.
    frames_per_combo = max(1, args.max_frames // len(drivers))

    print(f"\nSource stills      : {len(stills)}")
    print(f"Driving videos     : {len(drivers)}")
    print(f"Frames per combo   : {frames_per_combo}  (--max-frames {args.max_frames} / {len(drivers)} drivers)")
    print(f"Frame stride       : {args.stride}")
    print(f"Output             : {args.out_dir}\n")

    total_saved = 0
    t0 = time.monotonic()

    for still in stills:
        print(f"\n[{still.name}]")
        still_saved = 0

        for driver in drivers:
            # LP writes output under a temp subdir to avoid filename collisions.
            with tempfile.TemporaryDirectory(prefix="lp_out_") as tmp_dir:
                out_video = _run_liveportrait(
                    inference_py,
                    source=still,
                    driving=driver,
                    output_dir=Path(tmp_dir),
                )
                if out_video is None:
                    print(f"  Skipping {still.name} × {driver.name} — LP failed.")
                    continue

                # Use existing count as start_idx so filenames never collide.
                existing = len(list(args.out_dir.glob(f"lp_{still.stem}_*.jpg")))
                saved = _extract_frames_from_video(
                    video_path=out_video,
                    out_dir=args.out_dir,
                    stem=still.stem,
                    start_idx=existing,
                    max_frames=frames_per_combo,
                    stride=args.stride,
                    detector=detector,
                )

            still_saved += saved
            total_saved += saved
            print(f"  {driver.name}: {saved} frames  (still total: {still_saved}, overall: {total_saved})")

    elapsed = time.monotonic() - t0
    existing_classical = len(list(args.out_dir.glob("[123456]_*.jpg")))

    print(f"\n{'='*60}")
    print(f"LivePortrait frames added : {total_saved}")
    print(f"Classical augment frames  : {existing_classical}")
    print(f"Total DST dataset         : {total_saved + existing_classical} frames")
    print(f"Elapsed                   : {elapsed:.0f}s")
    print(f"\nOutput: {args.out_dir}/")
    print(f"\nNow proceed to Phase 2b (SRC recording) or Phase 3 (DFL training).")


if __name__ == "__main__":
    main()
