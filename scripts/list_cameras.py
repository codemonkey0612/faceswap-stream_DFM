"""Diagnostic: list all camera devices by name AND probe OpenCV modes.

Uses PowerShell to enumerate Windows camera devices by friendly name, then
tries each OpenCV backend+index pair at a few common resolutions.

Run:
    .venv\\Scripts\\python.exe scripts\\list_cameras.py
"""

from __future__ import annotations

import subprocess
import time

import cv2


BACKENDS = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
]

MODES: list[tuple[int, int]] = [
    (1920, 1080),
    (1280, 720),
    (3840, 2160),
    (640, 480),
]


def list_windows_cameras() -> list[str]:
    """Ask Windows (via PowerShell) for friendly names of connected cameras."""
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-PnpDevice -Class Camera,Image -Status OK | "
        "Select-Object -Property FriendlyName,Status | "
        "Format-Table -AutoSize | Out-String -Width 200",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [f"<PowerShell query failed: {exc}>"]


def probe_mode(backend: int, index: int, w: int, h: int) -> str:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return "open_failed"
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_val >> (8 * i)) & 0xFF) for i in range(4))

        # Only measure sustained FPS if we got the requested resolution.
        measured_fps: float | None = None
        if actual_w == w and actual_h == h:
            ok, _ = cap.read()
            if ok:
                t0 = time.monotonic()
                frames = 0
                for _ in range(15):
                    ok, _ = cap.read()
                    if ok:
                        frames += 1
                dt = time.monotonic() - t0
                if dt > 0:
                    measured_fps = frames / dt

        if measured_fps is not None:
            return f"{actual_w}x{actual_h} fmt={fourcc} set_fps={actual_fps:.1f} measured={measured_fps:.1f}fps  [OK]"
        return f"{actual_w}x{actual_h} fmt={fourcc} set_fps={actual_fps:.1f}  [resolution not honored]"
    finally:
        cap.release()


def probe_index(backend_name: str, backend: int, index: int) -> bool:
    cap = cv2.VideoCapture(index, backend)
    opened = cap.isOpened()
    name = ""
    if opened:
        try:
            name = cap.getBackendName()
        except Exception:
            pass
    cap.release()
    if not opened:
        return False

    print(f"\n[{backend_name}] index={index}  (backend={name})")
    for w, h in MODES:
        result = probe_mode(backend, index, w, h)
        print(f"  request {w:>4}x{h:<4}  ->  {result}")
    return True


def main() -> None:
    print("=" * 70)
    print("Windows camera devices (via PowerShell Get-PnpDevice):")
    print("=" * 70)
    for line in list_windows_cameras():
        print("  " + line)

    print()
    print("=" * 70)
    print("OpenCV probe (MSMF + DSHOW, indices 0-4, several modes)")
    print("=" * 70)
    for backend_name, backend in BACKENDS:
        for idx in range(5):
            probe_index(backend_name, backend, idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
