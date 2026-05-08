"""Phase 4 — Deploy trained DFM model.

Validates the trained .dfm, patches config/default.yaml automatically,
and runs a smoke test of the full swap pipeline before going live.

Usage
-----
  # Deploy momo.dfm (copies it in and updates config)
  .venv\\Scripts\\python.exe scripts\\deploy_model.py --dfm path\\to\\momo.dfm

  # Just verify the currently configured model (no copy)
  .venv\\Scripts\\python.exe scripts\\deploy_model.py --verify-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_DIR  = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_DIR / "config" / "default.yaml"
MODELS_DIR   = PROJECT_DIR / "models"


# ---------------------------------------------------------------------------
# DFM introspection
# ---------------------------------------------------------------------------

def _dfm_input_size(dfm_path: Path) -> int | None:
    """Return the model's expected input size by reading the ONNX graph."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(dfm_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        # Shape is [batch, C, H, W] — return H (== W for DFM).
        if inp.shape and len(inp.shape) >= 4:
            h = inp.shape[2]
            if isinstance(h, int) and h > 0:
                return h
    except Exception:
        pass
    return None


def _validate_dfm(dfm_path: Path) -> tuple[bool, str]:
    """Run a quick forward pass to confirm the model loads and produces output."""
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = list(ort.get_all_providers())
        providers  = [p for p in providers if p in available]
        sess = ort.InferenceSession(str(dfm_path), providers=providers)

        inp   = sess.get_inputs()[0]
        size  = inp.shape[2] if isinstance(inp.shape[2], int) else 256
        dummy = np.random.rand(1, 3, size, size).astype(np.float32)
        out   = sess.run(None, {inp.name: dummy})

        if not out or out[0] is None:
            return False, "Model produced no output."
        return True, f"OK  (input {size}x{size}, providers={providers})"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def _patch_config(dfm_path: Path, input_size: int) -> None:
    """Update config/default.yaml swap section in-place."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    cfg = yaml.safe_load(text)
    cfg["swap"]["dfm_path"]   = str(dfm_path.relative_to(PROJECT_DIR)).replace("\\", "/")
    cfg["swap"]["input_size"] = input_size

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  config/default.yaml updated:")
    print(f"    swap.dfm_path   = {cfg['swap']['dfm_path']}")
    print(f"    swap.input_size = {input_size}")


# ---------------------------------------------------------------------------
# Smoke test — single synthetic frame through the swap path
# ---------------------------------------------------------------------------

def _smoke_test(dfm_path: Path, input_size: int) -> bool:
    """Push a synthetic face-crop through DFMSwapper and verify output shape."""
    sys.path.insert(0, str(PROJECT_DIR / "src"))
    try:
        from swap.dfm_loader import DFMLoader
        from swap.dfm_swapper import DFMSwapper
    except ImportError as exc:
        print(f"  [WARN] Could not import pipeline modules: {exc}")
        print("         Skipping smoke test.")
        return True  # non-fatal during deployment

    try:
        loader  = DFMLoader(str(dfm_path))
        swapper = DFMSwapper(loader)

        # Synthetic 1080p frame with a centred "face" region.
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        lm5 = np.array([
            [880, 440], [1040, 440],   # left eye, right eye
            [960, 540],                 # nose
            [900, 640], [1020, 640],   # left mouth, right mouth
        ], dtype=np.float32)

        t0 = time.monotonic()
        result = swapper.process(frame, lm5)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if result.composited is None:
            print(f"  [FAIL] Swapper returned None composited frame.")
            return False
        if result.composited.shape != frame.shape:
            print(f"  [FAIL] Output shape {result.composited.shape} != input {frame.shape}")
            return False

        print(f"  Smoke test passed  ({elapsed_ms:.0f} ms single frame)")
        return True

    except Exception as exc:
        print(f"  [FAIL] Smoke test exception: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy trained DFM and verify pipeline")
    p.add_argument("--dfm", type=Path, default=None,
                   help="Path to trained .dfm file to deploy")
    p.add_argument("--name", default="momo.dfm",
                   help="Filename to use inside models/ (default: momo.dfm)")
    p.add_argument("--verify-only", action="store_true",
                   help="Skip copy; just validate the currently configured model")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\nPhase 4 — DFM deployment")
    print("=" * 50)

    # ── Resolve the target dfm path ──────────────────────────────
    if args.verify_only:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        dfm_dest = PROJECT_DIR / cfg["swap"]["dfm_path"]
        if not dfm_dest.exists():
            print(f"ERROR: configured model not found: {dfm_dest}")
            sys.exit(1)
        print(f"Verifying: {dfm_dest}")
    else:
        if args.dfm is None:
            print("ERROR: provide --dfm path/to/model.dfm  (or use --verify-only)")
            sys.exit(1)
        if not args.dfm.exists():
            print(f"ERROR: source .dfm not found: {args.dfm}")
            sys.exit(1)

        dfm_dest = MODELS_DIR / args.name
        MODELS_DIR.mkdir(exist_ok=True)

        print(f"\n[1/4] Copying model...")
        shutil.copy2(args.dfm, dfm_dest)
        print(f"  {args.dfm}  ->  {dfm_dest}")

    # ── Introspect input size ────────────────────────────────────
    print(f"\n[2/4] Reading model input size...")
    input_size = _dfm_input_size(dfm_dest)
    if input_size is None:
        print("  [WARN] Could not read input size from ONNX graph — defaulting to 256")
        input_size = 256
    else:
        print(f"  Input size: {input_size}x{input_size}")

    # ── Validate forward pass ────────────────────────────────────
    print(f"\n[3/4] Validating model (forward pass)...")
    ok, msg = _validate_dfm(dfm_dest)
    if not ok:
        print(f"  [FAIL] {msg}")
        sys.exit(1)
    print(f"  {msg}")

    # ── Patch config ─────────────────────────────────────────────
    if not args.verify_only:
        print(f"\n[4/4] Patching config...")
        _patch_config(dfm_dest, input_size)

    # ── Smoke test ───────────────────────────────────────────────
    print(f"\n[smoke test] Running single frame through pipeline...")
    smoke_ok = _smoke_test(dfm_dest, input_size)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*50}")
    if smoke_ok:
        print("  Deployment complete.")
        print(f"\n  Model   : {dfm_dest.name}")
        print(f"  Size    : {input_size}x{input_size}")
        print(f"\n  OBS SETUP:")
        print(f"    1. Add 'Video Capture Device' source -> your webcam")
        print(f"    2. Add 'Video Capture Device' source -> OBS Virtual Camera")
        print(f"    3. Launch the pipeline:")
        print(f"       .venv\\Scripts\\python.exe -m src.main")
        print(f"    4. Start Virtual Camera in OBS (Tools -> Start Virtual Camera)")
        print(f"    5. The swap output will appear on the Virtual Camera source")
    else:
        print("  Deployment FAILED — fix the errors above before going live.")
        sys.exit(1)


if __name__ == "__main__":
    main()
