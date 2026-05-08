"""Download the BiSeNet19 face-parser ONNX model.

Source: nznznz48/parsing_model on HuggingFace (public, no auth needed).
Output: models/face_parser.onnx  (~50 MB)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEST        = PROJECT_DIR / "models" / "face_parser.onnx"
REPO_ID     = "bluefoxcreation/Face_parsing_onnx"
FILENAME    = "faceparser_sim.onnx"   # ONNX-simplified variant — faster inference


def main() -> None:
    if DEST.exists():
        print(f"Already present: {DEST}")
        print("Delete it first if you want to re-download.")
        return

    print(f"Downloading BiSeNet19 face parser from HuggingFace...")
    print(f"  repo  : {REPO_ID}")
    print(f"  file  : {FILENAME}")
    print(f"  dest  : {DEST}\n")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Run:  .venv\\Scripts\\pip install huggingface_hub")
        sys.exit(1)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    import shutil
    shutil.copy2(tmp, DEST)
    print(f"\nSaved to: {DEST}")
    _verify(DEST)


def _verify(path: Path) -> None:
    print("\nVerifying model...")
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp  = sess.get_inputs()[0]
        size = inp.shape[2] if isinstance(inp.shape[2], int) else 512
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        out   = sess.run(None, {inp.name: dummy})[0]
        n_cls = out.shape[1]
        print(f"  Input  : {inp.shape}")
        print(f"  Output : {out.shape}  ({n_cls} classes)")
        if n_cls == 19:
            print("  Format : BiSeNet19  (correct)")
        else:
            print(f"  Format : {n_cls}-class model (check exclude_classes in FaceParser)")
        print("\nFace parser ready.")
    except Exception as exc:
        print(f"  [WARN] Verification failed: {exc}")
        print("  The file was saved but could not be tested.")


if __name__ == "__main__":
    main()
