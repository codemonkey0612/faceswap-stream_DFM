"""Make onnxruntime-gpu find its CUDA dependencies on Windows.

onnxruntime-gpu's provider DLL (onnxruntime_providers_cuda.dll) depends on
cuBLAS/cuDNN/cuFFT/etc. shipped as pip packages under site-packages/nvidia/*/bin.
Those dirs aren't on PATH by default, so CUDA silently fails to load and the
model falls back to CPU (~5 fps instead of GPU's 100+ fps).

Importing this module early adds every nvidia/*/bin dir to the DLL search path,
so CUDA works no matter how the app is launched (run_live.bat OR `python -m
src.main`). No-op on non-Windows or if the nvidia packages aren't installed.
"""

from __future__ import annotations

import os
import sys


def ensure_cuda_dll_path() -> list[str]:
    """Add site-packages/nvidia/*/bin to the DLL search path. Returns the dirs added."""
    added: list[str] = []
    try:
        import nvidia  # provided by nvidia-*-cu12 pip packages
    except Exception:
        return added

    for base in getattr(nvidia, "__path__", []):
        try:
            for pkg in os.listdir(base):
                bin_dir = os.path.join(base, pkg, "bin")
                if not os.path.isdir(bin_dir):
                    continue
                # Windows: register as a DLL directory (most reliable for ORT).
                if sys.platform == "win32":
                    try:
                        os.add_dll_directory(bin_dir)
                    except Exception:
                        pass
                # Also prepend to PATH (covers child processes / older loaders).
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                added.append(bin_dir)
        except Exception:
            continue
    return added
