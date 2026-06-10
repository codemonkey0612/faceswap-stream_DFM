"""CLI entry point."""

from __future__ import annotations

import signal
import sys

# Register CUDA DLL paths BEFORE importing anything that pulls in onnxruntime,
# so the DFM swap runs on GPU regardless of how this is launched.
from src.utils.cuda_paths import ensure_cuda_dll_path

_CUDA_DIRS = ensure_cuda_dll_path()

import typer

from src.capture.webcam import Webcam
from src.config import load_config
from src.detection.face_detector import FaceDetector
from src.failsafe import Gate
from src.occlusion.face_parser import FaceParser
from src.occlusion.hair_recolor import HairRecolor
from src.occlusion.hand_masker import HandMasker
from src.occlusion.xseg_masker import XSegMasker
from src.output.preview import PreviewSink
from src.output.virtual_camera import VirtualCamera
from src.pipeline import Pipeline
from src.swap.dfm_loader import DFMLoader
from src.swap.dfm_swapper import DFMSwapper
from src.utils.logger import configure_logging, get_logger

app = typer.Typer(add_completion=False)


@app.command()
def main(
    profile: str | None = typer.Option(None, help="Config profile to overlay: live | debug (omit to use default.yaml only)"),
    no_vcam: bool = typer.Option(False, "--no-vcam", help="Use local preview window instead of OBS virtual camera (dev only)"),
) -> None:
    config = load_config(profile=profile if profile else None)
    configure_logging(
        level=config.logging.level,
        json_format=config.logging.format == "json",
    )
    log = get_logger("main")
    log.info("starting", profile=profile, cuda_dll_dirs=len(_CUDA_DIRS))

    detector = FaceDetector(
        model_path=config.detection.model,
        confidence_threshold=config.detection.confidence_threshold,
        nms_threshold=config.detection.nms_threshold,
        top_k=config.detection.top_k,
        detect_scale=config.detection.detect_scale,
    )

    loader = DFMLoader(
        model_path=config.swap.dfm_path,
        providers=config.swap.providers,
    )
    swapper = DFMSwapper(loader, output_size=config.swap.input_size)

    hand_masker  = HandMasker()   # gracefully disabled if model not found
    xseg_masker  = XSegMasker()  # gracefully disabled if models/xseg.onnx absent
    # stride=1 on GPU (every frame); stride=4 on CPU (BiSeNet 512x512 is slow without CUDA)
    import onnxruntime as _ort
    _parser_stride = 1 if "CUDAExecutionProvider" in _ort.get_available_providers() else 4
    face_parser  = FaceParser(stride=_parser_stride)  # gracefully disabled if model absent

    hr_cfg = config.hair_recolor
    hair_recolor = HairRecolor(
        target_bgr=tuple(hr_cfg.target_bgr),  # type: ignore[arg-type]
        blend=hr_cfg.blend,
        mask_threshold=hr_cfg.mask_threshold,
        min_saturation=hr_cfg.min_saturation,
        enabled=hr_cfg.enable,
    )

    webcam = Webcam(config.capture)

    if no_vcam:
        log.info("preview_mode", note="local window only, not sending to OBS")
        sink = PreviewSink()
    else:
        sink = VirtualCamera(
            width=config.capture.width,
            height=config.capture.height,
            fps=config.capture.fps,
            fmt=config.output.virtual_camera.fmt,
        )
        sink.start()

    try:
        gate = Gate(sink, config.capture.width, config.capture.height)
        pipeline = Pipeline(config, gate, webcam, detector, swapper,
                            hand_masker, xseg_masker, face_parser,
                            hair_recolor=hair_recolor)

        def _handle_sigint(*_: object) -> None:
            log.info("sigint_received")
            pipeline.stop()

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            pipeline.run()
        except Exception:
            log.exception("pipeline_crashed")
            sys.exit(1)
    finally:
        hand_masker.close()
        if hasattr(sink, "close"):
            sink.close()


if __name__ == "__main__":
    app()
