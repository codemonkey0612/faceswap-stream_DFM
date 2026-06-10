"""Main pipeline orchestrator.

Phase 1g: webcam -> face detection -> pre-swap failsafe -> DFM swap
         -> hand occlusion mask -> post-composite failsafe -> gate -> vcam.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import structlog

from src.beauty.skin_smoother import SkinSmoother
from src.capture.webcam import Webcam
from src.config import AppConfig
from src.detection.face_detector import FaceDetector
from src.failsafe import Gate, Monitor
from src.failsafe.monitor import MonitorConfig
from src.occlusion.face_parser import FaceParser
from src.occlusion.hair_recolor import HairRecolor
from src.occlusion.hand_masker import HandMasker
from src.occlusion.hand_reshaper import HandReshaper
from src.occlusion.xseg_masker import XSegMasker
from src.swap.dfm_swapper import DFMSwapper


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        gate: Gate,
        webcam: Webcam,
        detector: FaceDetector,
        swapper: DFMSwapper,
        hand_masker: HandMasker,
        xseg_masker: XSegMasker | None = None,
        face_parser: FaceParser | None = None,
        hair_recolor: HairRecolor | None = None,
        hand_reshaper: "HandReshaper | None" = None,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        self.config = config
        self.gate = gate
        self.webcam = webcam
        self.detector = detector
        self.swapper = swapper
        self.hand_masker = hand_masker
        self.xseg_masker  = xseg_masker  or XSegMasker()   # disabled if model absent
        self.face_parser  = face_parser  or FaceParser()   # disabled if model absent
        self.hair_recolor = hair_recolor or HairRecolor()  # enabled=False by default
        self.hand_reshaper = hand_reshaper or HandReshaper()  # enabled=False by default
        sc = config.beauty.smoothing
        self.skin_smoother = SkinSmoother(
            d=sc.d,
            sigma_color=sc.sigma_color,
            sigma_space=sc.sigma_space,
            enabled=config.beauty.enable,
        )
        self._log = logger or structlog.get_logger("pipeline")
        self._stop = threading.Event()

        fs = config.failsafe.pre_swap
        self._monitor = Monitor(
            MonitorConfig(
                min_confidence=fs.min_confidence,
                min_face_area_ratio=fs.min_face_area_ratio,
            )
        )

    def run(self) -> None:
        self.webcam.start()
        self._log.info("pipeline_started")
        frame_idx = 0
        fps_t0 = time.monotonic()
        fps_count = 0

        try:
            while not self._stop.is_set():
                frame = self.webcam.read(timeout=1.0)
                if frame is None:
                    self.gate.emit_black(frame_idx=frame_idx)
                    frame_idx += 1
                    continue

                # --- Phase 1e: detection + pre-swap failsafe ---
                det_error: BaseException | None = None
                detections = []
                try:
                    detections = self.detector.detect(frame)
                except Exception as exc:
                    det_error = exc
                    self._log.warning("detector_exception", exc=str(exc))

                pre_result = self._monitor.check_pre_swap(
                    frame_shape=frame.shape,
                    detections=detections,
                    detector_error=det_error,
                )

                if pre_result.fired:
                    self.gate.write(None, trigger_result=pre_result, frame_idx=frame_idx)
                    frame_idx += 1
                    fps_count += 1
                    continue

                # --- Phase 1f: DFM align → swap → paste back ---
                best = detections[0]  # highest-confidence face (sorted by detector)
                comp_error: BaseException | None = None
                swap_result = None
                try:
                    swap_result = self.swapper.process(frame, best)
                except Exception as exc:
                    comp_error = exc
                    self._log.warning("swap_exception", exc=str(exc))

                # Short-circuit: swap itself failed.
                if comp_error is not None:
                    err_result = self._monitor.check_post_composite(
                        composited=None,  # type: ignore[arg-type]
                        mask=None,        # type: ignore[arg-type]
                        detected_bbox=best.bbox,
                        swap_bbox=best.bbox,
                        expected_shape=frame.shape,
                        compositor_error=comp_error,
                    )
                    self.gate.write(None, trigger_result=err_result, frame_idx=frame_idx)
                    frame_idx += 1
                    fps_count += 1
                    continue

                # --- XSeg: restore non-face occlusions (mic, headset, objects) ---
                # occluder_mask=1 where XSeg detected a foreign object over the face.
                composited = swap_result.composited
                mask_for_check = swap_result.mask_full
                if self.xseg_masker.enabled:
                    occ_mask = self.xseg_masker.get_occluder_mask_fullframe(
                        swap_result.aligned_crop,
                        swap_result.affine_M,
                        frame.shape,
                    )
                    if occ_mask.max() > 0.01:
                        composited     = _restore_occluded(composited, frame, occ_mask)
                        mask_for_check = mask_for_check * (1.0 - occ_mask)

                # --- BiSeNet: restore hair / headwear crossing the face ---
                # hair_mask=1 where BiSeNet detected hair/hat class pixels.
                hair_mask: np.ndarray | None = None
                if self.face_parser.enabled:
                    hair_mask = self.face_parser.get_hair_mask_fullframe(
                        swap_result.aligned_crop,
                        swap_result.affine_M,
                        frame.shape,
                    )
                    if hair_mask.max() > 0.01:
                        composited     = _restore_occluded(composited, frame, hair_mask)
                        mask_for_check = mask_for_check * (1.0 - hair_mask)

                # --- Hair recolor: tint restored hair to AI character color ---
                if hair_mask is not None:
                    composited = self.hair_recolor.apply(composited, hair_mask)

                # --- Phase 1g: hand occlusion — restore real hands over swap ---
                # hand_mask=1 where hands are; those pixels revert to original frame.
                hand_mask = self.hand_masker.detect(frame)
                if hand_mask.max() > 0.0:
                    composited     = _restore_occluded(composited, frame, hand_mask)
                    mask_for_check = mask_for_check * (1.0 - hand_mask)

                # --- Hand reshape — privacy warp for distinctive finger geometry ---
                # Runs on the restored (real) hand pixels using the cached
                # MediaPipe landmarks. No-op unless enabled in config.
                if self.hand_reshaper.enabled:
                    hands_px = self.hand_masker.last_landmarks_px(
                        frame.shape[1], frame.shape[0]
                    )
                    # Diagnostic: log hand-detection state periodically so we can
                    # tell "no hand detected" from "reshape ran but invisible".
                    if frame_idx % 30 == 0:
                        self._log.info(
                            "hand_reshape_state",
                            hands_detected=len(hands_px),
                            hand_mask_max=float(hand_mask.max()),
                            reshaper_enabled=self.hand_reshaper.enabled,
                        )
                    if hands_px:
                        composited = self.hand_reshaper.apply(composited, hands_px)

                # --- Phase 1h: skin smoother — soften user's skin outside swap ---
                composited = self.skin_smoother.apply(composited, mask_for_check)

                post_result = self._monitor.check_post_composite(
                    composited=composited,
                    mask=mask_for_check,
                    detected_bbox=best.bbox,
                    swap_bbox=swap_result.swap_bbox,
                    expected_shape=frame.shape,
                    compositor_error=None,
                )

                if post_result.fired:
                    self.gate.write(None, trigger_result=post_result, frame_idx=frame_idx)
                else:
                    self.gate.write(composited, trigger_result=None, frame_idx=frame_idx)

                frame_idx += 1
                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= 5.0:
                    fps = fps_count / (now - fps_t0)
                    self._log.info(
                        "pipeline_fps",
                        fps=round(fps, 2),
                        gate=self.gate.stats,
                        capture=self.webcam.stats,
                    )
                    fps_t0 = now
                    fps_count = 0
        finally:
            self.webcam.stop()
            self._log.info("pipeline_stopped", stats=self.gate.stats)

    def stop(self) -> None:
        self._stop.set()


def _restore_occluded(
    composited: np.ndarray,
    original: np.ndarray,
    occluder_mask: np.ndarray,
) -> np.ndarray:
    """Blend original frame back where an occluder (hand, hair, XSeg object) is detected.

    composited    : (H, W, 3) uint8 — frame after face swap
    original      : (H, W, 3) uint8 — unmodified webcam frame
    occluder_mask : (H, W) float32 in [0, 1] — 1.0 = occluder pixel

    Returns (H, W, 3) uint8 with swap pixels replaced by originals in occluded regions.
    """
    alpha = occluder_mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting
    out = alpha * original.astype(np.float32) + (1.0 - alpha) * composited.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
