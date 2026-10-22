"""Stage 2 - Skeleton (pose) model: MediaPipe BlazePose, Apache-2.0.

On court footage the player is often < 120 px tall, which is too small for
a full-frame pose model (the old pipeline detected a person in only ~60 %
of frames). We therefore run BlazePose on a zoomed crop around the player
box from the person detector, then map every landmark back to FULL-FRAME
pixel coordinates. The full frame is still used for ball + court, so
nothing is lost by zooming.

Output: ``(33, 3)`` array ``[x_px, y_px, confidence]`` in BlazePose order
(33 landmarks incl. heels and toes, which we need for foot placement).
``to_coco17`` converts to the standard COCO-17 layout for interoperability
(e.g. a Hailo YOLOX-pose / RTMPose model later).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_MODEL_PATH = os.path.join("models", "pose_landmarker.task")

# BlazePose landmark indices used by the analysis.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
    (27, 31), (28, 30), (30, 32), (28, 32), (0, 11), (0, 12),
]

# Standard COCO-17 order: 0 nose, 1-4 eyes/ears, 5/6 shoulders, 7/8 elbows,
# 9/10 WRISTS, 11/12 hips, 13/14 knees, 15/16 ANKLES.
# (The previous code put wrists at 15/16, which are ankles in real COCO.)
BLAZEPOSE_TO_COCO = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 11: 5, 12: 6, 13: 7, 14: 8,
                     15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16}


def to_coco17(pose33: np.ndarray) -> np.ndarray:
    """(33, 3) BlazePose array -> (17, 3) standard COCO-17 array."""
    out = np.zeros((17, 3), dtype=np.float64)
    for b, c in BLAZEPOSE_TO_COCO.items():
        out[c] = pose33[b]
    return out


def crop_window(box: Sequence[float], frame_w: int, frame_h: int,
                scale: float = 1.9):
    """Square crop around a player box, with head-room for raised arms."""
    x0, y0, x1, y1 = box[:4]
    h = max(y1 - y0, 1.0)
    side = h * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.12 * h  # shift up: arm/racket above head
    X0 = int(max(0, cx - side / 2)); Y0 = int(max(0, cy - side / 2))
    X1 = int(min(frame_w, cx + side / 2)); Y1 = int(min(frame_h, cy + side / 2))
    return X0, Y0, X1, Y1


class PoseModelBackend:
    """BlazePose (MediaPipe Tasks, VIDEO mode) on a zoomed player crop."""

    def __init__(self, model_path: Optional[str] = None, crop_px: int = 384,
                 min_detection_confidence: float = 0.3):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("mediapipe is not installed: pip install mediapipe") from e
        model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found at {model_path!r} - run: python tools/download_models.py")
        self._mp = mp
        self.crop_px = crop_px
        self._lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        ))
        self._last_ts = -1

    def detect(self, bgr: np.ndarray, box: Sequence[float],
               timestamp_ms: int) -> Optional[np.ndarray]:
        """Pose for the player inside ``box``; (33, 3) full-frame px or None."""
        H, W = bgr.shape[:2]
        X0, Y0, X1, Y1 = crop_window(box, W, H)
        crop = bgr[Y0:Y1, X0:X1]
        if crop.size == 0:
            return None
        s = self.crop_px / max(crop.shape[:2])
        big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        rgb = np.ascontiguousarray(big[..., ::-1])
        ts = max(int(timestamp_ms), self._last_ts + 1)  # VIDEO mode needs increasing ts
        self._last_ts = ts
        res = self._lm.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.pose_landmarks:
            return None
        lms = res.pose_landmarks[0]
        ch, cw = crop.shape[:2]
        return np.array([[X0 + p.x * cw, Y0 + p.y * ch,
                          float(max(p.visibility or 0.0, p.presence or 0.0))]
                         for p in lms], dtype=np.float64)

    def close(self) -> None:
        if getattr(self, "_lm", None) is not None:
            self._lm.close()
            self._lm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
