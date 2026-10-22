"""Player detector - YOLOX (Apache-2.0) via ONNX Runtime (MIT).

Why YOLOX and not Ultralytics YOLOv8/11: Ultralytics is AGPL-3.0, which
would force us to open-source the whole product (or buy a licence).
YOLOX code and its COCO weights are Apache-2.0, run on ONNX Runtime, and
YOLOX is in the Hailo Model Zoo, so the same model can move to the
Hailo-10H NPU later.

We only use the ``person`` class: on phone footage the ball (3-4 px) and
racket are too small for a COCO detector, so the ball is found by the
motion tracker instead (see ``ball_tracking/ball_tracker.py``).
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

PERSON = 0
TENNIS_RACKET = 38
Box = Tuple[float, float, float, float, float]  # x0, y0, x1, y1, score


class PersonDetector:
    def __init__(self, model_path: str = "models/yolox_tiny.onnx",
                 input_size: int = 416, score_threshold: float = 0.3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} missing - run: python tools/download_models.py")
        import onnxruntime as ort  # imported lazily: optional dependency

        # use a GPU / NPU when this machine has one (CoreML on a Mac, CUDA, DirectML),
        # always with the CPU as fallback
        fast = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                            "DmlExecutionProvider") if p in ort.get_available_providers()]
        try:
            self.session = ort.InferenceSession(model_path, providers=fast + ["CPUExecutionProvider"])
        except Exception:
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self.score_threshold = score_threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            n = input_size // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self._grids = np.concatenate(grids)
        self._strides = np.concatenate(strides)

    def detect(self, bgr: np.ndarray, cls: int = PERSON,
               score_threshold: Optional[float] = None) -> List[Box]:
        """All boxes of COCO class ``cls`` (default person), best first."""
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[: int(h * r), : int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)))
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.session.run(None, {self.input_name: x})[0][0]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(out[:, 2:4]) * self._strides
        thr = self.score_threshold if score_threshold is None else score_threshold
        score = out[:, 4] * out[:, 5 + cls]
        keep = score > thr
        if not keep.any():
            return []
        xy, wh, score = xy[keep] / r, wh[keep] / r, score[keep]
        rects = [[float(a - c / 2), float(b - d / 2), float(c), float(d)]
                 for (a, b), (c, d) in zip(xy, wh)]
        idx = np.array(cv2.dnn.NMSBoxes(rects, score.tolist(), thr, 0.45)).reshape(-1)
        boxes = [(rects[i][0], rects[i][1], rects[i][0] + rects[i][2],
                  rects[i][1] + rects[i][3], float(score[i])) for i in idx]
        return sorted(boxes, key=lambda b: -b[4])


def pick_player(boxes: List[Box], prev: Optional[Box]) -> Optional[Box]:
    """Choose the analysed player: closest to the previous box, else the
    tallest confident box (the server is usually the most prominent person)."""
    if not boxes:
        return None
    if prev is None:
        return max(boxes, key=lambda b: (b[3] - b[1]) * b[4])
    pcx, pcy = (prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2
    ph = prev[3] - prev[1]
    near = [b for b in boxes
            if abs((b[0] + b[2]) / 2 - pcx) < ph and abs((b[1] + b[3]) / 2 - pcy) < ph]
    if near:
        return max(near, key=lambda b: b[4])
    # lost track (player ran, or the box drifted): re-acquire the most prominent person
    return max(boxes, key=lambda b: (b[3] - b[1]) * b[4])


def find_racket(detector: "PersonDetector", bgr: np.ndarray, player_box: Sequence[float],
                wrist_xy: Sequence[float], min_score: float = 0.25):
    """Racquet on a zoomed crop around the player (it is too small full-frame).

    Returns ``(box, head_xy)`` in full-frame pixels or ``None``. The racquet
    head is the end of the box away from the hitting wrist.
    """
    x0, y0, x1, y1 = player_box[:4]
    h = y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = 1.8 * h
    X0, Y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    X1, Y1 = int(min(bgr.shape[1], cx + side / 2)), int(min(bgr.shape[0], cy + side / 2))
    crop = bgr[Y0:Y1, X0:X1]
    if crop.size == 0:
        return None
    boxes = detector.detect(crop, cls=TENNIS_RACKET, score_threshold=min_score)
    if not boxes:
        return None
    wx, wy = wrist_xy
    # prefer the racquet closest to the hitting wrist
    def dist(b):
        bx, by = X0 + (b[0] + b[2]) / 2, Y0 + (b[1] + b[3]) / 2
        return np.hypot(bx - wx, by - wy)
    b = min(boxes, key=dist)
    if dist(b) > 0.8 * h:
        return None
    box = (X0 + b[0], Y0 + b[1], X0 + b[2], Y0 + b[3], b[4])
    corners = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    far = max(corners, key=lambda c: np.hypot(c[0] - wx, c[1] - wy))
    head = (wx + 0.78 * (far[0] - wx), wy + 0.78 * (far[1] - wy))
    return box, (float(head[0]), float(head[1]))
