"""Court lines - find the baseline the server stands behind.

MVP scope: we only need the baseline near the server's feet to answer
"is the player behind / in front of the baseline" (foot fault, landing
inside the court) and to draw the court reference in the visuals.
White court lines = high brightness + low saturation; a probabilistic
Hough transform finds line segments in a band around the feet and we
keep the long, near-horizontal one closest to the feet.

An optional manual override (two clicked points) can be supplied from
the front end when automatic detection fails.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Line = Tuple[float, float, float, float]


def white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))


def line_y_at(line: Line, x: float) -> float:
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def detect_baseline(bgr: np.ndarray, feet_xy: Sequence[float],
                    player_h: float) -> Optional[Line]:
    """Long near-horizontal white line closest to the player's feet."""
    H, W = bgr.shape[:2]
    fx, fy = feet_xy
    y0, y1 = int(max(0, fy - 0.5 * player_h)), int(min(H, fy + 0.5 * player_h))
    band = white_mask(bgr)[y0:y1]
    segs = cv2.HoughLinesP(band, 1, np.pi / 360, threshold=25,
                           minLineLength=int(0.25 * W), maxLineGap=25)
    if segs is None:
        return None
    best, best_d = None, 1e9
    for x1, y1_, x2, y2_ in np.asarray(segs).reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2_ - y1_, x2 - x1))
        if abs(ang) > 30 and abs(abs(ang) - 180) > 30:
            continue
        line = (float(x1), float(y1_ + y0), float(x2), float(y2_ + y0))
        d = abs(line_y_at(line, fx) - fy)
        if d < best_d:
            best, best_d = line, d
    if best is None:
        return None
    # extend across the frame for drawing
    return (0.0, line_y_at(best, 0.0), float(W), line_y_at(best, float(W)))


def side_of_baseline(line: Optional[Line], foot_xy: Sequence[float],
                     camera_side: str = "far") -> Optional[str]:
    """'behind' or 'inside' (on the court side) for one foot.

    ``camera_side='far'``: the camera films from across the net, so the
    court is *below* the baseline in the image and "behind the baseline"
    means a smaller y. Use ``'near'`` when filming from behind the server.
    """
    if line is None:
        return None
    dy = foot_xy[1] - line_y_at(line, foot_xy[0])
    on_court = dy > 0 if camera_side == "far" else dy < 0
    return "inside" if on_court else "behind"


def detect_baseline_multi(frames, feet_xy, player_h: float) -> Optional[Line]:
    """Median baseline over several frames (single frames are noisy)."""
    lines = [detect_baseline(f, feet_xy, player_h) for f in frames]
    lines = [l for l in lines if l is not None]
    if len(lines) < max(2, len(frames) // 4):
        return None
    W = lines[0][2]
    return (0.0, float(np.median([l[1] for l in lines])), W,
            float(np.median([l[3] for l in lines])))
