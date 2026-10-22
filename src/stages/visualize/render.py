"""Visual outputs (OpenCV + Pillow, both permissive licences).

Per serve:
- ``keyframes.jpg``  ready / trophy / contact / finish, zoomed on the player,
  with skeleton, ball trajectory, contact point, baseline and a height ruler.
- ``skeleton.png``   the same four moments as clean skeletons with joint heights.
- ``serve.gif``      animated sequence (slowed down, pauses on contact).
- ``overview.jpg``   full frame at contact: player box, ball path, baseline.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..pose_model.pose_model_backend import (
    L_HEEL, L_HIP, L_KNEE, NOSE, R_ELBOW, R_HEEL, R_HIP, R_KNEE, R_WRIST, L_WRIST, SKELETON_EDGES)

HIT_ARM = {(12, 14), (14, 16)}
TOSS_ARM = {(11, 13), (13, 15)}
COL_BODY = (235, 200, 80)      # BGR - light blue
COL_HIT = (60, 90, 255)        # red-orange
COL_TOSS = (80, 220, 120)      # green
COL_BALL = (40, 230, 255)      # yellow
COL_LINE = (255, 255, 255)


def serve_crop(boxes: List[Sequence[float]], ball, W: int, H: int, aspect: float = 0.75,
               tight: bool = False):
    """Fixed crop that holds the player, the toss apex and the contact point."""
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]; ys1 = [b[3] for b in boxes]
    ph = float(np.median([b[3] - b[1] for b in boxes]))
    x0, y0, x1, y1 = min(xs0), min(ys0), max(xs1), max(ys1)
    if ball is not None:
        for p in ([ball.apex_xy] if ball.apex_xy else []) + ([ball.contact_xy] if ball.contact_xy else []):
            x0, x1, y0 = min(x0, p[0]), max(x1, p[0]), min(y0, p[1])
        bn = getattr(ball, "bounce", None)        # bounce near the player: keep it in view
        if bn is not None and x0 - 1.2 * ph <= bn[1] <= x1 + 1.2 * ph and bn[2] <= y1 + 1.0 * ph:
            x0, x1, y1 = min(x0, bn[1]), max(x1, bn[1]), max(y1, bn[2])
    if tight:
        x0 -= 0.35 * ph; x1 += 0.35 * ph; y0 -= 0.2 * ph; y1 += 0.12 * ph
    else:
        x0 -= 0.6 * ph; x1 += 0.6 * ph; y0 -= 0.35 * ph; y1 += 0.25 * ph
    h = y1 - y0; w = x1 - x0
    if w / h < aspect:
        cx = (x0 + x1) / 2; w = h * aspect; x0, x1 = cx - w / 2, cx + w / 2
    else:
        cy = (y0 + y1) / 2; h = w / aspect; y0, y1 = cy - h / 2, cy + h / 2
    # shift inside the frame
    dx = max(0, -x0) - max(0, x1 - W); dy = max(0, -y0) - max(0, y1 - H)
    x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
    return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))


class Painter:
    def __init__(self, crop, out_h: int = 520):
        self.x0, self.y0, x1, y1 = crop
        self.s = out_h / max(y1 - self.y0, 1)
        self.size = (int((x1 - self.x0) * self.s), out_h)
        self.crop = crop

    def P(self, x, y) -> Tuple[int, int]:
        return int((x - self.x0) * self.s), int((y - self.y0) * self.s)

    def base(self, frame: np.ndarray) -> np.ndarray:
        """``frame`` may be the full frame or an already-cut crop."""
        x0, y0, x1, y1 = self.crop
        if frame.shape[:2] != (y1 - y0, x1 - x0):
            frame = frame[y0:y1, x0:x1]
        return cv2.resize(frame, self.size, interpolation=cv2.INTER_CUBIC)


def _text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(img, pose, pt, gate=0.3, thick=3):
    if pose is None:
        return
    for a, b in SKELETON_EDGES:
        if pose[a, 2] < gate or pose[b, 2] < gate:
            continue
        col = COL_HIT if (a, b) in HIT_ARM else COL_TOSS if (a, b) in TOSS_ARM else COL_BODY
        cv2.line(img, pt(*pose[a, :2]), pt(*pose[b, :2]), col, thick, cv2.LINE_AA)
    for i in range(11, 33):
        if pose[i, 2] >= gate:
            cv2.circle(img, pt(*pose[i, :2]), thick, (255, 255, 255), -1, cv2.LINE_AA)


def draw_ruler(img, pt, scale, crop_x0):
    """Height ruler (metres at the player's depth) on the left edge."""
    for m in range(0, 4):
        y = scale.ground_y - m / scale.m_per_px
        _, py = pt(crop_x0, y)
        if 0 < py < img.shape[0]:
            cv2.line(img, (0, py), (18, py), COL_LINE, 2)
            _text(img, f"{m} m", (22, py + 5), 0.45)


def draw_ball(img, pt, ball, t: float, show_all_toss: bool = True):
    if ball is None or (not ball.toss and not ball.outgoing and not ball.contact_xy):
        return
    toss = ball.toss
    if show_all_toss:
        for (_, x, y) in toss:
            cv2.circle(img, pt(x, y), 2, (150, 150, 150), -1, cv2.LINE_AA)
    past = [(x, y) for (f, x, y) in toss if f <= t]
    for i in range(1, len(past)):
        cv2.line(img, pt(*past[i - 1]), pt(*past[i]), COL_BALL, 2, cv2.LINE_AA)
    cur = [p for p in toss if p[0] == int(round(t))]
    if cur:
        cv2.circle(img, pt(cur[0][1], cur[0][2]), 7, COL_BALL, 2, cv2.LINE_AA)
    if getattr(ball, "bounce", None) is not None and t >= ball.bounce[0]:
        bx, by = pt(ball.bounce[1], ball.bounce[2])
        cv2.drawMarker(img, (bx, by), (255, 200, 0), cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
        _text(img, "bounce", (bx + 8, by + 4), 0.4, (255, 220, 120))
    if ball.contact_xy and ball.contact_frame is not None and t >= ball.contact_frame - 0.5:
        cx, cy = pt(*ball.contact_xy)
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_STAR, 26, 2, cv2.LINE_AA)
        if abs(t - ball.contact_frame) <= 1.5:
            _text(img, "hitting point", (cx + 14, cy - 10), 0.45, (0, 255, 255))
        out = [(x, y) for (f, x, y) in ball.outgoing if f <= t + 0.01]
        if out:
            pts = [ball.contact_xy] + out
            for i in range(1, len(pts)):
                cv2.arrowedLine(img, pt(*pts[i - 1]), pt(*pts[i]), (0, 140, 255), 2,
                                cv2.LINE_AA, tipLength=0.25 if i == len(pts) - 1 else 0)


def draw_trajectory(img, ball, fps: float, scale=None, first="ball in"):
    """Full-frame ball trajectory: incoming path (fading in), bounce, hitting
    point and the ball leaving, each labelled; plus a small legend."""
    if ball is None:
        return img
    h, w = img.shape[:2]
    lw = max(2, int(round(w / 500)))
    fs = max(0.45, w / 2200)
    P = lambda x, y: (int(round(x)), int(round(y)))
    inc = [(f, x, y) for (f, x, y) in ball.toss if ball.contact_frame is None or f <= ball.contact_frame]
    for i in range(1, len(inc)):
        a = 0.35 + 0.65 * i / max(len(inc) - 1, 1)          # older = dimmer
        col = tuple(int(c * a) for c in COL_BALL)
        gap = inc[i][0] - inc[i - 1][0] > 1
        if gap:     # frames where the ball was not seen: dotted
            cv2.line(img, P(*inc[i - 1][1:]), P(*inc[i][1:]), col, 1, cv2.LINE_AA)
        else:
            cv2.line(img, P(*inc[i - 1][1:]), P(*inc[i][1:]), col, lw, cv2.LINE_AA)
    for (_, x, y) in inc[:: max(1, int(fps / 30))]:
        cv2.circle(img, P(x, y), lw + 1, COL_BALL, -1, cv2.LINE_AA)
    bn = getattr(ball, "bounce", None)
    if inc and (bn is None or np.hypot(inc[0][1] - bn[1], inc[0][2] - bn[2]) > 30):
        _text(img, first, (P(*inc[0][1:])[0] + 8, P(*inc[0][1:])[1] - 6), fs, COL_BALL)
    if getattr(ball, "bounce", None) is not None:
        b = P(ball.bounce[1], ball.bounce[2])
        cv2.drawMarker(img, b, (255, 200, 0), cv2.MARKER_TILTED_CROSS, 8 * lw, lw, cv2.LINE_AA)
        cv2.circle(img, b, 6 * lw, (255, 200, 0), lw, cv2.LINE_AA)
        _text(img, "bounce", (b[0] + 8 * lw, b[1] + 4), fs, (255, 220, 120))
    if ball.contact_xy:
        out = [ball.contact_xy] + [(x, y) for (_, x, y) in ball.outgoing]
        for i in range(1, len(out)):
            cv2.arrowedLine(img, P(*out[i - 1]), P(*out[i]), (0, 140, 255), lw, cv2.LINE_AA,
                            tipLength=0.3 if i == len(out) - 1 else 0)
        c = P(*ball.contact_xy)
        cv2.drawMarker(img, c, (0, 255, 255), cv2.MARKER_STAR, 12 * lw, lw, cv2.LINE_AA)
        txt = "hitting point"
        if scale is not None:
            txt += f" {scale.height_m(ball.contact_xy[1]):.2f} m"
        _text(img, txt, (c[0] + 8 * lw, c[1] - 6 * lw), fs, (0, 255, 255))
    # legend
    items = [(first, COL_BALL), ("bounce", (255, 200, 0)), ("hitting point", (0, 255, 255)),
             ("ball out", (0, 140, 255))]
    if getattr(ball, "bounce", None) is None:
        items = [it for it in items if it[0] != "bounce"]
    y = int(28 * fs * 2)
    for lab, col in items:
        cv2.line(img, (14, y - 5), (40, y - 5), col, lw + 1, cv2.LINE_AA)
        _text(img, lab, (48, y), fs, (255, 255, 255))
        y += int(26 * fs * 2)
    return img


COL_RACKET = (250, 250, 250)


def draw_racket(img, pt, racket, pose, scale_px: float = 1.0):
    """Racquet 'skeleton': handle from the wrist to the head + an oval head.

    ``racket`` = (box, head_xy) from the detector (None -> nothing drawn).
    """
    if racket is None or pose is None:
        return
    box, head = racket
    wx, wy = pose[R_WRIST, :2]
    w = pt(wx, wy); h = pt(*head)
    dx, dy = h[0] - w[0], h[1] - w[1]
    L = max(float(np.hypot(dx, dy)), 1.0)
    ux, uy = dx / L, dy / L
    neck = (int(w[0] + ux * 0.45 * L), int(w[1] + uy * 0.45 * L))
    cv2.line(img, w, neck, COL_RACKET, 3, cv2.LINE_AA)
    centre = (int(w[0] + ux * 0.8 * L), int(w[1] + uy * 0.8 * L))
    ang = float(np.degrees(np.arctan2(uy, ux)))
    cv2.ellipse(img, centre, (int(0.36 * L), int(0.24 * L)), ang, 0, 360, COL_RACKET, 2, cv2.LINE_AA)
    cv2.line(img, neck, (int(centre[0] - ux * 0.36 * L), int(centre[1] - uy * 0.36 * L)), COL_RACKET, 2, cv2.LINE_AA)


def draw_baseline(img, pt, baseline, W):
    if baseline is None:
        return
    a, b = pt(baseline[0], baseline[1]), pt(baseline[2], baseline[3])
    n = 24
    for i in range(n):
        if i % 2 == 0:
            p = (a[0] + (b[0] - a[0]) * i // n, a[1] + (b[1] - a[1]) * i // n)
            q = (a[0] + (b[0] - a[0]) * (i + 1) // n, a[1] + (b[1] - a[1]) * (i + 1) // n)
            cv2.line(img, p, q, COL_LINE, 2, cv2.LINE_AA)


def annotated_frame(frame, pose, painter, ball, t, baseline, scale, label=None, sub=None,
                    racket=None):
    img = painter.base(frame)
    pt = painter.P
    draw_baseline(img, pt, baseline, frame.shape[1])
    draw_ruler(img, pt, scale, painter.x0)
    draw_skeleton(img, pose, pt)
    draw_racket(img, pt, racket, pose)
    draw_ball(img, pt, ball, t)
    if label:
        _text(img, label, (10, 26), 0.75, (255, 255, 255), 2)
    if sub:
        for i, s in enumerate(sub):
            _text(img, s, (10, 52 + 22 * i), 0.5, (230, 230, 230))
    return img


def skeleton_panel(pose, painter, ball, t, scale, label, racket=None) -> np.ndarray:
    w, h = painter.size
    img = np.full((h, w, 3), 24, np.uint8)
    pt = painter.P
    gy = pt(0, scale.ground_y)[1]
    cv2.line(img, (0, gy), (w, gy), (90, 90, 90), 1)
    draw_ruler(img, pt, scale, painter.x0)
    draw_skeleton(img, pose, pt, thick=3)
    draw_racket(img, pt, racket, pose)
    draw_ball(img, pt, ball, t, show_all_toss=False)
    if pose is not None:
        for idx, name in ((R_WRIST, "R wrist"), (L_WRIST, "L wrist"), (L_HIP, "hip"), (L_KNEE, "knee")):
            if pose[idx, 2] >= 0.3:
                x, y = pt(*pose[idx, :2])
                _text(img, f"{scale.height_m(pose[idx, 1]):.2f} m", (x + 6, y), 0.38, (200, 200, 200))
    _text(img, label, (10, 26), 0.75, (255, 255, 255), 2)
    return img


def save_gif(frames_bgr: List[np.ndarray], durations_ms: List[int], path: str, width: int = 300):
    from PIL import Image

    ims = []
    for f in frames_bgr:
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ims.append(Image.fromarray(f[..., ::-1]).quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=durations_ms, loop=0,
                optimize=True)
