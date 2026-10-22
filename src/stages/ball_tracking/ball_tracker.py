"""Stage 1b - Ball finder, toss trajectory and contact point (serve).

On phone footage the ball is 2-4 px and video compression destroys its
yellow colour, so neither a colour filter nor a COCO detector finds it.
It *is* visible as a small moving blob, so we:

1. ``motion_candidates`` - compensate hand-held camera shake (phase
   correlation) and keep tiny blobs that moved both prev->cur AND
   cur->next (static edges and shake cancel out).
2. ``fit_toss`` - the toss is free fall, so its image path is a parabola
   y(t) = a t^2 + b t + c with 2a = g in pixels/frame^2, and g in pixels
   is known from the player's height and the video fps. A RANSAC search
   for that parabola through the (t, x, y) candidate cloud ignores the
   flickering leaves, fence and arm blobs that defeat frame-to-frame
   linking.
3. ``find_contact`` - the toss ends where the racket hits the ball; the
   hit ball is the fast blob that appears right after. Contact point =
   midpoint of the last toss position and the first outgoing position,
   contact time = midpoint of those two frames.

Pure numpy/OpenCV - no licence issues.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Cand = Tuple[float, float, float]  # x, y, strength
Point = Tuple[int, float, float]   # frame, x, y


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)


def frame_shift(ref: np.ndarray, img: np.ndarray, small: int = 640) -> Tuple[float, float]:
    """Camera shake between two gray frames (dx, dy), in full-resolution pixels.

    Estimated by phase correlation on the whole frame (the static background
    dominates there), on a copy at most ``small`` px wide: same answer (sub-pixel), ~3.5x faster on 1280 px.
    """
    f = min(1.0, small / float(ref.shape[1]))
    if f < 1.0:
        size = (int(ref.shape[1] * f), int(ref.shape[0] * f))
        ref = cv2.resize(ref, size, interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    (dx, dy), _ = cv2.phaseCorrelate(ref, img)
    return dx / f, dy / f


def _shift(img: np.ndarray, d: Tuple[float, float]) -> np.ndarray:
    if abs(d[0]) < 0.05 and abs(d[1]) < 0.05:      # rock-steady tripod: nothing to do
        return img
    m = np.float32([[1, 0, -d[0]], [0, 1, -d[1]]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def _align(ref: np.ndarray, img: np.ndarray, roi=None) -> np.ndarray:
    """Shift ``img`` to line up with ``ref`` (hand-held camera shake)."""
    return _shift(img, frame_shift(ref, img))


def search_roi(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """Image region where the toss, contact and outgoing ball can be."""
    x0, y0, x1, y1 = box[:4]
    ph = y1 - y0
    return (int(max(0, x0 - 2.0 * ph)), int(max(0, y0 - 3.0 * ph)),
            int(min(w, x1 + 2.0 * ph)), int(min(h, y0 + 0.45 * ph)))


def motion_candidates(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray,
                      roi, threshold: float = 14.0, max_area: int = 40,
                      max_side: int = 9, shifts=None) -> List[Cand]:
    """Small blobs that moved between prev->cur AND cur->next (gray frames).

    ``shifts`` = (shake cur->prev, shake cur->next) if already known.
    """
    x0, y0, x1, y1 = roi
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    sp, sn = shifts if shifts is not None else (frame_shift(cur, prev), frame_shift(cur, nxt))
    p, n = _shift(prev, sp), _shift(nxt, sn)
    d = np.minimum(np.abs(cur - p), np.abs(cur - n))[y0:y1, x0:x1]
    mask = (d > threshold).astype(np.uint8)
    num, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, num):
        if 2 <= st[i, 4] <= max_area and max(st[i, 2], st[i, 3]) <= max_side:
            out.append((float(cen[i][0] + x0), float(cen[i][1] + y0),
                        float(d[lab == i].max())))
    return out


def drop_static(cands: Dict[int, List[Cand]], radius: float = 3.0,
                window: int = 6, max_hits: int = 8) -> Dict[int, List[Cand]]:
    """Remove flicker that sits still (leaves, cables, compression noise).

    A real ball lingers within ``radius`` px for at most ~6 frames around
    its apex; static noise reappears at the same spot in most frames.
    """
    out: Dict[int, List[Cand]] = {}
    arr = {t: np.array([c[:2] for c in cs]) for t, cs in cands.items() if cs}
    for t, cs in cands.items():
        keep = []
        for c in cs:
            hits = 0
            for k in range(-window, window + 1):
                if k == 0 or (t + k) not in arr:
                    continue
                if np.min(np.hypot(*(arr[t + k] - c[:2]).T)) < radius:
                    hits += 1
            if hits < max_hits:
                keep.append(c)
        out[t] = keep
    return out


def gravity_px(player_h_px: float, fps: float, player_height_m: float = 1.8) -> float:
    """Image-space gravity (px/frame^2) at the player's depth."""
    return 9.81 / (fps * fps) * (player_h_px / player_height_m)


@dataclass
class Toss:
    points: List[Point]                 # inlier ball positions, one per frame
    coef_y: Tuple[float, float, float]  # y(t) = a t^2 + b t + c   (t = frame)
    coef_x: Tuple[float, float]         # x(t) = m t + k

    def y(self, t):
        a, b, c = self.coef_y
        return a * t * t + b * t + c

    def x(self, t):
        return self.coef_x[0] * t + self.coef_x[1]

    @property
    def apex_frame(self) -> float:
        a, b, _ = self.coef_y
        return -b / (2 * a)


def fit_toss(cands: Dict[int, List[Cand]], player_h: float, fps: float,
             player_height_m: float = 1.8, tol: float = 3.0,
             min_inliers: int = 7, iters: int = 4000, seed: int = 0,
             center_x: Optional[float] = None, head_y: Optional[float] = None) -> Optional[Toss]:
    """RANSAC a gravity parabola through the (t, x, y) candidate cloud.

    ``center_x`` / ``head_y`` (player's image x and head height) reject
    "tosses" that are not above the server - e.g. swaying tree branches.
    """
    rows = [(t, c[0], c[1]) for t, cs in cands.items() for c in cs]
    if len(rows) < min_inliers:
        return None
    P = np.array(rows, dtype=np.float64)
    T, X, Y = P[:, 0], P[:, 1], P[:, 2]
    g = gravity_px(player_h, fps, player_height_m)
    max_vx = 0.03 * player_h + 1.0          # a toss drifts only slightly sideways
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    n = len(P)
    for _ in range(iters):
        i = int(rng.integers(0, n))
        # guided 2nd sample: a later candidate a toss could plausibly reach
        dts = T - T[i]
        ok = ((dts >= 3) & (dts <= 14) & (np.abs(X - X[i]) <= max_vx * dts + tol)
              & (np.abs(Y - Y[i]) <= 0.8 * player_h))
        cand_j = np.flatnonzero(ok)
        if cand_j.size == 0:
            continue
        j = int(cand_j[rng.integers(0, cand_j.size)])
        dt = T[j] - T[i]
        a = 0.5 * g * rng.uniform(0.7, 1.4)       # allow perspective / height error
        # y = a t^2 + b t + c through both samples
        b = ((Y[j] - a * T[j] ** 2) - (Y[i] - a * T[i] ** 2)) / dt
        c = Y[i] - a * T[i] ** 2 - b * T[i]
        m = (X[j] - X[i]) / dt
        if abs(m) > max_vx:
            continue
        k = X[i] - m * T[i]
        t_apex = -b / (2 * a)
        if center_x is not None and abs(m * t_apex + k - center_x) > 0.7 * player_h:
            continue
        if head_y is not None and (c - b * b / (4 * a)) > head_y:
            continue  # apex below the head -> not a serve toss
        err = np.hypot(a * T * T + b * T + c - Y, m * T + k - X)
        inl = err < tol
        frames = np.unique(T[inl])
        if len(frames) < min_inliers:
            continue
        span = float(np.ptp(Y[inl]))
        score = len(frames) * (1.0 + min(span / (0.5 * player_h), 1.0))
        if score > best_n:
            best_n, best = score, (inl, err)
    if best is None:
        return None
    inl, err = best
    # one point per frame (the best fitting) then a least-squares refit
    pts: Dict[int, Tuple[float, float, float]] = {}
    for idx in np.where(inl)[0]:
        t = int(T[idx])
        if t not in pts or err[idx] < pts[t][2]:
            pts[t] = (X[idx], Y[idx], err[idx])
    ts = np.array(sorted(pts), dtype=np.float64)
    xs = np.array([pts[int(t)][0] for t in ts]); ys = np.array([pts[int(t)][1] for t in ts])
    coef_y = tuple(np.polyfit(ts, ys, 2)) if len(ts) >= 3 else (0.5 * g, 0.0, ys.mean())
    if not (0.25 * g <= 2 * coef_y[0] <= 3.0 * g):
        return None  # not a free-fall curve after all
    coef_x = tuple(np.polyfit(ts, xs, 1))
    if center_x is not None:
        ta = -coef_y[1] / (2 * coef_y[0])
        if abs(coef_x[0] * ta + coef_x[1] - center_x) > 0.7 * player_h:
            return None
    return Toss([(int(t), float(x), float(y)) for t, x, y in zip(ts, xs, ys)],
                coef_y, coef_x)


@dataclass
class ServeBall:
    toss: List[Point] = field(default_factory=list)
    outgoing: List[Point] = field(default_factory=list)
    contact_xy: Optional[Tuple[float, float]] = None
    contact_frame: Optional[float] = None       # may be fractional (midpoint)
    apex_xy: Optional[Tuple[float, float]] = None
    apex_frame: Optional[float] = None
    method: str = "none"                        # "ball-midpoint" | "toss-end" | "none"
    bounce: Optional[Point] = None              # where the incoming ball hit the court


def _follow(cands, start: Point, v: Tuple[float, float], n: int = 3) -> List[Point]:
    """Follow a fast ball along a straight line for up to ``n`` frames."""
    out = [start]
    vx, vy = v
    for _ in range(n):
        t, x, y = out[-1]
        ex, ey = x + vx, y + vy
        near = [(np.hypot(c[0] - ex, c[1] - ey), c) for c in cands.get(t + 1, [])]
        near = [z for z in near if z[0] < 0.35 * np.hypot(vx, vy) + 5]
        if not near:
            break
        c = min(near, key=lambda z: z[0])[1]
        vx, vy = c[0] - x, c[1] - y
        out.append((t + 1, c[0], c[1]))
    return out


def fast_tracks(cands: Dict[int, List[Cand]], t_from: int, t_to: int,
                min_step: float, max_step: float, min_len: int = 3) -> List[List[Point]]:
    """Straight, fast-moving blob tracks (a hit ball) starting in [t_from, t_to]."""
    tracks = []
    for t in range(t_from, t_to + 1):
        for c in cands.get(t, []):
            for c2 in cands.get(t + 1, []):
                d = float(np.hypot(c2[0] - c[0], c2[1] - c[1]))
                if not (min_step <= d <= max_step):
                    continue
                tr = _follow(cands, (t + 1, c2[0], c2[1]), (c2[0] - c[0], c2[1] - c[1]), n=4)
                tr = [(t, c[0], c[1])] + tr
                if len(tr) >= min_len:
                    tracks.append(tr)
    return tracks


def find_contact(toss: Optional[Toss], cands: Dict[int, List[Cand]],
                 player_h: float, hand_peak_frame: Optional[int] = None) -> ServeBall:
    """Contact = midpoint of the last toss position and the first hit-ball position.

    1. Find fast straight blob tracks after the toss apex (hit-ball candidates).
    2. Keep those whose first point, stepped one frame back along their own
       velocity, lands on the toss parabola (the ball left the toss there).
    3. Prefer the longest such track, then the smallest back-step residual.
    The toss is cut at the frame before the hit ball first appears.
    """
    if toss is None:
        return ServeBall()
    pts = list(toss.points)
    apex_t = float(np.clip(toss.apex_frame, pts[0][0], pts[-1][0]))
    sb = ServeBall(toss=pts, apex_frame=apex_t,
                   apex_xy=(float(toss.x(apex_t)), float(toss.y(apex_t))))

    t_from = int(np.ceil(apex_t)) + 1
    t_to = int(pts[-1][0]) + 3
    best = None
    for tr in fast_tracks(cands, t_from, t_to, min_step=0.12 * player_h, max_step=1.2 * player_h):
        (t0, x0, y0), (t1, x1, y1) = tr[0], tr[1]
        vx, vy = x1 - x0, y1 - y0
        tb = t0 - 1                                    # last frame on the toss
        tx, ty = float(toss.x(tb)), float(toss.y(tb))
        # the hit ball must not just be the toss itself continuing
        if np.hypot(x0 - float(toss.x(t0)), y0 - float(toss.y(t0))) < 5:
            continue
        resid = float(np.hypot(x0 - vx - tx, y0 - vy - ty))
        if resid > 0.6 * np.hypot(vx, vy) + 8:
            continue
        # a struck ball leaves much faster and at a sharp angle to the falling toss
        tvx = float(toss.x(tb + 1) - toss.x(tb)); tvy = float(toss.y(tb + 1) - toss.y(tb))
        tsp, hsp = np.hypot(tvx, tvy), np.hypot(vx, vy)
        if hsp < 2.0 * tsp:
            continue
        cosang = (vx * tvx + vy * tvy) / max(hsp * tsp, 1e-6)
        if cosang > np.cos(np.radians(35)):
            continue
        key = (len(tr), -resid)
        if best is None or key > best[0]:
            best = (key, tb, (tx, ty), tr)
    if best is not None:
        _, tb, (tx, ty), tr = best
        kept = [p for p in pts if p[0] <= tb]
        sb.toss = kept if len(kept) >= 3 else pts
        sb.outgoing = tr
        sb.contact_xy = ((tx + tr[0][1]) / 2.0, (ty + tr[0][2]) / 2.0)
        sb.contact_frame = (tb + tr[0][0]) / 2.0
        sb.method = "ball-midpoint"
        return sb
    t_e, x_e, y_e = pts[-1]
    sb.contact_xy, sb.contact_frame, sb.method = (x_e, y_e), float(t_e), "toss-end"
    return sb


# --------------------------------------------------------------------------- #
# Groundstrokes (forehand mode)
# --------------------------------------------------------------------------- #
def ground_roi(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """Region around the player where an incoming / hit groundstroke ball can be."""
    x0, y0, x1, y1 = box[:4]
    ph = y1 - y0
    return (int(max(0, x0 - 1.6 * ph)), int(max(0, y0 - 0.6 * ph)),
            int(min(w, x1 + 1.6 * ph)), int(min(h, y1 + 1.2 * ph)))   # below feet: bounces


def find_groundstroke_contact(cands: Dict[int, List[Cand]], racket: Dict[int, Tuple[float, float]],
                              peak: int, fps: float, m_per_px: float, stature_px: float,
                              head: Optional[Dict[int, Tuple[float, float]]] = None) -> ServeBall:
    """Contact = midpoint of the last incoming and first outgoing ball positions.

    Ball tracks are straight, fast blob tracks near the swing. An incoming
    track ends near the racket, an outgoing one starts near it 1-4 frames
    later and leaves at a clearly different angle (the ball was struck).
    """
    lo, hi = int(peak - 0.6 * fps), int(peak + 0.35 * fps)
    if head:   # a moving face / hair makes ball-sized blobs - ignore them
        r = 0.09 * stature_px
        cands = {t: [c for c in cs if t not in head or np.hypot(c[0] - head[t][0], c[1] - head[t][1]) > r]
                 for t, cs in cands.items()}
    min_step = 6.0 / fps / m_per_px            # >= 6 m/s (a struck / bounced ball)
    max_step = 70.0 / fps / m_per_px           # <= 70 m/s
    near = 0.6 * stature_px
    tracks = fast_tracks(cands, lo, hi, min_step, max_step, min_len=3)

    def dist_racket(pt):
        t, x, y = pt
        r = racket.get(t)
        return float(np.hypot(x - r[0], y - r[1])) if r is not None else 1e9

    incoming = [tr for tr in tracks if tr[-1][0] <= peak + 4 and dist_racket(tr[-1]) < near]
    outgoing = [tr for tr in tracks if tr[0][0] >= peak - 4 and dist_racket(tr[0]) < near]
    best = None
    for a in incoming:
        va = np.subtract(a[-1][1:], a[-2][1:])
        for b in outgoing:
            gap = b[0][0] - a[-1][0]
            if not (1 <= gap <= 4):
                continue
            if np.hypot(b[0][1] - a[-1][1], b[0][2] - a[-1][2]) > 0.5 * stature_px:
                continue
            vb = np.subtract(b[1][1:], b[0][1:])
            cos = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-6))
            if cos > 0.5:                         # < 60 deg change: not a hit
                continue
            score = len(a) + len(b) - gap - abs(a[-1][0] - peak) * 0.2
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is not None:
        _, a, b = best
        return ServeBall(toss=a, outgoing=b,
                         contact_xy=((a[-1][1] + b[0][1]) / 2, (a[-1][2] + b[0][2]) / 2),
                         contact_frame=(a[-1][0] + b[0][0]) / 2.0, method="ball-midpoint")
    one = sorted([(dist_racket(tr[-1]), tr, "in") for tr in incoming] +
                 [(dist_racket(tr[0]), tr, "out") for tr in outgoing], key=lambda z: z[0])
    if one:
        _, tr, kind = one[0]
        pt = tr[-1] if kind == "in" else tr[0]
        return ServeBall(toss=tr if kind == "in" else [], outgoing=tr if kind == "out" else [],
                         contact_xy=(pt[1], pt[2]), contact_frame=float(pt[0]), method="ball-one-side")
    r = racket.get(peak)
    return ServeBall(contact_xy=r, contact_frame=float(peak), method="pose-only")


def find_bounce(cands: Dict[int, List[Cand]], incoming: List[Point], fps: float,
                m_per_px: float, look_back_s: float = 1.0) -> Optional[Point]:
    """Where the incoming ball touched the court before the stroke.

    A bounce flips the ball's vertical image motion from falling (y grows) to
    rising (y shrinks). The straight-line tracker stops at that kink, so the
    incoming track usually *starts* just after the bounce; we look for an
    earlier falling track that ends right before it.
    """
    if len(incoming) < 2:
        return None
    t_in, x_in, y_in = incoming[0]
    vy_in = incoming[1][2] - incoming[0][2]
    min_step, max_step = 3.0 / fps / m_per_px, 60.0 / fps / m_per_px
    best = None
    for tr in fast_tracks(cands, int(t_in - look_back_s * fps), t_in - 1, min_step, max_step, min_len=3):
        gap = t_in - tr[-1][0]
        if not (1 <= gap <= 4):
            continue
        vy_pre = tr[-1][2] - tr[-2][2]
        if not (vy_pre > 0 and vy_in < 0):          # falling, then rising
            continue
        d = float(np.hypot(x_in - tr[-1][1], y_in - tr[-1][2]))
        if d > max_step * gap:
            continue
        if best is None or d < best[0]:
            best = (d, tr[-1])
    if best is not None:
        return best[1]
    # no falling segment found: if the incoming ball rises from its first
    # point, treat that first point as (just after) the bounce
    return incoming[0] if vy_in < 0 and len(incoming) >= 3 else None


# --------------------------------------------------------------------------- #
# Forehand contact from the racquet head (preferred when a racquet is detected)
# --------------------------------------------------------------------------- #
def is_ball_colour(bgr: np.ndarray, x: float, y: float, r: int = 2) -> bool:
    """Tennis-ball yellow-green at (x, y)? Works on HD footage; at 480p colour is lost."""
    h, w = bgr.shape[:2]
    x0, x1 = int(max(0, x - r)), int(min(w, x + r + 1))
    y0, y1 = int(max(0, y - r)), int(min(h, y + r + 1))
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    best = hsv[np.argmax(hsv[:, 1].astype(int) + hsv[:, 2].astype(int))]
    return 20 <= best[0] <= 50 and best[1] >= 60 and best[2] >= 110


def _chain(cands, start: Point, direction: int, n: int, max_step: float,
           max_gap: int = 6) -> List[Point]:
    """Ball chain from ``start`` forwards (+1) or backwards (-1).

    Uses a constant-velocity prediction (the Kalman idea from the original
    code, in its simplest form) and bridges up to ``max_gap`` frames where the
    blurred ball was not detected.
    """
    out = [start]
    v = None
    t = start[0]
    gap = 0
    while len(out) <= n and gap <= max_gap:
        t += direction
        lt, x, y = out[-1]
        k = abs(t - lt)
        ex, ey = (x + v[0] * k, y + v[1] * k) if v is not None else (x, y)
        tol = max_step * k if v is None else 0.5 * np.hypot(*v) * k + 6
        opts = [(np.hypot(c[0] - ex, c[1] - ey), c) for c in cands.get(t, [])]
        near = [o for o in opts if o[0] <= tol]
        if not near and v is not None and gap <= max_gap // 2:
            # the bounce: speed stays similar but the direction flips, so accept the
            # candidate the right distance from the last point, nearest the prediction
            sp = np.hypot(*v) * k
            near = [o for o in opts
                    if 0.4 * sp <= np.hypot(o[1][0] - x, o[1][1] - y) <= 1.6 * sp + 6]
        opts = near
        if not opts:
            gap += 1
            continue
        c = min(opts, key=lambda o: o[0])[1]
        v = ((c[0] - x) / k, (c[1] - y) / k)
        out.append((t, c[0], c[1]))
        gap = 0
    return out if direction > 0 else out[::-1]


def find_racquet_contact(cands: Dict[int, List[Cand]], heads: Dict[int, Tuple[float, float]],
                         peak: int, fps: float, m_per_px: float, stature_px: float) -> ServeBall:
    """Contact = midpoint of the ball in the two frames closest to the racquet head.

    ``cands`` should already be filtered to ball-coloured blobs off the body.
    Falls back to the racquet head at its fastest moment ('racquet-only').
    """
    lo, hi = int(peak - 0.25 * fps), int(peak + 0.25 * fps)
    gate = 0.35 * stature_px
    dist = {}
    for f in range(lo, hi + 1):
        if f not in heads or not cands.get(f):
            continue
        hx, hy = heads[f]
        d, c = min(((np.hypot(c[0] - hx, c[1] - hy), c) for c in cands[f]), key=lambda z: z[0])
        if d < gate:
            dist[f] = (d, c)
    max_step = 70.0 / fps / m_per_px
    if dist:
        # the two frames around the hit: among the frames where the ball is closest to the
        # racquet head, take the pair whose ball can be followed both into AND out of it
        gap = max(6, int(0.1 * fps))          # ball hidden behind the player for up to 0.1 s
        best = None
        for f1 in sorted(dist, key=lambda f: dist[f][0])[:4]:
            for f2 in (f1 - 1, f1 + 1, f1 - 2, f1 + 2):
                if f2 not in dist:
                    continue
                fa, fb = sorted((f1, f2))
                a, b = dist[fa][1], dist[fb][1]
                before = _chain(cands, (fa, a[0], a[1]), -1, int(1.4 * fps), max_step, gap)
                after = _chain(cands, (fb, b[0], b[1]), +1, int(0.5 * fps), max_step, gap)
                score = min(len(before), 12) + min(len(after), 12) - 0.01 * (dist[f1][0] + dist[f2][0])
                if best is None or score > best[0]:
                    best = (score, fa, fb, a, b, before, after)
        if best is None:
            f1 = min(dist, key=lambda f: dist[f][0])
            a = dist[f1][1]
            best = (0, f1, f1, a, a, _chain(cands, (f1, a[0], a[1]), -1, int(1.4 * fps), max_step, gap),
                    _chain(cands, (f1, a[0], a[1]), +1, int(0.5 * fps), max_step, gap))
        _, fa, fb, a, b, before, after = best
        return ServeBall(toss=before, outgoing=after[1:] if fb == fa else after,
                         contact_xy=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                         contact_frame=(fa + fb) / 2.0, method="ball-at-racquet")
    # no ball near the racquet: hitting point = racquet head at its fastest moment
    fs = sorted(f for f in heads if lo <= f <= hi)
    if len(fs) >= 3:
        sp = {fs[i]: np.hypot(heads[fs[i + 1]][0] - heads[fs[i - 1]][0],
                              heads[fs[i + 1]][1] - heads[fs[i - 1]][1]) for i in range(1, len(fs) - 1)}
        f = max(sp, key=sp.get)
        return ServeBall(contact_xy=heads[f], contact_frame=float(f), method="racquet-only")
    return ServeBall(contact_xy=heads.get(peak), contact_frame=float(peak), method="pose-only")


def bounce_from_chain(before: List[Point], contact_t: float, fps: float,
                      ground_y: float, stature_px: float) -> Optional[Point]:
    """Bounce = lowest point (max image y) of the incoming chain, near court level,
    at least 0.12 s before contact, with the ball falling into it and rising out.

    If the track starts at court level and only rises from there (the ball was
    not seen before it bounced), the first point is taken as the bounce.
    """
    pts = [p for p in before if contact_t - p[0] >= 0.12 * fps]
    if len(pts) < 5:
        return None
    k = int(np.argmax([p[2] for p in pts]))
    if pts[k][2] < ground_y - 0.25 * stature_px:       # too high above the court
        return None
    if k == len(pts) - 1:
        return None
    if k == 0:
        rising = all(pts[i + 1][2] <= pts[i][2] + 1 for i in range(min(5, len(pts) - 1)))
        if not (rising and contact_t - pts[0][0] >= 0.25 * fps and pts[0][2] >= ground_y - 0.1 * stature_px):
            return None
    return pts[k]


def confirms_hit(ball: ServeBall, heads: Dict[int, Tuple[float, float]], fps: float,
                 stature_px: float) -> bool:
    """A real hit: the racquet head is swinging fast at contact and the ball leaves
    in the direction of the swing. Rejects shadow swings and balls that merely
    pass the racquet (the ball must be followed out of the racquet)."""
    if ball.contact_xy is None or len(ball.outgoing) < 3:
        return False
    c = ball.contact_frame
    hs = sorted(t for t in heads if abs(t - c) <= max(2, 0.05 * fps))
    if len(hs) < 2:
        return False
    hv = np.subtract(heads[hs[-1]], heads[hs[0]])
    if np.hypot(*hv) < 0.2 * stature_px * (hs[-1] - hs[0]) / max(1.0, 0.1 * fps):
        return False                      # racquet barely moving: not a stroke
    ov = np.subtract(ball.outgoing[min(5, len(ball.outgoing) - 1)][1:], ball.contact_xy)
    cos = float(hv @ ov) / max(np.hypot(*hv) * np.hypot(*ov), 1e-6)
    return cos >= 0.3
