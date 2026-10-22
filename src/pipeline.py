"""End-to-end stroke analysis: video in -> report + images out.

Step 0  Split the video into one JPEG per frame on disk (e.g. 120 images per
        second for a 120 fps clip), scaled to <= ``max_width``. Every later
        step reads frames from disk, so long / high-fps videos never need to
        fit in memory and the frames can be reused for screenshots.
Pass 1  For each processed frame: player box (YOLOX) -> skeleton (BlazePose
        on a zoomed crop) -> ball candidates (motion, full-frame coordinates).
Per stroke (mode = "serve" or "forehand"): contact point, scale (px -> m),
        measurements, rules, Random-Forest suggestions, coach text.
Pass 2  Re-read only the frames around each stroke to draw keyframes,
        skeletons, the animated GIF and the overview.
"""
from __future__ import annotations

import glob
from dataclasses import asdict
import json
import os
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .pipeline_config import get, load_config, resolve_path
from .stages.ball_tracking.ball_tracker import (
    ServeBall, bounce_from_chain, confirms_hit, drop_static, find_bounce, find_contact,
    find_groundstroke_contact, find_racquet_contact, fit_toss, ground_roi, is_ball_colour,
    frame_shift, motion_candidates, search_roi, to_gray)
from .stages.court_segmentation.court_detector import detect_baseline_multi
from .stages.detection.person_detector import PersonDetector, find_racket, pick_player
from .stages.local_llm.coach import llm_feedback
from .stages.pose_model.pose_model_backend import (
    L_ANKLE, L_HEEL, L_WRIST, NOSE, R_ANKLE, R_HEEL, R_WRIST, PoseModelBackend)
from .stages.pose_model.smoothing import fill_gaps, smooth_poses
from .stages.serve_analysis.analysis import (
    Scale, ServeReport, apply_rules, estimate_scale, find_serve_peaks, knee_angle, measure_serve)
from .stages.serve_analysis.levels import (LEVELS, judge as judge_level,
                                           score as level_score, suggestions as level_suggestions)
from .stages.serve_analysis.forehand import (
    apply_fh_rules, facing_sign, find_swings, measure_forehand, racket_point, stroke_side)
from .stages.visualize import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Progress = Callable[[str, float], None]
MODES = ("serve", "forehand")
PHASE_LABELS = {
    "serve": [("ready", "1 Ready"), ("trophy", "2 Trophy"), ("contact", "3 Contact"), ("finish", "4 Finish")],
    "forehand": [("ready", "1 Ready"), ("backswing", "2 Backswing"), ("contact", "3 Contact"),
                 ("finish", "4 Follow-through")],
}


def _box_from_pose(pose: np.ndarray, W: int, H: int):
    ok = pose[:, 2] > 0.3
    if ok.sum() < 8:
        return None
    xs, ys = pose[ok, 0], pose[ok, 1]
    h = ys.max() - ys.min()
    return (max(0, xs.min() - 0.1 * h), max(0, ys.min() - 0.08 * h),
            min(W, xs.max() + 0.1 * h), min(H, ys.max() + 0.04 * h), 0.5)


# --------------------------------------------------------------------------- #
# Step 0: video -> frame images on disk
# --------------------------------------------------------------------------- #
def extract_frames(video_path: str, frames_dir: str, max_width: int = 1280,
                   progress: Optional[Progress] = None) -> dict:
    """Write every frame as ``frames_dir/000123.jpg``; returns meta (fps, n, size)."""
    meta_path = os.path.join(frames_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            return json.load(fh)
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    n, W, H = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]
        cv2.imwrite(os.path.join(frames_dir, f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
        if progress and n % 30 == 0:
            progress(f"Splitting video into images ({n}/{total})", 0.12 * n / max(total, 1))
    cap.release()
    if n == 0:
        raise ValueError("video has no readable frames")
    meta = {"fps": fps, "n": n, "size": [W, H]}
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return meta


def first_frame(video_path: str, max_width: int = 1280):
    """First frame, scaled exactly like ``extract_frames`` (for baseline clicks)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if frame.shape[1] > max_width:
        s = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return frame


def frame_path(frames_dir: str, i: int) -> str:
    return os.path.join(frames_dir, f"{i:06d}.jpg")


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def analyse_video(video_path: str, out_dir: str, cfg: Optional[dict] = None,
                  player_height_m: float = 1.80, baseline_override=None,
                  camera_side: str = "far", mode: str = "serve",
                  progress: Optional[Progress] = None, level: str = "club") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    cfg = cfg if cfg is not None else load_config(os.path.join(ROOT, "config.yaml"))
    progress = progress or (lambda stage, frac: None)
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    frames_dir = os.path.join(out_dir, "frames")
    meta = extract_frames(video_path, frames_dir, int(get(cfg, "pipeline.video.max_width", 1280)),
                          progress)
    src_fps, n_src = meta["fps"], meta["n"]
    W, H = meta["size"]
    stride = max(1, int(round(src_fps / float(get(cfg, "pipeline.video.max_fps", 120)))))
    fps = src_fps / stride
    proc = list(range(0, n_src, stride))          # processed frame t -> source image proc[t]

    def load(t: int) -> np.ndarray:
        return cv2.imread(frame_path(frames_dir, proc[t]))

    detector = PersonDetector(resolve_path(cfg, "pipeline.detector.model_path", ROOT,
                                           "models/yolox_tiny.onnx"))
    pose_model = PoseModelBackend(resolve_path(cfg, "pipeline.pose.model_path", ROOT,
                                               "models/pose_landmarker.task"))
    # heavy models run at a capped rate; the ball keeps every frame (speed on 120 fps)
    detect_every = max(int(get(cfg, "pipeline.detector.every_frames", 3)),
                       int(round(fps / float(get(cfg, "pipeline.detector.max_hz", 10)))))
    pose_every = max(1, int(round(fps / float(get(cfg, "pipeline.pose.max_hz", 60)))))
    full = lambda box, w, h: (0, 0, w, h)
    roi_fn = search_roi if mode == "serve" else full     # forehand: whole frame (bounce, incoming ball)

    # ---------------- pass 1: boxes, poses, ball candidates ----------------
    boxes: Dict[int, tuple] = {}
    poses: Dict[int, np.ndarray] = {}
    cands: Dict[int, list] = {}
    grays: List = []
    box = None
    lost = False          # skeleton not found -> re-detect the player on the next frame
    timings = {"frames": time.time() - t_start, "detect": 0.0, "pose": 0.0, "ball": 0.0}
    n_frames = len(proc)
    cache = os.path.join(out_dir, f"pass1_{mode}.pkl") if os.environ.get("PIPE_CACHE") else None
    if cache and os.path.exists(cache):
        import pickle
        with open(cache, "rb") as fh:
            boxes, poses, cands = pickle.load(fh)
    for t in range(n_frames if not (cache and os.path.exists(cache)) else 0):
        frame = load(t)
        if box is None or t % detect_every == 0 or lost:
            t0 = time.perf_counter()
            cand_box = pick_player(detector.detect(frame), box)
            timings["detect"] += time.perf_counter() - t0
            if cand_box is not None:
                box = cand_box
        if box is not None:
            boxes[t] = box
            if t % pose_every and not lost:
                pose = None
            else:
                t0 = time.perf_counter()
                pose = pose_model.detect(frame, box, int(t * 1000 / fps))
                timings["pose"] += time.perf_counter() - t0
                lost = pose is None
            if pose is not None:
                poses[t] = pose
                pb = _box_from_pose(pose, W, H)
                if pb is not None and t % detect_every != 0:
                    box = pb
        t0 = time.perf_counter()
        g = to_gray(frame)
        d = frame_shift(grays[-1][1], g) if grays else (0.0, 0.0)   # shake t-1 -> t, once per frame
        grays.append((t, g, d))
        if len(grays) > 3:
            grays.pop(0)
        if len(grays) == 3 and grays[1][0] in boxes:
            tm = grays[1][0]
            bh = boxes[tm][3] - boxes[tm][1]
            side = int(max(9, 0.07 * bh))          # ball blob limit grows with resolution
            cands[tm] = motion_candidates(grays[0][1], grays[1][1], grays[2][1],
                                          roi_fn(boxes[tm], W, H), max_area=max(40, side * side),
                                          max_side=side,
                                          shifts=((-grays[1][2][0], -grays[1][2][1]), grays[2][2]))
        timings["ball"] += time.perf_counter() - t0
        if t % 10 == 0:
            progress("Detecting player, skeleton and ball", 0.12 + 0.68 * t / max(n_frames, 1))
    pose_model.close()
    if cache and not os.path.exists(cache):
        import pickle
        with open(cache, "wb") as fh:
            pickle.dump((boxes, poses, cands), fh)
    if not poses:
        raise ValueError("No player found in the video.")
    if pose_every > 1:
        poses = fill_gaps(poses, pose_every)
    poses = smooth_poses(poses, fps)          # Stage 2b: steadier joints, no time lag

    # ---------------- strokes + contact ----------------
    progress("Finding strokes and contact points", 0.82)
    t_strokes = time.perf_counter()
    backhands = 0
    global_scale = estimate_scale(poses, list(poses), player_height_m)
    strokes = []          # (peak, scale, ball)
    rackets: Dict[int, object] = {}   # frame -> (box, head_xy) or None
    seen: List[float] = []
    if mode == "serve":
        for peak in find_serve_peaks(poses, fps, n_frames):
            lo, hi = int(max(0, peak - 2.0 * fps)), int(min(n_frames - 1, peak + 0.8 * fps))
            win = [f for f in range(lo, hi + 1) if f in poses]
            scale = estimate_scale(poses, win, player_height_m) or global_scale
            if scale is None:
                continue
            wc = drop_static({f: cands[f] for f in range(lo, hi + 1) if f in cands})
            cx = float(np.median([(boxes[f][0] + boxes[f][2]) / 2 for f in win if f in boxes]))
            head = float(np.median([poses[f][NOSE, 1] for f in win]))
            toss = fit_toss(wc, scale.stature_px, fps, player_height_m, center_x=cx, head_y=head)
            ball = find_contact(toss, wc, scale.stature_px)
            reliable = False
            if ball.contact_xy is not None:
                ratio = scale.height_m(ball.contact_xy[1]) / player_height_m
                above = (scale.height_m(ball.apex_xy[1]) - scale.height_m(ball.contact_xy[1])) if ball.apex_xy else 0
                reliable = 1.2 <= ratio <= 2.1 and above <= 1.6     # reach + racquet + jump is at most ~2x height
            if not reliable:
                ball = ServeBall(contact_frame=float(peak), method="pose-only")
            if any(abs(ball.contact_frame - c) < fps for c in seen):
                continue
            seen.append(ball.contact_frame)
            strokes.append((peak, scale, ball))
    else:
        if global_scale is None:
            raise ValueError("Could not measure the player's height in pixels.")
        # every racquet-hand speed peak is a candidate; on colour footage the ball
        # meeting the racquet confirms it (on low-res footage the hand-crossing test does)
        candidates = sorted(set(
            find_swings(poses, fps, global_scale.stature_px, min_speed=2.0, min_gap_s=0.25,
                        require_cross=False)))
        colour_ok = None          # does this footage keep the ball's colour?
        dbg, ball_memo = [], {}
        # which way is the player's right in the image? (median over the clip: the player
        # mostly faces the camera; during one backhand the pose model may swap shoulders)
        facing = facing_sign(poses)
        wanted = "backhand" if mode == "backhand" else "forehand"   # future backhand mode: the reverse
        backhands = 0         # strokes of the other side, skipped
        rstep = 2 if fps > 60 else 1      # racquet detector on every 2nd frame at high fps
        for peak in candidates:
            lo, hi = int(peak - 1.5 * fps), int(peak + 0.6 * fps)     # incoming ball + bounce + outgoing
            wc = drop_static({f: cands[f] for f in range(lo, hi + 1) if f in cands},
                             window=int(0.05 * fps) + 3, max_hits=int(0.05 * fps) + 5)

            def off_body(f, c):     # ignore blobs on the torso / head (shirt, face, arm)
                p = poses.get(f)
                if p is None:
                    return True
                xs = [p[k, 0] for k in (11, 12, 23, 24)]
                top, bot = p[NOSE, 1] - 0.05 * global_scale.stature_px, (p[23, 1] + p[24, 1]) / 2
                if min(xs) - 4 <= c[0] <= max(xs) + 4 and top <= c[1] <= bot:
                    return False
                feet = [p[k, :2] for k in (27, 28, 29, 30, 31, 32)]     # yellow shoes look like balls
                return all(np.hypot(c[0] - q[0], c[1] - q[1]) > 0.1 * global_scale.stature_px for q in feet)

            # ball-coloured blobs off the body (memoised: swing windows overlap)
            for f in range(max(0, lo), min(n_frames - 1, hi) + 1):
                if f not in ball_memo:
                    cs = cands.get(f) or []
                    img = load(f) if cs else None
                    ball_memo[f] = [c for c in cs if off_body(f, c) and is_ball_colour(img, c[0], c[1])]
            balls = {f: [c for c in ball_memo.get(f, []) if c in wc.get(f, [])] for f in range(lo, hi + 1)}
            near_hit = sum(len(balls.get(f, [])) for f in range(int(peak - 0.25 * fps), int(peak + 0.25 * fps) + 1))
            # racquet head on a zoomed crop, only around the hit and only if a ball is there
            racket, heads = {}, {}
            for f in range(int(peak - 0.3 * fps), int(peak + 0.3 * fps) + 1):
                if not (f in poses and f in boxes and 0 <= f < n_frames) or (colour_ok and not near_hit) \
                        or f % rstep:
                    continue
                det = rackets[f] if f in rackets else find_racket(detector, load(f), boxes[f],
                                                                  poses[f][R_WRIST, :2])
                rackets[f] = det
                racket[f] = det[1] if det else racket_point(poses[f], global_scale.stature_px)
                if det:
                    heads[f] = det[1]
            for f in list(heads):              # fill the frames skipped above
                for g in range(f + 1, f + rstep):
                    if f + rstep in heads:
                        w = (g - f) / rstep
                        heads[g] = tuple((1 - w) * np.array(heads[f]) + w * np.array(heads[f + rstep]))
            head = {t: (float(poses[t][NOSE, 0]), float(poses[t][NOSE, 1]))
                    for t in range(lo, hi + 1) if t in poses}
            if cache:
                dbg.append((peak, balls, heads))
            if colour_ok is None:
                colour_ok = sum(len(v) for v in balls.values()) >= 3
            if len(heads) >= 5 and colour_ok:
                ball = find_racquet_contact(balls, heads, peak, fps, global_scale.m_per_px,
                                            global_scale.stature_px)
                if ball.method != "ball-at-racquet" or max(len(ball.toss), len(ball.outgoing)) < 5 \
                        or not confirms_hit(ball, heads, fps, global_scale.stature_px):
                    continue          # ball followed out of a fast-moving racquet, else skip
            else:     # low-res / colourless footage: motion-only tracks
                ball = find_groundstroke_contact(wc, racket, peak, fps, global_scale.m_per_px,
                                                 global_scale.stature_px, head)
            if colour_ok is False and peak not in find_swings(poses, fps, global_scale.stature_px):
                continue          # low-res footage: keep the stricter forehand test
            if any(abs(ball.contact_frame - c) < 0.5 * fps for c in seen):
                continue
            if stroke_side(poses, ball.contact_xy, ball.contact_frame, fps, facing) != wanted:
                backhands += 1          # e.g. a backhand in forehand mode: not analysed
                seen.append(ball.contact_frame)
                continue
            seen.append(ball.contact_frame)
            pc_ = poses.get(int(round(ball.contact_frame)))
            ground = max(pc_[L_HEEL, 1], pc_[R_HEEL, 1]) if pc_ is not None else global_scale.ground_y
            ball.bounce = (bounce_from_chain(ball.toss, ball.contact_frame, fps, ground,
                                             global_scale.stature_px) if ball.toss else None)
            strokes.append((peak, global_scale, ball))
    strokes.sort(key=lambda s: s[2].contact_frame)
    if cache and mode != "serve":
        import pickle
        with open(os.path.join(out_dir, "strokes_dbg.pkl"), "wb") as fh:
            pickle.dump((dbg, global_scale.m_per_px, global_scale.stature_px), fh)
    timings["strokes"] = time.perf_counter() - t_strokes

    coach_cfg = get(cfg, "coach", {}) or {}
    reports = []
    t_rep = time.perf_counter()
    for i, (peak, scale, ball) in enumerate(strokes):
        progress(f"Analysing {mode} {i + 1}/{len(strokes)}", 0.85 + 0.13 * i / max(len(strokes), 1))
        reports.append(_report(i, mode, peak, scale, ball, poses, boxes, fps, n_frames, load,
                               proc, out_dir, W, H, baseline_override, camera_side,
                               coach_cfg, player_height_m, detector, rackets, level))

    timings["reports"] = time.perf_counter() - t_rep
    if not bool(get(cfg, "pipeline.video.keep_frames", True)):
        for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
            os.remove(f)
    result = {
        "video": os.path.basename(video_path), "mode": mode,
        "fps": fps, "source_fps": src_fps, "frames": n_frames, "size": [W, H],
        "frames_dir": os.path.relpath(frames_dir, out_dir),
        "player_height_m": player_height_m,
        "strokes": reports, "n_strokes": len(reports),
        "serves": reports, "n_serves": len(reports),          # backward compatible keys
        "judge": "coaching logic (priority-ordered rules)",
        "other_side_skipped": backhands if mode != "serve" else 0,      # backhands in forehand mode
        "level": level,
        "timing_s": {k: round(v, 2) for k, v in timings.items()},
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    progress("Done", 1.0)
    return result


# --------------------------------------------------------------------------- #
# One stroke: measurements, rules, RF, coach, images
# --------------------------------------------------------------------------- #
def _report(i, mode, peak, scale, ball, poses, boxes, fps, n_frames, load, proc, out_dir,
            W, H, baseline_override, camera_side, coach_cfg, player_height_m, detector, rackets,
            level="club"):
    ct = int(round(ball.contact_frame))
    if mode == "serve":
        a = int(max(0, (ball.toss[0][0] if ball.toss else ct - 1.5 * fps) - 10))
        b = int(min(n_frames - 1, ct + 0.6 * fps))
    else:
        start = ball.toss[0][0] - 5 if ball.toss else ct - 0.9 * fps     # from the ball coming in
        a = int(max(0, max(ct - 1.5 * fps, min(ct - 0.9 * fps, start))))
        b = int(min(n_frames - 1, ct + 0.6 * fps))

    baseline = None
    if baseline_override:
        baseline = tuple(baseline_override)
    else:
        p0 = poses.get(a + 10) if mode == "serve" else poses.get(ct)
        if p0 is None:
            p0 = poses.get(ct)
        if p0 is not None:
            feet = ((p0[L_HEEL, 0] + p0[R_HEEL, 0]) / 2,
                    scale.ground_y if mode == "serve" else max(p0[L_HEEL, 1], p0[R_HEEL, 1]))
            baseline = detect_baseline_multi([load(f) for f in range(a, ct, max(1, int(fps / 8)))],
                                             feet, scale.stature_px)

    for f in range(a, b + 1):      # racquet on a zoomed crop (forehand frames already done)
        if mode != "serve" and (f < ct - 0.45 * fps or (fps > 60 and f % 2)):
            continue               # forehand: racquet needed from the backswing on; every 2nd frame at 120 fps
        if f not in rackets and f in boxes and f in poses:
            rackets[f] = find_racket(detector, load(f), boxes[f], poses[f][R_WRIST, :2])
    if mode == "serve":
        phase, m = measure_serve(poses, ball, scale, fps, peak, baseline, camera_side,
                                 {f: rackets.get(f) for f in range(a, b + 1)})
    else:
        phase, m = measure_forehand(poses, ball, scale, fps, ball.contact_frame, camera_side,
                                    getattr(ball, "bounce", None), baseline,
                                    {f: rackets.get(f) for f in range(a, b + 1)})
    # Judge = coaching logic for the player's level: every failed check becomes a tip,
    # in that level's priority order. All three levels are judged so the UI can switch.
    by_level = {}
    for lv in LEVELS:
        lr = judge_level(mode, m, lv)
        by_level[lv] = {"rules": lr, "suggestions": level_suggestions(lr), "score": level_score(lr)}
    rules, suggestions = by_level[level]["rules"], by_level[level]["suggestions"]
    rep = ServeReport(index=i + 1, frames={k: (None if v is None else float(v)) for k, v in phase.items()},
                      contact_method=ball.method, measurements=m, rules=rules, suggestions=suggestions)
    if m.get("pose_confidence_contact") is not None and m["pose_confidence_contact"] < 0.6:
        rep.notes.append("Hitting arm is blurred at contact - arm angles are low-confidence.")
    if scale.stature_px < 150:
        rep.notes.append(f"Player is only {scale.stature_px:.0f}px tall - film closer "
                         "(player >= 1/3 of frame height) for better joint accuracy.")
    if ball.method == "pose-only":
        rep.notes.append("Ball not tracked reliably - contact timing comes from the skeleton; "
                         "ball-based checks use the racket-hand position instead.")
    elif ball.method == "ball-one-side":
        rep.notes.append("Only one side of the ball path was found - contact is the ball "
                         "position closest to the racket.")
    if mode == "serve" and baseline is None:
        rep.notes.append("Baseline not found automatically - click it in the demo to enable "
                         "foot-fault checks.")

    # timeline for the height chart
    tl = {"frame": [], "t": [], "ball_m": [], "right_wrist_m": [], "left_wrist_m": [], "knee_deg": []}
    path = list(ball.toss) + list(ball.outgoing)
    by_f = {f: y for f, _x, y in path}
    for seg in (ball.toss, ball.outgoing):
        if len(seg) >= 2:
            fs = [f for f, _x, _y in seg]
            for f in range(fs[0], fs[-1] + 1):
                by_f.setdefault(f, float(np.interp(f, fs, [y for _f, _x, y in seg])))
    for f in range(a, b + 1):
        p = poses.get(f)
        tl["frame"].append(f); tl["t"].append(round((f - ball.contact_frame) / fps, 3))
        tl["ball_m"].append(round(scale.height_m(by_f[f]), 2) if f in by_f else None)
        tl["right_wrist_m"].append(round(scale.height_m(p[R_WRIST, 1]), 2) if p is not None else None)
        tl["left_wrist_m"].append(round(scale.height_m(p[L_WRIST, 1]), 2) if p is not None else None)
        tl["knee_deg"].append(round(knee_angle(p), 1) if p is not None else None)
    rep.timeline = tl

    # images
    name = f"{mode}{i + 1}"
    sdir = os.path.join(out_dir, name)
    os.makedirs(sdir, exist_ok=True)
    crop = render.serve_crop([boxes[f] for f in range(a, b + 1) if f in boxes], ball, W, H,
                             tight=(mode == "forehand"))
    painter = render.Painter(crop)

    racket_at = {f: rackets.get(f) or rackets.get(f - 1) or rackets.get(f + 1) for f in range(a, b + 1)}

    def pose_at(f):
        for d in (0, 1, -1, 2, -2):
            if int(f) + d in poses:
                return poses[int(f) + d]
        return None

    labels = PHASE_LABELS[mode]
    key_imgs, skel_imgs = [], []
    for key, label in labels:
        f = phase.get(key)
        if f is None:
            continue
        fi = int(min(max(round(f), a), b))
        sub = []
        if key in ("trophy", "backswing") and m.get("knee_bend_deg") is not None:
            sub.append(f"knee bend {m['knee_bend_deg']:.0f} deg")
        if key == "contact" and m.get("contact_height_m") is not None:
            sub.append(f"contact {m['contact_height_m']:.2f} m ({m['contact_height_ratio']:.2f}x height)")
            if m.get("ball_speed_kmh") is not None:
                sub.append(f"ball >= {m['ball_speed_kmh']:.0f} km/h")
        if key == "contact" and m.get("right_foot_ahead_m") is not None:
            sub.append("left foot in front" if m["right_foot_ahead_m"] <= 0.05 else "right foot in front!")
        if key == "ready" and m.get("toss_apex_m") is not None:
            sub.append(f"toss apex {m['toss_apex_m']:.2f} m")
        img = load(fi)
        key_imgs.append(render.annotated_frame(img, pose_at(fi), painter, ball, f, baseline, scale, label, sub,
                                                 racket=racket_at.get(fi)))
        skel_imgs.append(render.skeleton_panel(pose_at(fi), painter, ball, f, scale, label, racket=racket_at.get(fi)))
    if key_imgs:
        cv2.imwrite(os.path.join(sdir, "keyframes.jpg"), np.hstack(key_imgs), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(os.path.join(sdir, "skeleton.png"), np.hstack(skel_imgs))
        rep.images["keyframes"] = f"{name}/keyframes.jpg"
        rep.images["skeleton"] = f"{name}/skeleton.png"
    # GIF: dense around contact, sparse elsewhere; ~ the same length for any fps
    near = max(8, int(0.12 * fps))
    sparse = max(2, int(round(fps / 15)))
    gif_frames, durations = [], []
    key_frames = {int(round(v)): lab for (k, lab) in labels for kk, v in phase.items()
                  if kk == k and v is not None}
    for f in range(a, b + 1):
        lab = key_frames.get(f)
        if lab is None and abs(f - ball.contact_frame) > near and (f - a) % sparse:
            continue
        if lab is None and abs(f - ball.contact_frame) <= near and fps > 60 and f % 2:
            continue
        gif_frames.append(render.annotated_frame(load(f), pose_at(f), painter, ball, f, baseline, scale,
                                                 lab or f"{(f - ball.contact_frame) / fps:+.2f}s",
                                                 racket=racket_at.get(f)))
        durations.append(900 if lab else (110 if abs(f - ball.contact_frame) <= near else 120))
    if gif_frames:
        render.save_gif(gif_frames, durations, os.path.join(sdir, "sequence.gif"))
        rep.images["gif"] = f"{name}/sequence.gif"
    ov = load(min(max(ct, 0), n_frames - 1))
    bx = boxes.get(ct)
    if bx:
        cv2.rectangle(ov, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0, 255, 0), 2)
    render.draw_baseline(ov, lambda x, y: (int(x), int(y)), baseline, W)
    render.draw_trajectory(ov, ball, fps, scale, "toss" if mode == "serve" else "ball in")          # whole ball path on the full frame
    cv2.imwrite(os.path.join(sdir, "overview.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep.images["overview"] = f"{name}/overview.jpg"
    rep.images["contact_frame_image"] = f"frames/{proc[min(max(ct, 0), n_frames - 1)]:06d}.jpg"

    d = rep.to_dict()
    d["mode"] = mode
    d["ball"] = {"before": [[f, round(x, 1), round(y, 1)] for f, x, y in ball.toss],
                 "after": [[f, round(x, 1), round(y, 1)] for f, x, y in ball.outgoing],
                 "contact_xy": ball.contact_xy, "apex_xy": ball.apex_xy,
                 "scale_m_per_px": round(scale.m_per_px, 4), "ground_y": scale.ground_y,
                 "baseline": baseline,
                 "bounce": getattr(ball, "bounce", None),
                 "racket_detected_frames": sum(1 for v in racket_at.values() if v)}
    d["level"] = level
    d["score"] = by_level[level]["score"]
    d["levels"] = {}
    for lv in LEVELS:
        sub = dict(d, level=lv, rules=[asdict(r) for r in by_level[lv]["rules"]],
                   suggestions=by_level[lv]["suggestions"], score=by_level[lv]["score"])
        sub.pop("levels", None)
        cfg_lv = dict(coach_cfg, stroke=mode, level=lv)
        cfg_lv["force_template"] = lv != level      # the language model only for the chosen level
        d["levels"][lv] = {"rules": sub["rules"], "suggestions": sub["suggestions"],
                           "score": sub["score"], "coach_text": llm_feedback(sub, cfg_lv)}
    d["coach_text"] = d["levels"][level]["coach_text"]
    return d
