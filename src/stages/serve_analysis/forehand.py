"""Forehand mode (right-handed): swing detection, measurements and rules.

Re-uses the scale (px -> m), angle helpers and RuleResult from the serve
analysis. Swings are found from the hitting wrist's speed; the contact
point comes from the ball (incoming + outgoing track midpoint, see
``ball_tracker.find_groundstroke_contact``) or, failing that, the racket
hand at the swing's speed peak.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..pose_model.pose_model_backend import (
    L_ANKLE, L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_ELBOW, R_HIP,
    R_SHOULDER, R_WRIST)
from .analysis import RuleResult, Scale, knee_angle

Pose = np.ndarray


def _side(p: Pose) -> float:
    """+1 if the player's right shoulder is to the image right, else -1."""
    return float(np.sign(p[R_SHOULDER, 0] - p[L_SHOULDER, 0]) or 1.0)


def facing_sign(poses: Dict[int, Pose]) -> float:
    """Which way the player's right side points in the image over the whole clip:
    +1 = image right, -1 = image left. The median makes it robust to the few frames
    where the pose model swaps left and right (often during a backhand)."""
    d = [p[R_SHOULDER, 0] - p[L_SHOULDER, 0] for p in poses.values()]
    return float(np.sign(np.median(d)) or 1.0) if d else 1.0


def stroke_side(poses: Dict[int, Pose], contact_xy, contact_t: float, fps: float,
                facing: float) -> str:
    """'forehand' or 'backhand' for a right-handed player.

    The ball is hit on the player's right (forehand) or left (backhand) of the hips;
    if the contact point is unknown, the hitting hand's side during the backswing decides.
    Works for any camera position that sees the player from the front or the side.
    """
    pc = _nearest(poses, contact_t)
    if contact_xy is not None and pc is not None:
        lateral = (contact_xy[0] - (pc[L_HIP, 0] + pc[R_HIP, 0]) / 2) * facing
    else:
        pb = _nearest(poses, contact_t - 0.25 * fps)
        if pb is None:
            return "forehand"
        lateral = (pb[R_WRIST, 0] - (pb[L_HIP, 0] + pb[R_HIP, 0]) / 2) * facing
    return "forehand" if lateral >= 0 else "backhand"


def _nearest(poses, t, max_d=3):
    ti = int(round(t))
    for d in range(max_d + 1):
        for c in (ti - d, ti + d):
            if c in poses:
                return poses[c]
    return None


def racket_point(p: Pose, stature_px: float) -> Tuple[float, float]:
    """Rough racket-face position: wrist + forearm direction * ~0.28 body heights."""
    w, e = p[R_WRIST, :2], p[R_ELBOW, :2]
    d = w - e
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return float(w[0]), float(w[1])
    q = w + d / n * 0.28 * stature_px
    return float(q[0]), float(q[1])


def find_swings(poses: Dict[int, Pose], fps: float, stature_px: float,
                min_speed: float = 2.5, min_gap_s: float = 1.0,
                require_cross: bool = True) -> List[int]:
    """Frames of peak hitting-wrist speed (in body heights / s) for forehands.

    A forehand moves the racket hand from the hitting side ACROSS the body,
    so the wrist's lateral position (toward the hitting side) must drop
    through the peak; backhands (the reverse) are skipped.
    """
    ts = sorted(poses)
    if len(ts) < 5:
        return []
    xy = np.array([poses[t][R_WRIST, :2] for t in ts])
    k = max(1, int(round(fps / 60)))                      # light smoothing
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sx = np.convolve(xy[:, 0], ker, mode="same"); sy = np.convolve(xy[:, 1], ker, mode="same")
    tt = np.array(ts, float)
    v = np.hypot(np.gradient(sx, tt), np.gradient(sy, tt)) * fps / stature_px
    order = np.argsort(-v)
    peaks: List[int] = []
    half = int(0.15 * fps)
    for i in order:
        if v[i] < min_speed:
            break
        t = ts[i]
        if any(abs(t - q) < min_gap_s * fps for q in peaks):
            continue
        a, b = poses.get(t - half), poses.get(t + half)
        if a is None or b is None:
            continue
        lat = lambda p: _side(p) * (p[R_WRIST, 0] - (p[L_HIP, 0] + p[R_HIP, 0]) / 2)
        if require_cross and lat(a) <= lat(b):   # hand did not cross the body -> not a forehand
            continue
        peaks.append(t)
    return sorted(peaks)


# Judgement = plain coaching logic, in priority order (first = most important).
# Each rule: key, title, measurement, test(value) -> good?, target text, tip.
# A tip may be a function of all measurements (e.g. depends on court position).

def _tip_too_high(m):
    pos = m.get("court_position")
    if pos == "behind":
        return ("You hit the ball above your head. You are behind the baseline - move FORWARD "
                "and take the ball earlier, at waist-to-chest height, instead of letting it climb.")
    if pos == "inside":
        return ("You hit the ball above your head while standing inside the court (no-man's land). "
                "Move back behind the baseline and wait for the ball to drop into your "
                "waist-to-chest zone.")
    return ("You hit the ball above your head. Behind the baseline: move forward and take it "
            "earlier. Inside the court: step back out of no-man's land and let it drop.")


FH_RULES = [   # (key, category, title, measurement, test, target, tip) - list order = priority
    # --- the coach's top three -------------------------------------------------
    ("contact_too_high", "Contact", "Contact not above your head", "contact_above_head_m",
     lambda v: v <= 0.0, "contact below the top of your head", _tip_too_high),
    # measured as time, which the camera angle cannot distort: a ball that bounces far in
    # front of you needs a long time to reach you
    ("bounce_too_far", "Footwork", "Ball bounced close enough", "bounce_to_contact_s",
     lambda v: v <= 0.8, "ball reaches you within 0.8 s of the bounce",
     "The ball bounced well in front of you - move forward to meet it, so you are not "
     "waiting deep for a ball that has lost its pace."),
    ("sideways", "Preparation", "Body sideways in preparation", "unit_turn_ratio",
     lambda v: v <= 0.80, "shoulders turned sideways (<= 0.8 of square-on)",
     "Turn your body sideways to the net before you swing (shoulders and hips turned) - "
     "it loads the swing and gives power without muscling the arm."),
    # --- contact ------------------------------------------------------------------
    ("contact_in_front", "Contact", "Contact in front of the body", "contact_rotation_ratio",
     lambda v: v >= 0.60, "chest turned toward the net at contact (>= 0.6)",
     "You were still sideways at contact - the ball got behind you. Start the swing earlier "
     "so you meet the ball in front of your front hip, with your chest turning to the net."),
    ("foot_order", "Footwork", "Left foot in front at contact", "right_foot_ahead_m",
     lambda v: v <= 0.05, "right foot not ahead of left",
     "At contact your right foot is in front of your left. Step in with the LEFT foot "
     "so your weight transfers forward through the ball."),
    ("contact_low", "Contact", "Contact not too low", "contact_height_ratio",
     lambda v: v >= 0.40, ">= 0.40 x your height (about waist)",
     "You are hitting the ball too low - bend your knees to get down to it or take it "
     "a little earlier."),
    ("spacing", "Contact", "Room between body and ball", "contact_spacing_m",
     lambda v: 0.25 <= v <= 0.9, "0.25-0.9 m to your right",
     "Give yourself room: meet the ball about an arm's length to your right, not jammed "
     "into the body or at full stretch."),
    ("arm_extension", "Contact", "Arm not cramped at contact", "hit_elbow_deg",
     lambda v: v >= 100, "hitting elbow >= 100 deg",
     "Your hitting arm is cramped at contact - move away from the ball so the arm can "
     "extend comfortably."),
    ("head_still", "Contact", "Head still at contact", "head_move_m",
     lambda v: v <= 0.15, "head moves <= 0.15 m around contact",
     "Keep your head still and your eyes on the contact point until the ball has left "
     "the strings."),
    ("head_level", "Contact", "Head stays level through the swing", "head_level_m",
     lambda v: v <= 0.07, "head height varies <= 0.07 m from backswing to contact",
     "Keep your head at the same height from the backswing through contact - bobbing up and down "
     "moves your eyes and your contact point."),
    ("spine_steady", "Contact", "Spine angle stays steady", "spine_change_deg",
     lambda v: v <= 10, "trunk lean changes <= 10 deg from backswing to contact",
     "Rotate around a steady spine - don't lean back or tip over as you swing."),
    # --- preparation --------------------------------------------------------------
    ("early_prep", "Preparation", "Early preparation", "prep_lead_s",
     lambda v: v >= 0.30, "turn completed >= 0.30 s before contact",
     "Prepare earlier: finish your turn and take the racquet back as soon as you read "
     "the ball, not when it bounces."),
    ("split_step", "Footwork", "Split step before the ball arrives", "split_step_m",
     lambda v: v >= 0.03, "both feet hop >= 0.03 m before the bounce",
     "Do a small split step (both feet hop) as your opponent hits - it gets you moving early."),
    ("coil", "Preparation", "Shoulders coil past the hips", "coil_sep",
     lambda v: v >= 0.12, "shoulders turned >= 0.12 more than hips at backswing",
     "Turn your shoulders further than your hips in the backswing - that stretch (coil) is free power."),
    ("left_arm", "Preparation", "Left arm tracks the ball", "left_arm_height_ratio",
     lambda v: v >= 0.75, "left hand near shoulder height at backswing",
     "Use your left arm: point it at the incoming ball during the turn - it keeps the "
     "shoulders turned and helps you time the contact."),
    ("left_arm_tuck", "Contact", "Left arm folds in at contact", "left_arm_tuck_m",
     lambda v: v <= 0.35, "left hand within 0.35 m of the chest at contact",
     "Pull the left arm in to your body as you swing - it stops the shoulders and whips the racquet through."),
    ("knee_bend", "Preparation", "Load the legs (squat)", "knee_bend_deg",
     lambda v: v >= 20, ">= 20 deg",
     "Squat a bit lower as you load, then drive up through the ball - the leg drive helps "
     "the racquet brush up the back of the ball for topspin."),
    ("knee_flex_contact", "Contact", "Still in your legs at contact", "knee_flex_contact_deg",
     lambda v: v >= 10, ">= 10 deg knee bend at contact",
     "Stay down through the ball - keep some bend in the knees at contact instead of standing up early."),
    # --- footwork -------------------------------------------------------------------
    ("stance_width", "Footwork", "Stable base", "stance_width_m",
     lambda v: v >= 0.45, "feet >= 0.45 m apart at contact",
     "Widen your base at contact - feet a bit more than shoulder-width apart for balance "
     "and power."),
    # --- swing and finish ---------------------------------------------------------
    ("racquet_drop", "Swing", "Racquet drops below the ball", "racquet_below_contact_m",
     lambda v: v >= 0.15, "racquet head >= 0.15 m below contact before the swing",
     "Let the racquet head drop below the height of the ball before swinging up - "
     "that is what lets you brush up the back of the ball for topspin."),
    ("low_to_high", "Swing", "Low-to-high swing (topspin)", "low_to_high_m",
     lambda v: v >= 0.40, ">= 0.40 m rise to the finish",
     "Swing low to high - the brushing path creates topspin and keeps the ball in."),
    ("kinetic_chain", "Swing", "Hips lead the arm (kinetic chain)", "chain_lead_s",
     lambda v: abs(v - 0.12) <= 0.08, "hips unwind 0.04-0.20 s before the hand's peak speed",
     "Start the forward swing with the hips, then shoulders, then arm - not all at once and not "
     "with the arm first."),
    ("leg_drive", "Swing", "Legs drive up into the ball", "leg_drive_m",
     lambda v: v >= 0.03, "hips rise >= 0.03 m into contact",
     "Push up from the ground as you swing - the hips should rise into the contact."),
    ("weight_transfer", "Swing", "Weight moves into the shot", "weight_transfer_m",
     lambda v: v >= 0.10, "hips move >= 0.10 m toward the target",
     "Move your weight forward into the ball - step in and let the hips travel toward the target."),
    ("follow_through", "Finish", "Finish over the left shoulder", "follow_through",
     lambda v: v >= 1, "hand ends across body at shoulder height",
     "Finish the swing across your body with the hand up by your left shoulder."),
    ("balance", "Finish", "Balanced finish", "balance_offset_m",
     lambda v: v <= 0.25, "head over the feet at the finish (<= 0.25 m)",
     "Stay balanced: at the finish your head should be over your feet, not falling away."),
]
FH_FEATURES = [r[3] for r in FH_RULES]


def _window(poses, a, b):
    return [t for t in range(int(a), int(b) + 1) if t in poses]


def measure_forehand(poses: Dict[int, Pose], ball, scale: Scale, fps: float,
                     contact_t: float, camera_side: str = "far", bounce=None,
                     baseline=None, rackets=None) -> Tuple[Dict, Dict]:
    ct = int(round(contact_t))
    mpp = scale.m_per_px
    pre = _window(poses, ct - 0.8 * fps, ct)
    post = _window(poses, ct, ct + 0.6 * fps)
    m: Dict[str, Optional[float]] = {k: None for k in FH_FEATURES + ["racquet_speed_ms",
        "contact_height_m", "ball_speed_kmh", "hit_elbow_deg", "head_height_m",
        "bounce_to_contact_s", "court_position", "baseline_distance_m"]}

    def width(t):
        p = poses[t]
        return abs(p[R_SHOULDER, 0] - p[L_SHOULDER, 0])

    back = min(pre, key=width) if pre else None            # most turned = backswing
    phase = {"ready": float(ct - 0.8 * fps), "backswing": None if back is None else float(back),
             "contact": float(contact_t), "finish": float(ct + 0.35 * fps)}

    pc = poses.get(ct)
    if pc is None and pre:
        pc = poses[pre[-1]]
    if pc is not None:
        dy = (pc[R_ANKLE, 1] - pc[L_ANKLE, 1]) * mpp
        m["right_foot_ahead_m"] = round(float(dy if camera_side == "far" else -dy), 2)
        cxy = ball.contact_xy if ball is not None and ball.contact_xy else racket_point(pc, scale.stature_px)
        m["contact_height_m"] = round(scale.height_m(cxy[1]), 2)
        m["contact_height_ratio"] = round(m["contact_height_m"] / scale.player_height_m, 2)
        # top of the head ~ nose height + 0.10 m
        m["head_height_m"] = round(scale.height_m(pc[NOSE, 1]) + 0.10, 2)
        m["contact_above_head_m"] = round(m["contact_height_m"] - m["head_height_m"], 2)
        hip_x = (pc[L_HIP, 0] + pc[R_HIP, 0]) / 2
        m["contact_spacing_m"] = round(_side(pc) * (cxy[0] - hip_x) * mpp, 2)
        from .analysis import angle
        if min(pc[R_SHOULDER, 2], pc[R_ELBOW, 2], pc[R_WRIST, 2]) >= 0.6 and scale.stature_px >= 150:
            m["hit_elbow_deg"] = round(angle(pc[R_SHOULDER], pc[R_ELBOW], pc[R_WRIST]), 1)
        sw = [width(t) for t in _window(poses, ct - 1.5 * fps, ct + 0.5 * fps)]
        if sw and max(sw) > 1e-6:
            m["contact_rotation_ratio"] = round(width(ct if ct in poses else pre[-1]) / max(sw), 2)
        m["stance_width_m"] = round(float(np.hypot(*(pc[L_ANKLE, :2] - pc[R_ANKLE, :2]))) * mpp, 2)
        near = _window(poses, ct - 0.06 * fps, ct + 0.06 * fps)
        if len(near) >= 2:
            a_, b_ = poses[near[0]][NOSE, :2], poses[near[-1]][NOSE, :2]
            m["head_move_m"] = round(float(np.hypot(*(b_ - a_))) * mpp, 2)
    if pre:
        # 10th percentile, not the minimum: one glitched skeleton must not decide
        m["knee_bend_deg"] = round(180.0 - float(np.nanpercentile([knee_angle(poses[t]) for t in pre], 10)), 1)
        low = max(poses[t][R_WRIST, 1] for t in pre)                 # lowest hand (max y)
        if post:
            high = min(poses[t][R_WRIST, 1] for t in post)
            m["low_to_high_m"] = round((low - high) * mpp, 2)
        sq = [width(t) for t in _window(poses, ct - 1.5 * fps, ct + 0.5 * fps)]
        if sq and max(sq) > 1e-6 and back is not None:
            m["unit_turn_ratio"] = round(width(back) / max(sq), 2)
        if back is not None:
            m["prep_lead_s"] = round((contact_t - back) / fps, 2)
            pb = poses[back]
            sh = scale.height_m((pb[L_SHOULDER, 1] + pb[R_SHOULDER, 1]) / 2)
            if sh > 0.1:
                m["left_arm_height_ratio"] = round(scale.height_m(pb[L_WRIST, 1]) / sh, 2)
        hip = lambda t: scale.height_m((poses[t][L_HIP, 1] + poses[t][R_HIP, 1]) / 2)
        if ct in poses:
            m["leg_drive_m"] = round(hip(ct) - min(hip(t) for t in pre), 2)
    if post:
        ok = 0.0
        for t in post:
            p = poses[t]
            body_x = (p[L_HIP, 0] + p[R_HIP, 0]) / 2
            crossed = _side(p) * (p[R_WRIST, 0] - body_x) < 0
            high = p[R_WRIST, 1] <= p[L_SHOULDER, 1] + 0.06 * scale.stature_px
            if crossed and high:
                ok = 1.0
                break
        m["follow_through"] = ok
        fin = poses.get(int(ct + 0.35 * fps))
        if fin is None:
            fin = poses[post[-1]]
        feet_mid = (fin[L_ANKLE, 0] + fin[R_ANKLE, 0]) / 2
        m["balance_offset_m"] = round(abs(fin[NOSE, 0] - feet_mid) * mpp, 2)
    if pc is not None:
        feet_x = (pc[L_ANKLE, 0] + pc[R_ANKLE, 0]) / 2
        feet_y = max(pc[L_ANKLE, 1], pc[R_ANKLE, 1])
        toward_net = 1.0 if camera_side == "far" else -1.0   # image +y = toward the net?
        if bounce is not None:
            m["bounce_to_contact_s"] = round((contact_t - bounce[0]) / fps, 2)
        if baseline is not None:
            from ..court_segmentation.court_detector import line_y_at
            d = toward_net * (feet_y - line_y_at(baseline, feet_x)) * mpp   # + = inside the court
            m["baseline_distance_m"] = round(d, 2)
            m["court_position"] = "inside" if d > 0.3 else "behind"
    if ball is not None and len(ball.outgoing) >= 2:
        (t0, x0, y0), (t1, x1, y1) = ball.outgoing[0], ball.outgoing[-1]
        m["ball_speed_kmh"] = round(np.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1) * fps * mpp * 3.6, 0)
        if m["ball_speed_kmh"] < 20:      # a hit ball is never this slow -> bad track
            m["ball_speed_kmh"] = None
    # racquet-based measurements (only with a detected racquet head)
    if rackets:
        heads = {t: r[1] for t, r in rackets.items() if r is not None}
        before = [t for t in heads if ct - 0.4 * fps <= t < ct]
        if before and m["contact_height_m"] is not None:
            lowest = max(heads[t][1] for t in before)
            m["racquet_below_contact_m"] = round(m["contact_height_m"] - scale.height_m(lowest), 2)
        around = sorted(t for t in heads if abs(t - contact_t) <= max(2, 0.03 * fps))
        if len(around) >= 2:
            (ta, (xa, ya)), (tb, (xb, yb)) = (around[0], heads[around[0]]), (around[-1], heads[around[-1]])
            m["racquet_speed_ms"] = round(float(np.hypot(xb - xa, yb - ya)) / max(tb - ta, 1) * fps * mpp, 1)
    from .joints import forehand_extras
    m.update(forehand_extras(poses, ball, scale, fps, contact_t, back))    # advanced joint analysis
    return phase, m


def apply_fh_rules(m: Dict[str, Optional[float]]) -> List[RuleResult]:
    out = []
    for key, cat, title, meas, ok, target, tip in FH_RULES:
        v = m.get(meas)
        tip = tip(m) if callable(tip) else tip
        if v is None:
            out.append(RuleResult(key, title, "unknown", None, meas, target, "Not measurable in this clip.", cat))
        else:
            out.append(RuleResult(key, title, "good" if ok(v) else "improve", v, meas, target, tip, cat))
    return out
