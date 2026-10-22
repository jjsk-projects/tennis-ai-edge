"""Stage 3 - Serve segmentation, phases, measurements and coaching rules.

Input: per-frame poses (BlazePose 33 x [x_px, y_px, conf], full frame),
ball contact info from the ball tracker, and the baseline line.
Output: one ``ServeReport`` per serve with phase frames, measurements in
real units (metres / degrees) and rule verdicts.

Heights in metres: the player's real height (default 1.80 m, set in the
UI) divided by their standing height in pixels gives metres-per-pixel at
the player's depth. That is accurate for things in the player's plane
(their joints, the toss, the contact point) - exactly what a serve needs.

Right-handed player (MVP): hitting arm = right (BlazePose 12/14/16),
toss arm = left (11/13/15).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..pose_model.pose_model_backend import (
    L_ANKLE, L_ELBOW, L_HEEL, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
    R_ANKLE, R_ELBOW, R_HEEL, R_HIP, R_KNEE, R_SHOULDER, R_WRIST)

Pose = np.ndarray  # (33, 3)
GATE = 0.4


def angle(a, b, c) -> float:
    """Angle ABC in degrees (2D)."""
    u, v = np.asarray(a[:2]) - np.asarray(b[:2]), np.asarray(c[:2]) - np.asarray(b[:2])
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))


def knee_angle(p: Pose) -> float:
    """Mean knee angle (180 = straight)."""
    return float(np.nanmean([angle(p[L_HIP], p[L_KNEE], p[L_ANKLE]),
                             angle(p[R_HIP], p[R_KNEE], p[R_ANKLE])]))


def _conf(p: Pose, *idx) -> float:
    return float(min(p[i, 2] for i in idx))


# --------------------------------------------------------------------------- #
# Scale: pixels -> metres at the player's depth
# --------------------------------------------------------------------------- #
@dataclass
class Scale:
    ground_y: float        # image y of the ground under the player (px)
    stature_px: float      # standing height in px
    player_height_m: float

    @property
    def m_per_px(self) -> float:
        return self.player_height_m / self.stature_px

    def height_m(self, y: float) -> float:
        return (self.ground_y - y) * self.m_per_px


def estimate_scale(poses: Dict[int, Pose], frames: Sequence[int],
                   player_height_m: float) -> Optional[Scale]:
    """Use frames where the player stands upright (knees nearly straight)."""
    def collect(min_knee):
        st, gr = [], []
        for t in frames:
            p = poses.get(t)
            if p is None or _conf(p, NOSE, L_HEEL, R_HEEL) < GATE or knee_angle(p) < min_knee:
                continue
            ground = max(p[L_HEEL, 1], p[R_HEEL, 1])
            st.append((ground - p[NOSE, 1]) / 0.92)   # nose sits at ~92 % of standing height
            gr.append(ground)
        return st, gr

    statures, grounds = collect(165)
    if len(statures) >= 3:
        return Scale(float(np.median(grounds)), float(np.median(statures)), player_height_m)
    # rallies: knees are rarely straight -> tallest postures (85th percentile)
    statures, grounds = collect(140)
    if len(statures) < 3:
        return None
    return Scale(float(np.median(grounds)), float(np.percentile(statures, 85)), player_height_m)


# --------------------------------------------------------------------------- #
# Serve segmentation
# --------------------------------------------------------------------------- #
def find_serve_peaks(poses: Dict[int, Pose], fps: float, n_frames: int,
                     min_gap_s: float = 2.0) -> List[int]:
    """Frames where a hand is far above the head (toss/contact moments)."""
    score = np.full(n_frames, -np.inf)
    for t, p in poses.items():
        if _conf(p, NOSE, L_HEEL) < GATE:
            continue
        body = max(p[L_HEEL, 1], p[R_HEEL, 1]) - p[NOSE, 1]
        if body <= 1:
            continue
        hand = min(p[L_WRIST, 1], p[R_WRIST, 1])
        score[t] = (p[NOSE, 1] - hand) / body       # >0: hand above the nose
    peaks: List[int] = []
    order = np.argsort(-score)
    for t in order:
        if not np.isfinite(score[t]) or score[t] < 0.08:
            break
        if all(abs(t - q) > min_gap_s * fps for q in peaks):
            peaks.append(int(t))
    return sorted(peaks)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class RuleResult:
    key: str
    title: str
    status: str          # "good" | "improve" | "unknown"
    value: Optional[float]
    unit: str
    target: str
    tip: str
    category: str = ""
    reached: str = ""                 # highest level whose target the value meets (levels.py)
    credit: Optional[float] = None    # 0-1 score credit for the chosen level (levels.py)
    ladder: str = ""                  # targets from Beginner to Club, e.g. "B >= 10 | I >= 15 | C >= 20 deg"


@dataclass
class ServeReport:
    index: int
    frames: Dict[str, Optional[float]]          # ready / trophy / contact / finish
    contact_method: str
    measurements: Dict[str, Optional[float]] = field(default_factory=dict)
    rules: List[RuleResult] = field(default_factory=list)
    suggestions: List[Dict] = field(default_factory=list)
    timeline: Dict[str, List] = field(default_factory=dict)
    images: Dict[str, str] = field(default_factory=dict)
    coach_text: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _nearest_pose(poses: Dict[int, Pose], t: float, max_d: int = 2) -> Optional[Pose]:
    ti = int(round(t))
    for d in range(max_d + 1):
        for c in (ti - d, ti + d):
            if c in poses:
                return poses[c]
    return None


def measure_serve(poses: Dict[int, Pose], ball, scale: Scale, fps: float,
                  peak: int, baseline=None, camera_side: str = "far",
                  rackets=None) -> Tuple[Dict, Dict]:
    """Phase frames + measurements for one serve."""
    from ..court_segmentation.court_detector import line_y_at

    contact_t = ball.contact_frame if ball and ball.contact_frame is not None else float(peak)
    ct = int(round(contact_t))
    win = [t for t in range(int(ct - 1.3 * fps), ct) if t in poses]
    # trophy = deepest knee bend before contact
    trophy = min(win, key=lambda t: knee_angle(poses[t])) if win else None
    ready = int(max(min(poses), (ball.toss[0][0] if ball and ball.toss else ct - 1.5 * fps) - 8))
    finish = int(ct + 0.5 * fps)
    frames = {"ready": ready, "trophy": trophy, "contact": contact_t, "finish": finish}

    m: Dict[str, Optional[float]] = {k: None for k in (
        "knee_bend_deg", "contact_height_m", "contact_height_ratio", "toss_apex_m",
        "toss_above_contact_m", "toss_side_offset", "hit_elbow_deg", "toss_arm_elbow_deg",
        "leg_drive_m", "airborne_m", "front_foot_vs_baseline_m", "ball_speed_kmh",
        "pose_confidence_contact", "trophy_elbow_vs_shoulder_m", "shoulder_tilt_m",
        "toss_arm_up_m", "head_drop_m", "landing_inside_m", "racquet_drop_m")}
    mpp = scale.m_per_px
    if trophy is not None:
        m["knee_bend_deg"] = round(180.0 - knee_angle(poses[trophy]), 1)
    pc = _nearest_pose(poses, contact_t)
    if ball and ball.contact_xy:
        m["contact_height_m"] = round(scale.height_m(ball.contact_xy[1]), 2)
        m["contact_height_ratio"] = round(m["contact_height_m"] / scale.player_height_m, 2)
    if ball and ball.apex_xy:
        m["toss_apex_m"] = round(scale.height_m(ball.apex_xy[1]), 2)
        if m["contact_height_m"] is not None:
            m["toss_above_contact_m"] = round(m["toss_apex_m"] - m["contact_height_m"], 2)
    if pc is not None:
        m["pose_confidence_contact"] = round(_conf(pc, R_SHOULDER, R_ELBOW, R_WRIST), 2)
        if m["pose_confidence_contact"] >= 0.6 and scale.stature_px >= 150:   # arm must be clear + big enough
            m["hit_elbow_deg"] = round(angle(pc[R_SHOULDER], pc[R_ELBOW], pc[R_WRIST]), 1)
        if ball and ball.contact_xy:
            # + = toward the hitting-arm side (right shoulder side in the image)
            side = np.sign(pc[R_SHOULDER, 0] - pc[L_SHOULDER, 0]) or 1.0
            m["toss_side_offset"] = round(float(side * (ball.contact_xy[0] - pc[NOSE, 0]) * mpp), 2)
        heel = min(scale.height_m(pc[L_HEEL, 1]), scale.height_m(pc[R_HEEL, 1]))
        m["airborne_m"] = round(max(0.0, heel), 2)
    if trophy is not None and pc is not None:
        hip_t = (poses[trophy][L_HIP, 1] + poses[trophy][R_HIP, 1]) / 2
        hip_c = (pc[L_HIP, 1] + pc[R_HIP, 1]) / 2
        m["leg_drive_m"] = round((hip_t - hip_c) * mpp, 2)
    if trophy is not None:
        pt_ = poses[trophy]
        h = scale.height_m
        m["trophy_elbow_vs_shoulder_m"] = round(h(pt_[R_ELBOW, 1]) - h(pt_[R_SHOULDER, 1]), 2)
        if scale.stature_px >= 150:     # a 5 cm tilt is under 3 px on a small player: too noisy
            m["shoulder_tilt_m"] = round(h(pt_[L_SHOULDER, 1]) - h(pt_[R_SHOULDER, 1]), 2)
        m["toss_arm_up_m"] = round(h(pt_[L_WRIST, 1]) - (h(pt_[NOSE, 1]) + 0.10), 2)
        if pc is not None:
            # head should stay up through contact (it rises with the jump, never drops)
            m["head_drop_m"] = round(h(pt_[NOSE, 1]) - h(pc[NOSE, 1]), 2)
    # racquet drop behind the back between trophy and contact (detected racquet only)
    if rackets and trophy is not None:
        drops = []
        for t in range(int(trophy), ct):
            r, p = rackets.get(t), poses.get(t)
            if r is not None and p is not None:
                drops.append((r[1][1] - p[R_WRIST, 1]) * mpp)   # + = head below the hand
        if len(drops) >= 2:
            m["racquet_drop_m"] = round(max(drops), 2)
    # toss arm extension at its highest point before contact
    tw = [t for t in range(int(ct - 1.3 * fps), ct) if t in poses]
    if tw:
        tr = min(tw, key=lambda t: poses[t][L_WRIST, 1])
        m["toss_arm_elbow_deg"] = round(angle(poses[tr][L_SHOULDER], poses[tr][L_ELBOW],
                                              poses[tr][L_WRIST]), 1)
    # foot vs baseline before contact (+ = over the line into the court)
    if baseline is not None:
        worst = None
        for t in range(ready, ct):
            p = poses.get(t)
            if p is None:
                continue
            for h in (L_HEEL, R_HEEL, L_ANKLE, R_ANKLE):
                dy = p[h, 1] - line_y_at(baseline, p[h, 0])
                over = dy if camera_side == "far" else -dy
                worst = over if worst is None else max(worst, over)
        if worst is not None:
            m["front_foot_vs_baseline_m"] = round(worst * mpp, 2)
        # landing: after contact the server should come down inside the court
        land = None
        for t in range(ct + 1, int(ct + 0.6 * fps)):
            p = poses.get(t)
            if p is None:
                continue
            for k in (L_HEEL, R_HEEL):
                dy = p[k, 1] - line_y_at(baseline, p[k, 0])
                inside = (dy if camera_side == "far" else -dy) * mpp
                land = inside if land is None else max(land, inside)
        if land is not None:
            m["landing_inside_m"] = round(land, 2)
    if ball and len(ball.outgoing) >= 2:
        (t0, x0, y0), (t1, x1, y1) = ball.outgoing[0], ball.outgoing[-1]
        px_per_frame = np.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1)
        # image-plane speed only -> a lower bound of the true ball speed
        m["ball_speed_kmh"] = round(px_per_frame * fps * mpp * 3.6, 0)
        if m["ball_speed_kmh"] < 20:      # a hit ball is never this slow -> bad track
            m["ball_speed_kmh"] = None
    from .joints import serve_extras
    m.update(serve_extras(poses, scale, fps, contact_t, trophy))     # advanced joint analysis
    return frames, m


# --------------------------------------------------------------------------- #
# Rules (right-handed serve). Targets are typical club-coaching ranges.
# --------------------------------------------------------------------------- #
RULES = [   # (key, category, title, measurement, test, target, tip) - list order = priority
    ("foot_fault", "Footwork", "No foot fault", "front_foot_vs_baseline_m", lambda v: v <= 0.10,
     "no foot over the baseline before contact (0.10 m tolerance)",
     "Your front foot creeps over the baseline before contact - that is a foot fault. "
     "Start a little further back and keep the front foot still until the ball is hit."),
    ("contact_height", "Contact", "Contact at full reach", "contact_height_ratio", lambda v: v >= 1.5,
     ">= 1.5 x your height (arm + racquet fully up)",
     "Hit the ball higher - reach up and contact it at full stretch, arm and racquet in one line."),
    ("toss_height", "Toss", "Toss height above contact", "toss_above_contact_m",
     lambda v: 0.15 <= v <= 1.0, "0.15-1.0 m above contact",
     "Make the toss consistent: it should peak a little (about one racquet-head to half a "
     "metre) above the contact point, so you hit it as it starts to drop."),
    ("toss_side", "Toss", "Toss on hitting side", "toss_side_offset", lambda v: -0.1 <= v <= 0.5,
     "0 to 0.5 m toward hitting arm",
     "Place the toss slightly to your hitting-arm side and in front, not behind or over the "
     "other shoulder."),
    ("knee_bend", "Trophy", "Knee bend at trophy", "knee_bend_deg", lambda v: v >= 18,
     ">= 18 deg (typical 20-45)",
     "Squat a bit lower in the trophy position. The leg drive pushes the racquet up "
     "the back of the ball - that brush is what creates topspin / kick and adds pace."),
    ("leg_drive", "Drive", "Leg drive (hip rise)", "leg_drive_m", lambda v: v >= 0.10, ">= 0.10 m",
     "Push up through the legs from the trophy position so the hips rise into the ball."),
    ("serve_chain", "Drive", "Legs fire before the arm (kinetic chain)", "chain_lead_s",
     lambda v: abs(v - 0.15) <= 0.10, "legs peak 0.05-0.25 s before the hand",
     "Let the legs start the swing: push up first, then the trunk turns, then the arm - "
     "the arm is the last link, not the first."),
    ("trophy_elbow", "Trophy", "Hitting elbow up at trophy", "trophy_elbow_vs_shoulder_m",
     lambda v: v >= -0.05, "elbow at or above shoulder height",
     "In the trophy position lift the hitting elbow to shoulder height - a low elbow "
     "shortens the swing and strains the shoulder."),
    ("shoulder_tilt", "Trophy", "Shoulders tilted up to the ball", "shoulder_tilt_m",
     lambda v: v >= 0.05, "tossing shoulder >= 0.05 m higher",
     "Tilt your shoulders at the trophy: tossing shoulder up, hitting shoulder down - "
     "the shoulders then rotate like a cartwheel up into the ball."),
    ("trophy_lean", "Trophy", "Body leans up and in at the trophy", "trophy_lean_deg",
     lambda v: v >= 10, "trunk leans >= 10 deg at the trophy",
     "At the trophy, lean the body up and into the court (hips forward, shoulders back) - "
     "it loads the trunk like a bow."),
    ("toss_arm", "Toss", "Toss arm straight", "toss_arm_elbow_deg", lambda v: v >= 150, ">= 150 deg",
     "Release the toss with a straight tossing arm - lift from the shoulder, not the elbow."),
    ("toss_arm_up", "Trophy", "Toss arm stays up", "toss_arm_up_m", lambda v: v >= 0.0,
     "tossing hand above the head at trophy",
     "Keep the tossing arm up (pointing at the ball) while the racquet drops - it keeps the "
     "shoulders tilted and the head up."),
    ("toss_arm_down", "Contact", "Toss arm pulls down before contact", "toss_arm_down_m",
     lambda v: v >= 0.10, "tossing hand >= 0.10 m below its shoulder at contact",
     "Pull the tossing arm down into your chest as the racquet goes up - it whips the hitting "
     "shoulder up into the ball."),
    ("racquet_drop", "Drive", "Racquet drops behind the back", "racquet_drop_m",
     lambda v: v >= 0.10, "racquet head >= 0.10 m below the hand",
     "Let the racquet head drop down behind your back before swinging up - a loose drop "
     "gives a longer, faster swing."),
    ("arm_extension", "Contact", "Hitting arm straight at contact", "hit_elbow_deg",
     lambda v: v >= 150, ">= 150 deg",
     "Reach up with a straight hitting arm at contact - a bent arm loses height and power."),
    ("body_line", "Contact", "Body stacked under the ball", "body_line_deg",
     lambda v: v <= 15, "feet-to-hand line within 15 deg of vertical",
     "Reach straight up at contact - feet, hips, shoulder and hand stacked in one tall line."),
    ("contact_tilt", "Contact", "Hitting shoulder up at contact (cartwheel)", "contact_tilt_m",
     lambda v: v >= 0.10, "hitting shoulder >= 0.10 m above the tossing shoulder",
     "Drive the hitting shoulder up through contact so the shoulders cartwheel - like throwing "
     "the racquet up at the ball."),
    ("head_up", "Contact", "Head up through contact", "head_drop_m", lambda v: v <= 0.10,
     "head does not drop before contact",
     "Keep your head up and eyes on the ball until after contact - dropping the head early "
     "pulls the ball into the net."),
    ("drive_up", "Drive", "Drive up off the ground", "airborne_m", lambda v: v >= 0.05,
     "feet >= 0.05 m off the ground at contact",
     "Drive up into the ball so your feet leave the ground at contact - the legs add pace "
     "and height."),
    ("landing", "Finish", "Land inside the court", "landing_inside_m", lambda v: v >= 0.10,
     "land >= 0.10 m inside the baseline",
     "Let the serve carry you forward - land on your front foot inside the court, ready "
     "for the next ball."),
]


def apply_rules(m: Dict[str, Optional[float]]) -> List[RuleResult]:
    out = []
    for key, cat, title, meas, ok, target, tip in RULES:
        v = m.get(meas)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(RuleResult(key, title, "unknown", None, meas, target,
                                  "Not measurable in this clip.", cat))
        else:
            good = bool(ok(v))
            out.append(RuleResult(key, title, "good" if good else "improve", v, meas, target, tip, cat))
    return out


FEATURES = [r[3] for r in RULES]
