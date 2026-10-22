"""Stage 4c - Advanced joint analysis (Club level, some Intermediate).

Looks at HOW the body moves, not only where things end up, from the 33-point
skeleton over time:

Forehand
- coil_sep           shoulders turn further than the hips at the backswing (hip-shoulder separation)
- chain_lead_s       the hips start unwinding before the hitting hand peaks (kinetic chain)
- spine_change_deg   the spine angle stays the same from backswing to contact
- head_level_m       the head stays level through the whole swing
- split_step_m       a small hop (both feet) before the ball arrives
- weight_transfer_m  the hips move into the shot, in the direction the ball leaves
- left_arm_tuck_m    the left arm folds in to the body at contact
- knee_flex_contact_deg  still in the legs at contact, not standing up early

Serve
- trophy_lean_deg        the trunk leans up and in at the trophy (not upright)
- chain_lead_s           the legs fire (knee extension peak) before the arm (wrist speed peak)
- body_line_deg          feet-to-hand line close to vertical at contact (stacked under the ball)
- contact_tilt_m         hitting shoulder above the tossing shoulder at contact (cartwheel)
- toss_arm_down_m        the tossing hand has pulled down below its shoulder at contact

All are 2D image measurements. They are only taken when the skeleton is reliable:
player at least MIN_PX tall and the joints used seen with confidence >= MIN_CONF;
otherwise they are left out ("not measurable") rather than guessed.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..pose_model.pose_model_backend import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST)

MIN_PX = 80        # body-level measures (trunk, hips, feet)
MIN_CONF = 0.5
FINE_PX = 150      # shoulder-height differences of a few cm need a bigger player

FH_KEYS = ["coil_sep", "chain_lead_s", "spine_change_deg", "head_level_m", "split_step_m",
           "weight_transfer_m", "left_arm_tuck_m", "knee_flex_contact_deg"]
SERVE_KEYS = ["trophy_lean_deg", "chain_lead_s", "body_line_deg", "contact_tilt_m", "toss_arm_down_m"]


def _f(x):
    return None if x is None else float(x)


def _win(poses, a, b):
    return [t for t in range(int(a), int(b) + 1) if t in poses]


def _ok(p, *idx) -> bool:
    return min(p[i, 2] for i in idx) >= MIN_CONF


def _mid(p, a, b):
    return (p[a, :2] + p[b, :2]) / 2


def _trunk_deg(p) -> float:
    """Trunk lean from vertical (degrees), hips -> shoulders."""
    v = _mid(p, L_SHOULDER, R_SHOULDER) - _mid(p, L_HIP, R_HIP)
    return float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))


def _knee(p) -> float:
    from .analysis import knee_angle
    return knee_angle(p)


def _speed(series: Dict[int, float], fps: float) -> Dict[int, float]:
    ts = sorted(series)
    if len(ts) < 3:
        return {}
    v = np.gradient(np.array([series[t] for t in ts]), np.array(ts, float)) * fps
    k = max(1, int(round(fps / 60)))
    v = np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    return dict(zip(ts, v))


def forehand_extras(poses, ball, scale, fps: float, contact_t: float, back: Optional[int]) -> Dict:
    m: Dict[str, Optional[float]] = {k: None for k in FH_KEYS}
    if scale.stature_px < MIN_PX:
        return m
    mpp, ct = scale.m_per_px, int(round(contact_t))
    pc = poses.get(ct)
    swing = _win(poses, (back if back is not None else ct - 0.5 * fps), ct)
    wide = _win(poses, ct - 1.5 * fps, ct + 0.5 * fps)
    sw = lambda p: abs(p[R_SHOULDER, 0] - p[L_SHOULDER, 0])
    hw = lambda p: abs(p[R_HIP, 0] - p[L_HIP, 0])

    if back is not None and wide:
        pb = poses[back]
        if _ok(pb, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            ms, mh = max(sw(poses[t]) for t in wide), max(hw(poses[t]) for t in wide)
            if ms > 1e-6 and mh > 1e-6:
                m["coil_sep"] = round(hw(pb) / mh - sw(pb) / ms, 2)
    if len(swing) >= 5:
        hip_open = _speed({t: hw(poses[t]) for t in swing}, fps)             # hips unwinding
        wrist = {t: float(poses[t][R_WRIST, 0]) for t in swing}
        wy = {t: float(poses[t][R_WRIST, 1]) for t in swing}
        vx, vy = _speed(wrist, fps), _speed(wy, fps)
        if hip_open and vx:
            t_hip = max(hip_open, key=hip_open.get)
            t_wr = max(vx, key=lambda t: np.hypot(vx[t], vy[t]))
            m["chain_lead_s"] = round((t_wr - t_hip) / fps, 2)
    if back is not None and pc is not None and _ok(poses[back], L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) \
            and _ok(pc, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        m["spine_change_deg"] = round(abs(_trunk_deg(pc) - _trunk_deg(poses[back])), 1)
    path = _win(poses, (back if back is not None else ct - 0.5 * fps), ct + 0.1 * fps)
    noses = [poses[t][NOSE, 1] for t in path if poses[t][NOSE, 2] >= MIN_CONF]
    if len(noses) >= 4:
        m["head_level_m"] = round((np.percentile(noses, 95) - np.percentile(noses, 5)) * mpp, 2)
    pre = _win(poses, ct - 1.3 * fps, ct - 0.35 * fps)
    if len(pre) >= 6:
        la = np.array([poses[t][L_ANKLE, 1] for t in pre]); ra = np.array([poses[t][R_ANKLE, 1] for t in pre])
        base_l, base_r = np.percentile(la, 85), np.percentile(ra, 85)        # feet on the ground (max y)
        hop = np.minimum(base_l - la, base_r - ra)                            # both feet up together
        m["split_step_m"] = round(float(max(0.0, hop.max())) * mpp, 3)
    if back is not None and pc is not None and ball is not None and len(getattr(ball, "outgoing", [])) >= 2:
        d = np.sign(ball.outgoing[-1][1] - ball.outgoing[0][1]) or 1.0
        hb, hc = _mid(poses[back], L_HIP, R_HIP)[0], _mid(pc, L_HIP, R_HIP)[0]
        m["weight_transfer_m"] = round(float(d * (hc - hb)) * mpp, 2)
    if pc is not None and _ok(pc, L_WRIST, L_SHOULDER, R_SHOULDER):
        torso_x = _mid(pc, L_SHOULDER, R_SHOULDER)[0]
        m["left_arm_tuck_m"] = round(abs(pc[L_WRIST, 0] - torso_x) * mpp, 2)
    if pc is not None and _ok(pc, L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE):
        m["knee_flex_contact_deg"] = round(180.0 - _knee(pc), 1)
    return {k: _f(v) for k, v in m.items()}


def serve_extras(poses, scale, fps: float, contact_t: float, trophy: Optional[int]) -> Dict:
    m: Dict[str, Optional[float]] = {k: None for k in SERVE_KEYS}
    if scale.stature_px < MIN_PX:
        return m
    mpp, ct = scale.m_per_px, int(round(contact_t))
    pc = poses.get(ct)
    if trophy is not None and _ok(poses[trophy], L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        m["trophy_lean_deg"] = round(_trunk_deg(poses[trophy]), 1)
    if trophy is not None:
        span = _win(poses, trophy, ct + 2)
        if len(span) >= 4:
            ext = _speed({t: _knee(poses[t]) for t in span}, fps)                  # knee straightening
            wr = _speed({t: -float(poses[t][R_WRIST, 1]) for t in span}, fps)      # hand going up
            if ext and wr:
                m["chain_lead_s"] = round((max(wr, key=wr.get) - max(ext, key=ext.get)) / fps, 2)
    if pc is not None and _ok(pc, L_ANKLE, R_ANKLE, R_WRIST):
        v = pc[R_WRIST, :2] - _mid(pc, L_ANKLE, R_ANKLE)
        if v[1] < 0:
            m["body_line_deg"] = round(float(np.degrees(np.arctan2(abs(v[0]), -v[1]))), 1)
    if pc is not None and _ok(pc, L_SHOULDER, R_SHOULDER) and scale.stature_px >= FINE_PX:
        m["contact_tilt_m"] = round(float(pc[L_SHOULDER, 1] - pc[R_SHOULDER, 1]) * mpp, 2)
    if pc is not None and _ok(pc, L_WRIST, L_SHOULDER):
        m["toss_arm_down_m"] = round(float(pc[L_WRIST, 1] - pc[L_SHOULDER, 1]) * mpp, 2)
    return {k: _f(v) for k, v in m.items()}
