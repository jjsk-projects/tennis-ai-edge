"""Ball toss / contact detection, serve rules and RF suggestions on synthetic data."""
import numpy as np
import pytest

from src.stages.ball_tracking.ball_tracker import (
    drop_static, find_contact, fit_toss, gravity_px)
from src.stages.serve_analysis.analysis import FEATURES, Scale, apply_rules, angle


FPS, PLAYER_H = 30.0, 90.0


def synthetic_serve(seed=0, noise_per_frame=12):
    """A toss parabola + a hit ball + random flicker + one static blob."""
    rng = np.random.default_rng(seed)
    g = gravity_px(PLAYER_H, FPS)
    a, apex_t, apex_y, x0 = g / 2, 35.0, 340.0, 180.0
    cands = {}
    for t in range(10, 70):
        cs = [(float(rng.uniform(0, 480)), float(rng.uniform(200, 500)), 20.0)
              for _ in range(noise_per_frame)]
        cs.append((300.0, 260.0, 30.0))                      # static flicker
        if 15 <= t <= 50:                                    # toss up and down
            cs.append((x0 - 0.3 * (t - 15), apex_y + a * (t - apex_t) ** 2, 40.0))
        if t >= 51:                                          # hit ball flies right
            cs.append((200.0 + 40.0 * (t - 51), 384.0, 50.0))
        cands[t] = cs
    return cands


def test_toss_is_found_through_noise():
    cands = drop_static(synthetic_serve())
    toss = fit_toss(cands, PLAYER_H, FPS)
    assert toss is not None
    assert abs(toss.apex_frame - 35.0) < 1.0
    assert len(toss.points) >= 25


def test_contact_is_midpoint_of_the_two_frames():
    cands = drop_static(synthetic_serve())
    ball = find_contact(fit_toss(cands, PLAYER_H, FPS), cands, PLAYER_H)
    assert ball.method == "ball-midpoint"
    assert ball.contact_frame == pytest.approx(50.5)
    toss_50 = (180.0 - 0.3 * 35, 340.0 + gravity_px(PLAYER_H, FPS) / 2 * 15 ** 2)
    expected = ((toss_50[0] + 200.0) / 2, (toss_50[1] + 384.0) / 2)
    assert ball.contact_xy[0] == pytest.approx(expected[0], abs=2.0)
    assert ball.contact_xy[1] == pytest.approx(expected[1], abs=2.0)


def test_no_toss_in_pure_noise():
    rng = np.random.default_rng(3)
    cands = {t: [(float(rng.uniform(0, 480)), float(rng.uniform(200, 500)), 20.0) for _ in range(5)]
             for t in range(60)}
    assert fit_toss(cands, PLAYER_H, FPS) is None


def test_scale_converts_pixels_to_metres():
    s = Scale(ground_y=500.0, stature_px=90.0, player_height_m=1.8)
    assert s.height_m(500.0) == 0.0
    assert s.height_m(410.0) == pytest.approx(1.8)


def test_angle():
    assert angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)
    assert angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)


def test_rules_flag_shallow_knee_bend_and_keep_unknowns():
    m = {f: None for f in FEATURES}
    m.update(knee_bend_deg=8.0, contact_height_ratio=1.6)
    res = {r.key: r for r in apply_rules(m)}
    assert res["knee_bend"].status == "improve"
    assert "squat" in res["knee_bend"].tip.lower()
    assert res["contact_height"].status == "good"
    assert res["toss_side"].status == "unknown"


def test_forehand_too_high_advice_depends_on_court_position():
    from src.stages.serve_analysis.forehand import FH_FEATURES, apply_fh_rules
    m = {f: None for f in FH_FEATURES}
    m.update(contact_above_head_m=0.2, court_position="behind")
    r = {x.key: x for x in apply_fh_rules(m)}["contact_too_high"]
    assert r.status == "improve" and "move forward" in r.tip.lower()
    m["court_position"] = "inside"
    r = {x.key: x for x in apply_fh_rules(m)}["contact_too_high"]
    assert "no-man's land" in r.tip and "wait for the ball to drop" in r.tip


def test_forehand_rules_cover_bounce_and_sideways_and_are_prioritised():
    from src.stages.serve_analysis.forehand import FH_RULES, FH_FEATURES, apply_fh_rules
    keys = [r[0] for r in FH_RULES]
    assert keys[:3] == ["contact_too_high", "bounce_too_far", "sideways"]
    m = {f: None for f in FH_FEATURES}
    m.update(bounce_to_contact_s=1.1, unit_turn_ratio=0.95)
    res = {x.key: x for x in apply_fh_rules(m)}
    assert res["bounce_too_far"].status == "improve" and "move forward" in res["bounce_too_far"].tip
    assert res["sideways"].status == "improve" and "sideways" in res["sideways"].tip
    assert res["spacing"].status == "unknown"


def test_every_rule_has_a_category():
    from src.stages.serve_analysis.analysis import RULES
    from src.stages.serve_analysis.forehand import FH_RULES
    assert len(RULES) >= 12 and len(FH_RULES) >= 15
    assert all(r[1] for r in RULES + FH_RULES)


# ---------------------------------------------------------------- levels
def test_levels_have_different_checks_targets_and_scores():
    from src.stages.serve_analysis.levels import LEVELS, judge, score, level_rules
    for mode in ("forehand", "serve"):
        sizes = [len(level_rules(mode, lv)) for lv in LEVELS]
        assert sizes[0] < sizes[1] < sizes[2]          # beginner < intermediate < club
    # knee bend 17 deg: fine for intermediate (>= 15), not for club (>= 20); not judged for beginners
    m = {"knee_bend_deg": 17.0}
    st = {lv: {r.key: r.status for r in judge("forehand", m, lv)} for lv in LEVELS}
    assert "knee_bend" not in st["beginner"]
    assert st["intermediate"]["knee_bend"] == "good" and st["club"]["knee_bend"] == "improve"
    # beginner wording is plain
    r = [r for r in judge("forehand", {"contact_rotation_ratio": 0.2}, "beginner") if r.key == "contact_in_front"][0]
    assert r.status == "improve" and "in front of you" in r.tip
    # score: weighted share of measurable checks, unknowns ignored
    # strict score: only "excellent" values earn full marks; just meeting a target earns 0.4
    best = judge("forehand", {"contact_rotation_ratio": 1.0, "contact_above_head_m": -0.8}, "beginner")
    assert score(best) == 100
    just = judge("forehand", {"contact_rotation_ratio": 0.45, "contact_above_head_m": 0.10}, "beginner")
    assert all(r.status == "good" for r in just if r.value is not None) and score(just) < 30
    bad = judge("forehand", {"contact_rotation_ratio": 0.2, "contact_above_head_m": -0.3}, "beginner")
    assert score(bad) < 50                             # the first (heavier) check failed
    assert score(judge("forehand", {}, "club")) is None
    # the same value scores lower against a higher level, and the ladder says where it sits
    one = {"knee_bend_deg": 17.0}
    sc = {lv: judge("forehand", one, lv) for lv in ("intermediate", "club")}
    kb = {lv: [r for r in sc[lv] if r.key == "knee_bend"][0] for lv in sc}
    assert kb["intermediate"].credit > kb["club"].credit and kb["club"].reached == "intermediate"
    assert "B >= 10" in kb["club"].ladder and "C >= 20" in kb["club"].ladder


def test_every_check_has_a_ladder_and_joint_checks_are_club_level():
    from src.stages.serve_analysis.levels import FH_LADDER, SERVE_LADDER, level_rules
    from src.stages.serve_analysis.forehand import FH_RULES
    from src.stages.serve_analysis.analysis import RULES
    assert {r[0] for r in FH_RULES} == set(FH_LADDER) and {r[0] for r in RULES} == set(SERVE_LADDER)
    club = [r[0] for r in level_rules("forehand", "club")]
    beg = [r[0] for r in level_rules("forehand", "beginner")]
    for k in ("coil", "kinetic_chain", "spine_steady", "head_level", "split_step", "weight_transfer",
              "left_arm_tuck", "knee_flex_contact"):
        assert k in club and k not in beg


def test_joint_extras_measure_split_step_and_knee_flex():
    from src.stages.serve_analysis.joints import forehand_extras
    from src.stages.serve_analysis.analysis import Scale
    from src.stages.pose_model.pose_model_backend import L_ANKLE, R_ANKLE, L_KNEE, R_KNEE, L_HIP, R_HIP
    fps = 60
    base = np.zeros((33, 3)); base[:, 2] = 1.0
    base[[11, 12], 1] = 100; base[11, 0], base[12, 0] = 110, 90       # shoulders
    base[[L_HIP, R_HIP], 1] = 200; base[L_HIP, 0], base[R_HIP, 0] = 105, 95
    base[[L_KNEE, R_KNEE], 1] = 300; base[L_KNEE, 0], base[R_KNEE, 0] = 115, 85   # bent knees
    base[[L_ANKLE, R_ANKLE], 1] = 400; base[L_ANKLE, 0], base[R_ANKLE, 0] = 105, 95
    poses = {}
    for t in range(0, 120):
        p = base.copy()
        if 20 <= t <= 24:                       # both feet hop 10 px ~ 0.045 m
            p[[L_ANKLE, R_ANKLE], 1] -= 10
        poses[t] = p
    sc = Scale(ground_y=400.0, stature_px=400.0, player_height_m=1.8)
    m = forehand_extras(poses, None, sc, fps, 90.0, 70)
    assert abs(m["split_step_m"] - 0.045) < 0.01
    assert m["knee_flex_contact_deg"] > 10

def test_coach_beginner_is_plain_and_club_shows_values():
    from src.stages.local_llm.coach import template_feedback
    from src.stages.serve_analysis.levels import judge, score, suggestions
    from dataclasses import asdict
    m = {"contact_rotation_ratio": 0.2, "knee_bend_deg": 5.0}
    for lv, must in (("beginner", "Try this next time"), ("club", "measured 5.0")):
        rules = judge("forehand", m, lv)
        rep = {"mode": "forehand", "level": lv, "rules": [asdict(r) for r in rules],
               "suggestions": suggestions(rules), "score": score(rules), "measurements": m}
        txt = template_feedback(rep)
        assert must in txt and "Score:" in txt


def test_backhand_is_recognised_from_contact_side():
    from src.stages.serve_analysis.forehand import facing_sign, stroke_side
    from src.stages.pose_model.pose_model_backend import L_HIP, R_HIP, L_SHOULDER, R_SHOULDER
    p = np.zeros((33, 3)); p[:, 2] = 1
    p[R_SHOULDER, 0], p[L_SHOULDER, 0] = 90, 110          # facing the camera: right side = image left
    p[R_HIP, 0], p[L_HIP, 0] = 95, 105
    poses = {t: p for t in range(0, 60)}
    f = facing_sign(poses)
    assert f == -1
    assert stroke_side(poses, (60, 50), 30, 30, f) == "forehand"     # hit on the image left
    assert stroke_side(poses, (140, 50), 30, 30, f) == "backhand"
