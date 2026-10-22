"""Stage 4b - Player level: Beginner -> Intermediate -> Club.

Every check has a LADDER of targets on one scale:

    poor  <  Beginner  <  Intermediate  <  Club  <  excellent

so each measured value can be placed on the range ("reached: Intermediate").
The chosen level then decides
- which checks are judged (Beginner: the basics, Club: everything),
- the pass mark: that level's rung of the ladder,
- how the tip is worded (Beginner: one simple picture, no jargon).

Score (strict, 0-100): each judged check earns credit on a curve -
0 at "poor", 0.4 when it just meets the chosen level's target, 1.0 only at
"excellent" (the standard of a strong competitive player, well past Club).
The credits are averaged with priority weights (first check 2, last 1) and the
average is raised to the power 1.5. So a player who only just meets every
target scores about 25; 70+ needs most checks well past the target, and 90+
is close to excellent on everything.
Checks that could not be measured are left out.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .analysis import RULES, RuleResult
from .forehand import FH_RULES

LEVELS = ["beginner", "intermediate", "club"]
LEVEL_NAMES = {"beginner": "Beginner", "intermediate": "Intermediate", "club": "Club level"}
PASS_CREDIT = 0.4          # credit for just meeting the chosen level's target
STRICTNESS = 1.5           # final curve: score = 100 * (weighted mean credit) ** STRICTNESS

# ladder: (direction, poor, beginner, intermediate, club, excellent, unit[, ideal])
#   "min"  = higher is better, "max" = lower is better,
#   "band" = closer to `ideal` is better; the rungs are the allowed distance from it.
FH_LADDER = {
    "contact_too_high": ("max", 0.40, 0.10, 0.05, 0.00, -0.70, "m above head"),
    "bounce_too_far":   ("max", 1.60, 1.20, 1.00, 0.80, 0.40, "s"),
    "sideways":         ("max", 1.00, 0.90, 0.85, 0.80, 0.40, "x square-on"),
    "contact_in_front": ("min", 0.20, 0.45, 0.55, 0.60, 1.00, "turn ratio"),
    "foot_order":       ("max", 0.40, 0.20, 0.10, 0.05, -0.30, "m"),
    "contact_low":      ("min", 0.20, 0.30, 0.35, 0.40, 0.55, "x height"),
    "spacing":          ("band", 0.80, 0.50, 0.40, 0.33, 0.05, "m from ideal", 0.575),
    "arm_extension":    ("min", 70, 85, 95, 100, 150, "deg"),
    "head_still":       ("max", 0.50, 0.30, 0.20, 0.15, 0.03, "m"),
    "early_prep":       ("min", 0.00, 0.15, 0.25, 0.30, 0.60, "s"),
    "left_arm":         ("min", 0.30, 0.50, 0.65, 0.75, 1.00, "x shoulder"),
    "knee_bend":        ("min", 3, 10, 15, 20, 45, "deg"),
    "stance_width":     ("min", 0.20, 0.30, 0.40, 0.45, 0.80, "m"),
    "racquet_drop":     ("min", -0.10, 0.00, 0.10, 0.15, 0.45, "m"),
    "low_to_high":      ("min", 0.05, 0.20, 0.30, 0.40, 0.90, "m"),
    "leg_drive":        ("min", -0.03, 0.00, 0.02, 0.03, 0.12, "m"),
    "follow_through":   ("min", 0, 1, 1, 1, 1, "(1 = yes)"),
    "balance":          ("max", 0.60, 0.35, 0.30, 0.25, 0.05, "m"),
    # advanced joint analysis (joints.py)
    "knee_flex_contact": ("min", 0, 3, 6, 10, 25, "deg"),
    "head_level":       ("max", 0.30, 0.18, 0.12, 0.07, 0.02, "m"),
    "spine_steady":     ("max", 30, 20, 14, 10, 3, "deg"),
    "split_step":       ("min", 0.0, 0.01, 0.02, 0.03, 0.08, "m"),
    "coil":             ("min", -0.05, 0.0, 0.05, 0.12, 0.35, "ratio"),
    "left_arm_tuck":    ("max", 0.80, 0.60, 0.45, 0.35, 0.10, "m"),
    "kinetic_chain":    ("band", 0.25, 0.18, 0.12, 0.08, 0.02, "s from ideal", 0.12),
    "weight_transfer":  ("min", -0.10, -0.02, 0.04, 0.10, 0.30, "m"),
}
SERVE_LADDER = {
    "foot_fault":     ("max", 0.30, 0.10, 0.10, 0.10, -0.10, "m over line"),
    "contact_height": ("min", 1.00, 1.20, 1.35, 1.50, 1.80, "x height"),
    "toss_height":    ("band", 1.30, 0.90, 0.60, 0.45, 0.10, "m from ideal", 0.50),
    "toss_side":      ("band", 0.80, 0.55, 0.40, 0.30, 0.05, "m from ideal", 0.20),
    "knee_bend":      ("min", 5, 8, 12, 18, 50, "deg"),
    "leg_drive":      ("min", 0.00, 0.03, 0.05, 0.10, 0.35, "m"),
    "trophy_elbow":   ("min", -0.30, -0.15, -0.10, -0.05, 0.10, "m"),
    "shoulder_tilt":  ("min", -0.10, 0.00, 0.03, 0.05, 0.25, "m"),
    "toss_arm":       ("min", 110, 130, 140, 150, 175, "deg"),
    "toss_arm_up":    ("min", -0.40, -0.20, -0.10, 0.00, 0.30, "m"),
    "racquet_drop":   ("min", -0.10, 0.00, 0.05, 0.10, 0.40, "m"),
    "arm_extension":  ("min", 110, 130, 140, 150, 175, "deg"),
    "head_up":        ("max", 0.40, 0.20, 0.15, 0.10, -0.05, "m"),
    "drive_up":       ("min", -0.02, 0.00, 0.02, 0.05, 0.30, "m"),
    "landing":        ("min", -0.40, -0.20, 0.00, 0.10, 0.80, "m"),
    # advanced joint analysis (joints.py)
    "serve_chain":    ("band", 0.30, 0.22, 0.15, 0.10, 0.03, "s from ideal", 0.15),
    "trophy_lean":    ("min", 0, 3, 6, 10, 25, "deg"),
    "toss_arm_down":  ("min", -0.40, -0.20, 0.00, 0.10, 0.40, "m"),
    "body_line":      ("max", 40, 30, 22, 15, 5, "deg"),
    "contact_tilt":   ("min", -0.10, 0.00, 0.05, 0.10, 0.30, "m"),
}

# which checks each level judges, in priority order, with level-specific wording
# (key, tip or None = the Club rulebook's tip)
FOREHAND = {
    "beginner": [
        ("contact_in_front", "Hit the ball in front of you, not beside or behind you."),
        ("contact_too_high", "Let the ball come down to waist height before you hit it."),
        ("sideways", "Turn sideways before you swing - point your left shoulder at the ball."),
        ("follow_through", "Finish with the racquet over your left shoulder."),
        ("balance", "Stay on your feet after you hit - don't fall away."),
    ],
    "intermediate": [
        ("contact_too_high", None), ("bounce_too_far", None), ("sideways", None),
        ("contact_in_front", None),
        ("knee_bend", "Squat down as you get ready to hit, then push up through the ball."),
        ("foot_order", None), ("spacing", None), ("early_prep", None),
        ("split_step", "Do a small split step (both feet hop) just before the ball arrives."),
        ("low_to_high", None),
        ("weight_transfer", "Step into the ball - your weight should move forward as you hit."),
        ("head_level", None),
        ("follow_through", None),
    ],
    "club": [(r[0], None) for r in FH_RULES],
}
SERVE = {
    "beginner": [
        ("foot_fault", "Keep your front foot behind the baseline until you have hit the ball."),
        ("toss_side", "Toss the ball in front of you, a little to your racquet side."),
        ("toss_height", "Toss the ball a little higher than you can reach with the racquet."),
        ("contact_height", "Reach up and hit the ball high, with your arm up."),
        ("head_up", "Watch the ball until you hit it."),
    ],
    "intermediate": [
        ("foot_fault", None), ("contact_height", None), ("toss_height", None), ("toss_side", None),
        ("knee_bend", "Bend your knees in the trophy position, then push up into the ball."),
        ("trophy_elbow", None), ("toss_arm_up", None), ("leg_drive", None),
        ("body_line", "Reach straight up at contact - stand tall under the ball."),
        ("head_up", None), ("landing", None),
    ],
    "club": [(r[0], None) for r in RULES],
}
_RUNG = {"beginner": 2, "intermediate": 3, "club": 4}


def _ladder(mode: str) -> dict:
    return SERVE_LADDER if mode == "serve" else FH_LADDER


def _norm(spec, v: float) -> float:
    """The value on a 'higher is better' scale (sign flip / distance from ideal)."""
    d = spec[0]
    if d == "min":
        return v
    if d == "max":
        return -v
    return -abs(v - spec[7])


def _rung(spec, i: int) -> float:
    return spec[i] if spec[0] == "min" else -spec[i]


def meets(spec, v: float, level: str) -> bool:
    return _norm(spec, v) >= _rung(spec, _RUNG[level]) - 1e-9


def reached(spec, v: float) -> str:
    for lv in ("club", "intermediate", "beginner"):
        if meets(spec, v, lv):
            return lv
    return "below beginner"


def credit(spec, v: float, level: str) -> float:
    """0 at 'poor', PASS_CREDIT at the level's target, 1 at 'excellent'."""
    x, poor, tgt, exc = _norm(spec, v), _rung(spec, 1), _rung(spec, _RUNG[level]), _rung(spec, 5)
    if x >= tgt:
        return 1.0 if exc <= tgt else min(1.0, PASS_CREDIT + (1 - PASS_CREDIT) * (x - tgt) / (exc - tgt))
    if tgt <= poor:
        return 0.0
    return max(0.0, PASS_CREDIT * (x - poor) / (tgt - poor))


def ladder_text(spec) -> str:
    d, u = spec[0], spec[6]
    f = lambda z: f"{z:g}"
    if d == "band":
        return (f"within B {f(spec[2])} | I {f(spec[3])} | C {f(spec[4])} of {f(spec[7])}"
                f" ({u.replace(' from ideal', '')})")
    op = ">=" if d == "min" else "<="
    return f"B {op} {f(spec[2])} | I {op} {f(spec[3])} | C {op} {f(spec[4])} {u}"


def level_rules(mode: str, level: str) -> List[tuple]:
    """The level's checks in priority order: (key, category, title, measurement, tip)."""
    base = {r[0]: r for r in (RULES if mode == "serve" else FH_RULES)}
    out = []
    for key, tip in (SERVE if mode == "serve" else FOREHAND)[level]:
        k, cat, title, meas, _test, _target, d_tip = base[key]
        out.append((k, cat, title, meas, tip or d_tip))
    return out


def judge(mode: str, m: Dict[str, Optional[float]], level: str = "club") -> List[RuleResult]:
    lad = _ladder(mode)
    out = []
    for key, cat, title, meas, tip in level_rules(mode, level):
        spec = lad[key]
        r_ = spec[_RUNG[level]]
        if spec[0] == "band":
            target = f"{LEVEL_NAMES[level]}: within {r_:g} m of {spec[7]:g}"
        else:
            target = f"{LEVEL_NAMES[level]}: {'>=' if spec[0] == 'min' else '<='} {r_:g} {spec[6]}"
        v = m.get(meas)
        if v is None or v != v:          # None or NaN
            out.append(RuleResult(key, title, "unknown", None, meas, target,
                                  "Not measurable in this clip.", cat, "", None, ladder_text(spec)))
            continue
        tip = tip(m) if callable(tip) else tip
        ok = meets(spec, v, level)
        out.append(RuleResult(key, title, "good" if ok else "improve", v, meas, target, tip, cat,
                              reached(spec, v), round(credit(spec, v, level), 2), ladder_text(spec)))
    return out


def _weights(n: int) -> List[float]:
    return [2.0 - (i / (n - 1) if n > 1 else 0.0) for i in range(n)]


def score(rules: List[RuleResult]) -> Optional[int]:
    num = den = 0.0
    for w, r in zip(_weights(len(rules)), rules):
        if r.status == "unknown" or r.credit is None:
            continue
        den += w
        num += w * r.credit
    return int(round(100 * (num / den) ** STRICTNESS)) if den else None


def suggestions(rules: List[RuleResult]) -> List[dict]:
    return [{"key": r.key, "title": r.title, "tip": r.tip, "priority": k + 1, "category": r.category}
            for k, r in enumerate([r for r in rules if r.status == "improve"])]
