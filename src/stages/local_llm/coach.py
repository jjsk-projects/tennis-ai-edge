"""Stage 4 - Coach: turns measurements + suggestions into spoken-style feedback.

Backends (pick in config.yaml -> coach.backend, or env COACH_BACKEND):
- ``template``  (default) built-in wording, no model, works offline.
- ``openai``    any OpenAI-compatible server, e.g. your local oMLX
                (http://127.0.0.1:8000/v1) or llama.cpp / vLLM on the edge box.
- ``anthropic`` Claude via the Anthropic Messages API (ANTHROPIC_API_KEY).

All HTTP is plain ``urllib`` - no SDK dependency. If an LLM call fails we
fall back to the template so the demo never breaks.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, List

SYSTEM = (
    "You are an encouraging professional tennis coach giving on-court feedback on a "
    "right-handed player's {stroke}. You receive measurements extracted by computer vision "
    "and the result of every technique check: good, improve (with a tip, in priority order) "
    "or unknown. Cover ALL of them: first one short line listing what is good, then EVERY "
    "'improve' check as a numbered fix in the given priority order, each with a simple cue "
    "the player can feel on the next {stroke_l}, e.g. 'squat a bit lower to brush up the "
    "back of the ball for topspin', then one line naming what could not be measured. "
    "Plain text, no markdown headings, no extra advice beyond the checks. {level_style}"
)

LEVEL_STYLE = {
    "beginner": "The player is a BEGINNER: use short, everyday words, no tennis jargon, "
                "one simple picture per fix (e.g. 'hit the ball in front of you, not behind').",
    "intermediate": "The player is INTERMEDIATE: name the technique (turn, squat, footwork) "
                    "and give one feel cue per fix.",
    "club": "The player is a CLUB-LEVEL player: be specific and technical, mention the "
            "measured value against its target for each fix.",
}


def _prompt(report: Dict) -> str:
    return json.dumps({
        "measurements": report["measurements"],
        "rules": [{k: r[k] for k in ("title", "status", "value", "target")} for r in report["rules"]],
        "suggestions": report["suggestions"],
        "level": report.get("level", "club"),
        "score_0_100": report.get("score"),
    }, indent=1)


def template_feedback(report: Dict) -> str:
    """Every check of the player's level, in priority order: what is good, all fixes,
    what was not measurable. Beginners get a shorter, friendlier note."""
    mode = report.get("mode", "serve")
    level = report.get("level", "club")
    good = [r["title"] for r in report["rules"] if r["status"] == "good"]
    fixes = sorted(report["suggestions"], key=lambda s: s.get("priority", 99))
    unknown = [r["title"] for r in report["rules"] if r["status"] == "unknown"]
    sc = report.get("score")
    lines = []
    if level == "beginner":
        if sc is not None:
            lines.append(f"Score: {sc}/100 (beginner).")
        if good:
            lines.append("Well done: " + "; ".join(t.lower() for t in good) + ".")
        if fixes:
            lines.append("Try this next time:")
            lines += [f"{i}. {s['tip']}" for i, s in enumerate(fixes, 1)]
        else:
            lines.append("Great hit - keep doing exactly that!")
        if unknown:
            lines.append("We couldn't see: " + "; ".join(t.lower() for t in unknown) +
                         " (stand the phone a little closer).")
        return "\n".join(lines)
    name = "intermediate" if level == "intermediate" else "club level"
    if sc is not None:
        lines.append(f"Score: {sc}/100 ({name}).")
    if good:
        lines.append(f"Good ({len(good)}): " + "; ".join(good) + ".")
    if fixes:
        lines.append(f"To improve on your next {mode}s ({len(fixes)}, most important first):")
        for i, s in enumerate(fixes, 1):
            cat = f"[{s['category']}] " if s.get("category") else ""
            extra = ""
            r = next((r for r in report["rules"] if r["key"] == s["key"]), None)
            if level == "club" and r and r.get("value") is not None:
                extra = f" (measured {r['value']}; {r.get('ladder') or r['target']})"
            elif r and r.get("reached"):
                extra = f" (now at {r['reached']} standard)"
            lines.append(f"{i}. {cat}{s['tip']}{extra}")
    else:
        lines.append("Every measured checkpoint is in range - keep grooving this motion.")
    # the score is strict: name the passed checks with the most room left
    room = sorted([r for r in report["rules"] if r["status"] == "good" and r.get("credit") is not None
                   and r["credit"] < 0.7], key=lambda r: r["credit"])[:3]
    if room:
        lines.append("To push the score up: " + "; ".join(
            f"{r['title'].lower()} (passed, {int(round(100 * r['credit']))}% of full marks)" for r in room) + ".")
    if unknown:
        lines.append(f"Could not measure ({len(unknown)}): " + "; ".join(unknown) +
                     " (film a little closer, steady, with the ball in view).")
    return "\n".join(lines)


def _post(url: str, body: Dict, headers: Dict, timeout: float) -> Dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _system(cfg: Dict) -> str:
    st = cfg.get("stroke", "serve")
    return SYSTEM.format(stroke=st.upper(), stroke_l=st,
                         level_style=LEVEL_STYLE.get(cfg.get("level", "club"), ""))


def llm_feedback(report: Dict, cfg: Dict) -> str:
    backend = os.environ.get("COACH_BACKEND", cfg.get("backend", "template"))
    if cfg.get("force_template"):
        backend = "template"
    try:
        if backend == "openai":
            base = os.environ.get("COACH_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8000/v1"))
            key = os.environ.get("COACH_API_KEY", cfg.get("api_key", ""))
            d = _post(base.rstrip("/") + "/chat/completions", {
                "model": os.environ.get("COACH_MODEL", cfg.get("model", "")),
                "messages": [{"role": "system", "content": _system(cfg)},
                             {"role": "user", "content": _prompt(report)}],
                "temperature": 0.3, "max_tokens": 900,
            }, {"Authorization": f"Bearer {key}"} if key else {}, cfg.get("timeout_s", 60))
            return d["choices"][0]["message"]["content"].strip()
        if backend == "anthropic":
            d = _post("https://api.anthropic.com/v1/messages", {
                "model": os.environ.get("COACH_MODEL", cfg.get("model", "claude-sonnet-5-5")),
                "max_tokens": 900, "system": _system(cfg),
                "messages": [{"role": "user", "content": _prompt(report)}],
            }, {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01"}, cfg.get("timeout_s", 60))
            return "".join(b.get("text", "") for b in d["content"]).strip()
    except Exception as e:  # never break the demo because of the LLM
        return template_feedback(report) + f"\n(LLM '{backend}' unavailable: {e.__class__.__name__})"
    return template_feedback(report)
