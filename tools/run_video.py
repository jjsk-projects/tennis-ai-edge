#!/usr/bin/env python3
"""Analyse a serve video from the command line.

    python tools/run_video.py clips/IMG_3025.mov --height 1.80 --out runs/cli
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import analyse_video  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="output folder (default runs/<video name>)")
    ap.add_argument("--height", type=float, default=1.80, help="player height in metres")
    ap.add_argument("--mode", choices=["serve", "forehand"], default="serve")
    ap.add_argument("--level", choices=["beginner", "intermediate", "club"], default="club",
                    help="player level: which checks, how strict, how the coach talks")
    ap.add_argument("--camera-side", choices=["far", "near"], default="far",
                    help="far = filmed from across the net, near = from behind the server")
    a = ap.parse_args()
    out = a.out or os.path.join("runs", os.path.splitext(os.path.basename(a.video))[0])
    res = analyse_video(a.video, out, player_height_m=a.height, camera_side=a.camera_side, mode=a.mode, level=a.level,
                        progress=lambda s, f: print(f"\r[{f:5.0%}] {s:<45}", end="", flush=True))
    print()
    for s in res["strokes"]:
        print(f"\n{a.mode.title()} {s['index']}  contact frame {s['frames']['contact']:.1f} ({s['contact_method']})"
              f"  score {s['score']}/100 ({a.level})")
        for r in s["rules"]:
            print(f"  {r['status']:8s} {r['title']:28s} {r['value']}  target {r['target']}")
        print("  Coach:", s["coach_text"].replace("\n", "\n         "))
    if res.get("other_side_skipped"):
        print(f"\n{res['other_side_skipped']} backhand(s) recognised and skipped")
    print(f"\n{res['n_strokes']} {a.mode}(s); results in {out}/result.json ({res['elapsed_s']} s)")
