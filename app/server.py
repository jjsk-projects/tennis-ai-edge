#!/usr/bin/env python3
"""Demo web server - Python standard library only (no Flask/FastAPI needed).

    python app/server.py            # then open http://127.0.0.1:8080

Flow: upload a video (or pick an example clip) -> optional: click the
baseline on the first frame -> start -> poll progress -> show results.
Analysis jobs run one at a time in a background thread.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from src.pipeline import MODES, analyse_video, first_frame  # noqa: E402
from src.stages.serve_analysis.levels import LEVELS  # noqa: E402

STATIC = os.path.join(ROOT, "app", "static")
RUNS = os.path.join(ROOT, "runs")
CLIPS = os.path.join(ROOT, "clips")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
MAX_UPLOAD = 300 * 1024 * 1024

JOBS: dict = {}
RUN_LOCK = threading.Lock()   # one analysis at a time (CPU bound)


def _new_job(src_path: str, name: str) -> dict:
    jid = uuid.uuid4().hex[:10]
    d = os.path.join(RUNS, jid)
    os.makedirs(d, exist_ok=True)
    frame = first_frame(src_path)   # same scaling as the pipeline -> clicks match
    ok = frame is not None
    w, h = (frame.shape[1], frame.shape[0]) if ok else (0, 0)
    if ok:
        cv2.imwrite(os.path.join(d, "first_frame.jpg"), frame)
    job = {"id": jid, "name": name, "video": src_path, "status": "uploaded",
           "stage": "Ready to analyse", "progress": 0.0, "size": [w, h],
           "first_frame": f"/runs/{jid}/first_frame.jpg" if ok else None}
    JOBS[jid] = job
    return job


def _run(job: dict, height: float, camera: str, baseline, mode: str, level: str = "club"):
    def prog(stage, frac):
        job["stage"], job["progress"] = stage, round(frac, 3)

    job["status"], job["stage"] = "queued", "Waiting for a free worker"
    with RUN_LOCK:
        job["status"] = "running"
        try:
            res = analyse_video(job["video"], os.path.join(RUNS, job["id"]),
                                player_height_m=height, camera_side=camera,
                                baseline_override=baseline, mode=mode, progress=prog,
                                level=level)
            job["result"], job["status"] = res, "done"
        except Exception as e:  # report, don't crash the server
            traceback.print_exc()
            job["status"], job["error"] = "error", f"{e.__class__.__name__}: {e}"


class Handler(BaseHTTPRequestHandler):
    server_version = "ServeLab/0.1"

    def log_message(self, fmt, *args):
        if "/api/jobs/" not in (args[0] if args else ""):
            sys.stderr.write("%s\n" % (fmt % args))

    # -- helpers --------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        real = os.path.realpath(path)
        if not (real.startswith(STATIC) or real.startswith(RUNS)) or not os.path.isfile(real):
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(real)[0] or "application/octet-stream"
        size = os.path.getsize(real)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(real, "rb") as fh:
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- routes ---------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if p.startswith("/static/"):
            return self._file(os.path.join(STATIC, p[len("/static/"):]))
        if p.startswith("/runs/"):
            return self._file(os.path.join(RUNS, p[len("/runs/"):]))
        if p == "/api/examples":
            names = sorted(f for f in os.listdir(CLIPS) if f.lower().endswith(VIDEO_EXT)) \
                if os.path.isdir(CLIPS) else []
            return self._json({"examples": names})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)", p)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._json({"error": "unknown job"}, 404)
            return self._json({k: v for k, v in job.items() if k != "video"})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        if p == "/api/upload":
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_UPLOAD:
                return self._json({"error": "empty or too large upload"}, 400)
            name = os.path.basename(q.get("name", "upload.mp4"))
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXT:
                return self._json({"error": f"unsupported file type {ext}"}, 400)
            os.makedirs(os.path.join(RUNS, "_uploads"), exist_ok=True)
            dst = os.path.join(RUNS, "_uploads", uuid.uuid4().hex[:8] + ext)
            remaining = n
            with open(dst, "wb") as fh:
                while remaining:
                    chunk = self.rfile.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            job = _new_job(dst, name)
            return self._json({k: v for k, v in job.items() if k != "video"})
        if p == "/api/example":
            name = os.path.basename(q.get("name", ""))
            src = os.path.join(CLIPS, name)
            if not os.path.isfile(src):
                return self._json({"error": "unknown example"}, 404)
            job = _new_job(src, name)
            return self._json({k: v for k, v in job.items() if k != "video"})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/start", p)
        if m:
            job = JOBS.get(m.group(1))
            if not job:
                return self._json({"error": "unknown job"}, 404)
            if job["status"] in ("queued", "running"):
                return self._json({"error": "already running"}, 409)
            try:
                height = float(q.get("height", 1.8))
                camera = q.get("camera", "far")
                mode = q.get("mode", "serve")
                if mode not in MODES:
                    raise ValueError(f"mode must be one of {MODES}")
                level = q.get("level", "club")
                if level not in LEVELS:
                    raise ValueError(f"level must be one of {LEVELS}")
                baseline = [float(v) for v in q["baseline"].split(",")] if q.get("baseline") else None
                if baseline is not None and len(baseline) != 4:
                    raise ValueError("baseline needs x1,y1,x2,y2")
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            job.pop("result", None); job.pop("error", None)
            threading.Thread(target=_run, args=(job, height, camera, baseline, mode, level), daemon=True).start()
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    os.makedirs(RUNS, exist_ok=True)
    os.chdir(ROOT)
    print(f"Serve Lab demo on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
