#!/usr/bin/env python3
"""One-time model download (the runtime itself never touches the network).

    python tools/download_models.py            # YOLOX-tiny + BlazePose full
    python tools/download_models.py --heavy    # also BlazePose heavy (better arms)
"""
import argparse
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    # Apache-2.0, Megvii YOLOX release
    "yolox_tiny.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
    # Apache-2.0, Google MediaPipe
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}
HEAVY = {"pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                       "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heavy", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    for name, url in {**MODELS, **(HEAVY if a.heavy else {})}.items():
        path = os.path.join(ROOT, "models", name)
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            print(f"ok   {name}")
            continue
        print(f"get  {name} ...")
        urllib.request.urlretrieve(url, path)
    print("models ready")
