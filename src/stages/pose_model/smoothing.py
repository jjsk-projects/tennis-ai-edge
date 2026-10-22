"""Stage 2b - Skeleton smoothing (adapted from the original PoseTracker).

The original ``PoseTracker`` blended each new pose into a running estimate,
weighted by keypoint confidence, so low-confidence (blurred / occluded)
keypoints barely moved it. We keep that idea but, because the whole clip
is available, use a *centred* window: no time lag at the contact moment,
which a forward-only running average would introduce.

Window: +/-1 frame at 30 fps, +/-2 at 60 fps, +/-4 at 120 fps (~33 ms each
side) - long enough to remove jitter, short enough to keep the swing.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def smooth_poses(poses: Dict[int, np.ndarray], fps: float,
                 half_window_s: float = 0.034) -> Dict[int, np.ndarray]:
    """Confidence-weighted centred average of each keypoint over time."""
    k = max(1, int(round(half_window_s * fps)))
    out: Dict[int, np.ndarray] = {}
    for t, p in poses.items():
        stack = [poses[t + d] for d in range(-k, k + 1) if (t + d) in poses]
        arr = np.stack(stack)                                # (n, 33, 3)
        w = np.clip(arr[:, :, 2], 1e-3, None)                # confidence weights
        q = p.copy()
        q[:, 0] = (arr[:, :, 0] * w).sum(0) / w.sum(0)
        q[:, 1] = (arr[:, :, 1] * w).sum(0) / w.sum(0)
        q[:, 2] = arr[:, :, 2].mean(0)
        out[t] = q
    return out


def fill_gaps(poses: Dict[int, np.ndarray], max_gap: int) -> Dict[int, np.ndarray]:
    """Linear interpolation for frames where the pose model was skipped
    (at 120 fps it runs on every 2nd frame) or briefly failed."""
    out = dict(poses)
    ts = sorted(poses)
    for a, b in zip(ts, ts[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (1 - w) * poses[a] + w * poses[b]
    return out
