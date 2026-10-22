"""Stage 2 - pose model: COCO-17 conversion and crop geometry (no model needed)."""
import os

import numpy as np
import pytest

from src.stages.pose_model.pose_model_backend import (
    BLAZEPOSE_TO_COCO, crop_window, to_coco17)


def test_coco17_uses_standard_layout():
    pose33 = np.zeros((33, 3))
    pose33[15] = (1, 2, 0.9)   # BlazePose left wrist
    pose33[16] = (3, 4, 0.9)   # BlazePose right wrist
    pose33[27] = (5, 6, 0.9)   # BlazePose left ankle
    coco = to_coco17(pose33)
    assert coco.shape == (17, 3)
    np.testing.assert_allclose(coco[9], (1, 2, 0.9))    # COCO 9 = left wrist
    np.testing.assert_allclose(coco[10], (3, 4, 0.9))   # COCO 10 = right wrist
    np.testing.assert_allclose(coco[15], (5, 6, 0.9))   # COCO 15 = left ANKLE


def test_mapping_is_one_to_one():
    assert sorted(BLAZEPOSE_TO_COCO.values()) == list(range(17))


def test_crop_window_inside_frame_and_above_player():
    x0, y0, x1, y1 = crop_window((100, 400, 140, 500), 480, 854)
    assert 0 <= x0 < x1 <= 480 and 0 <= y0 < y1 <= 854
    assert y0 < 400          # head-room for the raised arm and racket
    assert x0 < 100 and x1 > 140


@pytest.mark.skipif(not os.path.exists("models/pose_landmarker.task"), reason="model not downloaded")
def test_backend_detects_sample_person():
    pytest.importorskip("mediapipe")
    import cv2
    from src.stages.pose_model.pose_model_backend import PoseModelBackend
    img = cv2.imread("models/pose_sample.jpg")
    if img is None:
        pytest.skip("no sample image")
    h, w = img.shape[:2]
    with PoseModelBackend() as pm:
        pose = pm.detect(img, (0.25 * w, 0.1 * h, 0.75 * w, 0.95 * h), 0)
    assert pose is not None and pose.shape == (33, 3)
