"""Stage 2: skeleton model (BlazePose on a zoomed player crop)."""
from .pose_model_backend import PoseModelBackend, to_coco17, SKELETON_EDGES

__all__ = ["PoseModelBackend", "to_coco17", "SKELETON_EDGES"]
