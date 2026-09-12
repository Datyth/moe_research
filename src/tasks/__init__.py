"""Learning-task contracts and built-in task implementations."""

from .base import Task, TaskStepOutput
from .mask_reconstruction import MaskReconstructionTask
from .phase_b_fuse import PhaseBFuseTask
from .phase_b_moe import PhaseBMoETask
from .phase_b_router import PhaseBRouterTask
from .segmentation import SegmentationTask

__all__ = [
    "Task",
    "TaskStepOutput",
    "MaskReconstructionTask",
    "PhaseBFuseTask",
    "PhaseBMoETask",
    "PhaseBRouterTask",
    "SegmentationTask",
]
