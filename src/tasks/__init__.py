"""Learning-task contracts and built-in task implementations."""

from .base import Task, TaskStepOutput
from .mask_reconstruction import MaskReconstructionTask
from .phase_b_a1_no_expert import PhaseBA1NoExpertTask
from .phase_b_fuse import PhaseBFuseTask
from .phase_b_moe import PhaseBMoETask
from .phase_b_router import PhaseBRouterTask
from .segmentation import SegmentationTask

__all__ = [
    "Task",
    "TaskStepOutput",
    "MaskReconstructionTask",
    "PhaseBA1NoExpertTask",
    "PhaseBFuseTask",
    "PhaseBMoETask",
    "PhaseBRouterTask",
    "SegmentationTask",
]
