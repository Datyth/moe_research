"""Learning-task contracts and built-in task implementations."""

from .base import Task, TaskStepOutput
from .joint_prior_posterior import JointPriorPosteriorTask
from .mask_reconstruction import MaskReconstructionTask
from .phase_b_a1_no_expert import PhaseBA1NoExpertTask
from .phase_b_build_up import PhaseBBuildUpTask
from .phase_b_fuse import PhaseBFuseTask
from .phase_b_moe import PhaseBMoETask
from .phase_b_router import PhaseBRouterTask
from .phase_c_distill import PhaseCDistillTask
from .segmentation import SegmentationTask

__all__ = [
    "Task",
    "TaskStepOutput",
    "JointPriorPosteriorTask",
    "MaskReconstructionTask",
    "PhaseBA1NoExpertTask",
    "PhaseBBuildUpTask",
    "PhaseBFuseTask",
    "PhaseBMoETask",
    "PhaseBRouterTask",
    "PhaseCDistillTask",
    "SegmentationTask",
]
