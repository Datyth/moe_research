"""Phase C: posterior-to-prior distillation for deployable routing."""

from .adaptive_student import PhaseCAdaptiveStudent
from .c5_trainable_student_routing import PhaseCC5TrainableStudentRouting
from .b6_prior_distill import (
    PhaseCB6PriorDistill,
    PhaseCDistillationState,
    PhaseCSegmentationOutput,
)
from .metrics import (
    categorical_routing_kl,
    gamma_distance,
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)

__all__ = [
    "PhaseCAdaptiveStudent",
    "PhaseCC5TrainableStudentRouting",
    "PhaseCB6PriorDistill",
    "PhaseCDistillationState",
    "PhaseCSegmentationOutput",
    "categorical_routing_kl",
    "gamma_distance",
    "latent_transfer_metrics",
    "routing_js",
    "topk_transfer_metrics",
]
