"""Phase C: posterior-to-prior distillation for deployable routing."""

from .b6_prior_distill import (
    PhaseCB6PriorDistill,
    PhaseCDistillationState,
    PhaseCSegmentationOutput,
)
from .metrics import (
    categorical_routing_kl,
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)

__all__ = [
    "PhaseCB6PriorDistill",
    "PhaseCDistillationState",
    "PhaseCSegmentationOutput",
    "categorical_routing_kl",
    "latent_transfer_metrics",
    "routing_js",
    "topk_transfer_metrics",
]
