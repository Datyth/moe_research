"""Posterior-to-prior transfer metrics for Phase C."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from src.models.phase_b.posterior import DiagonalGaussian


def _validate_probabilities(
    probabilities: Tensor,
    *,
    name: str,
) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"{name} must be [B, K], got {tuple(probabilities.shape)}.")
    if probabilities.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one category.")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError(f"{name} must be finite.")
    if bool((probabilities < 0).any()):
        raise ValueError(f"{name} must be non-negative.")
    totals = probabilities.sum(dim=1)
    if not torch.allclose(
        totals,
        torch.ones_like(totals),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(f"{name} must sum to one.")


def categorical_routing_kl(
    teacher_dense_probs: Tensor,
    student_logits: Tensor,
) -> Tensor:
    """Return batch-mean ``KL(stop_gradient[teacher] || student)``."""

    _validate_probabilities(teacher_dense_probs, name="teacher_dense_probs")
    if student_logits.shape != teacher_dense_probs.shape:
        raise ValueError(
            "student_logits must match teacher_dense_probs, got "
            f"{tuple(student_logits.shape)} and "
            f"{tuple(teacher_dense_probs.shape)}."
        )
    if not torch.isfinite(student_logits).all():
        raise FloatingPointError("student_logits must be finite.")
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        teacher_dense_probs.detach(),
        reduction="batchmean",
    )


def routing_js(
    first_dense_probs: Tensor,
    second_dense_probs: Tensor,
) -> Tensor:
    """Jensen-Shannon divergence between two dense routing distributions."""

    _validate_probabilities(first_dense_probs, name="first_dense_probs")
    _validate_probabilities(second_dense_probs, name="second_dense_probs")
    if first_dense_probs.shape != second_dense_probs.shape:
        raise ValueError(
            "Routing distributions must share shape, got "
            f"{tuple(first_dense_probs.shape)} and "
            f"{tuple(second_dense_probs.shape)}."
        )
    first = first_dense_probs.detach()
    second = second_dense_probs.detach()
    mixture = 0.5 * (first + second)
    tiny = torch.finfo(mixture.dtype).tiny

    def kl(probabilities: Tensor, target: Tensor) -> Tensor:
        return (
            probabilities
            * (
                probabilities.clamp_min(tiny).log()
                - target.clamp_min(tiny).log()
            )
        ).sum(dim=1)

    return (0.5 * (kl(first, mixture) + kl(second, mixture))).mean()


def topk_transfer_metrics(
    teacher_indices: Tensor,
    student_indices: Tensor,
    *,
    num_experts: int,
) -> dict[str, Tensor]:
    """Order-insensitive exact agreement and overlap over active expert sets."""

    if teacher_indices.ndim != 2 or student_indices.ndim != 2:
        raise ValueError("Top-K indices must both be [B, k_e].")
    if teacher_indices.shape != student_indices.shape:
        raise ValueError(
            "Teacher/student Top-K indices must share shape, got "
            f"{tuple(teacher_indices.shape)} and {tuple(student_indices.shape)}."
        )
    if teacher_indices.dtype != torch.long or student_indices.dtype != torch.long:
        raise ValueError("Top-K indices must use torch.long dtype.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if teacher_indices.shape[1] == 0:
        raise ValueError("Top-K indices must contain at least one active expert.")
    for name, indices in (
        ("teacher_indices", teacher_indices),
        ("student_indices", student_indices),
    ):
        if bool(((indices < 0) | (indices >= num_experts)).any()):
            raise ValueError(f"{name} contains an out-of-range expert.")

    teacher_set = F.one_hot(
        teacher_indices,
        num_classes=num_experts,
    ).amax(dim=1).bool()
    student_set = F.one_hot(
        student_indices,
        num_classes=num_experts,
    ).amax(dim=1).bool()
    dtype = torch.float32
    exact = (teacher_set == student_set).all(dim=1).to(dtype)
    overlap = (teacher_set & student_set).sum(dim=1).to(dtype)
    overlap = overlap / teacher_indices.shape[1]
    return {
        "exact_topk_set_agreement": exact.mean(),
        "topk_overlap": overlap.mean(),
    }


def latent_transfer_metrics(
    posterior: DiagonalGaussian,
    prior: DiagonalGaussian,
) -> dict[str, Tensor]:
    """Distances between posterior and prior diagonal-Gaussian parameters."""

    if posterior.mean.shape != prior.mean.shape:
        raise ValueError("Posterior/prior means must share shape.")
    if posterior.std.shape != prior.std.shape:
        raise ValueError("Posterior/prior standard deviations must share shape.")
    mean_difference = posterior.mean.detach() - prior.mean.detach()
    std_difference = posterior.std.detach() - prior.std.detach()
    return {
        "mean_distance": mean_difference.norm(dim=1).mean(),
        "std_distance": std_difference.norm(dim=1).mean(),
        "wasserstein2_squared": (
            mean_difference.pow(2).sum(dim=1)
            + std_difference.pow(2).sum(dim=1)
        ).mean(),
    }


__all__ = [
    "categorical_routing_kl",
    "latent_transfer_metrics",
    "routing_js",
    "topk_transfer_metrics",
]
