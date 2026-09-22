"""Learning objective and evaluation semantics for Phase-C distillation."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from src.models.phase_b.posterior import gaussian_kl
from src.models.phase_c.b6_prior_distill import PhaseCDistillationState
from src.models.phase_c.metrics import (
    categorical_routing_kl,
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)

from .base import TaskStepOutput
from .segmentation import SegmentationTask


class PhaseCDistillTask(SegmentationTask):
    """Fit only p(z|I) to the fixed B6 posterior and routing policy."""

    evaluation_mode = "image_only"
    teacher_evaluation_mode = "posterior_oracle"

    def __init__(
        self,
        *,
        criterion: nn.Module,
        lambda_latent: float,
        lambda_route: float,
        lambda_deploy: float,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
        task: str = "binary",
    ) -> None:
        super().__init__(
            criterion=criterion,
            threshold=threshold,
            boundary_tolerance=boundary_tolerance,
            task=task,
        )
        if task != "binary":
            raise ValueError("Phase-C B6 distillation supports binary masks only.")
        for name, value in (
            ("lambda_latent", lambda_latent),
            ("lambda_route", lambda_route),
            ("lambda_deploy", lambda_deploy),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be non-negative.")
        self.lambda_latent = float(lambda_latent)
        self.lambda_route = float(lambda_route)
        self.lambda_deploy = float(lambda_deploy)

    @staticmethod
    def _distillation_forward(
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        *,
        decode_prior: bool,
        decode_posterior: bool,
    ) -> PhaseCDistillationState:
        forward = getattr(model, "distillation_forward", None)
        if not callable(forward):
            raise TypeError(
                "PhaseCDistillTask requires model.distillation_forward()."
            )
        state = forward(
            images,
            targets,
            decode_prior=decode_prior,
            decode_posterior=decode_posterior,
        )
        if not isinstance(state, PhaseCDistillationState):
            raise TypeError(
                "distillation_forward must return PhaseCDistillationState."
            )
        return state

    def _loss_components(
        self,
        state: PhaseCDistillationState,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        latent_kl = gaussian_kl(state.posterior.detach(), state.prior).mean()
        route_kl = categorical_routing_kl(
            state.posterior_routing.dense_probs,
            state.prior_routing.logits,
        )
        total = self.lambda_latent * latent_kl + self.lambda_route * route_kl
        raw = {
            "latent_kl": latent_kl,
            "route_kl": route_kl,
        }
        if self.lambda_deploy > 0.0:
            if state.prior_logits is None:
                raise ValueError(
                    "lambda_deploy > 0 requires decoded prior segmentation logits."
                )
            deploy_loss = self.criterion(state.prior_logits, targets)
            total = total + self.lambda_deploy * deploy_loss
            raw["deploy_seg_loss"] = deploy_loss
        raw["total_loss"] = total.detach()
        return total, raw

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        state = self._distillation_forward(
            model,
            images,
            targets,
            decode_prior=self.lambda_deploy > 0.0,
            decode_posterior=False,
        )
        total, metrics = self._loss_components(state, targets)
        return TaskStepOutput(
            loss=total,
            metrics=metrics,
            batch_size=images.shape[0],
        )

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        state = self._distillation_forward(
            model,
            images,
            targets,
            decode_prior=True,
            decode_posterior=True,
        )
        if state.prior_logits is None or state.posterior_logits is None:
            raise ValueError("Phase-C evaluation requires both decoded paths.")
        total, metrics = self._loss_components(state, targets)

        metrics.update(self._segmentation_metrics(state.prior_logits, targets))
        posterior_metrics = self._segmentation_metrics(
            state.posterior_logits,
            targets,
        )
        metrics.update(
            {
                f"posterior_{name}": value
                for name, value in posterior_metrics.items()
            }
        )
        metrics.update(latent_transfer_metrics(state.posterior, state.prior))
        metrics.update(
            topk_transfer_metrics(
                state.posterior_routing.expert_indices,
                state.prior_routing.expert_indices,
                num_experts=state.posterior_routing.dense_probs.shape[1],
            )
        )
        metrics["routing_js"] = routing_js(
            state.posterior_routing.dense_probs,
            state.prior_routing.dense_probs,
        )
        return TaskStepOutput(
            loss=total,
            metrics=metrics,
            batch_size=images.shape[0],
        )

    def finalize_evaluation_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        finalized = dict(metrics)
        finalized["transfer_gap_dice"] = (
            finalized["posterior_dice"] - finalized["dice"]
        )
        return finalized


__all__ = ["PhaseCDistillTask"]
