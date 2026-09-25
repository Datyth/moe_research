"""Learning objective and evaluation semantics for Phase-C distillation."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.phase_b.posterior import gaussian_kl
from src.models.phase_c.b6_prior_distill import PhaseCDistillationState
from src.models.phase_c.metrics import (
    categorical_routing_kl,
    gamma_distance,
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
        latent_objective: str = "gaussian_kl",
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
        if (
            not isinstance(latent_objective, str)
            or latent_objective not in {"gaussian_kl", "mean_mse", "none"}
        ):
            raise ValueError(
                "latent_objective must be one of: gaussian_kl, mean_mse, none."
            )
        self.lambda_latent = float(lambda_latent)
        self.lambda_route = float(lambda_route)
        self.lambda_deploy = float(lambda_deploy)
        self.latent_objective = latent_objective
        if self.latent_objective == "none" and self.lambda_latent != 0.0:
            raise ValueError(
                "lambda_latent must be 0 when latent_objective is 'none'."
            )
        if (
            self.lambda_latent == 0.0
            and self.lambda_route == 0.0
            and self.lambda_deploy == 0.0
        ):
            raise ValueError("Phase-C objective must enable at least one loss term.")

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
        zero = state.prior.mean.detach().new_zeros(())
        weighted_latent = zero
        weighted_route = zero
        weighted_deploy = zero
        metrics: dict[str, Tensor] = {}

        if self.lambda_latent > 0.0:
            if self.latent_objective == "gaussian_kl":
                latent_loss = gaussian_kl(
                    state.posterior.detach(),
                    state.prior,
                ).mean()
                metrics["gaussian_kl"] = latent_loss
                # Backward-compatible alias: this key has always meant the
                # raw full-Gaussian KL in Phase C.
                metrics["latent_kl"] = latent_loss
            elif self.latent_objective == "mean_mse":
                latent_loss = F.mse_loss(
                    state.prior.mean,
                    state.posterior.mean.detach(),
                    reduction="mean",
                )
                metrics["mean_mse"] = latent_loss
            else:  # guarded by constructor validation
                raise RuntimeError(
                    "An active latent loss requires a concrete latent_objective."
                )
            weighted_latent = self.lambda_latent * latent_loss

        if self.lambda_route > 0.0:
            route_kl = categorical_routing_kl(
                state.posterior_routing.dense_probs,
                state.prior_routing.logits,
            )
            metrics["route_kl"] = route_kl
            weighted_route = self.lambda_route * route_kl

        if self.lambda_deploy > 0.0:
            if state.prior_logits is None:
                raise ValueError(
                    "lambda_deploy > 0 requires decoded prior segmentation logits."
                )
            deploy_loss = self.criterion(state.prior_logits, targets)
            metrics["deploy_seg_loss"] = deploy_loss
            weighted_deploy = self.lambda_deploy * deploy_loss

        total = weighted_latent + weighted_route + weighted_deploy
        metrics.update(
            {
                "weighted_latent_loss": weighted_latent.detach(),
                "weighted_route_loss": weighted_route.detach(),
                "weighted_deploy_seg_loss": weighted_deploy.detach(),
                "total_loss": total.detach(),
            }
        )
        return total, metrics

    def _prior_only_loss_components(
        self,
        prior_logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute a deployment loss without constructing the teacher branch."""

        if self.lambda_deploy <= 0.0:
            raise RuntimeError(
                "Prior-only training requires a positive lambda_deploy."
            )
        deploy_loss = self.criterion(prior_logits, targets)
        weighted_deploy = self.lambda_deploy * deploy_loss
        zero = deploy_loss.detach().new_zeros(())
        return weighted_deploy, {
            "deploy_seg_loss": deploy_loss,
            "weighted_latent_loss": zero,
            "weighted_route_loss": zero,
            "weighted_deploy_seg_loss": weighted_deploy.detach(),
            "total_loss": weighted_deploy.detach(),
        }

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        teacher_required = self.lambda_latent > 0.0 or self.lambda_route > 0.0
        if teacher_required:
            state = self._distillation_forward(
                model,
                images,
                targets,
                decode_prior=self.lambda_deploy > 0.0,
                decode_posterior=False,
            )
            total, metrics = self._loss_components(state, targets)
        else:
            # The mask is deliberately not passed to the model: for C4 it is
            # only the segmentation target, never privileged model input.
            prior_output = model(images)
            prior_logits = self._extract_logits(prior_output)
            total, metrics = self._prior_only_loss_components(
                prior_logits,
                targets,
            )
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

        # Preserve the original Phase-C evaluation diagnostics even when a
        # component is disabled for optimization. These values are detached
        # and never participate in the training objective.
        metrics["latent_kl"] = gaussian_kl(
            state.posterior.detach(),
            state.prior,
        ).mean().detach()
        metrics["route_kl"] = categorical_routing_kl(
            state.posterior_routing.dense_probs,
            state.prior_routing.logits,
        ).detach()

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
        if state.posterior_gamma is None or state.prior_gamma is None:
            raise ValueError(
                "Phase-C evaluation requires decoded posterior/prior gamma."
            )
        metrics["gamma_distance"] = gamma_distance(
            state.posterior_gamma,
            state.prior_gamma,
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
