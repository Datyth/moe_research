"""Task semantics for joint privileged-posterior and image-prior training."""

from __future__ import annotations

import math
from typing import Any

from torch import Tensor, nn

from src.models.joint_prior_posterior import JointPriorPosteriorState
from src.models.phase_b.posterior import gaussian_kl
from src.models.phase_c.metrics import (
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)

from .base import TaskStepOutput
from .segmentation import SegmentationTask


class JointPriorPosteriorTask(SegmentationTask):
    """Optimize posterior segmentation and a bidirectional latent-space KL."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        lambda_balance: float,
        kl_beta_max: float,
        kl_zero_until_epoch: int,
        kl_ramp_end_epoch: int,
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
            raise ValueError("Joint prior-posterior training supports binary masks only.")
        for name, value in (
            ("lambda_balance", lambda_balance),
            ("kl_beta_max", kl_beta_max),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number.")
        for name, value in (
            ("kl_zero_until_epoch", kl_zero_until_epoch),
            ("kl_ramp_end_epoch", kl_ramp_end_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if kl_zero_until_epoch >= kl_ramp_end_epoch:
            raise ValueError(
                "kl_zero_until_epoch must be less than kl_ramp_end_epoch."
            )
        self.lambda_balance = float(lambda_balance)
        self.kl_beta_max = float(kl_beta_max)
        self.kl_zero_until_epoch = int(kl_zero_until_epoch)
        self.kl_ramp_end_epoch = int(kl_ramp_end_epoch)
        self.current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer.")
        self.current_epoch = epoch

    @property
    def kl_beta(self) -> float:
        if self.current_epoch <= self.kl_zero_until_epoch:
            return 0.0
        if self.current_epoch >= self.kl_ramp_end_epoch:
            return self.kl_beta_max
        progress = (
            (self.current_epoch - self.kl_zero_until_epoch)
            / (self.kl_ramp_end_epoch - self.kl_zero_until_epoch)
        )
        return self.kl_beta_max * progress

    @staticmethod
    def _joint_forward(
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        *,
        decode_posterior: bool,
        decode_prior: bool,
    ) -> JointPriorPosteriorState:
        forward = getattr(model, "joint_forward", None)
        if not callable(forward):
            raise TypeError(
                "JointPriorPosteriorTask requires model.joint_forward()."
            )
        state = forward(
            images,
            targets,
            decode_posterior=decode_posterior,
            decode_prior=decode_prior,
        )
        if not isinstance(state, JointPriorPosteriorState):
            raise TypeError(
                "joint_forward must return JointPriorPosteriorState."
            )
        return state

    def _loss_components(
        self,
        state: JointPriorPosteriorState,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if state.posterior_logits is None:
            raise ValueError("Joint loss requires decoded posterior logits.")
        segmentation = self.criterion(state.posterior_logits, targets)
        latent_kl_sum = gaussian_kl(state.posterior, state.prior).mean()
        latent_dim = state.posterior.mean.shape[1]
        if latent_dim <= 0:
            raise ValueError("Joint latent dimension must be positive.")
        latent_kl_per_dim = latent_kl_sum / latent_dim
        total = (
            segmentation
            + self.kl_beta * latent_kl_per_dim
            + self.lambda_balance * state.posterior_balance
        )
        metrics: dict[str, Tensor | float] = {
            "seg_loss": segmentation.detach(),
            "latent_kl_sum": latent_kl_sum.detach(),
            "latent_kl_per_dim": latent_kl_per_dim.detach(),
            "balance_loss": state.posterior_balance.detach(),
            "kl_beta": self.kl_beta,
        }
        return total, metrics

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        state = self._joint_forward(
            model,
            images,
            targets,
            decode_posterior=True,
            decode_prior=False,
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
        state = self._joint_forward(
            model,
            images,
            targets,
            decode_posterior=True,
            decode_prior=True,
        )
        if state.prior_logits is None or state.posterior_logits is None:
            raise ValueError("Joint evaluation requires both decoded paths.")
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

        latent_metrics = latent_transfer_metrics(state.posterior, state.prior)
        metrics.update(latent_metrics)
        metrics["latent_mean_l2"] = latent_metrics["mean_distance"]
        metrics["latent_std_l2"] = latent_metrics["std_distance"]

        topk_metrics = topk_transfer_metrics(
            state.posterior_routing.expert_indices,
            state.prior_routing.expert_indices,
            num_experts=state.posterior_routing.dense_probs.shape[1],
        )
        metrics.update(topk_metrics)
        metrics["topk_set_agreement"] = topk_metrics[
            "exact_topk_set_agreement"
        ]
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


__all__ = ["JointPriorPosteriorTask"]
