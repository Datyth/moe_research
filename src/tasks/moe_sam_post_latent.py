"""Task semantics for deterministic post-MoE mean conditioning."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from src.models.moe_sam_post_latent import MoeSamPostLatentState

from .base import TaskStepOutput
from .segmentation import SegmentationTask


class MoeSamPostLatentMeanTask(SegmentationTask):
    """Posterior segmentation plus stop-gradient posterior-to-prior mean MSE."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        lambda_mean_align: float = 1.0,
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
            raise ValueError(
                "MoE-SAM post-latent mean conditioning supports binary masks only."
            )
        if (
            isinstance(lambda_mean_align, bool)
            or not isinstance(lambda_mean_align, (int, float))
            or not math.isfinite(float(lambda_mean_align))
            or float(lambda_mean_align) <= 0.0
        ):
            raise ValueError(
                "lambda_mean_align must be a finite positive number."
            )
        self.lambda_mean_align = float(lambda_mean_align)

    @staticmethod
    def _require_finite(name: str, tensor: Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"{name} contains NaN or Inf.")

    @staticmethod
    def _mean_metrics(
        state: MoeSamPostLatentState,
    ) -> tuple[Tensor, Tensor]:
        difference = (state.prior_mean - state.posterior_mean.detach()).float()
        mean_mse = difference.pow(2).mean()
        mean_l2_distance = difference.norm(dim=1).mean()
        return mean_mse, mean_l2_distance

    @staticmethod
    def _joint_forward(
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        *,
        decode_posterior: bool,
        decode_prior: bool,
    ) -> MoeSamPostLatentState:
        forward = getattr(model, "joint_forward", None)
        if not callable(forward):
            raise TypeError(
                "MoeSamPostLatentMeanTask requires model.joint_forward()."
            )
        state = forward(
            images,
            targets,
            decode_posterior=decode_posterior,
            decode_prior=decode_prior,
        )
        if not isinstance(state, MoeSamPostLatentState):
            raise TypeError(
                "joint_forward must return MoeSamPostLatentState."
            )
        return state

    def _objective(
        self,
        state: MoeSamPostLatentState,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if state.posterior_logits is None:
            raise ValueError("Training requires posterior logits.")
        posterior_seg = self.criterion(state.posterior_logits, targets)
        mean_mse, mean_l2_distance = self._mean_metrics(state)
        weighted_mean = self.lambda_mean_align * mean_mse
        total = posterior_seg + weighted_mean
        for name, tensor in (
            ("posterior segmentation loss", posterior_seg),
            ("mean MSE", mean_mse),
            ("weighted mean alignment", weighted_mean),
            ("total loss", total),
        ):
            self._require_finite(name, tensor)
        return total, {
            "posterior_seg_loss": posterior_seg.detach(),
            "mean_mse": mean_mse.detach(),
            "mean_l2_distance": mean_l2_distance.detach(),
            "weighted_mean_align_loss": weighted_mean.detach(),
            "total_loss": total.detach(),
        }

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
        total, metrics = self._objective(state, targets)
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
        if state.posterior_logits is None or state.prior_logits is None:
            raise ValueError("Evaluation requires posterior and prior logits.")

        total, metrics = self._objective(state, targets)
        prior_seg = self.criterion(state.prior_logits, targets)
        self._require_finite("prior segmentation loss", prior_seg)
        metrics["prior_seg_loss"] = prior_seg.detach()
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
        if state.prior_delta_norm_ratio is not None:
            metrics["post_conditioning_delta_norm_ratio"] = (
                state.prior_delta_norm_ratio.detach().mean()
            )
        if state.posterior_delta_norm_ratio is not None:
            metrics["posterior_post_conditioning_delta_norm_ratio"] = (
                state.posterior_delta_norm_ratio.detach().mean()
            )
        return TaskStepOutput(
            loss=prior_seg,
            metrics=metrics,
            batch_size=images.shape[0],
        )

    def finalize_evaluation_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        finalized = dict(metrics)
        if "posterior_dice" in finalized and "dice" in finalized:
            finalized["transfer_gap_dice"] = (
                finalized["posterior_dice"] - finalized["dice"]
            )
        return finalized


__all__ = ["MoeSamPostLatentMeanTask"]
