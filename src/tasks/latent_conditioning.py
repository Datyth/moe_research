"""Task semantics for the C1/C2/C3 latent-conditioning experiments."""

from __future__ import annotations

import math
import warnings
from typing import Any

import torch
from torch import Tensor, nn

from src.models.latent_conditioning import LatentConditioningState
from src.models.phase_b.posterior import gaussian_kl
from src.models.phase_b.router import RoutingOutput
from src.models.phase_c.metrics import (
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)

from .base import TaskStepOutput
from .segmentation import SegmentationTask


class LatentConditioningTask(SegmentationTask):
    """Posterior segmentation plus joint prior/posterior Gaussian KL."""

    _COLLAPSE_THRESHOLD = 0.8
    _COLLAPSE_EPOCHS = 3

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
            raise ValueError("Latent conditioning supports binary masks only.")
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
        self._collapse_state: dict[str, dict[str, int | bool]] = {}

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

    def _require_finite(self, name: str, value: Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(
                f"Non-finite {name} at epoch {self.current_epoch}."
            )

    def _joint_forward(
        self,
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
        *,
        decode_posterior: bool,
        decode_prior: bool,
    ) -> LatentConditioningState:
        forward = getattr(model, "joint_forward", None)
        if not callable(forward):
            raise TypeError("LatentConditioningTask requires model.joint_forward().")
        try:
            state = forward(
                images,
                targets,
                decode_posterior=decode_posterior,
                decode_prior=decode_prior,
            )
        except FloatingPointError as error:
            raise FloatingPointError(
                f"{error} Model call failed at epoch {self.current_epoch}."
            ) from error
        if not isinstance(state, LatentConditioningState):
            raise TypeError(
                "joint_forward must return LatentConditioningState."
            )
        return state

    def _loss_components(
        self,
        state: LatentConditioningState,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if state.posterior_logits is None:
            raise ValueError("Latent-conditioning loss requires posterior logits.")
        segmentation = self.criterion(state.posterior_logits, targets)
        self._require_finite("segmentation loss", segmentation)

        latent_kl_sum = gaussian_kl(state.posterior, state.prior).mean()
        self._require_finite("Gaussian KL", latent_kl_sum)
        latent_dim = state.posterior.mean.shape[1]
        if latent_dim != 8:
            raise ValueError(f"Latent conditioning requires d_z=8, got {latent_dim}.")
        latent_kl_per_dim = latent_kl_sum / latent_dim
        total = segmentation + self.kl_beta * latent_kl_per_dim
        metrics: dict[str, Tensor | float] = {
            "seg_loss": segmentation.detach(),
            "latent_kl_sum": latent_kl_sum.detach(),
            "latent_kl_per_dim": latent_kl_per_dim.detach(),
            "kl_beta": self.kl_beta,
        }

        if state.use_moe:
            if state.posterior_balance is None:
                raise ValueError("MoE training requires posterior balance loss.")
            self._require_finite("load-balance loss", state.posterior_balance)
            total = total + self.lambda_balance * state.posterior_balance
            metrics["balance_loss"] = state.posterior_balance.detach()
        elif state.posterior_balance is not None:
            raise ValueError("C2 must not produce a load-balance loss.")

        self._require_finite("total loss", total)
        return total, metrics

    @staticmethod
    def _routing_diagnostics(routing: RoutingOutput) -> dict[str, Tensor]:
        dense_probs = routing.dense_probs.detach()
        expert_indices = routing.expert_indices.detach()
        if dense_probs.ndim != 2 or expert_indices.ndim != 2:
            raise ValueError("Routing tensors must be [B,K] and [B,k_e].")
        num_experts = dense_probs.shape[1]
        if num_experts <= 1:
            raise ValueError("Routing diagnostics require at least two experts.")
        slot_counts = torch.bincount(
            expert_indices.reshape(-1),
            minlength=num_experts,
        ).to(dtype=dense_probs.dtype, device=dense_probs.device)
        fractions = slot_counts / max(int(expert_indices.numel()), 1)
        tiny = torch.finfo(fractions.dtype).tiny
        entropy = -(fractions.clamp_min(tiny).log() * fractions).sum()
        metrics = {
            "expert_utilization": (fractions > 0).to(dense_probs.dtype).mean(),
            "routing_entropy_normalized": entropy / math.log(num_experts),
        }
        for expert_index in range(num_experts):
            metrics[f"expert_usage_fraction_{expert_index}"] = fractions[
                expert_index
            ]
        return metrics

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
            raise ValueError("Evaluation requires both posterior and prior logits.")
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

        if state.use_moe and state.use_pre_moe_latent:
            if state.posterior_routing is None or state.prior_routing is None:
                raise ValueError("C3 evaluation requires both routing outputs.")
            transfer = topk_transfer_metrics(
                state.posterior_routing.expert_indices,
                state.prior_routing.expert_indices,
                num_experts=state.posterior_routing.dense_probs.shape[1],
            )
            metrics.update(transfer)
            metrics["routing_js"] = routing_js(
                state.posterior_routing.dense_probs,
                state.prior_routing.dense_probs,
            )
            metrics.update(
                {
                    f"posterior_{name}": value
                    for name, value in self._routing_diagnostics(
                        state.posterior_routing
                    ).items()
                }
            )
            metrics.update(
                {
                    f"prior_{name}": value
                    for name, value in self._routing_diagnostics(
                        state.prior_routing
                    ).items()
                }
            )
        elif state.use_moe:
            if state.posterior_routing is None:
                raise ValueError("C1 evaluation requires its shared image route.")
            metrics.update(self._routing_diagnostics(state.posterior_routing))

        return TaskStepOutput(
            loss=total,
            metrics=metrics,
            batch_size=images.shape[0],
        )

    def _update_collapse_warning(
        self,
        label: str,
        fractions: list[float],
    ) -> None:
        maximum = max(fractions, default=0.0)
        state = self._collapse_state.setdefault(
            label,
            {"last_epoch": 0, "streak": 0, "warned": False},
        )
        last_epoch = int(state["last_epoch"])
        if maximum <= self._COLLAPSE_THRESHOLD:
            state.update(
                {"last_epoch": self.current_epoch, "streak": 0, "warned": False}
            )
            return
        if last_epoch == self.current_epoch:
            return
        streak = int(state["streak"])
        streak = streak + 1 if last_epoch == self.current_epoch - 1 else 1
        state.update({"last_epoch": self.current_epoch, "streak": streak})
        if streak >= self._COLLAPSE_EPOCHS and not bool(state["warned"]):
            warnings.warn(
                f"{label} routing collapse: one expert received {maximum:.1%} "
                f"of active slots for {streak} consecutive validation epochs "
                f"through epoch {self.current_epoch}.",
                RuntimeWarning,
                stacklevel=3,
            )
            state["warned"] = True

    def finalize_evaluation_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        finalized = dict(metrics)
        finalized["transfer_gap_dice"] = (
            finalized["posterior_dice"] - finalized["dice"]
        )
        routing_prefixes = {
            "shared": "expert_usage_fraction_",
            "posterior": "posterior_expert_usage_fraction_",
            "prior": "prior_expert_usage_fraction_",
        }
        for label, prefix in routing_prefixes.items():
            fractions = [
                value
                for name, value in finalized.items()
                if name.startswith(prefix)
            ]
            if fractions:
                self._update_collapse_warning(label, fractions)
        return finalized


__all__ = ["LatentConditioningTask"]
