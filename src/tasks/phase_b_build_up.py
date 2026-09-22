"""Task semantics and diagnostics for the isolated build-up B1-B6 ladder."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from src.models.phase_b.studies.build_up.common import (
    IMAGE_ONLY,
    POSTERIOR_ORACLE,
    SUPPORTED_EVALUATION_MODES,
)

from .base import TaskStepOutput
from .phase_b_diagnostics import routing_metrics
from .segmentation import SegmentationTask


class PhaseBBuildUpTask(SegmentationTask):
    """Segmentation plus optional load balance and variant-aware metrics."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        evaluation_mode: str,
        lambda_balance: float,
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
        if evaluation_mode not in SUPPORTED_EVALUATION_MODES:
            raise ValueError(
                "evaluation_mode must be 'image_only' or 'posterior_oracle'."
            )
        if lambda_balance < 0:
            raise ValueError("lambda_balance must be non-negative.")
        self.evaluation_mode = evaluation_mode
        self.lambda_balance = float(lambda_balance)

    def _forward(
        self,
        model: nn.Module,
        images: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Any], Any]:
        model_mode = getattr(model, "evaluation_mode", None)
        if model_mode != self.evaluation_mode:
            raise ValueError(
                "Build-up task/model evaluation_mode mismatch: "
                f"task={self.evaluation_mode!r}, model={model_mode!r}."
            )
        if self.evaluation_mode == POSTERIOR_ORACLE:
            output = model(images, masks=targets)
        else:
            output = model(images)
        logits = self._extract_logits(output)
        diagnostics = getattr(output, "diagnostics", {}) or {}
        stage = diagnostics.get("phase_b_build_up")
        if stage is None:
            raise ValueError(
                "Build-up task expected a 'phase_b_build_up' diagnostic; got "
                f"{sorted(diagnostics)}."
            )
        if stage.evaluation_mode != self.evaluation_mode:
            raise ValueError("Build-up diagnostic evaluation_mode is inconsistent.")
        return logits, diagnostics, stage

    def _loss(self, logits: Tensor, targets: Tensor, stage: Any) -> Tensor:
        loss = self.criterion(logits, targets)
        routing = stage.routing
        if routing is not None:
            loss = loss + self.lambda_balance * routing.balance
        return loss

    @staticmethod
    def _alpha_metrics(diagnostics: dict[str, Any]) -> dict[str, Tensor]:
        alpha = diagnostics.get("level_weights")
        if not torch.is_tensor(alpha):
            return {}
        detached = alpha.detach()
        entropy = -(
            detached.clamp_min(torch.finfo(detached.dtype).tiny).log()
            * detached
        ).sum(dim=1)
        metrics: dict[str, Tensor] = {
            "level_weight_entropy": entropy.mean(),
            "level_weight_max": detached.max(dim=1).values.mean(),
        }
        levels = diagnostics.get("level_ids")
        if levels is not None:
            for index, level in enumerate(levels):
                value = detached[:, index].mean()
                metrics[f"layer_weight_{int(level)}"] = value
                metrics[f"mean_alpha_layer_{int(level)}"] = value
        return metrics

    @staticmethod
    def _enhancement_metrics(stage: Any) -> dict[str, Tensor]:
        return {
            "enhancement_aux_ratio": stage.aux_norm_ratio.detach().mean(),
            "fused_token_norm": stage.fused_token_norm.detach().mean(),
        }

    @staticmethod
    def _hierarchical_metrics(
        stage: Any,
        diagnostics: dict[str, Any],
    ) -> dict[str, Tensor]:
        beta = stage.expert_layer_weights
        routing = stage.routing
        if not torch.is_tensor(beta) or routing is None:
            return {}
        if beta.ndim != 3:
            raise ValueError("expert_layer_weights must be [B, k_e, L].")
        indices = routing.routing.expert_indices
        if indices.shape != beta.shape[:2]:
            raise ValueError("beta active slots must align with expert_indices.")
        levels = tuple(int(level) for level in diagnostics["level_ids"])
        if len(levels) != beta.shape[2]:
            raise ValueError("beta levels must align with level_ids.")

        detached = beta.detach()
        entropy = -(
            detached.clamp_min(torch.finfo(detached.dtype).tiny).log()
            * detached
        ).sum(dim=2)
        metrics: dict[str, Tensor] = {
            "expert_layer_weight_entropy": entropy.mean(),
        }

        gamma = stage.fusion_weights.detach()
        gamma_entropy = -(
            gamma.clamp_min(torch.finfo(gamma.dtype).tiny).log() * gamma
        ).sum(dim=1)
        metrics["hierarchical_layer_weight_entropy"] = gamma_entropy.mean()
        metrics["hierarchical_layer_weight_max"] = gamma.max(dim=1).values.mean()
        for level_index, level in enumerate(levels):
            metrics[f"mean_gamma_layer_{level}"] = gamma[:, level_index].mean()

        batch_size = beta.shape[0]
        num_experts = routing.routing.dense_probs.shape[1]
        for expert_id in range(num_experts):
            active = (indices == expert_id).to(beta.dtype)
            metrics[f"__beta_denominator_expert_{expert_id}"] = (
                active.sum() / batch_size
            )
            for level_index, level in enumerate(levels):
                numerator = (
                    detached[:, :, level_index] * active
                ).sum() / batch_size
                metrics[
                    f"__beta_numerator_expert_{expert_id}_layer_{level}"
                ] = numerator
        return metrics

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, _, stage = self._forward(model, images, targets)
        return TaskStepOutput(
            loss=self._loss(logits, targets, stage),
            metrics={},
            batch_size=images.shape[0],
        )

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics, stage = self._forward(model, images, targets)
        metrics = self._segmentation_metrics(logits, targets)
        metrics.update(self._alpha_metrics(diagnostics))
        metrics.update(self._enhancement_metrics(stage))
        if stage.routing is not None:
            metrics.update(routing_metrics(stage.routing))
        metrics.update(self._hierarchical_metrics(stage, diagnostics))
        return TaskStepOutput(
            loss=self._loss(logits, targets, stage),
            metrics=metrics,
            batch_size=images.shape[0],
        )

    def finalize_evaluation_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """Convert beta sufficient statistics to exact conditional means."""

        finalized = dict(metrics)
        denominators = {
            key.removeprefix("__beta_denominator_expert_"): finalized.pop(key)
            for key in tuple(finalized)
            if key.startswith("__beta_denominator_expert_")
        }
        numerator_prefix = "__beta_numerator_expert_"
        for key in tuple(finalized):
            if not key.startswith(numerator_prefix):
                continue
            numerator = finalized.pop(key)
            suffix = key.removeprefix(numerator_prefix)
            expert_id, level = suffix.split("_layer_", maxsplit=1)
            denominator = denominators[expert_id]
            finalized[f"mean_beta_expert_{expert_id}_layer_{level}"] = (
                numerator / denominator if denominator > 0.0 else 0.0
            )
        return finalized


__all__ = ["IMAGE_ONLY", "POSTERIOR_ORACLE", "PhaseBBuildUpTask"]
