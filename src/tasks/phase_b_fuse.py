"""Phase B fuse-stage task semantics.

Differs from plain segmentation in one respect: the ground-truth mask is also
handed to the model as privileged input, so the Shape Teacher can produce h_M
and the model can form h_q = [h_I ; h_M].

The loss is still the segmentation loss. At the fuse stage there is no
posterior yet, so nothing trains h_I: the descriptor and its level attention
receive no gradient until the posterior q(z | I, M) and its KL/routing terms
land in the router stage (`phase_b_router`), and the segmentation loss
itself only reaches them through the enhancement stage (`phase_b_moe`).
`strict_fuse` therefore defaults to True so a run that
silently stopped producing h_q fails loudly instead of quietly degrading into
an ordinary MoE-SAM run.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import TaskStepOutput
from .segmentation import SegmentationTask


class PhaseBFuseTask(SegmentationTask):
    """Segmentation, with the ground-truth mask passed as privileged input."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
        task: str = "binary",
        strict_fuse: bool = True,
    ) -> None:
        super().__init__(
            criterion=criterion,
            threshold=threshold,
            boundary_tolerance=boundary_tolerance,
            task=task,
        )
        if task != "binary":
            raise ValueError(
                "The Shape Teacher reconstructs single-channel masks, so the "
                "Phase B fuse stage supports task='binary' only."
            )
        self.strict_fuse = bool(strict_fuse)

    def _forward_with_privileged_mask(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output = model(images, masks=targets)
        logits = self._extract_logits(output)
        diagnostics = getattr(output, "diagnostics", {}) or {}
        if self.strict_fuse and "fuse_stage" not in diagnostics:
            raise ValueError(
                "Phase B expected the model to return a 'fuse_stage' "
                "diagnostic (h_q); got keys "
                f"{sorted(diagnostics)}. Is this a phase_b_fuse model?"
            )
        return logits, diagnostics

    @staticmethod
    def _level_weight_metrics(diagnostics: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Report how peaked the level attention is.

        Entropy near log(L) means the router weights every SAM level equally
        and the level attention has not specialized; near 0 means it collapsed
        onto one level. Both are worth seeing during training.
        """

        weights = diagnostics.get("level_weights")
        if weights is None:
            return {}
        entropy = -(weights.clamp_min(1e-9).log() * weights).sum(dim=1)
        return {
            "level_weight_entropy": entropy.mean(),
            "level_weight_max": weights.max(dim=1).values.mean(),
        }

    def training_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, _ = self._forward_with_privileged_mask(model, images, targets)
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics={},
            batch_size=images.shape[0],
        )

    def evaluation_step(self, model: nn.Module, batch: Any, device: Any) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, diagnostics = self._forward_with_privileged_mask(
            model, images, targets
        )
        metrics = self._segmentation_metrics(logits, targets)
        metrics.update(self._level_weight_metrics(diagnostics))
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics=metrics,
            batch_size=images.shape[0],
        )
