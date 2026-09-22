"""Segmentation-only task for A1 no-expert Phase B ablation."""

from __future__ import annotations

from typing import Any

from torch import nn

from .base import TaskStepOutput
from .phase_b_diagnostics import enhancement_metrics, layer_fusion_metrics
from .segmentation import SegmentationTask


class PhaseBA1NoExpertTask(SegmentationTask):
    """Train A1 with segmentation loss while reporting fusion diagnostics."""

    def _forward(self, model: nn.Module, images):
        output = model(images)
        logits = self._extract_logits(output)
        diagnostics = getattr(output, "diagnostics", {}) or {}
        if "phase_b_ablation" not in diagnostics:
            raise ValueError(
                "A1 task expected a 'phase_b_ablation' diagnostic; got keys "
                f"{sorted(diagnostics)}."
            )
        return logits, diagnostics

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits, _ = self._forward(model, images)
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
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
        logits, diagnostics = self._forward(model, images)
        metrics = self._segmentation_metrics(logits, targets)
        metrics.update(layer_fusion_metrics(diagnostics))
        metrics.update(enhancement_metrics(diagnostics["phase_b_ablation"]))
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics=metrics,
            batch_size=images.shape[0],
        )
