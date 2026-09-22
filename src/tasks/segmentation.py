"""Binary and multiclass segmentation task semantics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.metrics import (
    compute_binary_dice_iou,
    compute_binary_surface_metrics,
    compute_multiclass_dice_iou,
    compute_multiclass_surface_metrics,
)
from src.models import SegmentationOutput

from .base import TaskStepOutput


SUPPORTED_TASKS = ("binary", "multiclass")


class SegmentationTask:
    """Train and evaluate image-to-mask segmentation models.

    `task="binary"` expects `[B, 1, H, W]` float targets and thresholds the
    sigmoid at `threshold`. `task="multiclass"` expects `[B, H, W]` long class
    indices and argmaxes the logits; its metrics are class-mean with background
    excluded and classes absent from both prediction and target skipped.
    """

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
        task: str = "binary",
    ) -> None:
        if task not in SUPPORTED_TASKS:
            raise ValueError("task must be 'binary' or 'multiclass'.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if isinstance(boundary_tolerance, bool):
            raise ValueError("boundary_tolerance must be a non-negative number.")
        try:
            resolved_boundary_tolerance = float(boundary_tolerance)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "boundary_tolerance must be a non-negative number."
            ) from error
        if (
            not math.isfinite(resolved_boundary_tolerance)
            or resolved_boundary_tolerance < 0.0
        ):
            raise ValueError("boundary_tolerance must be a non-negative number.")

        self.criterion = criterion
        self.threshold = float(threshold)
        self.boundary_tolerance = resolved_boundary_tolerance
        self.task = task

    @property
    def target_dtype(self) -> torch.dtype:
        """Class indices for multiclass, soft/binary masks otherwise."""

        return torch.long if self.task == "multiclass" else torch.float32

    def _prepare_batch(self, batch: Any, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if "image" not in batch or "mask" not in batch:
            raise KeyError("Each segmentation batch must contain 'image' and 'mask'.")
        images = batch["image"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = batch["mask"].to(
            device,
            dtype=self.target_dtype,
            non_blocking=True,
        )
        return images, targets

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if not isinstance(output, SegmentationOutput):
            raise TypeError(
                "Segmentation model forward must return SegmentationOutput, got "
                f"{type(output).__name__}."
            )
        return output.logits

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits = self._extract_logits(model(images))
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics={},
            batch_size=images.shape[0],
        )

    def _segmentation_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Region and surface metrics as batch means, per this task's mode."""

        if self.task == "multiclass":
            dice, iou = compute_multiclass_dice_iou(logits, targets)
            hd, hd95, assd, boundary_f1 = compute_multiclass_surface_metrics(
                logits,
                targets,
                boundary_tolerance=self.boundary_tolerance,
            )
        else:
            dice, iou = compute_binary_dice_iou(
                logits,
                targets,
                threshold=self.threshold,
            )
            hd, hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                logits,
                targets,
                threshold=self.threshold,
                boundary_tolerance=self.boundary_tolerance,
            )
        return {
            "dice": dice.mean(),
            "iou": iou.mean(),
            "hd": hd.mean(),
            "hd95": hd95.mean(),
            "assd": assd.mean(),
            "boundary_f1": boundary_f1.mean(),
        }

    def finalize_evaluation_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """Finalize aggregated metrics; ordinary segmentation is identity."""

        return metrics

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits = self._extract_logits(model(images))
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics=self._segmentation_metrics(logits, targets),
            batch_size=images.shape[0],
        )
