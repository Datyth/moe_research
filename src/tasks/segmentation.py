"""Binary segmentation task semantics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.metrics import compute_binary_dice_iou, compute_binary_surface_metrics
from src.models import SegmentationOutput

from .base import TaskStepOutput


class SegmentationTask:
    """Train and evaluate image-to-mask binary segmentation models."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
        boundary_tolerance: float = 2,
    ) -> None:
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

    @staticmethod
    def _prepare_batch(batch: Any, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if "image" not in batch or "mask" not in batch:
            raise KeyError("Each segmentation batch must contain 'image' and 'mask'.")
        images = batch["image"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = batch["mask"].to(
            device,
            dtype=torch.float32,
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

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        images, targets = self._prepare_batch(batch, device)
        logits = self._extract_logits(model(images))
        dice, iou = compute_binary_dice_iou(
            logits,
            targets,
            threshold=self.threshold,
        )
        hd95, assd, boundary_f1 = compute_binary_surface_metrics(
            logits,
            targets,
            threshold=self.threshold,
            boundary_tolerance=self.boundary_tolerance,
        )
        return TaskStepOutput(
            loss=self.criterion(logits, targets),
            metrics={
                "dice": dice.mean(),
                "iou": iou.mean(),
                "hd95": hd95.mean(),
                "assd": assd.mean(),
                "boundary_f1": boundary_f1.mean(),
            },
            batch_size=images.shape[0],
        )
