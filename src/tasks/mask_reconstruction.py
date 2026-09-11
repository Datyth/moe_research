"""Binary mask reconstruction task semantics."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.metrics import compute_binary_dice_iou
from src.models.shape import ShapeAutoencoderOutput

from .base import TaskStepOutput


class MaskReconstructionTask:
    """Train and evaluate mask-to-mask latent reconstruction models."""

    def __init__(
        self,
        *,
        criterion: nn.Module,
        threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        self.criterion = criterion
        self.threshold = float(threshold)

    @staticmethod
    def _prepare_masks(batch: Any, device: Any) -> torch.Tensor:
        if "mask" not in batch:
            raise KeyError("Each mask reconstruction batch must contain 'mask'.")
        return batch["mask"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if not isinstance(output, ShapeAutoencoderOutput):
            raise TypeError(
                "Shape reconstruction model forward must return "
                f"ShapeAutoencoderOutput, got {type(output).__name__}."
            )
        return output.reconstruction_logits

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        masks = self._prepare_masks(batch, device)
        logits = self._extract_logits(model(masks))
        return TaskStepOutput(
            loss=self.criterion(logits, masks),
            metrics={},
            batch_size=masks.shape[0],
        )

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        masks = self._prepare_masks(batch, device)
        logits = self._extract_logits(model(masks))
        dice, _ = compute_binary_dice_iou(
            logits,
            masks,
            threshold=self.threshold,
        )
        return TaskStepOutput(
            loss=self.criterion(logits, masks),
            metrics={"dice": dice.mean()},
            batch_size=masks.shape[0],
        )
