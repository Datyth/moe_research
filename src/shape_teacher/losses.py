"""Reconstruction-only objective for the Gaussian Shape Teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def soft_dice_score(
    logits: Tensor,
    targets: Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Mean per-sample soft Dice for binary mask logits."""

    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("logits and targets must have the same [B, C, H, W] shape.")
    probabilities = logits.sigmoid()
    targets = targets.to(dtype=probabilities.dtype)
    spatial = tuple(range(2, logits.ndim))
    intersection = (probabilities * targets).sum(dim=spatial)
    denominator = probabilities.sum(dim=spatial) + targets.sum(dim=spatial)
    score = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return score.mean()


@dataclass
class ShapeTeacherLosses:
    total: Tensor
    bce: Tensor
    dice: Tensor
    soft_dice: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "bce": float(self.bce.detach()),
            "dice_loss": float(self.dice.detach()),
            "soft_dice": float(self.soft_dice.detach()),
        }


class ShapeTeacherLoss(nn.Module):
    """``L_shape = lambda_bce * BCEWithLogits + lambda_dice * DiceLoss``."""

    def __init__(
        self,
        *,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        dice_epsilon: float = 1.0e-6,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        if lambda_bce < 0.0 or lambda_dice < 0.0:
            raise ValueError("loss weights must be non-negative.")
        if lambda_bce == 0.0 and lambda_dice == 0.0:
            raise ValueError("at least one loss weight must be positive.")
        if dice_epsilon <= 0.0:
            raise ValueError("dice_epsilon must be positive.")
        weight = None if pos_weight is None else torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight", weight)
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        self.dice_epsilon = float(dice_epsilon)

    def forward(self, logits: Tensor, targets: Tensor) -> ShapeTeacherLosses:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have the same shape.")
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )
        dice_score = soft_dice_score(
            logits,
            targets,
            epsilon=self.dice_epsilon,
        )
        dice = 1.0 - dice_score
        total = self.lambda_bce * bce + self.lambda_dice * dice
        return ShapeTeacherLosses(total, bce, dice, dice_score)
