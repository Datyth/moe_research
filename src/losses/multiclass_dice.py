"""Soft Dice loss for multiclass segmentation logits."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .registry import register_loss


@register_loss("multiclass_dice")
class MulticlassDiceLoss(nn.Module):
    """Mean soft Dice loss over classes, computed from softmax probabilities."""

    def __init__(
        self,
        *,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
        ignore_background: bool = False,
    ) -> None:
        super().__init__()

        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.smooth = smooth
        self.epsilon = epsilon
        self.ignore_background = ignore_background

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(logits, targets)

        num_classes = logits.shape[1]
        probabilities = logits.softmax(dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        if self.ignore_background:
            probabilities = probabilities[:, 1:]
            targets_one_hot = targets_one_hot[:, 1:]

        spatial_dimensions = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * targets_one_hot).sum(dim=spatial_dimensions)
        denominator = probabilities.sum(dim=spatial_dimensions) + targets_one_hot.sum(
            dim=spatial_dimensions
        )

        dice_score = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth + self.epsilon
        )
        return 1.0 - dice_score.mean()

    @staticmethod
    def _validate_inputs(logits: Tensor, targets: Tensor) -> None:
        if logits.ndim != 4:
            raise ValueError(
                f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}."
            )
        if targets.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError(
                "targets must have shape [B, H, W] matching logits' batch/spatial "
                f"dims, got {tuple(targets.shape)} for logits {tuple(logits.shape)}."
            )
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor.")
        if targets.dtype not in (torch.long, torch.int):
            raise TypeError("targets must be an integer class-index tensor.")
