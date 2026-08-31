"""Cross-entropy plus soft Dice loss for multiclass segmentation."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .registry import register_loss


@register_loss("ce_dice")
class CEDiceLoss(nn.Module):
    """Combine cross-entropy and soft Dice losses for multiclass segmentation.

    Follows the paper's training objective ``L = (1 - alpha) * L_ce + L_dice``
    where ``ce_weight`` plays the role of ``(1 - alpha)``. Dice is computed
    from softmax probabilities over the foreground classes (class indices
    ``1 .. C-1``) so the background does not dominate the overlap term.
    """

    def __init__(
        self,
        *,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weight: list[float] | None = None,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1e-7,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if ce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")
        if dice_smooth < 0:
            raise ValueError("dice_smooth must be non-negative.")
        if dice_epsilon <= 0:
            raise ValueError("dice_epsilon must be positive.")

        weight_tensor = (
            None
            if class_weight is None
            else torch.as_tensor(class_weight, dtype=torch.float32)
        )
        self.register_buffer("class_weight", weight_tensor)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.dice_epsilon = dice_epsilon
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(logits, targets)

        labels = targets.reshape(targets.shape[0], *targets.shape[2:]).long()

        if self.ce_weight > 0:
            ce_loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weight,
                ignore_index=(
                    -100 if self.ignore_index is None else self.ignore_index
                ),
            )
        else:
            ce_loss = logits.new_zeros(())

        if self.dice_weight > 0:
            dice_loss = self._dice_loss(logits, labels)
        else:
            dice_loss = logits.new_zeros(())

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

    def _dice_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        num_classes = logits.shape[1]
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).to(
            dtype=logits.dtype
        )

        # Foreground classes only: skip the background channel.
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

        spatial_dimensions = tuple(range(2, logits.ndim))
        intersection = (probabilities * one_hot).sum(dim=spatial_dimensions)
        denominator = probabilities.sum(
            dim=spatial_dimensions
        ) + one_hot.sum(dim=spatial_dimensions)

        dice_score = (
            2.0 * intersection + self.dice_smooth
        ) / (
            denominator + self.dice_smooth + self.dice_epsilon
        )
        return 1.0 - dice_score.mean()

    @staticmethod
    def _validate_inputs(logits: Tensor, targets: Tensor) -> None:
        if logits.ndim != 4 or logits.shape[1] <= 1:
            raise ValueError(
                "ce_dice expects multiclass logits with shape [B, C, H, W] "
                "and C > 1, got "
                f"{tuple(logits.shape)}."
            )
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor.")
        if (
            targets.ndim != 4
            or targets.shape[0] != logits.shape[0]
            or targets.shape[1] != 1
            or targets.shape[2:] != logits.shape[2:]
        ):
            raise ValueError(
                "targets must be class indices with shape [B, 1, H, W] "
                f"matching logits spatial size, got {tuple(targets.shape)}."
            )
