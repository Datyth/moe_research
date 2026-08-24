"""Weighted combination of BCE and Dice losses."""

from torch import Tensor, nn

from .bce import BCELoss
from .dice import DiceLoss
from .registry import register_loss


@register_loss("bce_dice")
class BCEDiceLoss(nn.Module):
    """Combine BCE and Dice losses for binary segmentation."""

    def __init__(
        self,
        *,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | Tensor | None = None,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1e-7,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss(
            smooth=dice_smooth,
            epsilon=dice_epsilon,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )
