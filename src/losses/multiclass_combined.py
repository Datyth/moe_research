"""Weighted combination of cross-entropy and multiclass Dice losses."""

from torch import Tensor, nn

from .multiclass_dice import MulticlassDiceLoss
from .registry import register_loss


@register_loss("ce_dice")
class CEDiceLoss(nn.Module):
    """Combine cross-entropy and Dice losses for multiclass segmentation."""

    def __init__(
        self,
        *,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: list[float] | None = None,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1e-7,
        ignore_background: bool = False,
    ) -> None:
        super().__init__()

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if ce_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive.")

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weight_tensor = (
            None if class_weights is None else Tensor(class_weights)
        )
        self.register_buffer("class_weights", weight_tensor)
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice = MulticlassDiceLoss(
            smooth=dice_smooth,
            epsilon=dice_epsilon,
            ignore_background=ignore_background,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
