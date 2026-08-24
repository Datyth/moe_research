"""Soft Dice loss for binary segmentation logits."""

from torch import Tensor, nn

from .registry import register_loss


@register_loss("dice")
class DiceLoss(nn.Module):
    """Compute mean soft Dice loss over samples and channels."""

    def __init__(
        self,
        *,
        smooth: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()

        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.smooth = smooth
        self.epsilon = epsilon

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(logits, targets)

        probabilities = logits.sigmoid()
        targets = targets.to(dtype=logits.dtype)
        spatial_dimensions = tuple(range(2, logits.ndim))

        intersection = (
            probabilities * targets
        ).sum(dim=spatial_dimensions)
        denominator = probabilities.sum(
            dim=spatial_dimensions
        ) + targets.sum(dim=spatial_dimensions)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth + self.epsilon
        )

        return 1.0 - dice_score.mean()

    @staticmethod
    def _validate_inputs(logits: Tensor, targets: Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}."
            )
        if logits.ndim != 4:
            raise ValueError(
                f"Expected tensors with shape [B, C, H, W], got "
                f"{tuple(logits.shape)}."
            )
        if not logits.is_floating_point():
            raise TypeError("logits must be a floating-point tensor.")
