"""Binary cross-entropy loss for segmentation logits."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BCELoss(nn.Module):
    """Numerically stable BCE loss operating directly on raw logits."""

    def __init__(
        self,
        *,
        pos_weight: float | Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(
                "reduction must be one of: 'none', 'mean', or 'sum'."
            )

        weight_tensor = (
            None
            if pos_weight is None
            else torch.as_tensor(pos_weight, dtype=torch.float32)
        )
        self.register_buffer("pos_weight", weight_tensor)
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        self._validate_inputs(logits, targets)

        return F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

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
