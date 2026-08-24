"""Loss functions for binary segmentation."""

from .bce import BCELoss
from .combined import BCEDiceLoss
from .dice import DiceLoss
from .registry import LOSS_REGISTRY, build_loss, register_loss

__all__ = [
    "BCELoss",
    "BCEDiceLoss",
    "DiceLoss",
    "LOSS_REGISTRY",
    "build_loss",
    "register_loss",
]
