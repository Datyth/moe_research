"""Loss functions for binary and multiclass segmentation."""

from .bce import BCELoss
from .ce_dice import CEDiceLoss
from .combined import BCEDiceLoss
from .dice import DiceLoss
from .multiclass_dice import MulticlassDiceLoss
from .registry import LOSS_REGISTRY, build_loss, register_loss

__all__ = [
    "BCELoss",
    "BCEDiceLoss",
    "CEDiceLoss",
    "DiceLoss",
    "MulticlassDiceLoss",
    "LOSS_REGISTRY",
    "build_loss",
    "register_loss",
]
