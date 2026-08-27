"""Loss functions for binary segmentation."""

from .bce import BCELoss
from .combined import BCEDiceLoss
from .dice import DiceLoss
from .registry import LOSS_REGISTRY, build_loss, register_loss
from .shapemoe import (
    ShapeMoELoss,
    ShapeMoELosses,
    expert_balance_cv2_loss,
    gaussian_kl_divergence,
)
from .vae import MaskVAELoss, MaskVAELosses, gaussian_kl_to_standard_normal

__all__ = [
    "BCELoss",
    "BCEDiceLoss",
    "DiceLoss",
    "LOSS_REGISTRY",
    "MaskVAELoss",
    "MaskVAELosses",
    "ShapeMoELoss",
    "ShapeMoELosses",
    "build_loss",
    "expert_balance_cv2_loss",
    "gaussian_kl_divergence",
    "gaussian_kl_to_standard_normal",
    "register_loss",
]
