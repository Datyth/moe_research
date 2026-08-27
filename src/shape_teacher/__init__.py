"""Mask-only Gaussian Shape Teacher pretraining components."""

from .corruptions import MaskCorruptor
from .data import MaskOnlyDataset, audit_mask_splits, build_mask_datasets
from .losses import ShapeTeacherLoss, ShapeTeacherLosses, soft_dice_score
from .model import ShapeTeacher, ShapeTeacherOutput
from .qualitative import (
    QUALITATIVE_CATEGORIES,
    collect_qualitative_records,
    mask_shape_statistics,
    select_representatives,
)

__all__ = [
    "MaskCorruptor",
    "MaskOnlyDataset",
    "ShapeTeacher",
    "ShapeTeacherLoss",
    "ShapeTeacherLosses",
    "ShapeTeacherOutput",
    "QUALITATIVE_CATEGORIES",
    "audit_mask_splits",
    "build_mask_datasets",
    "collect_qualitative_records",
    "mask_shape_statistics",
    "select_representatives",
    "soft_dice_score",
]
