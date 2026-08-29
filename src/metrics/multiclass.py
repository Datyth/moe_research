"""Region and surface metrics for multiclass segmentation masks."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .segmentation import _sample_surface_metrics, _validate_boundary_tolerance


def _validate_multiclass_shapes(logits: Tensor, targets: Tensor) -> None:
    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits with shape [B, C, H, W], got {tuple(logits.shape)}."
        )
    if targets.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError(
            "targets must have shape [B, H, W] matching logits' batch/spatial "
            f"dims, got {tuple(targets.shape)} for logits {tuple(logits.shape)}."
        )


def compute_multiclass_dice_iou(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_background: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute per-sample, class-mean Dice and IoU from multiclass logits.

    `logits` is `[B, C, H, W]`; `targets` is `[B, H, W]` with integer class
    indices. A class absent from both the prediction and target for a given
    sample is excluded from that sample's mean (not counted as a perfect or
    zero score) so classes that simply don't appear in a slice don't skew
    the average.
    """

    _validate_multiclass_shapes(logits, targets)

    num_classes = logits.shape[1]
    predictions = logits.argmax(dim=1)

    predictions_one_hot = F.one_hot(predictions, num_classes=num_classes).permute(0, 3, 1, 2)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

    first_class = 1 if ignore_background else 0
    predictions_one_hot = predictions_one_hot[:, first_class:].float()
    targets_one_hot = targets_one_hot[:, first_class:].float()

    spatial_dimensions = (2, 3)
    intersection = (predictions_one_hot * targets_one_hot).sum(dim=spatial_dimensions)
    prediction_size = predictions_one_hot.sum(dim=spatial_dimensions)
    target_size = targets_one_hot.sum(dim=spatial_dimensions)

    dice_denominator = prediction_size + target_size
    union = prediction_size + target_size - intersection

    class_present = (prediction_size + target_size) > 0
    dice_per_class = torch.where(
        dice_denominator == 0,
        torch.ones_like(intersection),
        2.0 * intersection / dice_denominator.clamp(min=1e-7),
    )
    iou_per_class = torch.where(
        union == 0,
        torch.ones_like(intersection),
        intersection / union.clamp(min=1e-7),
    )

    class_count = class_present.sum(dim=1).clamp(min=1)
    dice = (dice_per_class * class_present).sum(dim=1) / class_count
    iou = (iou_per_class * class_present).sum(dim=1) / class_count
    return dice, iou


def compute_multiclass_surface_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_background: bool = True,
    boundary_tolerance: float = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-sample, class-mean HD95, ASSD and Boundary F1.

    Generalizes `compute_binary_surface_metrics` to multiclass masks: each
    class is scored as its own binary problem (that class vs. everything
    else) via the same per-sample surface-distance logic used for binary
    segmentation, then averaged across classes present in that sample —
    matching `compute_multiclass_dice_iou`'s convention of excluding a class
    absent from both prediction and target rather than scoring it as
    perfect or zero.
    """

    _validate_multiclass_shapes(logits, targets)
    tolerance = _validate_boundary_tolerance(boundary_tolerance)

    num_classes = logits.shape[1]
    first_class = 1 if ignore_background else 0
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    target_masks = targets.detach().cpu().numpy()

    batch_size = logits.shape[0]
    values = torch.empty((batch_size, 3), dtype=torch.float64)

    for sample_index in range(batch_size):
        prediction_sample = predictions[sample_index]
        target_sample = target_masks[sample_index]

        class_values: list[tuple[float, float, float]] = []
        for class_id in range(first_class, num_classes):
            prediction_mask = prediction_sample == class_id
            target_mask = target_sample == class_id
            if not prediction_mask.any() and not target_mask.any():
                continue
            class_values.append(
                _sample_surface_metrics(
                    prediction_mask,
                    target_mask,
                    boundary_tolerance=tolerance,
                )
            )

        if not class_values:
            values[sample_index] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        else:
            values[sample_index] = torch.tensor(
                class_values, dtype=torch.float64
            ).mean(dim=0)

    return values[:, 0], values[:, 1], values[:, 2]
