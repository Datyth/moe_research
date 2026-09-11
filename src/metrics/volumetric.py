"""Case-level (3D-volume) region and surface metrics for CT multiclass segmentation.

Complements `multiclass.py`'s per-2D-slice metrics. Synapse/BTCV literature —
including every row of MoE-SAM's Table 1 — reports Dice/HD per 3D case
volume (reconstruct the patient's volume, score each organ once over the
whole volume, average over organs then over patients), not as a flat mean
over individual axial slices. `src/engine/evaluate()` only does the latter,
so its `dice`/`hd` numbers are not directly comparable to a paper's reported
DSC/HD even when the model itself is correct.

This module reconstructs each test case's prediction/target volume from its
2D slice predictions — grouped by the `case_id` and slice index already
carried by the manifest (see `scripts/data/ct_conversion.py`) — and scores
it with the natural 3D generalization of `multiclass.py`'s per-slice logic
(same class-present convention: a class absent from both prediction and
target for a case is excluded from that case's mean rather than scored as
perfect or zero).

Protocol confirmed by MoE-SAM's authors (email, 2026-09-07): their
evaluation "assembles a 3D prediction array and computes each foreground
class's metrics over that volume", scoring Dice and HD95 with
`medpy.metric.binary.dc` and `medpy.metric.binary.hd95`. They also confirmed
that the `hd95` call is made without voxel spacing, so their reported figure
is in *voxel units on the evaluation grid*, not millimetres. That resolves
what was previously an open caveat here: this module likewise never reads
the NIfTI affine, so its voxel-unit HD95 is directly comparable to the
paper's HD column after all. `dice`/`hd95` below therefore call medpy
directly, so the two agree exactly rather than merely in definition.

`hd`, `assd` and `boundary_f1` are this repo's own additions and have no
counterpart in the paper. Note that the paper's Table 1 column labelled "HD"
is the `hd95` key here, not `hd`.
"""

from __future__ import annotations

import math

import numpy as np
from medpy import metric as medpy_metric
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt


BoolVolume = NDArray[np.bool_]


def _extract_surface_3d(mask: BoolVolume) -> BoolVolume:
    if not mask.any():
        return np.zeros_like(mask)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return np.asarray(mask & ~eroded, dtype=bool)


def _directed_surface_distances_3d(
    prediction_surface: BoolVolume,
    target_surface: BoolVolume,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prediction_to_target = distance_transform_edt(~target_surface)[prediction_surface]
    target_to_prediction = distance_transform_edt(~prediction_surface)[target_surface]
    return (
        np.asarray(prediction_to_target, dtype=np.float64),
        np.asarray(target_to_prediction, dtype=np.float64),
    )


def _class_metrics_3d(
    prediction: BoolVolume,
    target: BoolVolume,
    *,
    boundary_tolerance: float,
) -> dict[str, float]:
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())

    intersection = float(np.logical_and(prediction, target).sum())
    prediction_size = float(prediction.sum())
    target_size = float(target.sum())
    dice_denominator = prediction_size + target_size
    union = dice_denominator - intersection

    # medpy.metric.binary.dc is the exact call the MoE-SAM authors use, but it
    # divides by zero when both volumes are empty; that case is defined as a
    # perfect score here (the class simply does not occur in this patient) and
    # is filtered out one level up by compute_case_metrics_3d anyway.
    dice = (
        1.0
        if dice_denominator == 0.0
        else float(medpy_metric.binary.dc(prediction, target))
    )
    iou = 1.0 if union == 0.0 else intersection / union

    if not prediction_nonempty and not target_nonempty:
        hd = hd95 = assd = 0.0
        boundary_f1 = 1.0
    elif prediction_nonempty != target_nonempty:
        depth, height, width = prediction.shape
        max_distance = math.sqrt((depth - 1) ** 2 + (height - 1) ** 2 + (width - 1) ** 2)
        hd = hd95 = assd = max_distance
        boundary_f1 = 0.0
    else:
        prediction_surface = _extract_surface_3d(prediction)
        target_surface = _extract_surface_3d(target)
        prediction_to_target, target_to_prediction = _directed_surface_distances_3d(
            prediction_surface, target_surface
        )
        combined = np.concatenate((prediction_to_target, target_to_prediction))
        hd = float(combined.max())
        # Same call (and therefore same voxel units) as the authors' own
        # evaluation; medpy requires both masks non-empty, which this branch
        # already guarantees.
        hd95 = float(medpy_metric.binary.hd95(prediction, target))
        assd = float(
            (prediction_to_target.sum() + target_to_prediction.sum())
            / (prediction_to_target.size + target_to_prediction.size)
        )
        precision = float(np.mean(prediction_to_target <= boundary_tolerance))
        recall = float(np.mean(target_to_prediction <= boundary_tolerance))
        precision_recall_sum = precision + recall
        boundary_f1 = (
            0.0 if precision_recall_sum == 0.0 else 2.0 * precision * recall / precision_recall_sum
        )

    return {
        "dice": dice,
        "iou": iou,
        "hd": hd,
        "hd95": hd95,
        "assd": assd,
        "boundary_f1": boundary_f1,
    }


_METRIC_KEYS = ("dice", "iou", "hd", "hd95", "assd", "boundary_f1")


def compute_case_metrics_3d(
    prediction_volume: NDArray[np.int64],
    target_volume: NDArray[np.int64],
    num_classes: int,
    *,
    ignore_background: bool = True,
    boundary_tolerance: float = 2.0,
) -> dict:
    """Score one reconstructed case volume, class-mean over organs present.

    `prediction_volume`/`target_volume` are `[Z, H, W]` integer class maps
    for a single case (Z from stacking that case's 2D slice predictions at
    their true slice index; missing slices — dropped at conversion time for
    having no foreground label — are implicitly background, see module
    docstring). Returns per-class breakdown plus the class-mean used for
    the case-level score, so a caller can inspect which organs the model is
    actually failing on.
    """

    if prediction_volume.shape != target_volume.shape:
        raise ValueError(
            "prediction_volume and target_volume must have the same shape, got "
            f"{prediction_volume.shape} and {target_volume.shape}."
        )
    if prediction_volume.ndim != 3:
        raise ValueError(
            f"Expected a [Z, H, W] volume, got shape {prediction_volume.shape}."
        )

    first_class = 1 if ignore_background else 0
    per_class: dict[int, dict[str, float]] = {}
    for class_id in range(first_class, num_classes):
        prediction_mask = prediction_volume == class_id
        target_mask = target_volume == class_id
        if not prediction_mask.any() and not target_mask.any():
            continue
        per_class[class_id] = _class_metrics_3d(
            prediction_mask, target_mask, boundary_tolerance=boundary_tolerance
        )

    if not per_class:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "hd": 0.0,
            "hd95": 0.0,
            "assd": 0.0,
            "boundary_f1": 1.0,
            "num_classes_present": 0,
            "per_class": {},
        }

    means = {
        key: float(np.mean([values[key] for values in per_class.values()]))
        for key in _METRIC_KEYS
    }
    means["num_classes_present"] = len(per_class)
    means["per_class"] = {str(class_id): values for class_id, values in per_class.items()}
    return means


__all__ = ["compute_case_metrics_3d"]
