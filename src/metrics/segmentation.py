"""Region and surface metrics for binary 2D segmentation masks."""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import Tensor


BinaryArray = NDArray[np.bool_]


def _validate_metric_inputs(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float,
) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "Binary metrics require logits and targets with shape [B, 1, H, W]."
        )
    if logits.shape[0] == 0 or logits.shape[2] == 0 or logits.shape[3] == 0:
        raise ValueError("Binary metrics require non-empty batch and spatial dimensions.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")


def _validate_boundary_tolerance(boundary_tolerance: float) -> float:
    if isinstance(boundary_tolerance, bool):
        raise ValueError("boundary_tolerance must be a non-negative number.")
    try:
        tolerance = float(boundary_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "boundary_tolerance must be a non-negative number."
        ) from error
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("boundary_tolerance must be a non-negative number.")
    return tolerance


def _as_binary_2d_mask(mask: ArrayLike, *, name: str) -> BinaryArray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty spatial dimensions.")
    return np.asarray(array, dtype=bool)


def extract_binary_surface(mask: ArrayLike) -> BinaryArray:
    """Extract the one-pixel inner surface of a 2D binary mask.

    The surface is ``mask & ~binary_erosion(mask)`` using 4-connectivity. Pixels
    outside the image are background, so foreground touching an image edge is
    included in the surface.
    """

    binary_mask = _as_binary_2d_mask(mask, name="mask")
    if not binary_mask.any():
        return np.zeros_like(binary_mask)
    eroded = binary_erosion(binary_mask, iterations=1, border_value=0)
    return np.asarray(binary_mask & ~eroded, dtype=bool)


def _directed_surface_distances(
    prediction_surface: BinaryArray,
    target_surface: BinaryArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prediction_to_target = distance_transform_edt(~target_surface)[
        prediction_surface
    ]
    target_to_prediction = distance_transform_edt(~prediction_surface)[
        target_surface
    ]
    return (
        np.asarray(prediction_to_target, dtype=np.float64),
        np.asarray(target_to_prediction, dtype=np.float64),
    )


def compute_binary_surface_distances(
    prediction: ArrayLike,
    target: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Euclidean pixel distances from prediction surface to target and back.

    The returned arrays contain one nearest-neighbor distance per surface pixel.
    If either mask is empty, both arrays are empty; public aggregate metrics apply
    the documented finite image-diagonal penalty for that case.
    """

    prediction_mask = _as_binary_2d_mask(prediction, name="prediction")
    target_mask = _as_binary_2d_mask(target, name="target")
    if prediction_mask.shape != target_mask.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{prediction_mask.shape} and {target_mask.shape}."
        )
    if not prediction_mask.any() or not target_mask.any():
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy()
    return _directed_surface_distances(
        extract_binary_surface(prediction_mask),
        extract_binary_surface(target_mask),
    )


def _sample_surface_metrics(
    prediction: BinaryArray,
    target: BinaryArray,
    *,
    boundary_tolerance: float,
) -> tuple[float, float, float, float]:
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not prediction_nonempty and not target_nonempty:
        return 0.0, 0.0, 0.0, 1.0
    if prediction_nonempty != target_nonempty:
        height, width = prediction.shape
        maximum_distance = math.hypot(height - 1, width - 1)
        return maximum_distance, maximum_distance, maximum_distance, 0.0

    prediction_surface = extract_binary_surface(prediction)
    target_surface = extract_binary_surface(target)
    prediction_to_target, target_to_prediction = _directed_surface_distances(
        prediction_surface,
        target_surface,
    )
    combined_distances = np.concatenate(
        (prediction_to_target, target_to_prediction)
    )
    max_hd = float(combined_distances.max())
    hd95 = float(np.percentile(combined_distances, 95))
    assd = float(
        (prediction_to_target.sum() + target_to_prediction.sum())
        / (prediction_to_target.size + target_to_prediction.size)
    )
    boundary_precision = float(
        np.mean(prediction_to_target <= boundary_tolerance)
    )
    boundary_recall = float(
        np.mean(target_to_prediction <= boundary_tolerance)
    )
    precision_recall_sum = boundary_precision + boundary_recall
    boundary_f1 = (
        0.0
        if precision_recall_sum == 0.0
        else 2.0
        * boundary_precision
        * boundary_recall
        / precision_recall_sum
    )
    return max_hd, hd95, assd, boundary_f1


def compute_binary_surface_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute per-sample HD, HD95, ASSD and Boundary F1 for binary logits.

    HD is the maximum of the concatenated prediction-to-target and
    target-to-prediction nearest-surface distances (classic Hausdorff
    distance). HD95 is the 95th percentile of the same distance set. ASSD is
    the summed distance divided by the total number of surface pixels. All
    three are Euclidean distances in pixels. Boundary F1 matches a surface
    pixel when its Euclidean distance to the other surface is less than or
    equal to ``boundary_tolerance`` pixels.

    Two empty masks return ``(0, 0, 0, 1)``. If exactly one mask is empty, HD,
    HD95 and ASSD use the finite image-diagonal penalty and Boundary F1 is
    zero.
    """

    _validate_metric_inputs(logits, targets, threshold=threshold)
    tolerance = _validate_boundary_tolerance(boundary_tolerance)
    predictions = (
        torch.sigmoid(logits).ge(threshold).detach().cpu().numpy()[:, 0]
    )
    target_masks = targets.ge(0.5).detach().cpu().numpy()[:, 0]

    values = [
        _sample_surface_metrics(
            np.asarray(prediction, dtype=bool),
            np.asarray(target, dtype=bool),
            boundary_tolerance=tolerance,
        )
        for prediction, target in zip(predictions, target_masks)
    ]
    metrics = torch.tensor(values, dtype=torch.float64)
    return metrics[:, 0], metrics[:, 1], metrics[:, 2], metrics[:, 3]


def compute_binary_hd(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """Compute per-sample maximum Hausdorff distance in pixel units."""

    max_hd, _, _, _ = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
    )
    return max_hd


def compute_binary_hd95_assd(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute per-sample symmetric HD95 and ASSD in pixel units."""

    _, hd95, assd, _ = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
    )
    return hd95, assd


def compute_multiclass_dice_iou(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute per-sample mean Dice and IoU over foreground classes.

    ``logits`` must have shape ``[B, C, H, W]`` with ``C > 1`` and ``targets``
    must contain integer class indices with shape ``[B, 1, H, W]`` or
    ``[B, H, W]``. Predictions are taken as the argmax over the class
    dimension. Metrics are computed independently for every foreground class
    (all classes except background ``0``) and then averaged per sample. A
    class that is absent in both prediction and target contributes a perfect
    score, mirroring the binary implementation.
    """

    predictions, labels, num_classes = _resolve_multiclass(logits, targets)

    dice_scores = []
    iou_scores = []
    ones = torch.ones(
        predictions.shape[0],
        dtype=torch.float32,
        device=predictions.device,
    )
    for class_index in range(1, num_classes):
        prediction_masks = (predictions == class_index).flatten(start_dim=1)
        target_masks = (labels == class_index).flatten(start_dim=1)

        intersection = torch.logical_and(
            prediction_masks,
            target_masks,
        ).sum(dim=1).float()
        prediction_size = prediction_masks.sum(dim=1).float()
        target_size = target_masks.sum(dim=1).float()

        dice_denominator = prediction_size + target_size
        union = prediction_size + target_size - intersection

        dice_scores.append(
            torch.where(dice_denominator == 0, ones, 2.0 * intersection / dice_denominator)
        )
        iou_scores.append(
            torch.where(union == 0, ones, intersection / union)
        )

    dice = torch.stack(dice_scores, dim=1).mean(dim=1)
    iou = torch.stack(iou_scores, dim=1).mean(dim=1)
    return dice, iou


def compute_multiclass_surface_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    boundary_tolerance: float = 2,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute per-sample HD, HD95, ASSD and Boundary F1 for multiclass logits.

    Surface metrics are computed per foreground class with the same rules as
    the binary implementation (finite image-diagonal penalty when exactly one
    of the two masks is empty) and then averaged per sample over all
    foreground classes.
    """

    predictions, labels, num_classes = _resolve_multiclass(logits, targets)
    tolerance = _validate_boundary_tolerance(boundary_tolerance)

    predictions_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    values = []
    for sample_predictions, sample_labels in zip(predictions_np, labels_np):
        class_values = [
            _sample_surface_metrics(
                np.asarray(sample_predictions == class_index, dtype=bool),
                np.asarray(sample_labels == class_index, dtype=bool),
                boundary_tolerance=tolerance,
            )
            for class_index in range(1, num_classes)
        ]
        values.append(np.mean(class_values, axis=0))

    metrics = torch.tensor(np.asarray(values), dtype=torch.float64)
    return metrics[:, 0], metrics[:, 1], metrics[:, 2], metrics[:, 3]


def _resolve_multiclass(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Validate multiclass inputs and return (predictions, labels, C).

    Class-index targets arrive as `[B, H, W]` (what `src/data/ct_slice.py`,
    `src/data/acdc.py` and `src/metrics/multiclass.py` use); a redundant
    single-channel `[B, 1, H, W]` layout is accepted and squeezed.
    """

    if logits.ndim != 4 or logits.shape[1] <= 1:
        raise ValueError(
            "Multiclass metrics require logits with shape [B, C, H, W] and C > 1."
        )
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    if (
        targets.ndim != 3
        or targets.shape[0] != logits.shape[0]
        or targets.shape[1:] != logits.shape[2:]
    ):
        raise ValueError(
            "targets must be class indices with shape [B, H, W] or [B, 1, H, W] "
            f"matching logits spatial size, got {tuple(targets.shape)} and "
            f"{tuple(logits.shape)}."
        )

    num_classes = int(logits.shape[1])
    predictions = logits.argmax(dim=1)
    labels = targets.long()
    if labels.shape != predictions.shape:
        raise ValueError(
            f"targets shape {tuple(labels.shape)} does not match prediction "
            f"shape {tuple(predictions.shape)}."
        )
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"targets contain class indices outside [0, {num_classes - 1}]."
        )
    return predictions, labels, num_classes


def compute_binary_boundary_f1(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
) -> Tensor:
    """Compute per-sample boundary F1 within an inclusive pixel tolerance."""

    _, _, _, boundary_f1 = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
    )
    return boundary_f1
