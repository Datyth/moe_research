"""Surface metrics for binary 2D segmentation masks."""

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
) -> tuple[float, float, float]:
    prediction_nonempty = bool(prediction.any())
    target_nonempty = bool(target.any())
    if not prediction_nonempty and not target_nonempty:
        return 0.0, 0.0, 1.0
    if prediction_nonempty != target_nonempty:
        height, width = prediction.shape
        maximum_distance = math.hypot(height - 1, width - 1)
        return maximum_distance, maximum_distance, 0.0

    prediction_surface = extract_binary_surface(prediction)
    target_surface = extract_binary_surface(target)
    prediction_to_target, target_to_prediction = _directed_surface_distances(
        prediction_surface,
        target_surface,
    )
    combined_distances = np.concatenate(
        (prediction_to_target, target_to_prediction)
    )
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
    return hd95, assd, boundary_f1


def compute_binary_surface_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-sample HD95, ASSD and Boundary F1 for binary logits.

    HD95 is the 95th percentile of the concatenated prediction-to-target and
    target-to-prediction nearest-surface distances. ASSD is their summed distance
    divided by the total number of surface pixels. Both are Euclidean distances
    in pixels. Boundary F1 matches a surface pixel when its Euclidean distance to
    the other surface is less than or equal to ``boundary_tolerance`` pixels.

    Two empty masks return ``(0, 0, 1)``. If exactly one mask is empty, HD95 and
    ASSD use the finite image-diagonal penalty and Boundary F1 is zero.
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
    return metrics[:, 0], metrics[:, 1], metrics[:, 2]


def compute_binary_hd95_assd(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute per-sample symmetric HD95 and ASSD in pixel units."""

    hd95, assd, _ = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
    )
    return hd95, assd


def compute_binary_boundary_f1(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
) -> Tensor:
    """Compute per-sample boundary F1 within an inclusive pixel tolerance."""

    _, _, boundary_f1 = compute_binary_surface_metrics(
        logits,
        targets,
        threshold=threshold,
        boundary_tolerance=boundary_tolerance,
    )
    return boundary_f1
