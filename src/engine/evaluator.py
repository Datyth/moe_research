"""Evaluation utilities for binary segmentation models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.metrics import (
    compute_binary_boundary_f1,
    compute_binary_hd,
    compute_binary_hd95_assd,
    compute_binary_surface_distances,
    compute_binary_surface_metrics,
    compute_multiclass_dice_iou,
    compute_multiclass_surface_metrics,
    extract_binary_surface,
)
from src.models import SegmentationOutput


__all__ = [
    "compute_binary_boundary_f1",
    "compute_binary_dice_iou",
    "compute_binary_hd",
    "compute_binary_hd95_assd",
    "compute_binary_surface_distances",
    "compute_binary_surface_metrics",
    "evaluate",
    "extract_binary_surface",
]


def compute_binary_dice_iou(
    logits: Tensor,
    targets: Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute per-sample Dice and IoU from binary-segmentation logits."""

    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have the same shape, got "
            f"{tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            "Binary metrics require logits and targets with shape [B, 1, H, W]."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    predictions = torch.sigmoid(logits) >= threshold
    target_masks = targets >= 0.5
    predictions = predictions.flatten(start_dim=1)
    target_masks = target_masks.flatten(start_dim=1)

    intersection = torch.logical_and(predictions, target_masks).sum(dim=1).float()
    prediction_size = predictions.sum(dim=1).float()
    target_size = target_masks.sum(dim=1).float()

    dice_denominator = prediction_size + target_size
    union = prediction_size + target_size - intersection
    ones = torch.ones_like(intersection)

    dice = torch.where(
        dice_denominator == 0,
        ones,
        2.0 * intersection / dice_denominator,
    )
    iou = torch.where(union == 0, ones, intersection / union)
    return dice, iou

def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
    threshold: float = 0.5,
    boundary_tolerance: float = 2,
    task: str = "binary",
) -> dict[str, float]:
    """Return sample-mean loss, region metrics (Dice, IoU) and surface
    metrics (HD95, ASSD, boundary F1) via src.metrics. For `task="multiclass"`,
    all metrics are class-mean (background excluded, absent classes skipped)."""

    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'.")
    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if isinstance(boundary_tolerance, bool):
        raise ValueError("boundary_tolerance must be a non-negative number.")
    try:
        resolved_boundary_tolerance = float(boundary_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "boundary_tolerance must be a non-negative number."
        ) from error
    if (
        not math.isfinite(resolved_boundary_tolerance)
        or resolved_boundary_tolerance < 0.0
    ):
        raise ValueError("boundary_tolerance must be a non-negative number.")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    was_training = model.training
    model.to(resolved_device)
    criterion.to(resolved_device)
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_hd = 0.0
    total_hd95 = 0.0
    total_assd = 0.0
    total_boundary_f1 = 0.0
    total_samples = 0
    target_dtype = torch.long if task == "multiclass" else torch.float32

    try:
        with torch.inference_mode():
            for batch in loader:
                if "image" not in batch or "mask" not in batch:
                    raise KeyError(
                        "Each evaluation batch must contain 'image' and 'mask'."
                    )
                images = batch["image"].to(
                    resolved_device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                targets = batch["mask"].to(
                    resolved_device,
                    dtype=target_dtype,
                    non_blocking=True,
                )
                batch_size = images.shape[0]

                output = model(images)
                if not isinstance(output, SegmentationOutput):
                    raise TypeError(
                        "Model forward must return SegmentationOutput, got "
                        f"{type(output).__name__}."
                    )
                logits = output.logits
                loss = criterion(logits, targets)
                if loss.ndim != 0:
                    raise ValueError(
                        "criterion must return a scalar loss, got "
                        f"shape {tuple(loss.shape)}."
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite evaluation loss detected.")

                if task == "multiclass":
                    dice, iou = compute_multiclass_dice_iou(logits, targets)
                    max_hd, hd95, assd, boundary_f1 = compute_multiclass_surface_metrics(
                        logits,
                        targets,
                        boundary_tolerance=resolved_boundary_tolerance,
                    )
                else:
                    dice, iou = compute_binary_dice_iou(
                        logits,
                        targets,
                        threshold=threshold,
                    )
                    max_hd, hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                        logits,
                        targets,
                        threshold=threshold,
                        boundary_tolerance=resolved_boundary_tolerance,
                    )
                if not torch.isfinite(dice).all() or not torch.isfinite(iou).all():
                    raise FloatingPointError("Non-finite evaluation metric detected.")
                if not (
                    torch.isfinite(max_hd).all()
                    and torch.isfinite(hd95).all()
                    and torch.isfinite(assd).all()
                    and torch.isfinite(boundary_f1).all()
                ):
                    raise FloatingPointError("Non-finite evaluation metric detected.")

                total_loss += loss.item() * batch_size
                total_dice += dice.sum().item()
                total_iou += iou.sum().item()
                total_hd += max_hd.sum().item()
                total_hd95 += hd95.sum().item()
                total_assd += assd.sum().item()
                total_boundary_f1 += boundary_f1.sum().item()
                total_samples += batch_size
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("loader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
        "hd": total_hd / total_samples,
        "hd95": total_hd95 / total_samples,
        "assd": total_assd / total_samples,
        "boundary_f1": total_boundary_f1 / total_samples,
    }
