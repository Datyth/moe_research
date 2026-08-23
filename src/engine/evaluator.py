"""Evaluation utilities for binary segmentation models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


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


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(
        "Model output must be a Tensor or expose a 'logits' attribute."
    )


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a model and return sample-weighted loss, Dice and IoU."""

    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

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
    total_samples = 0

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
                    dtype=torch.float32,
                    non_blocking=True,
                )
                batch_size = images.shape[0]

                logits = _extract_logits(model(images))
                loss = criterion(logits, targets)
                if loss.ndim != 0:
                    raise ValueError(
                        "criterion must return a scalar loss, got "
                        f"shape {tuple(loss.shape)}."
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite evaluation loss detected.")

                dice, iou = compute_binary_dice_iou(
                    logits,
                    targets,
                    threshold=threshold,
                )
                if not torch.isfinite(dice).all() or not torch.isfinite(iou).all():
                    raise FloatingPointError("Non-finite evaluation metric detected.")

                total_loss += loss.item() * batch_size
                total_dice += dice.sum().item()
                total_iou += iou.sum().item()
                total_samples += batch_size
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("loader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
    }
