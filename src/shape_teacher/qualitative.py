"""Deterministic representative-mask selection for qualitative evaluation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


QUALITATIVE_CATEGORIES = (
    "small",
    "large",
    "smooth",
    "irregular",
    "difficult",
)


def mask_shape_statistics(mask: Tensor) -> dict[str, float]:
    """Measure normalized area and lattice perimeter/compactness of one mask."""

    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("mask must have shape [1, H, W].")
    binary = mask[0].detach().float().cpu().ge(0.5).float()
    height, width = binary.shape
    area = float(binary.sum())
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1), value=0.0)
    vertical = (padded[1:, :] - padded[:-1, :]).abs().sum()
    horizontal = (padded[:, 1:] - padded[:, :-1]).abs().sum()
    perimeter = float(vertical + horizontal)
    area_ratio = area / float(height * width)
    irregularity = (
        perimeter * perimeter / (4.0 * math.pi * area)
        if area > 0.0
        else float("inf")
    )
    return {
        "area_ratio": area_ratio,
        "perimeter": perimeter,
        "irregularity": irregularity,
    }


@torch.no_grad()
def collect_qualitative_records(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epsilon: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Score every test mask using deterministic clean reconstruction."""

    records: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    for batch in loader:
        targets = batch["mask"].float()
        outputs = model(targets.to(device), sample=False)
        probabilities = outputs.logits.sigmoid().detach().cpu()
        dimensions = tuple(range(1, targets.ndim))
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(
            dim=dimensions
        )
        dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
        indices = batch["dataset_index"].tolist()
        for row, dataset_index in enumerate(indices):
            statistics = mask_shape_statistics(targets[row])
            records.append(
                {
                    "dataset_index": int(dataset_index),
                    "sample_id": str(batch["sample_id"][row]),
                    "mask_path": str(batch["mask_path"][row]),
                    **statistics,
                    "clean_soft_dice": float(dice[row]),
                }
            )
    model.train(was_training)
    return records


def _ranked(
    records: Iterable[dict[str, Any]],
    *,
    metric: str,
    reverse: bool,
) -> list[dict[str, Any]]:
    direction = -1.0 if reverse else 1.0
    return sorted(
        records,
        key=lambda record: (
            direction * float(record[metric]),
            str(record["mask_path"]),
            int(record["dataset_index"]),
        ),
    )


def select_representatives(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select five categories with deterministic ties and unique rows if possible."""

    if not records:
        raise ValueError("At least one qualitative record is required.")
    criteria = (
        ("small", "area_ratio", False),
        ("large", "area_ratio", True),
        ("smooth", "irregularity", False),
        ("irregular", "irregularity", True),
        ("difficult", "clean_soft_dice", False),
    )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category, metric, reverse in criteria:
        ranking = _ranked(records, metric=metric, reverse=reverse)
        candidate = next(
            (
                record
                for record in ranking
                if int(record["dataset_index"]) not in used
            ),
            ranking[0],
        )
        output = dict(candidate)
        output["category"] = category
        output["selection_metric"] = metric
        selected.append(output)
        used.add(int(candidate["dataset_index"]))
    return selected
