"""Reproducible morphology-like binary-mask corruptions."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class CorruptionSummary:
    samples: int = 0
    changed: int = 0
    skipped_degenerate: int = 0
    operations_applied: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "corruption_changed_fraction": self.changed / self.samples
            if self.samples
            else 0.0,
            "corruption_skipped_degenerate": float(self.skipped_degenerate),
            "corruption_operations_applied": float(self.operations_applied),
        }


def _disk(radius: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    coordinates = torch.arange(-radius, radius + 1, device=device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx.square() + yy.square()) <= radius * radius).to(dtype=dtype)


def _morphology(mask: Tensor, *, radius: int, operation: str) -> Tensor:
    kernel = _disk(radius, device=mask.device, dtype=mask.dtype)
    response = F.conv2d(
        mask.unsqueeze(0),
        kernel.view(1, 1, *kernel.shape),
        padding=radius,
    ).squeeze(0)
    if operation == "dilation":
        return (response > 0).to(mask.dtype)
    if operation == "erosion":
        return (response >= kernel.sum()).to(mask.dtype)
    raise ValueError(f"Unknown morphology operation: {operation}")


def _remove_disks(
    mask: Tensor,
    *,
    rng: random.Random,
    count_range: tuple[int, int],
    radius_range: tuple[int, int],
) -> Tensor:
    result = mask.clone()
    foreground = torch.nonzero(result[0] > 0.5, as_tuple=False)
    if foreground.numel() == 0:
        return result
    height, width = result.shape[-2:]
    yy = torch.arange(height, device=result.device).view(-1, 1)
    xx = torch.arange(width, device=result.device).view(1, -1)
    for _ in range(rng.randint(*count_range)):
        point = foreground[rng.randrange(foreground.shape[0])]
        center_y, center_x = int(point[0]), int(point[1])
        radius = rng.randint(*radius_range)
        disk = (yy - center_y).square() + (xx - center_x).square() <= radius**2
        result[0, disk] = 0.0
    return result


def _affine_jitter(
    mask: Tensor,
    *,
    rng: random.Random,
    translate: float,
    scale_range: tuple[float, float],
    rotate: float,
) -> Tensor:
    height, width = mask.shape[-2:]
    angle = math.radians(rng.uniform(-rotate, rotate))
    scale = rng.uniform(*scale_range)
    tx = 2.0 * rng.uniform(-translate, translate) / max(width - 1, 1)
    ty = 2.0 * rng.uniform(-translate, translate) / max(height - 1, 1)
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    theta = mask.new_tensor(
        [[cosine, sine, -tx], [-sine, cosine, -ty]]
    ).unsqueeze(0)
    grid = F.affine_grid(theta, size=(1, 1, height, width), align_corners=False)
    transformed = F.grid_sample(
        mask.unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    return transformed.squeeze(0)


class MaskCorruptor:
    """Apply stochastic training corruption or path-keyed fixed eval corruption."""

    OPERATIONS = (
        "erosion",
        "dilation",
        "random_holes",
        "blob_removal",
        "boundary_jitter",
    )

    def __init__(
        self,
        config: dict[str, Any],
        *,
        seed: int,
        evaluation: bool = False,
    ) -> None:
        self.config = dict(config)
        self.seed = int(seed)
        self.evaluation = bool(evaluation)
        self.rng = random.Random(self.seed)
        self.summary = CorruptionSummary()

    def reset_summary(self) -> None:
        self.summary = CorruptionSummary()

    def _fixed_rng(self, split: str, key: str) -> random.Random:
        payload = f"{self.seed}\0{split}\0{key}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(value)

    def _settings(self) -> dict[str, Any]:
        key = "evaluation" if self.evaluation else "training"
        return dict(self.config.get(key, {}))

    def _apply_operation(
        self,
        mask: Tensor,
        operation: str,
        rng: random.Random,
        settings: dict[str, Any],
    ) -> Tensor:
        radius_range = tuple(settings.get("morphology_radius", [1, 3 if self.evaluation else 5]))
        if operation in {"erosion", "dilation"}:
            return _morphology(
                mask,
                radius=rng.randint(int(radius_range[0]), int(radius_range[1])),
                operation=operation,
            )
        if operation == "random_holes":
            return _remove_disks(
                mask,
                rng=rng,
                count_range=tuple(settings.get("hole_count", [1, 3 if self.evaluation else 5])),
                radius_range=tuple(settings.get("hole_radius", [2, 6 if self.evaluation else 10])),
            )
        if operation == "blob_removal":
            return _remove_disks(
                mask,
                rng=rng,
                count_range=tuple(settings.get("blob_count", [1, 2 if self.evaluation else 3])),
                radius_range=tuple(settings.get("blob_radius", [3, 7 if self.evaluation else 12])),
            )
        if operation == "boundary_jitter":
            return _affine_jitter(
                mask,
                rng=rng,
                translate=float(settings.get("translate", 2 if self.evaluation else 4)),
                scale_range=tuple(settings.get("scale", [0.98, 1.02] if self.evaluation else [0.95, 1.05])),
                rotate=float(settings.get("rotate", 5 if self.evaluation else 10)),
            )
        raise ValueError(f"Unknown corruption operation: {operation}")

    def corrupt_one(
        self,
        mask: Tensor,
        *,
        key: str = "",
        split: str = "train",
    ) -> Tensor:
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError("mask must have shape [1, H, W].")
        rng = self._fixed_rng(split, key) if self.evaluation else self.rng
        settings = self._settings()
        probabilities = dict(
            settings.get(
                "probabilities",
                {
                    "erosion": 0.35,
                    "dilation": 0.35,
                    "random_holes": 0.25,
                    "blob_removal": 0.20,
                    "boundary_jitter": 0.25,
                },
            )
        )
        minimum, maximum = settings.get(
            "operation_count", [1, 2 if self.evaluation else 3]
        )
        selected = rng.sample(
            list(self.OPERATIONS),
            k=rng.randint(int(minimum), int(maximum)),
        )
        original = (mask > 0.5).to(dtype=mask.dtype)
        result = original.clone()
        was_nonempty = bool(original.any())
        for operation in selected:
            if rng.random() >= float(probabilities.get(operation, 0.0)):
                continue
            candidate = self._apply_operation(result, operation, rng, settings)
            candidate = (candidate.clamp(0, 1) > 0.5).to(result.dtype)
            if was_nonempty and (not bool(candidate.any()) or bool(candidate.all())):
                self.summary.skipped_degenerate += 1
                continue
            result = candidate
            self.summary.operations_applied += 1
        self.summary.samples += 1
        self.summary.changed += int(not torch.equal(result, original))
        return result

    def __call__(
        self,
        masks: Tensor,
        *,
        keys: list[str] | tuple[str, ...] | None = None,
        split: str = "train",
    ) -> Tensor:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError("masks must have shape [B, 1, H, W].")
        if keys is None:
            keys = [str(index) for index in range(masks.shape[0])]
        if len(keys) != masks.shape[0]:
            raise ValueError("keys must contain one value per mask.")
        return torch.stack(
            [
                self.corrupt_one(mask, key=str(key), split=split)
                for mask, key in zip(masks, keys)
            ]
        )
