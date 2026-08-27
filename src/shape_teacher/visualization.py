"""Qualitative grids for Shape Teacher corruption and reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def save_reconstruction_grid(
    path: str | Path,
    *,
    targets: Tensor,
    inputs: Tensor,
    logits: Tensor,
    threshold: float = 0.5,
    max_samples: int = 8,
    row_labels: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Save target, teacher input, probability and thresholded output columns."""

    count = min(int(targets.shape[0]), int(max_samples))
    if count <= 0:
        raise ValueError("At least one sample is required for visualization.")
    if row_labels is not None and len(row_labels) < count:
        raise ValueError("row_labels must contain one label per displayed sample.")
    probabilities = logits.sigmoid().detach().cpu()
    targets = targets.detach().cpu()
    inputs = inputs.detach().cpu()
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    titles = (
        "Clean target",
        "Teacher input",
        "Reconstructed probability",
        "Thresholded reconstruction",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row in range(count):
        images = (
            targets[row, 0],
            inputs[row, 0],
            probabilities[row, 0],
            probabilities[row, 0].ge(threshold),
        )
        for column, image in enumerate(images):
            axes[row, column].imshow(image.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        if row_labels is not None:
            axes[row, 0].set_ylabel(str(row_labels[row]), rotation=90)
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_corruption_grid(
    path: str | Path,
    *,
    targets: Tensor,
    corrupted: Tensor,
    max_samples: int = 8,
) -> None:
    """Save a compact proof that denoising inputs differ from clean targets."""

    count = min(int(targets.shape[0]), int(max_samples))
    figure, axes = plt.subplots(count, 3, figsize=(9, 3 * count), squeeze=False)
    axes[0, 0].set_title("Clean target")
    axes[0, 1].set_title("Corrupted input")
    axes[0, 2].set_title("Changed pixels")
    for row in range(count):
        difference = targets[row, 0].ne(corrupted[row, 0])
        for column, image in enumerate(
            (targets[row, 0], corrupted[row, 0], difference)
        ):
            axes[row, column].imshow(image.detach().cpu().numpy(), cmap="gray")
            axes[row, column].axis("off")
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
