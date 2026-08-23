#!/usr/bin/env python3
"""Evaluate a binary segmentation checkpoint and save visualizations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs.dataset import DatasetConfig, DatasetSplitConfig
from src.data import build_dataset
from src.engine import evaluate
from src.losses import BCEDiceLoss
from src.models import build_model


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "name": "unet",
    "in_channels": 3,
    "num_classes": 1,
    "task": "binary",
    "base_channels": 32,
}
DEFAULT_DATA_CONFIG: dict[str, Any] = {
    "name": "isic2018",
    "task": "binary",
    "num_classes": 1,
    "in_channels": 3,
    "image_size": [256, 256],
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "mask_threshold": 0.5,
}
DEFAULT_LOSS_CONFIG: dict[str, Any] = {
    "name": "bce_dice",
    "bce_weight": 0.5,
    "dice_weight": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a UNet checkpoint on ISIC 2018.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "isic2018_task1",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_best.pt",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "unet_test",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-visualizations", type=int, default=12)
    return parser.parse_args()


def _checkpoint_configs(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    model_config = DEFAULT_MODEL_CONFIG.copy()
    saved_model_config = metadata.get("model_config")
    if isinstance(saved_model_config, dict):
        model_config.update(saved_model_config)
    if args.base_channels is not None:
        model_config["base_channels"] = args.base_channels

    data_config = DEFAULT_DATA_CONFIG.copy()
    saved_data_config = metadata.get("data_config")
    if isinstance(saved_data_config, dict):
        data_config.update(saved_data_config)
    if args.image_size is not None:
        data_config["image_size"] = [args.image_size, args.image_size]

    loss_config = DEFAULT_LOSS_CONFIG.copy()
    saved_loss_config = metadata.get("loss_config")
    if isinstance(saved_loss_config, dict):
        loss_config.update(saved_loss_config)
    if loss_config.get("name") != "bce_dice":
        raise ValueError(
            f"Unsupported loss configuration: {loss_config.get('name')!r}."
        )

    saved_trainer_config = checkpoint.get("trainer_config")
    saved_threshold = None
    if isinstance(saved_trainer_config, dict):
        saved_threshold = saved_trainer_config.get("prediction_threshold")
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(saved_threshold if saved_threshold is not None else 0.5)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    return model_config, data_config, loss_config, threshold


def _build_dataset_config(
    *,
    data_root: Path,
    data_config: dict[str, Any],
) -> DatasetConfig:
    return DatasetConfig(
        name=str(data_config["name"]),
        root=data_root,
        task=str(data_config["task"]),
        num_classes=int(data_config["num_classes"]),
        in_channels=int(data_config["in_channels"]),
        image_size=tuple(int(value) for value in data_config["image_size"]),
        image_mean=tuple(float(value) for value in data_config["image_mean"]),
        image_std=tuple(float(value) for value in data_config["image_std"]),
        mask_threshold=float(data_config["mask_threshold"]),
        splits={
            "val": DatasetSplitConfig("images/train", "labels/train"),
            "test": DatasetSplitConfig("images/test", "labels/test"),
        },
    )


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(
        "Model output must be a Tensor or expose a 'logits' attribute."
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _denormalize_image(
    image: Tensor,
    *,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    image = image.detach().cpu() * std_tensor + mean_tensor
    return image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a visible two-pixel boundary for a binary mask."""

    binary_mask = np.asarray(mask >= 0.5, dtype=bool)
    if not binary_mask.any():
        return np.zeros_like(binary_mask)
    eroded = binary_erosion(binary_mask, iterations=1, border_value=0)
    one_pixel_boundary = binary_mask & ~eroded
    return binary_dilation(one_pixel_boundary, iterations=1)


def _build_boundary_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """Overlay GT (green), prediction (red) and shared boundary (yellow)."""

    overlay = image.copy()
    ground_truth_boundary = _mask_boundary(ground_truth)
    prediction_boundary = _mask_boundary(prediction)
    shared_boundary = ground_truth_boundary & prediction_boundary

    overlay[ground_truth_boundary] = np.array([0.0, 1.0, 0.0])
    overlay[prediction_boundary] = np.array([1.0, 0.0, 0.0])
    overlay[shared_boundary] = np.array([1.0, 1.0, 0.0])
    return overlay


def save_visualizations(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    output_dir: Path,
    limit: int,
) -> None:
    if limit < 0:
        raise ValueError("num_visualizations must be non-negative.")
    if limit == 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            logits = _extract_logits(model(images))
            predictions = (torch.sigmoid(logits) >= threshold).float().cpu()

            for index in range(images.shape[0]):
                image = _denormalize_image(
                    images[index],
                    mean=image_mean,
                    std=image_std,
                )
                ground_truth = batch["mask"][index, 0].cpu().numpy()
                prediction = predictions[index, 0].numpy()
                overlay = _build_boundary_overlay(
                    image,
                    ground_truth,
                    prediction,
                )

                figure, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(image)
                axes[0].set_title("Input")
                axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Prediction")
                axes[3].imshow(overlay)
                axes[3].set_title(
                    "Boundary Overlay\nGT: green | Pred: red | Shared: yellow"
                )
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout()

                sample_id = str(batch["sample_id"][index])
                safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
                figure.savefig(
                    output_dir / f"{safe_sample_id}.png",
                    dpi=140,
                    bbox_inches="tight",
                )
                plt.close(figure)

                saved += 1
                if saved >= limit:
                    return


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.num_visualizations < 0:
        raise ValueError("num_visualizations must be non-negative.")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")

    model_config, data_config, loss_config, threshold = _checkpoint_configs(
        checkpoint,
        args,
    )
    dataset_config = _build_dataset_config(
        data_root=args.data_root.expanduser().resolve(),
        data_config=data_config,
    )
    dataset = build_dataset(dataset_config, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = build_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
    )

    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        threshold=threshold,
    )
    output_dir = args.output_dir.expanduser().resolve()
    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "loss": metrics["loss"],
        "dice": metrics["dice"],
        "iou": metrics["iou"],
    }
    _save_json(output_dir / "metrics.json", metrics_payload)

    save_visualizations(
        model=model,
        loader=loader,
        device=device,
        threshold=threshold,
        image_mean=dataset_config.image_mean,
        image_std=dataset_config.image_std,
        output_dir=output_dir / "visualizations",
        limit=args.num_visualizations,
    )

    split_label = "Validation" if args.split == "val" else "Test"
    print(f"{split_label} Loss : {metrics['loss']:.6f}")
    print(f"{split_label} Dice : {metrics['dice']:.6f}")
    print(f"{split_label} IoU  : {metrics['iou']:.6f}")
    print(f"Metrics   : {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
