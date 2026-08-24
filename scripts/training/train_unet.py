#!/usr/bin/env python3
"""Deprecated UNet CLI backed by the generic experiment runner."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import resolve_experiment_config
from src.experiment import execute_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated: use scripts/run_experiment.py --config instead.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "isic2018_task1",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-prefix", default="unet")
    parser.add_argument("--history-path", type=Path)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.warn(
        "scripts/training/train_unet.py is deprecated; use "
        "scripts/run_experiment.py --config configs/unet.yaml. "
        "Checkpoint/history path options no longer control individual files.",
        DeprecationWarning,
        stacklevel=2,
    )
    if args.checkpoint_dir is not None or args.history_path is not None:
        print(
            "Warning: legacy output paths are ignored; all artifacts are saved "
            "under the standardized run folder."
        )

    raw_config = {
        "experiment": {
            "name": args.checkpoint_prefix,
            "output_root": "runs",
        },
        "seed": args.seed,
        "dataset": {
            "name": "isic2018",
            "root": str(args.data_root),
            "manifest": "manifests/isic2018_task1_v1.json",
            "version": "isic2018-task1-v1",
            "task": "binary",
            "num_classes": 1,
            "in_channels": 3,
            "image_size": [args.image_size, args.image_size],
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "mask_threshold": 0.5,
        },
        "model": {
            "name": "unet",
            "base_channels": args.base_channels,
        },
        "loss": {
            "name": "bce_dice",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "optimizer": {
            "name": "adamw",
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {"name": "none"},
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "amp": not args.no_amp,
            "prediction_threshold": args.prediction_threshold,
            "log_interval": 20,
            "gradient_clip_norm": None,
        },
    }
    config = resolve_experiment_config(raw_config, project_root=PROJECT_ROOT)
    execute_experiment(config)


if __name__ == "__main__":
    main()
