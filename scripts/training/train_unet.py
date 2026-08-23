#!/usr/bin/env python3
"""Train the binary-segmentation UNet on ISIC 2018."""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.configs.dataset import DatasetConfig, DatasetSplitConfig
from src.data import build_dataset
from src.engine import Trainer, TrainerConfig
from src.losses import BCEDiceLoss
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train UNet on the ISIC 2018 training split.",
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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
    )
    parser.add_argument("--checkpoint-prefix", default="unet")
    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "unet_training_history.json",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_config = DatasetConfig(
        name="isic2018",
        root=args.data_root,
        task="binary",
        num_classes=1,
        in_channels=3,
        image_size=(args.image_size, args.image_size),
        splits={
            "train": DatasetSplitConfig(
                images_dir="images/train",
                masks_dir="labels/train",
            ),
            "val": DatasetSplitConfig(
                images_dir="images/train",
                masks_dir="labels/train",
            ),
        },
    )
    train_dataset = build_dataset(dataset_config, split="train")
    val_dataset = build_dataset(dataset_config, split="val")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model_config = {
        "name": "unet",
        "in_channels": 3,
        "num_classes": 1,
        "task": "binary",
        "base_channels": args.base_channels,
    }
    model = build_model(model_config)
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    last_checkpoint_path = (
        checkpoint_dir / f"{args.checkpoint_prefix}_last.pt"
    )
    best_checkpoint_path = (
        checkpoint_dir / f"{args.checkpoint_prefix}_best.pt"
    )
    checkpoint_metadata = {
        "model_config": model_config,
        "data_config": {
            "name": dataset_config.name,
            "task": dataset_config.task,
            "num_classes": dataset_config.num_classes,
            "in_channels": dataset_config.in_channels,
            "image_size": list(dataset_config.image_size),
            "image_mean": list(dataset_config.image_mean),
            "image_std": list(dataset_config.image_std),
            "mask_threshold": dataset_config.mask_threshold,
        },
        "loss_config": {
            "name": "bce_dice",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
    }

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainerConfig(
            epochs=args.epochs,
            device=args.device,
            last_checkpoint_path=last_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            history_path=args.history_path.expanduser().resolve(),
            prediction_threshold=args.prediction_threshold,
            use_amp=not args.no_amp,
            log_interval=20,
        ),
        checkpoint_metadata=checkpoint_metadata,
    )

    print("=== UNet training ===")
    print(f"Device      : {args.device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Image size  : {args.image_size}x{args.image_size}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Epochs      : {args.epochs}")
    print(f"AMP         : {not args.no_amp}")
    print(f"Last checkpoint: {last_checkpoint_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"History        : {args.history_path.expanduser().resolve()}")

    trainer.train()


if __name__ == "__main__":
    main()
