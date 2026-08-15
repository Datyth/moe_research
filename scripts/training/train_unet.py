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
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "unet_initial.pt",
    )
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
        },
    )
    train_dataset = build_dataset(dataset_config, split="train")

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

    model = build_model({
        "name": "unet",
        "in_channels": 3,
        "num_classes": 1,
        "task": "binary",
        "base_channels": args.base_channels,
    })
    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        config=TrainerConfig(
            epochs=args.epochs,
            device=args.device,
            checkpoint_path=args.checkpoint,
            use_amp=not args.no_amp,
            log_interval=20,
        ),
    )

    print("=== UNet training ===")
    print(f"Device      : {args.device}")
    print(f"Samples     : {len(train_dataset)}")
    print(f"Batches     : {len(train_loader)}")
    print(f"Image size  : {args.image_size}x{args.image_size}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Epochs      : {args.epochs}")
    print(f"AMP         : {not args.no_amp}")
    print(f"Checkpoint  : {args.checkpoint}")

    trainer.train()


if __name__ == "__main__":
    main()
