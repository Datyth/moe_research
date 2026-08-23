#!/usr/bin/env python3
"""Prepare or deterministically split the ISIC 2018 dataset manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from SegMoTE.tools.prepare_isic2018 import convert_split


DEFAULT_RAW_ROOT = PROJECT_ROOT / "SegMoTE" / "raw_data" / "isic2018"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "isic2018_task1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare ISIC 2018 and split official training records into "
            "deterministic train/validation manifests."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-images",
        type=Path,
        default=DEFAULT_RAW_ROOT / "ISIC2018_Task1-2_Training_Input",
    )
    parser.add_argument(
        "--train-masks",
        type=Path,
        default=DEFAULT_RAW_ROOT / "ISIC2018_Task1_Training_GroundTruth",
    )
    parser.add_argument(
        "--test-images",
        type=Path,
        default=DEFAULT_RAW_ROOT / "ISIC2018_Task1-2_Validation_Input",
    )
    parser.add_argument(
        "--test-masks",
        type=Path,
        default=DEFAULT_RAW_ROOT / "ISIC2018_Task1_Validation_GroundTruth",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symbolic links during conversion.",
    )
    return parser.parse_args()


def split_training_records(
    records: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return disjoint deterministic training and validation records."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1.")

    records_by_image: dict[str, dict[str, Any]] = {}
    for record in records:
        image_path = record.get("image")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("Every training record must contain an image path.")
        existing = records_by_image.get(image_path)
        if existing is not None and existing != record:
            raise ValueError(f"Conflicting duplicate record for {image_path}.")
        records_by_image[image_path] = record

    ordered_records = [records_by_image[key] for key in sorted(records_by_image)]
    if len(ordered_records) < 2:
        raise ValueError("At least two training records are required for a split.")

    shuffled_records = ordered_records.copy()
    random.Random(seed).shuffle(shuffled_records)

    validation_count = round(len(shuffled_records) * val_ratio)
    validation_count = max(1, min(validation_count, len(shuffled_records) - 1))

    validation_records = shuffled_records[:validation_count]
    training_records = shuffled_records[validation_count:]

    training_images = {record["image"] for record in training_records}
    validation_images = {record["image"] for record in validation_records}
    if training_images & validation_images:
        raise RuntimeError("Training and validation splits overlap.")
    if len(training_images | validation_images) != len(ordered_records):
        raise RuntimeError("Training split did not preserve all unique records.")

    return training_records, validation_records


def _load_or_convert_dataset(
    *,
    data_root: Path,
    train_images: Path,
    train_masks: Path,
    test_images: Path,
    test_masks: Path,
    copy_images: bool,
) -> dict[str, Any]:
    manifest_path = data_root / "dataset.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Dataset manifest must contain an object: {manifest_path}")
        return manifest

    data_root.mkdir(parents=True, exist_ok=True)
    training_records = convert_split(
        image_dir=train_images.expanduser().resolve(),
        mask_dir=train_masks.expanduser().resolve(),
        output_root=data_root,
        split_name="train",
        make_pseudo=True,
        copy_images=copy_images,
    )
    test_records = convert_split(
        image_dir=test_images.expanduser().resolve(),
        mask_dir=test_masks.expanduser().resolve(),
        output_root=data_root,
        split_name="test",
        make_pseudo=False,
        copy_images=copy_images,
    )
    return {
        "labels": {"0": "background", "1": "skin lesion"},
        "training": training_records,
        "test": test_records,
    }


def prepare_manifest(
    *,
    data_root: Path,
    val_ratio: float,
    seed: int,
    train_images: Path = DEFAULT_RAW_ROOT / "ISIC2018_Task1-2_Training_Input",
    train_masks: Path = DEFAULT_RAW_ROOT / "ISIC2018_Task1_Training_GroundTruth",
    test_images: Path = DEFAULT_RAW_ROOT / "ISIC2018_Task1-2_Validation_Input",
    test_masks: Path = DEFAULT_RAW_ROOT / "ISIC2018_Task1_Validation_GroundTruth",
    copy_images: bool = False,
) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve()
    manifest = _load_or_convert_dataset(
        data_root=data_root,
        train_images=train_images,
        train_masks=train_masks,
        test_images=test_images,
        test_masks=test_masks,
        copy_images=copy_images,
    )

    existing_training = manifest.get("training", [])
    existing_validation = manifest.get("validation", [])
    test_records = manifest.get("test")
    if not isinstance(existing_training, list):
        raise ValueError("Manifest field 'training' must be a list.")
    if not isinstance(existing_validation, list):
        raise ValueError("Manifest field 'validation' must be a list.")
    if not isinstance(test_records, list) or not test_records:
        raise ValueError("Manifest field 'test' must be a non-empty list.")

    training_records, validation_records = split_training_records(
        existing_training + existing_validation,
        val_ratio=val_ratio,
        seed=seed,
    )
    manifest["training"] = training_records
    manifest["validation"] = validation_records

    manifest_path = data_root / "dataset.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(manifest_path)

    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_manifest(
        data_root=args.data_root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_images=args.train_images,
        train_masks=args.train_masks,
        test_images=args.test_images,
        test_masks=args.test_masks,
        copy_images=args.copy_images,
    )
    print(f"Dataset manifest: {args.data_root.expanduser().resolve() / 'dataset.json'}")
    print(f"Training samples: {len(manifest['training'])}")
    print(f"Validation samples: {len(manifest['validation'])}")
    print(f"Test samples: {len(manifest['test'])}")
    print(f"Split seed: {args.seed}")


if __name__ == "__main__":
    main()
