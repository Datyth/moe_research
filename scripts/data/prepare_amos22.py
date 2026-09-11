#!/usr/bin/env python3
"""Prepare AMOS22 CT: filter CT-only cases, split, convert to 2D slices, write manifest.

AMOS22's official test split has no public labels (Zenodo record 7262581),
so it can't be used for a scored ablation. Instead, this pools every labeled
CT case (imagesTr + imagesVa, case id <= 500 — ids 501-600 are MRI) and
carves out a fresh case-level 70/15/15 train/val/test split, so no 2D slice
from a test-split patient ever appears in train or val.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.amos22_conversion import AMOS_LABELS, convert_cases, write_dataset_json


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RAW_ROOT = DEFAULT_DATASET_ROOT / "raw" / "amos22" / "amos22"
DEFAULT_DATA_ROOT = DEFAULT_DATASET_ROOT / "amos22_ct"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifests" / "amos22_ct_v1.json"

CT_MAX_CASE_ID = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _case_id_from_filename(path: Path) -> int:
    # amos_0001.nii.gz -> 1
    stem = path.name.split(".")[0]
    return int(stem.split("_")[1])


def collect_labeled_ct_cases(raw_root: Path) -> list[dict[str, Any]]:
    """List every CT case (id <= 500) with both an image and a label file."""

    cases = []
    for images_dir, labels_dir in (
        (raw_root / "imagesTr", raw_root / "labelsTr"),
        (raw_root / "imagesVa", raw_root / "labelsVa"),
    ):
        for image_path in sorted(images_dir.glob("amos_*.nii.gz")):
            case_numeric_id = _case_id_from_filename(image_path)
            if case_numeric_id > CT_MAX_CASE_ID:
                continue
            label_path = labels_dir / image_path.name
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            cases.append(
                {
                    "case_id": f"amos_{case_numeric_id:04d}",
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

    if not cases:
        raise RuntimeError(f"No labeled CT cases found under {raw_root}")
    return cases


def split_cases(
    cases: list[dict[str, Any]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio and test_ratio must be strictly between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    ordered_cases = sorted(cases, key=lambda case: case["case_id"])
    shuffled_cases = ordered_cases.copy()
    random.Random(seed).shuffle(shuffled_cases)

    total = len(shuffled_cases)
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))

    test_cases = shuffled_cases[:test_count]
    val_cases = shuffled_cases[test_count : test_count + val_count]
    train_cases = shuffled_cases[test_count + val_count :]

    if not train_cases:
        raise RuntimeError("Split produced an empty training set.")
    return train_cases, val_cases, test_cases


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    cases = collect_labeled_ct_cases(raw_root)
    train_cases, val_cases, test_cases = split_cases(
        cases,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"Labeled CT cases: {len(cases)} total -> "
        f"{len(train_cases)} train / {len(val_cases)} val / {len(test_cases)} test"
    )

    training_records = convert_cases(cases=train_cases, output_root=data_root, split_name="train")
    validation_records = convert_cases(cases=val_cases, output_root=data_root, split_name="val")
    test_records = convert_cases(cases=test_cases, output_root=data_root, split_name="test")

    manifest = {
        "labels": AMOS_LABELS,
        "version": "amos22-ct-v1",
        "split_metadata": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "case_count": {
                "train": len(train_cases),
                "validation": len(val_cases),
                "test": len(test_cases),
            },
        },
        "training": training_records,
        "validation": validation_records,
        "test": test_records,
    }

    dataset_json_path = write_dataset_json(data_root, manifest)
    print(f"Dataset JSON: {dataset_json_path}")

    manifest_output = args.manifest_output.expanduser().resolve()
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = manifest_output.with_suffix(".json.tmp")
    with temporary_output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_output.replace(manifest_output)

    print(f"Tracked manifest: {manifest_output}")
    print(f"Training slices: {len(training_records)}")
    print(f"Validation slices: {len(validation_records)}")
    print(f"Test slices: {len(test_records)}")


if __name__ == "__main__":
    main()
