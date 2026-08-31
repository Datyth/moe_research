#!/usr/bin/env python3
"""Prepare Synapse/BTCV: pair image/label volumes, split, convert to 2D slices.

Requires the raw NIfTI files under --raw-root (default
dataset/raw/synapse/RawData/Training), downloaded manually from
https://www.synapse.org/Synapse:syn3193805 (Synapse account + Data Use
Agreement required — this script only converts already-downloaded files).

BTCV's official test split has no public labels, so — exactly like
scripts/data/prepare_amos22.py — this pools every labeled case under
Training/img + Training/label and carves out a fresh case-level 70/15/15
train/val/test split, so no 2D slice from a test-split patient ever appears
in train or val.

File pairing is by numeric case id extracted from the filename (e.g.
`img0001.nii.gz` <-> `label0001.nii.gz`), not by an exact naming convention,
since Synapse downloads have used slightly different prefixes over time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.prepare_amos22 import split_cases
from scripts.data.synapse_conversion import SYNAPSE_LABELS, convert_cases, write_dataset_json


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RAW_ROOT = DEFAULT_DATASET_ROOT / "raw" / "synapse" / "RawData" / "Training"
DEFAULT_DATA_ROOT = DEFAULT_DATASET_ROOT / "synapse_btcv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifests" / "synapse_btcv_v1.json"

CASE_ID_PATTERN = re.compile(r"(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--images-subdir", default="img")
    parser.add_argument("--labels-subdir", default="label")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_case_number(path: Path) -> str:
    match = CASE_ID_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not find a numeric case id in filename: {path.name}")
    return match.group(1)


def collect_labeled_cases(raw_root: Path, *, images_subdir: str, labels_subdir: str) -> list[dict[str, Any]]:
    """Pair every image/label NIfTI file under raw_root by numeric case id."""

    images_dir = raw_root / images_subdir
    labels_dir = raw_root / labels_subdir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    labels_by_case = {
        _extract_case_number(path): path
        for path in sorted(labels_dir.glob("*.nii.gz"))
    }

    cases = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        case_number = _extract_case_number(image_path)
        label_path = labels_by_case.get(case_number)
        if label_path is None:
            raise FileNotFoundError(
                f"No matching label for {image_path} (case id {case_number}) "
                f"in {labels_dir}"
            )
        cases.append(
            {
                "case_id": f"synapse_{case_number}",
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    if not cases:
        raise RuntimeError(f"No labeled cases found under {raw_root}")
    return cases


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    cases = collect_labeled_cases(
        raw_root, images_subdir=args.images_subdir, labels_subdir=args.labels_subdir
    )
    train_cases, val_cases, test_cases = split_cases(
        cases,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(
        f"Labeled cases: {len(cases)} total -> "
        f"{len(train_cases)} train / {len(val_cases)} val / {len(test_cases)} test"
    )

    training_records = convert_cases(cases=train_cases, output_root=data_root, split_name="train")
    validation_records = convert_cases(cases=val_cases, output_root=data_root, split_name="val")
    test_records = convert_cases(cases=test_cases, output_root=data_root, split_name="test")

    manifest = {
        "labels": SYNAPSE_LABELS,
        "version": "synapse-btcv-v1",
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
