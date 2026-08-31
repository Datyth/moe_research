#!/usr/bin/env python3
"""Prepare the ACDC dataset: convert raw NIfTI patients into 2D samples
and write a frozen, patient-level train/val/test manifest.

The MoE-SAM paper follows the official ACDC protocol, but the labels of the
50 official test patients were never publicly released. Following the
standard practice of the SAM-medical literature it compares against
(MedSAM, SAMed, SAMUS), the 100 labeled patients are re-split by patient:
train / validation / test shares are controlled with --val-ratio and
--test-ratio (defaults 0.2/0.2 -> 60/20/20). No slice from a patient ever
appears in more than one split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.acdc_conversion import (
    CLASSES,
    convert_patients,
)


DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_RAW_ROOT = DEFAULT_DATASET_ROOT / "raw" / "acdc"
DEFAULT_DATA_ROOT = DEFAULT_DATASET_ROOT / "acdc"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "manifests" / "acdc_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw ACDC patient folders into 2D train/val/test samples "
            "with a deterministic patient-level split manifest."
        ),
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-foreground-pixels",
        type=int,
        default=10,
        help="Drop 2D slices with fewer labeled pixels than this value.",
    )
    return parser.parse_args()


def discover_patient_dirs(raw_root: Path) -> list[Path]:
    """Return patient directories that contain annotated ED/ES frames."""

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw ACDC root not found: {raw_root}")

    patient_dirs = [
        path
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and path.name.startswith("patient")
        and any(path.glob("*_gt.nii.gz"))
    ]
    if not patient_dirs:
        raise RuntimeError(
            f"No labeled patient folders (*_gt.nii.gz) found under {raw_root}."
        )
    return patient_dirs


def split_patients(
    patient_ids: list[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Deterministically split patient IDs into train/val/test.

    Patient-level splitting guarantees that no 2D slice from one patient
    leaks across splits. Returns disjoint, sorted ID lists per split.
    """

    for name, ratio in (("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"{name} must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be below 1.0.")

    ordered_ids = sorted(set(patient_ids))
    if len(ordered_ids) < 3:
        raise ValueError("At least three patients are required for a split.")

    shuffled_ids = ordered_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)

    validation_count = round(len(shuffled_ids) * val_ratio)
    test_count = round(len(shuffled_ids) * test_ratio)

    validation_ids = sorted(shuffled_ids[:validation_count])
    test_ids = sorted(shuffled_ids[validation_count:validation_count + test_count])
    training_ids = sorted(
        shuffled_ids[validation_count + test_count:]
    )

    if not training_ids:
        raise RuntimeError("Patient split produced an empty training set.")
    if len(training_ids) + len(validation_ids) + len(test_ids) != len(ordered_ids):
        raise RuntimeError("Patient split did not preserve all patients.")

    return {
        "training": training_ids,
        "validation": validation_ids,
        "test": test_ids,
    }


def build_manifest(
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    split_to_records: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    label_map = {str(index): name for index, name in CLASSES.items()}
    return {
        "version": "acdc-v1",
        "split_metadata": {
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_level": "patient",
            "notes": (
                "2D slices from labeled ACDC ED/ES frames; the official 50 "
                "test patients have no public labels, so the 100 labeled "
                "patients are re-split by patient."
            ),
        },
        "labels": label_map,
        "training": split_to_records["training"],
        "validation": split_to_records["validation"],
        "test": split_to_records["test"],
    }


def main() -> None:
    args = parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()

    patient_dirs = discover_patient_dirs(raw_root)
    patient_ids = [path.name for path in patient_dirs]
    print(f"Found {len(patient_ids)} labeled ACDC patients under {raw_root}")

    split_to_ids = split_patients(
        patient_ids,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    id_to_dir = {path.name: path for path in patient_dirs}

    split_to_records: dict[str, list[dict[str, str]]] = {}
    split_names = {
        "training": "train",
        "validation": "val",
        "test": "test",
    }
    for split_key, split_name in split_names.items():
        ids = split_to_ids[split_key]
        print(
            f"Converting {len(ids)} patients for split '{split_name}' "
            f"({', '.join(ids[:3])}{'...' if len(ids) > 3 else ''})"
        )
        split_to_records[split_key] = convert_patients(
            [id_to_dir[patient_id] for patient_id in ids],
            data_root,
            split_name,
            min_foreground_pixels=args.min_foreground_pixels,
        )
        print(
            f"  -> {len(split_to_records[split_key])} 2D samples written to "
            f"{data_root / 'images' / split_name}"
        )

    manifest = build_manifest(
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_to_records=split_to_records,
    )

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written to {manifest_output}")

    sizes = {
        key: len(records) for key, records in split_to_records.items()
    }
    print(
        "Sample counts: "
        + ", ".join(f"{key}={value}" for key, value in sizes.items())
    )


if __name__ == "__main__":
    main()
