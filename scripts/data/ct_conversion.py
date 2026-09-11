#!/usr/bin/env python3
"""Shared NIfTI CT volume -> 2D axial slice conversion, used by AMOS22 and
Synapse/BTCV (scripts/data/amos22_conversion.py, synapse_conversion.py).

Mirrors scripts/data/isic2018_conversion.py's on-disk convention: each slice
is stored at native resolution as a PNG image plus a sparse-encoded label
(`sparse.csr_matrix`, works for any integer label range, not just binary).
Resizing to a model's input resolution happens later, in
src/data/transforms.py, exactly like ISIC2018.

2D slicing is required here, not just a preprocessing choice: `esam` wraps
SAM, whose ViT backbone only processes 2D images (this is why every
SAM-adapter paper for medical imaging slices 3D volumes the same way).

CT-specific steps this repo's other converters don't need:
  - HU windowing: clip to a dataset-specific window then normalize to [0, 1].
    The window is NOT the same for every CT benchmark, so callers pass it in
    (see `HU_WINDOW` in each dataset's conversion module). MoE-SAM's authors
    confirmed by email (2026-09-07) that they used [-150, 500] for BTCV and
    [-125, 275] for the 8-organ Synapse benchmark; [-125, 275] is also the
    TransUNet/SAMed convention this repo previously applied to both.
  - Slices with no foreground label at all are dropped for every split
    (train/val/test alike) — most axial slices in an abdominal CT are pure
    air/background outside the labeled organs, and keeping them would both
    waste compute and bias 2D per-slice Dice toward trivial all-background
    predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import sparse


# Fallback window, kept as the historical default so AMOS22 (which the
# MoE-SAM paper does not cover) keeps the abdominal soft-tissue window this
# repo already converted it with.
HU_WINDOW_MIN = -125.0
HU_WINDOW_MAX = 275.0


def window_and_normalize(
    volume: np.ndarray,
    *,
    hu_window: tuple[float, float] = (HU_WINDOW_MIN, HU_WINDOW_MAX),
) -> np.ndarray:
    """Clip Hounsfield units to `hu_window` and scale that range to [0, 1]."""

    hu_min, hu_max = float(hu_window[0]), float(hu_window[1])
    if not hu_max > hu_min:
        raise ValueError(f"hu_window must satisfy max > min, got {hu_window}.")

    clipped = np.clip(volume, hu_min, hu_max)
    return (clipped - hu_min) / (hu_max - hu_min)


def convert_case(
    *,
    case_id: str,
    image_path: Path,
    label_path: Path,
    output_root: Path,
    split_name: str,
    hu_window: tuple[float, float] = (HU_WINDOW_MIN, HU_WINDOW_MAX),
) -> list[dict[str, str]]:
    """Convert one CT volume into per-slice PNG/label pairs. Returns manifest records."""

    image_volume = nib.load(image_path).get_fdata().astype(np.float32)
    label_volume = nib.load(label_path).get_fdata().astype(np.int64)

    if image_volume.shape != label_volume.shape:
        raise ValueError(
            f"{case_id}: image shape {image_volume.shape} != "
            f"label shape {label_volume.shape}."
        )

    normalized_volume = window_and_normalize(image_volume, hu_window=hu_window)

    image_output_dir = output_root / "images" / split_name
    label_output_dir = output_root / "labels" / split_name
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    num_slices = normalized_volume.shape[2]

    for slice_index in range(num_slices):
        label_slice = label_volume[:, :, slice_index]
        if not label_slice.any():
            continue

        image_slice = normalized_volume[:, :, slice_index]
        height, width = image_slice.shape

        sample_id = f"{case_id}_slice{slice_index:04d}"

        image_uint8 = (image_slice * 255.0).round().astype(np.uint8)
        image_output_path = image_output_dir / f"{sample_id}.png"
        Image.fromarray(image_uint8, mode="L").save(image_output_path)

        shape_string = f"(1,{height},{width})"
        label_output_path = label_output_dir / f"{sample_id}.{shape_string}.npz"
        sparse_label = sparse.csr_matrix(
            label_slice.reshape(1, -1).astype(np.int64)
        )
        sparse.save_npz(label_output_path, sparse_label)

        records.append(
            {
                "image": image_output_path.relative_to(output_root).as_posix(),
                "label": label_output_path.relative_to(output_root).as_posix(),
                "case_id": case_id,
            }
        )

    return records


def convert_cases(
    *,
    cases: list[dict[str, Path]],
    output_root: Path,
    split_name: str,
    hu_window: tuple[float, float] = (HU_WINDOW_MIN, HU_WINDOW_MAX),
) -> list[dict[str, str]]:
    """Convert a list of {case_id, image_path, label_path} entries."""

    records: list[dict[str, str]] = []
    for case in cases:
        case_records = convert_case(
            case_id=case["case_id"],
            image_path=case["image_path"],
            label_path=case["label_path"],
            output_root=output_root,
            split_name=split_name,
            hu_window=hu_window,
        )
        records.extend(case_records)
        print(f"[{split_name}] {case['case_id']}: {len(case_records)} labeled slices.")

    return records


def write_dataset_json(output_root: Path, manifest: dict[str, Any]) -> Path:
    dataset_json_path = output_root / "dataset.json"
    with dataset_json_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return dataset_json_path
