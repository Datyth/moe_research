"""Mask-only datasets and preflight auditing for Shape Teacher training."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


_SHAPE_PATTERN = re.compile(r"\.\((\d+),(\d+),(\d+)\)\.npz$")
_SPLIT_KEYS = {"train": "training", "val": "validation", "test": "test"}
_MASK_EXTENSIONS = {
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def _has_binary_values(values: np.ndarray) -> bool:
    observed = {float(value) for value in values.tolist()}
    return observed.issubset({0.0, 1.0}) or observed.issubset({0.0, 255.0})


def _shape_from_path(path: Path) -> tuple[int, int, int]:
    match = _SHAPE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            f"Sparse mask filename must encode shape as '.(C,H,W).npz': {path}"
        )
    return tuple(int(value) for value in match.groups())


def _resolve_mask_path(root: Path, record: Any) -> Path:
    if isinstance(record, str):
        value = record
    elif isinstance(record, dict):
        value = record.get("mask_path", record.get("label"))
    else:
        value = None
    if not isinstance(value, str) or not value:
        raise ValueError("Each manifest record needs 'mask_path' or 'label'.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _json_records(path: Path, split: str) -> list[Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"JSON mask manifest must contain a list or object: {path}")
    records = payload.get(_SPLIT_KEYS[split], payload.get(split))
    if not isinstance(records, list):
        raise ValueError(f"JSON manifest {path} has no list for split '{split}'.")
    return records


def _csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "mask_path" not in reader.fieldnames:
            raise ValueError(f"CSV manifest must contain a 'mask_path' column: {path}")
        records = [
            {"mask_path": str(row["mask_path"]).strip()}
            for row in reader
            if row.get("mask_path") and str(row["mask_path"]).strip()
        ]
    if not records:
        raise ValueError(f"CSV manifest has no mask paths: {path}")
    return records


def _text_records(path: Path) -> list[str]:
    records = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not records:
        raise ValueError(f"Text manifest has no mask paths: {path}")
    return records


def _directory_records(path: Path) -> list[str]:
    records = [
        str(candidate.resolve())
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in _MASK_EXTENSIONS
    ]
    if not records:
        raise ValueError(f"Mask directory contains no supported mask files: {path}")
    return records


def _records_from_source(source: Any, split: str) -> list[Any]:
    if isinstance(source, dict):
        available = [key for key in ("manifest", "directory") if key in source]
        if len(available) != 1:
            raise ValueError(
                f"Split '{split}' needs exactly one of 'manifest' or 'directory'."
            )
        source = source[available[0]]
    if not isinstance(source, (str, Path)) or not str(source):
        raise ValueError(f"Invalid mask source for split '{split}'.")
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return _directory_records(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask source not found for split '{split}': {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _json_records(path, split)
    if suffix == ".csv":
        return _csv_records(path)
    if suffix in {".txt", ".list"}:
        return _text_records(path)
    raise ValueError(
        f"Unsupported mask source '{path}'. Use JSON, CSV, TXT, or a directory."
    )


class MaskOnlyDataset(Dataset):
    """Read only masks from an authoritative split source."""

    def __init__(
        self,
        *,
        root: str | Path,
        records: list[Any],
        split: str,
        image_size: tuple[int, int] | list[int] = (256, 256),
        foreground_threshold: float = 0.0,
        allow_non_binary_source: bool = False,
    ) -> None:
        if split not in _SPLIT_KEYS:
            raise ValueError("split must be train, val or test.")
        if not records:
            raise ValueError(f"split '{split}' has no mask records.")
        self.root = Path(root).expanduser().resolve()
        self.records = list(records)
        self.split = split
        self.image_size = tuple(int(value) for value in image_size)
        self.foreground_threshold = float(foreground_threshold)
        self.allow_non_binary_source = bool(allow_non_binary_source)
        self.mask_paths = [_resolve_mask_path(self.root, record) for record in records]

    def __len__(self) -> int:
        return len(self.mask_paths)

    def _read_sparse(self, path: Path) -> tuple[Tensor, tuple[int, int]]:
        channels, height, width = _shape_from_path(path)
        if channels != 1:
            raise ValueError(f"Shape Teacher requires one-channel masks, got {path}.")
        matrix = sparse.load_npz(path)
        stored_elements = int(matrix.shape[0]) * int(matrix.shape[1])
        if stored_elements != channels * height * width:
            raise ValueError(f"Encoded shape does not match sparse mask elements: {path}")
        values = np.unique(matrix.data)
        if not self.allow_non_binary_source and not _has_binary_values(values):
            raise ValueError(f"Non-binary sparse values in {path}: {values.tolist()}")
        dense = np.asarray(matrix.toarray(), dtype=np.float32).reshape(
            channels, height, width
        )
        return torch.from_numpy(dense), (height, width)

    def _read_image(self, path: Path) -> tuple[Tensor, tuple[int, int]]:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        unique = np.unique(array)
        if not self.allow_non_binary_source and not _has_binary_values(unique):
            raise ValueError(
                f"Non-binary source mask {path} has values {unique.tolist()}."
            )
        return torch.from_numpy(array.copy()).unsqueeze(0), tuple(array.shape)

    def read_original(self, index: int) -> tuple[Tensor, tuple[int, int]]:
        path = self.mask_paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Mask not found: {path}")
        if path.suffix.lower() == ".npz":
            return self._read_sparse(path)
        return self._read_image(path)

    def source_statistics(self, index: int) -> tuple[tuple[int, int], int, int]:
        """Return size, foreground count and pixel count without dense NPZ decode."""

        path = self.mask_paths[index]
        if not path.is_file():
            raise FileNotFoundError(f"Mask not found: {path}")
        if path.suffix.lower() != ".npz":
            source, size = self._read_image(path)
            foreground = int((source > self.foreground_threshold).sum())
            return size, foreground, source.numel()

        channels, height, width = _shape_from_path(path)
        if channels != 1:
            raise ValueError(f"Shape Teacher requires one-channel masks, got {path}.")
        matrix = sparse.load_npz(path)
        stored_elements = int(matrix.shape[0]) * int(matrix.shape[1])
        total = channels * height * width
        if stored_elements != total:
            raise ValueError(f"Encoded shape does not match sparse mask elements: {path}")
        values = np.unique(matrix.data)
        if not self.allow_non_binary_source and not _has_binary_values(values):
            raise ValueError(f"Non-binary sparse values in {path}: {values.tolist()}")
        foreground = int(
            np.count_nonzero(np.asarray(matrix.data) > self.foreground_threshold)
        )
        return (height, width), foreground, total

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, original_size = self.read_original(index)
        binary = (source > self.foreground_threshold).float()
        if tuple(binary.shape[1:]) != self.image_size:
            binary = F.interpolate(
                binary.unsqueeze(0),
                size=self.image_size,
                mode="nearest",
            ).squeeze(0)
        path = self.mask_paths[index]
        return {
            "mask": binary.contiguous(),
            "dataset_index": index,
            "sample_id": path.name.split(".")[0],
            "mask_path": str(path),
            "split": self.split,
            "original_size": torch.tensor(original_size, dtype=torch.int64),
        }


def build_mask_datasets(config: dict[str, Any]) -> dict[str, MaskOnlyDataset]:
    dataset_config = config["dataset"]
    source = dataset_config["manifest"]
    if isinstance(source, dict):
        missing = [split for split in _SPLIT_KEYS if split not in source]
        if missing:
            raise ValueError(
                f"dataset.manifest split mapping is missing: {', '.join(missing)}."
            )
        records_by_split = {
            split: _records_from_source(source[split], split) for split in _SPLIT_KEYS
        }
    else:
        records_by_split = {
            split: _records_from_source(source, split) for split in _SPLIT_KEYS
        }
    common = {
        "root": dataset_config["root"],
        "image_size": dataset_config["image_size"],
        "foreground_threshold": dataset_config.get("foreground_threshold", 0.0),
        "allow_non_binary_source": dataset_config.get(
            "allow_non_binary_source", False
        ),
    }
    datasets = {
        split: MaskOnlyDataset(
            records=records_by_split[split], split=split, **common
        )
        for split in _SPLIT_KEYS
    }
    audit_mask_splits(datasets, scan_masks=False)
    return datasets


def audit_mask_splits(
    datasets: dict[str, MaskOnlyDataset],
    *,
    scan_masks: bool = True,
) -> dict[str, Any]:
    """Validate split isolation and optionally collect mask statistics."""

    owners: dict[str, str] = {}
    for split, dataset in datasets.items():
        for path in dataset.mask_paths:
            key = str(path)
            if key in owners:
                raise ValueError(
                    f"Duplicate mask path across splits '{owners[key]}' and "
                    f"'{split}': {path}"
                )
            owners[key] = split

    report: dict[str, Any] = {}
    for split, dataset in datasets.items():
        split_report: dict[str, Any] = {"count": len(dataset)}
        if scan_masks:
            empty = 0
            full = 0
            dimensions: Counter[str] = Counter()
            foreground_fractions: list[float] = []
            for index in range(len(dataset)):
                size, count, total = dataset.source_statistics(index)
                empty += int(count == 0)
                full += int(count == total)
                dimensions[f"{size[0]}x{size[1]}"] += 1
                foreground_fractions.append(count / total)
            fractions = np.asarray(foreground_fractions, dtype=np.float64)
            split_report.update(
                empty_masks=empty,
                all_foreground_masks=full,
                original_dimensions=dict(sorted(dimensions.items())),
                foreground_fraction={
                    "min": float(fractions.min()),
                    "mean": float(fractions.mean()),
                    "median": float(np.median(fractions)),
                    "max": float(fractions.max()),
                },
            )
        report[split] = split_report
    return report
