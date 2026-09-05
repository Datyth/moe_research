"""Native loader for the 8-organ Synapse CT/TransUNet dataset layout.

The dataset keeps training samples as 2D ``.npz`` containers and held-out
cases as HDF5 volumes. Split descriptors live in a tracked JSON manifest and
point to the audited ``train.txt``/``val.txt`` files under the external data
root. Validation and test intentionally share ``val.txt`` because this
18-case/12-case protocol has no separate validation cohort.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from src.configs.dataset import DatasetConfig

from .base import BaseSegmentationDataset
from .registry import register_dataset
from .transforms import build_segmentation_transform

_CASE_PATTERN = re.compile(r"case\d+", re.IGNORECASE)


@register_dataset("synapse_ct")
class SynapseCTDataset(BaseSegmentationDataset):
    """Read native Synapse CT NPZ slices and HDF5 evaluation volumes."""

    MANIFEST_SPLITS = {
        "train": "training",
        "val": "validation",
        "test": "test",
    }

    def __init__(self, config: DatasetConfig, split: str, transform=None) -> None:
        super().__init__(config=config, split=split, transform=transform)
        if config.task != "multiclass":
            raise ValueError("SynapseCTDataset supports multiclass segmentation only.")

        self.root = Path(config.root).expanduser().resolve()
        self.transform = transform or build_segmentation_transform(config, split)
        self._h5_handles: dict[Path, h5py.File] = {}

        manifest = self._load_manifest()
        manifest_split = self.MANIFEST_SPLITS.get(split, split)
        descriptors = manifest.get(manifest_split)
        if not isinstance(descriptors, list) or not descriptors:
            raise ValueError(
                f"Split '{manifest_split}' is missing or empty in {config.manifest}."
            )

        self.records: list[dict[str, Any]] = []
        for descriptor in descriptors:
            self.records.extend(self._expand_descriptor(descriptor))
        if not self.records:
            raise ValueError(f"Split '{split}' contains no samples.")

    def _load_manifest(self) -> dict[str, Any]:
        import json

        manifest_path = Path(self.config.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"Dataset manifest must be a mapping: {manifest_path}")
        return manifest

    def _expand_descriptor(self, descriptor: Any) -> list[dict[str, Any]]:
        if not isinstance(descriptor, dict):
            raise ValueError("Synapse CT split descriptors must be mappings.")
        kind = descriptor.get("kind")
        if kind not in {"npz_slices", "hdf5_volumes"}:
            raise ValueError(
                "Synapse CT descriptor.kind must be 'npz_slices' or "
                "'hdf5_volumes'."
            )

        list_path = self._resolve_list_path(str(descriptor.get("list", "")))
        entries = [
            line.strip()
            for line in list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_samples = descriptor.get("expected_samples")
        if expected_samples is not None and len(entries) != int(expected_samples):
            raise ValueError(
                f"{list_path} contains {len(entries)} entries; expected "
                f"{expected_samples}."
            )
        expected_cases = descriptor.get("expected_cases")
        observed_cases = {
            match.group(0).lower()
            for entry in entries
            if (match := _CASE_PATTERN.search(entry)) is not None
        }
        if expected_cases is not None and len(observed_cases) != int(expected_cases):
            raise ValueError(
                f"{list_path} contains {len(observed_cases)} cases; expected "
                f"{expected_cases}."
            )

        data_dir = str(descriptor.get("data_dir", ""))
        if kind == "npz_slices":
            return [
                {
                    "kind": kind,
                    "path": self._resolve_data_path(entry, data_dir, kind),
                    "sample_id": Path(entry).name.removesuffix(".npz"),
                }
                for entry in entries
            ]

        records: list[dict[str, Any]] = []
        for entry in entries:
            path = self._resolve_data_path(entry, data_dir, kind)
            with h5py.File(path, "r") as handle:
                self._validate_h5(handle, path)
                depth = int(handle["image"].shape[0])
            case_id = _case_id(entry)
            records.extend(
                {
                    "kind": kind,
                    "path": path,
                    "slice_index": slice_index,
                    "sample_id": f"{case_id}_slice{slice_index:04d}",
                }
                for slice_index in range(depth)
            )
        return records

    def _resolve_list_path(self, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError("Synapse CT descriptor.list must not be empty.")
        direct = self.root / relative_path
        if direct.is_file():
            return direct
        matches = sorted(
            path for path in self.root.rglob(Path(relative_path).name) if path.is_file()
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"List file '{relative_path}' was not found under {self.root}."
            )
        raise ValueError(
            f"List file '{relative_path}' is ambiguous under {self.root}: {matches}"
        )

    def _resolve_data_path(self, entry: str, data_dir: str, kind: str) -> Path:
        entry_path = Path(entry)
        names = [entry_path.name]
        if kind == "npz_slices" and not entry_path.name.endswith(".npz"):
            names.append(f"{entry_path.name}.npz")
        if kind == "hdf5_volumes" and not entry_path.name.endswith(".h5"):
            names.extend((f"{entry_path.name}.h5", f"{entry_path.name}.npy.h5"))

        directories = [self.root / data_dir] if data_dir else []
        directories.append(self.root)
        for directory in directories:
            for name in names:
                candidate = directory / name
                if candidate.is_file():
                    return candidate

        suffix = ".npz" if kind == "npz_slices" else ".h5"
        entry_stem = entry_path.name.removesuffix(".npz").removesuffix(".npy.h5").removesuffix(".h5")
        matches = sorted(
            path
            for path in self.root.rglob(f"{entry_stem}*")
            if path.is_file() and path.name.endswith(suffix)
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Data entry '{entry}' ({kind}) was not found under {self.root}."
            )
        raise ValueError(f"Data entry '{entry}' is ambiguous: {matches}")

    @staticmethod
    def _validate_h5(handle: h5py.File, path: Path) -> None:
        for key in ("image", "label"):
            if key not in handle:
                raise KeyError(f"HDF5 volume {path} is missing dataset '{key}'.")
        if handle["image"].shape != handle["label"].shape:
            raise ValueError(
                f"HDF5 image/label shape mismatch in {path}: "
                f"{handle['image'].shape} vs {handle['label'].shape}."
            )
        if len(handle["image"].shape) != 3:
            raise ValueError(f"HDF5 volume must have shape [D,H,W]: {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path: Path = record["path"]
        if record["kind"] == "npz_slices":
            with np.load(path) as container:
                image = np.asarray(container["image"])
                label = np.asarray(container["label"])
        else:
            handle = self._h5_handles.get(path)
            if handle is None:
                handle = h5py.File(path, "r")
                self._h5_handles[path] = handle
            slice_index = int(record["slice_index"])
            image = np.asarray(handle["image"][slice_index])
            label = np.asarray(handle["label"][slice_index])

        image = np.squeeze(image)
        label = np.squeeze(label)
        if image.ndim != 2 or label.ndim != 2 or image.shape != label.shape:
            raise ValueError(
                f"Image/label must be matching 2D arrays in {path}, got "
                f"{image.shape} and {label.shape}."
            )
        if not np.isfinite(image).all():
            raise ValueError(f"Image contains non-finite values: {path}")
        if image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"Image values must be normalized to [0,1]: {path}")
        if label.min() < 0 or label.max() >= self.config.num_classes:
            raise ValueError(
                f"Label values must be in [0,{self.config.num_classes - 1}]: {path}"
            )

        image_u8 = np.rint(image.astype(np.float32) * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(image_u8, mode="L").convert("RGB")
        mask = torch.from_numpy(label.astype(np.int64, copy=False)).unsqueeze(0)
        image_tensor, mask_tensor = self.transform(pil_image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor.round().long().squeeze(0),
            "sample_id": record["sample_id"],
            "image_path": str(path),
            "mask_path": str(path),
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass


def _case_id(entry: str) -> str:
    match = _CASE_PATTERN.search(entry)
    return match.group(0).lower() if match is not None else Path(entry).stem
