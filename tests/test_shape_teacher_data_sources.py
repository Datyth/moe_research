"""Tests for generic split mask sources used by Shape Teacher training."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.configs.experiment import resolve_experiment_config
from src.shape_teacher.data import MaskOnlyDataset, build_mask_datasets


class TestShapeTeacherDataSources(unittest.TestCase):
    def _mask(self, path: Path, *, rgb: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 255
        array = np.repeat(mask[..., None], 3, axis=2) if rgb else mask
        Image.fromarray(array).save(path)

    def _base_config(self, root: Path, manifest: object) -> dict:
        return {
            "experiment": {"name": "sources", "output_root": "runs"},
            "seed": 7,
            "dataset": {
                "name": "masks",
                "root": str(root),
                "manifest": manifest,
                "version": "fixture-v1",
                "task": "binary",
                "num_classes": 1,
                "in_channels": 1,
                "image_size": [16, 20],
                "image_mean": [0.0],
                "image_std": [1.0],
                "foreground_threshold": 0.0,
            },
            "model": {"name": "shape_teacher"},
            "loss": {"name": "shape_teacher"},
            "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
            "scheduler": {"name": "none"},
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
            },
        }

    def test_csv_txt_and_directory_split_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_mask = root / "masks/train/A.png"
            val_mask = root / "masks/val/B.png"
            test_mask = root / "masks/test/C.png"
            self._mask(train_mask)
            self._mask(val_mask)
            self._mask(test_mask)

            train_csv = root / "train.csv"
            with train_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["mask_path"])
                writer.writeheader()
                writer.writerow({"mask_path": "masks/train/A.png"})
            val_txt = root / "val.txt"
            val_txt.write_text("# validation masks\nmasks/val/B.png\n", encoding="utf-8")

            raw = self._base_config(
                root,
                {
                    "train": {"manifest": str(train_csv)},
                    "val": str(val_txt),
                    "test": {"directory": str(test_mask.parent)},
                },
            )
            config = resolve_experiment_config(raw, project_root=root)
            datasets = build_mask_datasets(config)

            self.assertEqual({key: len(value) for key, value in datasets.items()}, {
                "train": 1,
                "val": 1,
                "test": 1,
            })
            self.assertEqual(datasets["train"][0]["mask"].shape, (1, 16, 20))
            self.assertEqual(datasets["val"].mask_paths, [val_mask.resolve()])
            self.assertEqual(datasets["test"].mask_paths, [test_mask.resolve()])
            self.assertTrue(Path(config["dataset"]["manifest"]["val"]).is_absolute())
            self.assertTrue(
                Path(config["dataset"]["manifest"]["test"]["directory"]).is_absolute()
            )

    def test_directory_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("z.png", "a.png", "nested/m.png"):
                self._mask(root / "test" / name)
            config = {
                "dataset": {
                    "root": str(root),
                    "manifest": {
                        split: {"directory": str(root / split)}
                        for split in ("train", "val", "test")
                    },
                    "image_size": [8, 10],
                }
            }
            self._mask(root / "train/train.png")
            self._mask(root / "val/val.png")
            datasets = build_mask_datasets(config)
            names = [path.name for path in datasets["test"].mask_paths]
            self.assertEqual(names, ["a.png", "m.png", "z.png"])

    def test_rgb_mask_is_grayscaled_and_resized_with_nearest_neighbor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rgb.png"
            self._mask(path, rgb=True)
            dataset = MaskOnlyDataset(
                root=root,
                records=[str(path)],
                split="train",
                image_size=(16, 20),
            )
            sample = dataset[0]
            self.assertEqual(set(sample["mask"].unique().tolist()), {0.0, 1.0})
            self.assertEqual(int(sample["mask"].sum()), 80)
            self.assertEqual(sample["dataset_index"], 0)

    def test_non_binary_source_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "non_binary.png"
            Image.fromarray(np.array([[0, 127]], dtype=np.uint8)).save(path)
            dataset = MaskOnlyDataset(
                root=root,
                records=[str(path)],
                split="train",
                image_size=(1, 2),
            )
            with self.assertRaisesRegex(ValueError, "Non-binary source mask"):
                _ = dataset[0]

    def test_split_mapping_requires_exact_train_val_test_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self._base_config(root, {"train": "train.txt", "val": "val.txt"})
            with self.assertRaisesRegex(ValueError, "missing test"):
                resolve_experiment_config(raw, project_root=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
