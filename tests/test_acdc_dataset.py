"""Smoke tests for the multiclass ACDC dataset and joint transforms."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.configs.dataset import DatasetConfig
from src.data import ACDCDataset, build_dataset
from src.data.transforms import SegmentationTransform


class TestACDCDataset(unittest.TestCase):
    """Test loading, transforming and batching without the real dataset."""

    LABEL_VALUES = [0, 1, 2, 3]

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._create_split("train", "patient001_frame01_slice001")
        self._create_split("train", "patient001_frame01_slice002")
        self._create_split("val", "patient021_frame01_slice001")
        self._create_split("test", "patient081_frame01_slice001")

        manifest = {
            "training": [self._record("train", "patient001_frame01_slice001"),
                         self._record("train", "patient001_frame01_slice002")],
            "validation": [self._record("val", "patient021_frame01_slice001")],
            "test": [self._record("test", "patient081_frame01_slice001")],
        }
        (self.root / "acdc.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        self.config = DatasetConfig(
            name="acdc",
            root=self.root,
            manifest=self.root / "acdc.json",
            version="acdc-v1",
            task="multiclass",
            num_classes=4,
            image_size=(32, 32),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_split(self, split: str, sample_id: str):
        image_directory = self.root / "images" / split
        label_directory = self.root / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(len(sample_id))
        image = (rng.integers(0, 256, size=(48, 64))).astype(np.uint8)
        mask = rng.choice(self.LABEL_VALUES, size=(48, 64), p=[0.7, 0.1, 0.1, 0.1])
        mask[0, 0] = 0

        Image.fromarray(image, mode="L").convert("RGB").save(
            image_directory / f"{sample_id}.png"
        )
        np.savez_compressed(
            label_directory / f"{sample_id}.npz",
            mask=mask.astype(np.uint8),
        )

    @staticmethod
    def _record(split: str, sample_id: str) -> dict[str, str]:
        return {
            "image": f"images/{split}/{sample_id}.png",
            "label": f"labels/{split}/{sample_id}.npz",
        }

    def _dataset(self, split: str) -> ACDCDataset:
        return ACDCDataset(self.config, split=split)

    def test_len_and_registry(self):
        self.assertEqual(len(self._dataset("train")), 2)
        self.assertEqual(len(self._dataset("val")), 1)
        built = build_dataset(self.config, split="test")
        self.assertIsInstance(built, ACDCDataset)

    def test_rejects_binary_task(self):
        binary_config = DatasetConfig(
            name="acdc",
            root=self.root,
            manifest=self.root / "acdc.json",
            version="acdc-v1",
            task="binary",
            num_classes=1,
            image_size=(32, 32),
        )
        with self.assertRaises(ValueError):
            ACDCDataset(binary_config, split="train")

    def test_sample_containing_multiclass_mask(self):
        sample = self._dataset("train")[0]

        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertEqual(sample["mask"].shape, (1, 32, 32))
        self.assertEqual(sample["mask"].dtype, torch.long)
        self.assertTrue(bool(torch.isin(sample["mask"], torch.tensor(self.LABEL_VALUES)).all()))

    def test_dataloader_batches(self):
        loader = DataLoader(self._dataset("train"), batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (2, 3, 32, 32))
        self.assertEqual(batch["mask"].shape, (2, 1, 32, 32))

    def test_evaluation_transform_is_deterministic(self):
        dataset = self._dataset("test")
        first = dataset[0]["image"]
        second = dataset[0]["image"]
        self.assertTrue(torch.equal(first, second))

    def test_training_transform_applies_augmentation(self):
        augmented = SegmentationTransform(
            image_size=(32, 32),
            image_mean=(0.5,),
            image_std=(0.5,),
            training=True,
            rotation_range=15.0,
            scaling_range=(0.8, 1.2),
            intensity_shift_range=0.1,
        )
        image = Image.new("RGB", (48, 64))
        mask = torch.randint(0, 4, (1, 48, 64))
        _, transformed_mask = augmented(image, mask)

        self.assertEqual(transformed_mask.shape, (1, 32, 32))
        self.assertTrue(
            bool(torch.isin(transformed_mask.unique(), torch.tensor(self.LABEL_VALUES)).all())
        )


if __name__ == "__main__":
    unittest.main()
