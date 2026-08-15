"""Smoke tests for the ISIC 2018 dataset and joint transforms."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch.utils.data import DataLoader

from src.configs.dataset import DatasetConfig, DatasetSplitConfig
from src.data import (
    ISIC2018Dataset,
    SegmentationTransform,
    build_dataset,
)


class TestISIC2018Dataset(unittest.TestCase):
    """Test loading, transforming and batching without the real dataset."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._create_split("train", "ISIC_TEST_TRAIN")
        self._create_split("test", "ISIC_TEST")

        manifest = {
            "training": [self._record("train", "ISIC_TEST_TRAIN")],
            "test": [self._record("test", "ISIC_TEST")],
        }
        (self.root / "dataset.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        self.config = DatasetConfig(
            name="isic2018",
            root=self.root,
            task="binary",
            num_classes=1,
            image_size=(32, 32),
            splits={
                "train": DatasetSplitConfig(
                    "images/train",
                    "labels/train",
                ),
                "test": DatasetSplitConfig(
                    "images/test",
                    "labels/test",
                ),
            },
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_split(self, split: str, sample_id: str):
        image_directory = self.root / "images" / split
        mask_directory = self.root / "labels" / split
        image_directory.mkdir(parents=True)
        mask_directory.mkdir(parents=True)

        image = np.zeros((24, 40, 3), dtype=np.uint8)
        image[:, :20, 0] = 255
        Image.fromarray(image).save(
            image_directory / f"{sample_id}.jpg"
        )

        mask = np.zeros((1, 24, 40), dtype=np.uint8)
        mask[:, 6:18, 10:30] = 1
        sparse.save_npz(
            mask_directory / f"{sample_id}.(1,24,40).npz",
            sparse.csr_matrix(mask.reshape(1, -1)),
        )

    @staticmethod
    def _record(split: str, sample_id: str) -> dict[str, str]:
        return {
            "image": f"images/{split}/{sample_id}.jpg",
            "label": f"labels/{split}/{sample_id}.(1,24,40).npz",
        }

    def test_sample_contract(self):
        dataset = build_dataset(self.config, split="test")
        sample = dataset[0]

        self.assertIsInstance(dataset, ISIC2018Dataset)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertEqual(sample["mask"].shape, (1, 32, 32))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["mask"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample["image"]).all())
        self.assertTrue(
            set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
        )
        self.assertEqual(sample["sample_id"], "ISIC_TEST")

        print("\n=== ISIC 2018 dataset smoke test ===")
        print(f"Dataset class : {type(dataset).__name__}")
        print(f"Samples       : {len(dataset)}")
        print(f"Image shape   : {tuple(sample['image'].shape)}")
        print(f"Mask shape    : {tuple(sample['mask'].shape)}")
        print(f"Mask values   : {sample['mask'].unique().tolist()}")
        print("Result        : PASS")

    def test_transform_keeps_image_and_mask_aligned(self):
        image_array = np.zeros((8, 10, 3), dtype=np.uint8)
        image_array[2:6, 1:4, 0] = 255
        image = Image.fromarray(image_array)

        mask = torch.zeros((1, 8, 10), dtype=torch.float32)
        mask[:, 2:6, 1:4] = 1.0

        transform = SegmentationTransform(
            image_size=(8, 10),
            image_mean=(0.0, 0.0, 0.0),
            image_std=(1.0, 1.0, 1.0),
            training=True,
            horizontal_flip_probability=1.0,
            vertical_flip_probability=0.0,
        )

        transformed_image, transformed_mask = transform(image, mask)
        red_region = (transformed_image[0] > 0.5).float()

        self.assertTrue(
            torch.equal(red_region, transformed_mask[0]),
            "Image and mask became misaligned after horizontal flip.",
        )

    def test_dataloader_batch(self):
        dataset = build_dataset(self.config, split="test")
        batch = next(
            iter(DataLoader(dataset, batch_size=1, shuffle=False))
        )

        self.assertEqual(batch["image"].shape, (1, 3, 32, 32))
        self.assertEqual(batch["mask"].shape, (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
