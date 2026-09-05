"""Smoke tests for the ISIC 2018 dataset and joint transforms."""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch.utils.data import DataLoader

from src.configs.dataset import DatasetConfig
from src.data import (
    AMOS22Dataset,
    ISIC2018Dataset,
    SegmentationTransform,
    SynapseCTDataset,
    SynapseDataset,
    build_dataset,
)
from scripts.data.ct_conversion import window_and_normalize
from scripts.data.prepare_amos22 import split_cases
from scripts.data.prepare_isic2018 import split_training_records
from scripts.data.prepare_synapse import _extract_case_number


class TestISIC2018Dataset(unittest.TestCase):
    """Test loading, transforming and batching without the real dataset."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._create_split("train", "ISIC_TEST_TRAIN")
        self._create_split("train", "ISIC_TEST_VAL")
        self._create_split("test", "ISIC_TEST")

        manifest = {
            "training": [self._record("train", "ISIC_TEST_TRAIN")],
            "validation": [self._record("train", "ISIC_TEST_VAL")],
            "test": [self._record("test", "ISIC_TEST")],
        }
        (self.root / "dataset.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        self.config = DatasetConfig(
            name="isic2018",
            root=self.root,
            manifest=self.root / "dataset.json",
            version="test-v1",
            task="binary",
            num_classes=1,
            image_size=(32, 32),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_split(self, split: str, sample_id: str):
        image_directory = self.root / "images" / split
        mask_directory = self.root / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        mask_directory.mkdir(parents=True, exist_ok=True)

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

    def test_validation_split_contract(self):
        dataset = build_dataset(self.config, split="val")
        sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["sample_id"], "ISIC_TEST_VAL")
        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertEqual(sample["mask"].shape, (1, 32, 32))

    def test_manifest_split_is_deterministic_and_disjoint(self):
        records = [
            {"image": f"images/train/ISIC_{index:04d}.jpg", "label": str(index)}
            for index in range(10)
        ]

        first_train, first_val = split_training_records(
            records,
            val_ratio=0.2,
            seed=42,
        )
        repeated_train, repeated_val = split_training_records(
            first_val + first_train,
            val_ratio=0.2,
            seed=42,
        )
        _, different_val = split_training_records(
            records,
            val_ratio=0.2,
            seed=43,
        )

        first_train_ids = {record["image"] for record in first_train}
        first_val_ids = {record["image"] for record in first_val}
        self.assertEqual(len(first_train), 8)
        self.assertEqual(len(first_val), 2)
        self.assertFalse(first_train_ids & first_val_ids)
        self.assertEqual(first_train_ids | first_val_ids, {
            record["image"] for record in records
        })
        self.assertEqual(first_train, repeated_train)
        self.assertEqual(first_val, repeated_val)
        self.assertNotEqual(first_val, different_val)


class TestAMOS22Dataset(unittest.TestCase):
    """Test loading multiclass CT slices without the real AMOS22 dataset."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._create_slice("train", "amos_0001_slice0010")
        self._create_slice("test", "amos_0002_slice0005")

        manifest = {
            "training": [self._record("train", "amos_0001_slice0010")],
            "validation": [self._record("train", "amos_0001_slice0010")],
            "test": [self._record("test", "amos_0002_slice0005")],
        }
        (self.root / "dataset.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        self.config = DatasetConfig(
            name="amos22_ct",
            root=self.root,
            manifest=self.root / "dataset.json",
            version="test-v1",
            task="multiclass",
            num_classes=16,
            image_size=(32, 32),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_slice(self, split: str, sample_id: str):
        image_directory = self.root / "images" / split
        mask_directory = self.root / "labels" / split
        image_directory.mkdir(parents=True, exist_ok=True)
        mask_directory.mkdir(parents=True, exist_ok=True)

        image = np.zeros((24, 40), dtype=np.uint8)
        image[:, :20] = 200
        Image.fromarray(image, mode="L").save(image_directory / f"{sample_id}.png")

        mask = np.zeros((1, 24, 40), dtype=np.int64)
        mask[:, 6:18, 10:30] = 7
        sparse.save_npz(
            mask_directory / f"{sample_id}.(1,24,40).npz",
            sparse.csr_matrix(mask.reshape(1, -1)),
        )

    @staticmethod
    def _record(split: str, sample_id: str) -> dict[str, str]:
        return {
            "image": f"images/{split}/{sample_id}.png",
            "label": f"labels/{split}/{sample_id}.(1,24,40).npz",
            "case_id": sample_id.split("_slice")[0],
        }

    def test_sample_contract(self):
        dataset = build_dataset(self.config, split="test")
        sample = dataset[0]

        self.assertIsInstance(dataset, AMOS22Dataset)
        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertEqual(sample["mask"].shape, (32, 32))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["mask"].dtype, torch.long)
        self.assertTrue(set(sample["mask"].unique().tolist()) <= set(range(16)))
        self.assertEqual(sample["sample_id"], "amos_0002_slice0005")

    def test_binary_task_is_rejected(self):
        binary_config = DatasetConfig(
            name="amos22_ct",
            root=self.root,
            manifest=self.root / "dataset.json",
            version="test-v1",
            task="binary",
            num_classes=1,
            image_size=(32, 32),
        )
        with self.assertRaisesRegex(ValueError, "multiclass"):
            build_dataset(binary_config, split="test")


class TestAMOS22Conversion(unittest.TestCase):
    def test_window_and_normalize_clips_and_scales_to_unit_range(self):
        volume = np.array([-2000.0, -125.0, 75.0, 275.0, 3000.0])
        normalized = window_and_normalize(volume)

        self.assertTrue(np.allclose(normalized, [0.0, 0.0, 0.5, 1.0, 1.0]))

    def test_split_cases_is_deterministic_disjoint_and_case_level(self):
        cases = [{"case_id": f"amos_{index:04d}"} for index in range(20)]

        train, val, test = split_cases(cases, val_ratio=0.15, test_ratio=0.15, seed=42)
        train_ids = {case["case_id"] for case in train}
        val_ids = {case["case_id"] for case in val}
        test_ids = {case["case_id"] for case in test}

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, {c["case_id"] for c in cases})

        repeated_train, repeated_val, repeated_test = split_cases(
            cases, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        self.assertEqual(train, repeated_train)
        self.assertEqual(val, repeated_val)
        self.assertEqual(test, repeated_test)


class TestSynapseDataset(unittest.TestCase):
    """SynapseDataset shares all loading logic with AMOS22Dataset via
    CTSliceDataset — just confirm it's independently registered and usable."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        image_directory = self.root / "images" / "test"
        mask_directory = self.root / "labels" / "test"
        image_directory.mkdir(parents=True)
        mask_directory.mkdir(parents=True)

        image = np.zeros((16, 16), dtype=np.uint8)
        Image.fromarray(image, mode="L").save(
            image_directory / "synapse_0001_slice0003.png"
        )
        mask = np.zeros((1, 16, 16), dtype=np.int64)
        mask[:, 4:10, 4:10] = 6
        sparse.save_npz(
            mask_directory / "synapse_0001_slice0003.(1,16,16).npz",
            sparse.csr_matrix(mask.reshape(1, -1)),
        )

        manifest = {
            "training": [],
            "validation": [],
            "test": [
                {
                    "image": "images/test/synapse_0001_slice0003.png",
                    "label": "labels/test/synapse_0001_slice0003.(1,16,16).npz",
                    "case_id": "synapse_0001",
                }
            ],
        }
        (self.root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

        self.config = DatasetConfig(
            name="synapse_btcv",
            root=self.root,
            manifest=self.root / "dataset.json",
            version="test-v1",
            task="multiclass",
            num_classes=14,
            image_size=(16, 16),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_sample_contract(self):
        dataset = build_dataset(self.config, split="test")
        sample = dataset[0]

        self.assertIsInstance(dataset, SynapseDataset)
        self.assertEqual(sample["image"].shape, (3, 16, 16))
        self.assertEqual(sample["mask"].shape, (16, 16))
        self.assertIn(6, sample["mask"].unique().tolist())


class TestSynapseCTDataset(unittest.TestCase):
    """Load the native TransUNet NPZ/HDF5 layout through audited lists."""

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "lists" / "lists_Synapse").mkdir(parents=True)
        (self.root / "train_npz").mkdir()
        (self.root / "test_vol_h5").mkdir()

        train_id = "case0005_slice000"
        image = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        label = np.zeros((16, 16), dtype=np.uint8)
        label[4:12, 4:12] = 5
        np.savez(self.root / "train_npz" / f"{train_id}.npz", image=image, label=label)
        (self.root / "lists" / "lists_Synapse" / "train.txt").write_text(
            f"{train_id}\n", encoding="utf-8"
        )

        volume_id = "case0001"
        volume_image = np.stack((image, image[::-1].copy()))
        volume_label = np.stack((label, label))
        with h5py.File(
            self.root / "test_vol_h5" / f"{volume_id}.npy.h5", "w"
        ) as handle:
            handle.create_dataset("image", data=volume_image)
            handle.create_dataset("label", data=volume_label)
        (self.root / "lists" / "lists_Synapse" / "val.txt").write_text(
            f"{volume_id}\n", encoding="utf-8"
        )

        descriptor_train = {
            "kind": "npz_slices",
            "list": "lists/lists_Synapse/train.txt",
            "data_dir": "train_npz",
            "expected_samples": 1,
            "expected_cases": 1,
        }
        descriptor_eval = {
            "kind": "hdf5_volumes",
            "list": "lists/lists_Synapse/val.txt",
            "data_dir": "test_vol_h5",
            "expected_cases": 1,
        }
        manifest = {
            "training": [descriptor_train],
            "validation": [descriptor_eval],
            "test": [descriptor_eval],
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.config = DatasetConfig(
            name="synapse_ct",
            root=self.root,
            manifest=self.root / "manifest.json",
            version="test-v1",
            task="multiclass",
            num_classes=9,
            image_size=(16, 16),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_npz_training_slice_contract(self):
        dataset = build_dataset(self.config, split="train")
        sample = dataset[0]

        self.assertIsInstance(dataset, SynapseCTDataset)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(sample["image"].shape), (3, 16, 16))
        self.assertEqual(tuple(sample["mask"].shape), (16, 16))
        self.assertEqual(sample["mask"].dtype, torch.long)
        self.assertEqual(sample["sample_id"], "case0005_slice000")
        self.assertIn(5, sample["mask"].unique().tolist())

    def test_hdf5_volume_expands_to_2d_evaluation_slices(self):
        validation = build_dataset(self.config, split="val")
        test = build_dataset(self.config, split="test")

        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual(validation[1]["sample_id"], "case0001_slice0001")
        self.assertEqual(tuple(validation[1]["image"].shape), (3, 16, 16))
        self.assertEqual(tuple(validation[1]["mask"].shape), (16, 16))


class TestSynapseConversion(unittest.TestCase):
    def test_extract_case_number_finds_digits_regardless_of_prefix(self):
        self.assertEqual(_extract_case_number(Path("img0007.nii.gz")), "0007")
        self.assertEqual(_extract_case_number(Path("label0007.nii.gz")), "0007")

    def test_extract_case_number_raises_without_digits(self):
        with self.assertRaises(ValueError):
            _extract_case_number(Path("scan.nii.gz"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
