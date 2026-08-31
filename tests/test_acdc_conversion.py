"""Tests for ACDC conversion utilities and patient-level splitting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.data.acdc_conversion import (
    annotated_frame_numbers,
    normalize_slice,
    parse_info_cfg,
)
from scripts.data.prepare_acdc import discover_patient_dirs, split_patients


class TestInfoCfgParsing(unittest.TestCase):
    def test_parse_official_style_cfg(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "Info.cfg"
            path.write_text(
                "Diastole:  46\nEsistole:  78\nWeight:    82\nHeight:    171\n",
                encoding="utf-8",
            )
            values = parse_info_cfg(path)
        self.assertEqual(values["diastole"], "46")
        self.assertEqual(values["esistole"], "78")
        self.assertEqual(values["weight"], "82")

    def test_annotated_frame_numbers(self):
        frames = annotated_frame_numbers({"diastole": "3", "esistole": "12"})
        self.assertEqual(frames, [3, 12])

    def test_annotated_frame_numbers_requires_frames(self):
        with self.assertRaises(ValueError):
            annotated_frame_numbers({"weight": "82"})


class TestNormalizeSlice(unittest.TestCase):
    def test_output_is_uint8_full_range(self):
        rng = np.random.default_rng(0)
        slice2d = rng.normal(loc=100.0, scale=25.0, size=(32, 32))
        normalized = normalize_slice(slice2d)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertGreaterEqual(int(normalized.min()), 0)
        self.assertLessEqual(int(normalized.max()), 255)

    def test_constant_slice_does_not_crash(self):
        normalized = normalize_slice(np.full((8, 8), 42.0))
        self.assertEqual(normalized.dtype, np.uint8)


class TestPatientSplit(unittest.TestCase):
    def _ids(self, count: int) -> list[str]:
        return [f"patient{index:03d}" for index in range(1, count + 1)]

    def test_split_is_patient_level_and_disjoint(self):
        splits = split_patients(
            self._ids(20),
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )
        training = set(splits["training"])
        validation = set(splits["validation"])
        test = set(splits["test"])

        self.assertEqual(len(training), 12)
        self.assertEqual(len(validation), 4)
        self.assertEqual(len(test), 4)
        self.assertFalse(training & validation)
        self.assertFalse(training & test)
        self.assertFalse(validation & test)
        self.assertEqual(
            len(training | validation | test),
            20,
        )

    def test_split_is_deterministic(self):
        first = split_patients(self._ids(15), val_ratio=0.2, test_ratio=0.2, seed=7)
        second = split_patients(self._ids(15), val_ratio=0.2, test_ratio=0.2, seed=7)
        self.assertEqual(first, second)

    def test_different_seeds_change_split(self):
        first = split_patients(self._ids(15), val_ratio=0.2, test_ratio=0.2, seed=1)
        second = split_patients(self._ids(15), val_ratio=0.2, test_ratio=0.2, seed=2)
        self.assertNotEqual(first, second)

    def test_invalid_ratios_rejected(self):
        with self.assertRaises(ValueError):
            split_patients(self._ids(10), val_ratio=0.6, test_ratio=0.6, seed=1)
        with self.assertRaises(ValueError):
            split_patients(self._ids(10), val_ratio=-0.1, test_ratio=0.2, seed=1)


class TestDiscoverPatients(unittest.TestCase):
    def test_only_labeled_patients_are_discovered(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            labeled = root / "patient001"
            labeled.mkdir()
            (labeled / "patient001_frame01_gt.nii.gz").touch()
            (labeled / "Info.cfg").touch()

            unlabeled = root / "patient101"
            unlabeled.mkdir()
            (unlabeled / "patient101_4d.nii.gz").touch()

            not_a_patient = root / "database"
            not_a_patient.mkdir()

            patients = discover_patient_dirs(root)

        self.assertEqual([path.name for path in patients], ["patient001"])


if __name__ == "__main__":
    unittest.main()
