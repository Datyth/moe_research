"""Tests for deterministic qualitative representative selection."""

from __future__ import annotations

import unittest

import torch

from src.shape_teacher.qualitative import (
    QUALITATIVE_CATEGORIES,
    mask_shape_statistics,
    select_representatives,
)


class TestQualitativeSelection(unittest.TestCase):
    def _record(
        self,
        index: int,
        *,
        area: float,
        irregularity: float,
        dice: float,
    ) -> dict:
        return {
            "dataset_index": index,
            "sample_id": f"sample_{index}",
            "mask_path": f"/masks/{index}.png",
            "area_ratio": area,
            "perimeter": 10.0,
            "irregularity": irregularity,
            "clean_soft_dice": dice,
        }

    def test_selects_all_five_categories_uniquely_and_deterministically(self):
        records = [
            self._record(0, area=0.1, irregularity=3.0, dice=0.9),
            self._record(1, area=0.9, irregularity=4.0, dice=0.8),
            self._record(2, area=0.4, irregularity=1.0, dice=0.7),
            self._record(3, area=0.5, irregularity=10.0, dice=0.6),
            self._record(4, area=0.6, irregularity=2.0, dice=0.1),
        ]
        expected_indices = [0, 1, 2, 3, 4]
        first = select_representatives(records)
        second = select_representatives(list(reversed(records)))

        self.assertEqual([item["category"] for item in first], list(QUALITATIVE_CATEGORIES))
        self.assertEqual([item["dataset_index"] for item in first], expected_indices)
        self.assertEqual(first, second)
        self.assertEqual(len({item["dataset_index"] for item in first}), 5)

    def test_path_breaks_equal_metric_ties(self):
        records = [
            self._record(0, area=0.2, irregularity=2.0, dice=0.5),
            self._record(1, area=0.2, irregularity=2.0, dice=0.5),
        ]
        records[0]["mask_path"] = "/masks/z.png"
        records[1]["mask_path"] = "/masks/a.png"
        selected = select_representatives(records)
        self.assertEqual(selected[0]["dataset_index"], 1)

    def test_irregular_mask_has_higher_compactness_penalty(self):
        smooth = torch.zeros(1, 32, 32)
        smooth[:, 8:24, 8:24] = 1
        irregular = torch.zeros(1, 32, 32)
        irregular[:, 8:24:2, 8:24] = 1
        irregular[:, 8:24, 8:24:2] = 1

        smooth_stats = mask_shape_statistics(smooth)
        irregular_stats = mask_shape_statistics(irregular)
        self.assertGreater(irregular_stats["irregularity"], smooth_stats["irregularity"])
        self.assertGreater(irregular_stats["perimeter"], smooth_stats["perimeter"])

    def test_rejects_empty_record_collection(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            select_representatives([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
