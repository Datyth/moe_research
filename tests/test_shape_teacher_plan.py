"""Contract tests for the clean-vs-denoising Shape Teacher plan."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import sparse

from src.shape_teacher import (
    MaskCorruptor,
    MaskOnlyDataset,
    ShapeTeacher,
    ShapeTeacherLoss,
    audit_mask_splits,
)


def small_teacher() -> ShapeTeacher:
    return ShapeTeacher(
        image_size=(32, 32),
        encoder_channels=(8, 16),
        feature_dim=16,
        latent_dim=8,
        decoder_channels=(8, 4),
    )


def corruption_config() -> dict:
    settings = {
        "operation_count": [5, 5],
        "probabilities": {
            "erosion": 1.0,
            "dilation": 0.0,
            "random_holes": 0.0,
            "blob_removal": 0.0,
            "boundary_jitter": 0.0,
        },
        "morphology_radius": [1, 1],
    }
    return {"enabled": True, "training": settings, "evaluation": settings}


class TestShapeTeacherModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = small_teacher()
        self.masks = (torch.rand(2, 1, 32, 32) > 0.5).float()

    def test_forward_contract_and_positive_scale(self):
        logits, z, mu, sigma = self.model(self.masks)
        self.assertEqual(logits.shape, self.masks.shape)
        self.assertEqual(z.shape, (2, 8))
        self.assertEqual(mu.shape, (2, 8))
        self.assertEqual(sigma.shape, (2, 8))
        self.assertTrue(torch.all(torch.isfinite(sigma)))
        self.assertTrue(torch.all(sigma > 0))

    def test_sampling_contract(self):
        deterministic = self.model(self.masks, sample=False)
        self.assertTrue(torch.equal(deterministic.z, deterministic.mu))

        torch.manual_seed(123)
        first = self.model(self.masks, sample=True).z
        torch.manual_seed(123)
        second = self.model(self.masks, sample=True).z
        self.assertTrue(torch.equal(first, second))

    def test_shape_loss_has_no_kl_and_reaches_encoder_and_decoder(self):
        output = self.model(self.masks, sample=True)
        losses = ShapeTeacherLoss()(output.logits, self.masks)
        self.assertEqual(
            set(losses.as_dict()), {"loss", "bce", "dice_loss", "soft_dice"}
        )
        self.assertTrue(torch.allclose(losses.total, losses.bce + losses.dice))
        losses.total.backward()
        self.assertGreater(self.model.encoder[0][0].weight.grad.abs().sum(), 0)
        self.assertGreater(self.model.decoder_output.weight.grad.abs().sum(), 0)

    def test_checkpoint_round_trip_is_deterministic_in_eval_mode(self):
        self.model.eval()
        expected = self.model(self.masks, sample=False).logits
        clone = small_teacher()
        clone.load_state_dict(self.model.state_dict())
        clone.eval()
        actual = clone(self.masks, sample=False).logits
        self.assertTrue(torch.equal(actual, expected))


class TestMaskCorruption(unittest.TestCase):
    def test_fixed_eval_corruption_is_reproducible_and_preserves_target(self):
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1
        original = target.clone()
        first = MaskCorruptor(corruption_config(), seed=42, evaluation=True)
        second = MaskCorruptor(corruption_config(), seed=42, evaluation=True)
        corrupted_a = first(target, keys=["sample"], split="val")
        corrupted_b = second(target, keys=["sample"], split="val")

        self.assertTrue(torch.equal(corrupted_a, corrupted_b))
        self.assertFalse(torch.equal(corrupted_a, target))
        self.assertTrue(torch.equal(target, original))
        self.assertEqual(set(torch.unique(corrupted_a).tolist()), {0.0, 1.0})


class TestMaskOnlyDataset(unittest.TestCase):
    def _record(self, root: Path, split: str, sample_id: str) -> dict[str, str]:
        directory = root / "labels" / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sample_id}.(1,16,20).npz"
        mask = np.zeros((1, 16, 20), dtype=np.uint8)
        mask[:, 4:12, 5:15] = 1
        sparse.save_npz(path, sparse.csr_matrix(mask.reshape(1, -1)))
        return {"label": str(path.relative_to(root))}

    def test_preprocessing_and_clean_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._record(root, "train", "A")
            dataset = MaskOnlyDataset(
                root=root,
                records=[record],
                split="train",
                image_size=(32, 32),
            )
            sample = dataset[0]
            self.assertEqual(sample["mask"].shape, (1, 32, 32))
            self.assertEqual(set(torch.unique(sample["mask"]).tolist()), {0.0, 1.0})
            teacher_input = sample["mask"]
            target = sample["mask"]
            self.assertTrue(torch.equal(teacher_input, target))

    def test_duplicate_path_across_splits_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self._record(root, "train", "A")
            datasets = {
                split: MaskOnlyDataset(
                    root=root,
                    records=[record],
                    split=split,
                    image_size=(32, 32),
                )
                for split in ("train", "val")
            }
            with self.assertRaisesRegex(ValueError, "Duplicate mask path"):
                audit_mask_splits(datasets, scan_masks=False)


class TestTeacherConfigs(unittest.TestCase):
    def test_clean_and_denoise_configs_only_change_allowed_experiment_fields(self):
        root = Path(__file__).resolve().parents[1]
        clean = yaml.safe_load(
            (root / "configs/teacher_clean_isic2018.yaml").read_text()
        )
        denoise = yaml.safe_load(
            (root / "configs/teacher_denoise_isic2018.yaml").read_text()
        )
        self.assertEqual(clean["model"], denoise["model"])
        self.assertEqual(clean["loss"], denoise["loss"])
        self.assertEqual(clean["dataset"], denoise["dataset"])
        self.assertEqual(clean["optimizer"], denoise["optimizer"])
        self.assertEqual(clean["scheduler"], denoise["scheduler"])
        clean_training = dict(clean["training"])
        denoise_training = dict(denoise["training"])
        self.assertEqual(clean_training.pop("input_mode"), "clean")
        self.assertEqual(denoise_training.pop("input_mode"), "denoise")
        self.assertEqual(clean_training, denoise_training)
        clean_corruption = dict(clean["corruption"])
        denoise_corruption = dict(denoise["corruption"])
        self.assertFalse(clean_corruption.pop("enabled"))
        self.assertTrue(denoise_corruption.pop("enabled"))
        self.assertEqual(clean_corruption, denoise_corruption)


if __name__ == "__main__":
    unittest.main(verbosity=2)
