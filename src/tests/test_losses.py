"""Tests for binary segmentation losses and UNet integration."""

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import torch
from torch.nn import functional as F

from src.losses import BCEDiceLoss, BCELoss, DiceLoss
from src.models import build_model


class TestBinarySegmentationLosses(unittest.TestCase):
    def setUp(self):
        self.targets = torch.tensor(
            [[[[0.0, 1.0], [1.0, 0.0]]]]
        )
        self.good_logits = torch.tensor(
            [[[[-20.0, 20.0], [20.0, -20.0]]]],
            requires_grad=True,
        )
        self.bad_logits = -self.good_logits.detach()

    def test_bce_matches_pytorch(self):
        criterion = BCELoss()
        actual = criterion(self.good_logits, self.targets)
        expected = F.binary_cross_entropy_with_logits(
            self.good_logits,
            self.targets,
        )

        self.assertTrue(torch.allclose(actual, expected))

    def test_dice_rewards_better_predictions(self):
        criterion = DiceLoss()
        good_loss = criterion(self.good_logits, self.targets)
        bad_loss = criterion(self.bad_logits, self.targets)

        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertLess(good_loss.item(), 1e-5)

    def test_combined_loss_uses_configured_weights(self):
        bce_weight = 0.25
        dice_weight = 0.75
        criterion = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        combined = criterion(self.good_logits, self.targets)
        expected = (
            bce_weight * BCELoss()(self.good_logits, self.targets)
            + dice_weight * DiceLoss()(self.good_logits, self.targets)
        )

        self.assertTrue(torch.allclose(combined, expected))

    def test_unet_forward_and_loss_backward(self):
        model = build_model({
            "name": "unet",
            "in_channels": 3,
            "num_classes": 1,
            "task": "binary",
            "base_channels": 4,
        })
        images = torch.randn(2, 3, 32, 32)
        targets = torch.randint(
            0,
            2,
            (2, 1, 32, 32),
            dtype=torch.float32,
        )

        output = model(images)
        loss = BCEDiceLoss()(output.logits, targets)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )

        print("\n=== UNet loss smoke test ===")
        print(f"Logits shape : {tuple(output.logits.shape)}")
        print(f"Targets shape: {tuple(targets.shape)}")
        print(f"BCE+Dice loss: {loss.item():.6f}")
        print("Backward     : PASS")

    def test_shape_mismatch_raises_clear_error(self):
        wrong_targets = torch.zeros(1, 2, 2)

        for criterion in (BCELoss(), DiceLoss(), BCEDiceLoss()):
            with self.subTest(criterion=type(criterion).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "must have the same shape",
                ):
                    criterion(self.good_logits, wrong_targets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
