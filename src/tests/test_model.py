"""Smoke test for constructing UNet and running one inference forward pass.

Run from the repository root with either:
    python src/tests/test_model.py
    python -m src.tests.test_model
"""

import sys
import unittest
from pathlib import Path

# Support direct execution without requiring the project to be installed first.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import torch

from src.models import SegmentationOutput, UNetModel, build_model


class TestUNetModel(unittest.TestCase):
    """Verify the common model factory and UNet forward contract."""

    def test_build_and_forward_unet(self):
        model_config = {
            "name": "unet",
            "in_channels": 3,
            "num_classes": 1,
            "task": "binary",
            "base_channels": 8,
        }

        model = build_model(model_config)
        model.eval()
        images = torch.randn(2, 3, 64, 64)

        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        print("\n=== UNet smoke test ===")
        print(f"Model class          : {type(model).__name__}")
        print(f"Model config         : {model_config}")
        print(f"Total parameters     : {total_parameters:,}")
        print(f"Trainable parameters : {trainable_parameters:,}")
        print(f"Model training mode  : {model.training}")
        print(f"Input shape          : {tuple(images.shape)}")
        print(f"Input dtype/device   : {images.dtype} / {images.device}")

        with torch.inference_mode():
            output = model(images)

        print(f"Output type          : {type(output).__name__}")
        print(f"Logits shape         : {tuple(output.logits.shape)}")
        print(f"Logits dtype/device  : {output.logits.dtype} / {output.logits.device}")
        print(f"Gradient tracking    : {output.logits.requires_grad}")

        self.assertIsInstance(model, UNetModel)
        self.assertIsInstance(output, SegmentationOutput)
        self.assertEqual(output.logits.shape, (2, 1, 64, 64))
        self.assertEqual(output.logits.dtype, images.dtype)
        self.assertFalse(output.logits.requires_grad)

        print("Result               : PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
