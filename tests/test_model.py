"""Smoke test for constructing UNet and running one inference forward pass.

Run from the repository root with unittest discovery.
"""

import unittest

import torch

from src.models import EsamModel, SegmentationOutput, UNetModel, build_model


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


class TestEsamModel(unittest.TestCase):
    """Verify the E-SAM model factory, forward contract and backbone freezing."""

    def _build(self, *, num_classes: int, task: str):
        return build_model(
            {
                "name": "esam",
                "in_channels": 3,
                "num_classes": num_classes,
                "task": task,
                "image_size": 64,
                "checkpoint": None,
                "moe_num_experts": 2,
                "moe_top_k_ratio": 0.5,
            }
        )

    def test_build_and_forward_binary(self):
        model = self._build(num_classes=1, task="binary")
        model.eval()
        images = torch.randn(2, 3, 64, 64)

        with torch.inference_mode():
            output = model(images)

        self.assertIsInstance(model, EsamModel)
        self.assertIsInstance(output, SegmentationOutput)
        self.assertEqual(output.logits.shape, (2, 1, 64, 64))
        self.assertIn("iou_predictions", output.diagnostics)

    def test_use_moe_false_drops_moe_module_and_still_runs(self):
        model = build_model(
            {
                "name": "esam",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "image_size": 64,
                "checkpoint": None,
                "use_moe": False,
            }
        )
        model.train()
        images = torch.randn(2, 3, 64, 64)

        output = model(images)
        output.logits.mean().backward()

        self.assertFalse(hasattr(model.network, "ExpertChoiceTokenMoE"))
        self.assertIsNone(output.diagnostics["moe_expert_indices"])
        self.assertEqual(output.logits.shape, (2, 1, 64, 64))

    def test_forward_multiclass_channel_count(self):
        model = self._build(num_classes=4, task="multiclass")
        model.eval()
        images = torch.randn(2, 3, 64, 64)

        with torch.inference_mode():
            output = model(images)

        self.assertEqual(output.logits.shape, (2, 4, 64, 64))

    def test_backward_updates_only_adapter_and_head_params(self):
        model = self._build(num_classes=1, task="binary")
        model.train()
        images = torch.randn(2, 3, 64, 64)

        output = model(images)
        loss = output.logits.mean()
        loss.backward()

        for name, param in model.network.image_encoder.named_parameters():
            self.assertEqual(param.requires_grad, "Adapter" in name, msg=name)
        for param in model.network.prompt_encoder.parameters():
            self.assertFalse(param.requires_grad)

        trainable_with_grad = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "iou_prediction_head" not in name
        ]
        missing_grad = [
            name
            for name in trainable_with_grad
            if dict(model.named_parameters())[name].grad is None
        ]
        self.assertEqual(missing_grad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
