"""Architecture and forward-contract tests for the promptless SAM baseline."""

from __future__ import annotations

import unittest

import torch

from src.models import PromptlessSamModel, SegmentationOutput, build_model
from src.models.esam._vendor.image_encoder import AdapterBlock, Block


class TestPromptlessSamModel(unittest.TestCase):
    def _build(self, *, num_classes: int = 1, task: str = "binary"):
        return build_model(
            {
                "name": "sam",
                "in_channels": 3,
                "num_classes": num_classes,
                "task": task,
                "image_size": 64,
                "checkpoint": None,
                "freeze_image_encoder": True,
                "freeze_prompt_encoder": True,
            }
        )

    def test_architecture_is_adapter_free_and_freezes_only_encoders(self):
        model = self._build()

        self.assertIsInstance(model, PromptlessSamModel)
        self.assertTrue(
            all(isinstance(block, Block) for block in model.network.image_encoder.blocks)
        )
        self.assertFalse(
            any(isinstance(module, AdapterBlock) for module in model.modules())
        )
        self.assertFalse(hasattr(model.network, "ExpertChoiceTokenMoE"))
        self.assertIsNone(model.network.prompt_encoder.lpeg)
        self.assertFalse(model.network.use_moe)
        self.assertFalse(model.network.use_lpeg)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.network.image_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.network.prompt_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.network.mask_decoder.parameters()
            )
        )

    def test_forward_uses_empty_sparse_prompt_and_returns_binary_logits(self):
        model = self._build()
        model.eval()
        captured: dict[str, torch.Tensor] = {}

        def capture_prompt_output(module, inputs, output):
            captured["sparse"] = output[0]
            captured["dense"] = output[1]

        handle = model.network.prompt_encoder.register_forward_hook(
            capture_prompt_output
        )
        try:
            with torch.inference_mode():
                output = model(torch.randn(2, 3, 64, 64))
        finally:
            handle.remove()

        self.assertIsInstance(output, SegmentationOutput)
        self.assertEqual(tuple(output.logits.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(captured["sparse"].shape), (2, 0, 256))
        self.assertEqual(tuple(captured["dense"].shape), (2, 256, 4, 4))
        self.assertIn("iou_predictions", output.diagnostics)
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_multiclass_decoder_emits_requested_channel_count(self):
        model = self._build(num_classes=4, task="multiclass")
        model.eval()

        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64))

        self.assertEqual(tuple(output.logits.shape), (1, 4, 64, 64))

    def test_rejects_non_rgb_and_non_patch_aligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "in_channels=3"):
            build_model(
                {
                    "name": "sam",
                    "in_channels": 1,
                    "num_classes": 1,
                    "task": "binary",
                    "image_size": 64,
                }
            )
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            build_model(
                {
                    "name": "sam",
                    "in_channels": 3,
                    "num_classes": 1,
                    "task": "binary",
                    "image_size": 62,
                }
            )


if __name__ == "__main__":
    unittest.main()
