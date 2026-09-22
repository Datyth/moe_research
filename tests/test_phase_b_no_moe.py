"""Controlled Phase B multi-level enhancement without Shape-MoE."""

import os
import unittest

import torch

from src.models.phase_b import MultiLevelImageDescriptor, PhaseBNoMoEStage


BATCH_SIZE = 2
EMBED_DIM = 768
DESCRIPTOR_DIM = 32
LEVELS = (3, 6, 9, 12)
PATCH_GRID = 4


def synthetic_block_outputs(depth: int = 12):
    return [
        torch.randn(BATCH_SIZE, PATCH_GRID, PATCH_GRID, EMBED_DIM)
        for _ in range(depth)
    ]


class TestPhaseBNoMoETokenFusion(unittest.TestCase):
    def test_descriptor_alpha_and_raw_token_fusion_contract(self):
        descriptor = MultiLevelImageDescriptor(
            embed_dim=EMBED_DIM,
            descriptor_dim=DESCRIPTOR_DIM,
            levels=LEVELS,
            scoring_hidden_dim=16,
        )
        output = descriptor(synthetic_block_outputs())
        fused = PhaseBNoMoEStage.fuse_level_tokens(
            output.level_tokens,
            output.level_weights,
        )

        self.assertEqual(tuple(output.level_weights.shape), (BATCH_SIZE, 4))
        torch.testing.assert_close(
            output.level_weights.sum(dim=1), torch.ones(BATCH_SIZE)
        )
        self.assertEqual(
            tuple(fused.shape),
            (BATCH_SIZE, PATCH_GRID * PATCH_GRID, EMBED_DIM),
        )
        expected = sum(
            output.level_weights[:, level].view(BATCH_SIZE, 1, 1)
            * output.level_tokens[level]
            for level in range(len(LEVELS))
        )
        torch.testing.assert_close(fused, expected)

    def test_token_fusion_rejects_misaligned_inputs(self):
        tokens = tuple(
            torch.randn(BATCH_SIZE, 8, 16) for _ in range(len(LEVELS))
        )
        weights = torch.full((BATCH_SIZE, len(LEVELS)), 0.25)

        with self.assertRaises(ValueError):
            PhaseBNoMoEStage.fuse_level_tokens(tokens[:-1], weights)
        with self.assertRaises(ValueError):
            PhaseBNoMoEStage.fuse_level_tokens(
                tokens[:-1] + (torch.randn(BATCH_SIZE, 7, 16),),
                weights,
            )
        with self.assertRaises(ValueError):
            PhaseBNoMoEStage.fuse_level_tokens(tokens, weights.unsqueeze(-1))

    def test_model_rejects_moe_or_disabled_lpeg_before_backbone_build(self):
        with self.assertRaisesRegex(ValueError, "use_moe=False"):
            PhaseBNoMoEStage(use_moe=True)
        with self.assertRaisesRegex(ValueError, "use_lpeg=True"):
            PhaseBNoMoEStage(use_lpeg=False)


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone in this test.",
)
class TestPhaseBNoMoEStageModel(unittest.TestCase):
    def test_forward_contract_and_absence_of_moe_modules(self):
        from src.models.phase_b import (
            ExpertBank,
            GaussianParameterHead,
            LayerPreferenceScorer,
            ShapeTeacher,
            TopKRouter,
        )

        torch.manual_seed(0)
        model = PhaseBNoMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.eval()

        forbidden_types = (
            ShapeTeacher,
            GaussianParameterHead,
            TopKRouter,
            ExpertBank,
            LayerPreferenceScorer,
        )
        self.assertFalse(
            any(isinstance(module, forbidden_types) for module in model.modules())
        )
        for attribute in (
            "shape_teacher",
            "posterior_head",
            "prior_head",
            "router",
            "experts",
        ):
            self.assertFalse(hasattr(model, attribute))

        network = model.backbone.network
        self.assertFalse(network.use_moe)
        self.assertTrue(network.use_lpeg)
        self.assertFalse(hasattr(network, "ExpertChoiceTokenMoE"))

        captured_aux_shape = None

        def capture_aux_shape(_module, _inputs, output):
            nonlocal captured_aux_shape
            captured_aux_shape = tuple(output.shape)

        handle = model.enhancement_neck.register_forward_hook(capture_aux_shape)
        try:
            with torch.no_grad():
                output = model(torch.randn(1, 3, 256, 256))
        finally:
            handle.remove()

        diagnostics = output.diagnostics
        alpha = diagnostics["level_weights"]
        self.assertEqual(tuple(alpha.shape), (1, len(LEVELS)))
        torch.testing.assert_close(alpha.sum(dim=1), torch.ones(1))
        self.assertEqual(captured_aux_shape, (1, 256, 16, 16))
        self.assertEqual(
            tuple(diagnostics["enhancement_aux_ratio"].shape), (1,)
        )
        self.assertGreater(
            float(diagnostics["enhancement_aux_ratio"].mean()), 0.0
        )
        self.assertEqual(tuple(diagnostics["level_weight_entropy"].shape), (1,))
        self.assertNotIn("phase_b_router", diagnostics)
        self.assertNotIn("phase_b_moe", diagnostics)
        self.assertNotIn("moe_expert_indices", diagnostics)
        self.assertEqual(tuple(output.logits.shape), (1, 1, 256, 256))

    def test_segmentation_gradient_reaches_fusion_and_neck_only_allowed_backbone(self):
        torch.manual_seed(0)
        model = PhaseBNoMoEStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.train()
        output = model(torch.randn(1, 3, 256, 256))
        output.logits.sum().backward()

        for projection in model.image_descriptor.level_projections:
            gradient = projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        for layer_index in (0, 2):
            gradient = model.image_descriptor.level_scoring[layer_index].weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)
            neck_gradient = model.enhancement_neck[layer_index].weight.grad
            self.assertIsNotNone(neck_gradient)
            self.assertGreater(float(neck_gradient.abs().sum()), 0.0)

        adapter_received_gradient = False
        for name, parameter in model.backbone.network.image_encoder.named_parameters():
            if "Adapter" in name:
                if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0:
                    adapter_received_gradient = True
            else:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
        self.assertTrue(adapter_received_gradient)


if __name__ == "__main__":
    unittest.main()
