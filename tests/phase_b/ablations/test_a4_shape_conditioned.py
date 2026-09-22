"""A4 dedicated registry initialization and sparse-compute contract."""

import os
import unittest
from unittest.mock import patch

import torch

from src.models.base import BaseSegmentationModel
from src.models.phase_b import (
    PhaseBA4ShapeConditionedStage,
    ShapeConditionedFusionMoEEnhancement,
)


def _fake_a0_init(instance, **kwargs):
    BaseSegmentationModel.__init__(
        instance,
        in_channels=3,
        num_classes=1,
        task="binary",
    )
    instance.enhancement_mode = kwargs["enhancement_mode"]
    instance.enhancement = ShapeConditionedFusionMoEEnhancement(
        embed_dim=8,
        num_experts=4,
        num_levels=4,
        latent_dim=5,
        expert_hidden_ratio=1,
    )


class TestA4ShapeConditioned(unittest.TestCase):
    def test_registry_specific_final_scorer_is_zero_initialized(self):
        with patch(
            "src.models.phase_b.ablations.a4_shape_conditioned_fusion."
            "PhaseBMoEStage.__init__",
            new=_fake_a0_init,
        ):
            model = PhaseBA4ShapeConditionedStage()

        final = model.enhancement.g_fuse[-1]
        torch.testing.assert_close(final.weight, torch.zeros_like(final.weight))
        torch.testing.assert_close(final.bias, torch.zeros_like(final.bias))

        tokens = tuple(torch.randn(2, 4, 8) for _ in range(4))
        latent = torch.randn(2, 5)
        indices = torch.tensor([[0, 2], [0, 2]])
        probabilities = torch.zeros(2, 4).scatter(1, indices, 0.5)
        output = model.enhancement(
            tokens,
            latent,
            probabilities,
            indices,
        )
        torch.testing.assert_close(
            output.layer_weights,
            torch.full((2, 4), 0.25),
        )

    def test_cannot_override_fixed_mode(self):
        with self.assertRaisesRegex(ValueError, "fixes enhancement_mode"):
            PhaseBA4ShapeConditionedStage(enhancement_mode="hierarchical")


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone in this test.",
)
class TestA4FullBackbone(unittest.TestCase):
    def test_posterior_and_prior_keep_decoder_contract(self):
        torch.manual_seed(0)
        model = PhaseBA4ShapeConditionedStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.eval()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()
        with torch.no_grad():
            posterior = model(images, masks=masks)
            prior = model(images)

        self.assertEqual(
            posterior.diagnostics["phase_b_router"].source,
            "posterior",
        )
        self.assertEqual(
            prior.diagnostics["phase_b_router"].source,
            "prior",
        )
        for output in (posterior, prior):
            alpha = output.diagnostics["phase_b_moe"].layer_weights
            torch.testing.assert_close(alpha, torch.full((1, 4), 0.25))
            self.assertEqual(
                tuple(output.logits.shape),
                (1, 1, 256, 256),
            )


if __name__ == "__main__":
    unittest.main()
