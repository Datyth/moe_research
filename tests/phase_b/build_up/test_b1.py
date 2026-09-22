"""B1 multi-level enhancement tests."""

import unittest

import torch

from src.models.phase_b import ExpertBank, ShapeTeacher, TopKRouter
from src.models.phase_b.studies.build_up import IdentityEnhancer, PhaseBB1MultiLevel


class TestB1(unittest.TestCase):
    def test_identity_enhancement_fuses_raw_tokens(self):
        module = IdentityEnhancer(embed_dim=8, num_levels=4)
        tokens = tuple(torch.randn(2, 6, 8) for _ in range(4))
        alpha = torch.softmax(torch.randn(2, 4), dim=1)
        output = module(tokens, alpha)
        expected = sum(alpha[:, index, None, None] * value for index, value in enumerate(tokens))
        torch.testing.assert_close(output.fused_tokens, expected)
        self.assertEqual(tuple(output.fused_tokens.shape), (2, 6, 8))

    def test_gradient_reaches_level_descriptor_and_neck(self):
        from src.models.phase_b import MultiLevelImageDescriptor
        from src.models.phase_b.studies.build_up.common import tokens_to_spatial

        descriptor = MultiLevelImageDescriptor(
            embed_dim=24,
            descriptor_dim=8,
            levels=(1, 2, 3, 4),
            scoring_hidden_dim=4,
        )
        enhancer = IdentityEnhancer(embed_dim=24, num_levels=4)
        neck = torch.nn.Conv2d(24, 8, 1, bias=False)
        block_outputs = [torch.randn(2, 2, 2, 24) for _ in range(4)]
        output = descriptor(block_outputs)
        enhanced = enhancer(output.level_tokens, output.level_weights)
        embeddings = torch.zeros(2, 8, 2, 2)
        spatial = tokens_to_spatial(
            enhanced.fused_tokens, embeddings, embed_dim=24
        )
        neck(spatial).square().mean().backward()

        for projection in descriptor.level_projections:
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)
        for index in (0, 2):
            gradient = descriptor.level_scoring[index].weight.grad
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertGreater(float(neck.weight.grad.abs().sum()), 0.0)

    def test_b1_rejects_legacy_moe_or_unfrozen_backbone_early(self):
        with self.assertRaisesRegex(ValueError, "use_moe=False"):
            PhaseBB1MultiLevel(use_moe=True)
        with self.assertRaisesRegex(ValueError, "use_lpeg=True"):
            PhaseBB1MultiLevel(use_lpeg=False)
        with self.assertRaisesRegex(ValueError, "freeze_backbone=True"):
            PhaseBB1MultiLevel(freeze_backbone=False)

    def test_identity_enhancer_contains_no_forbidden_module(self):
        module = IdentityEnhancer(embed_dim=8, num_levels=4)
        forbidden = (ExpertBank, ShapeTeacher, TopKRouter)
        self.assertFalse(any(isinstance(child, forbidden) for child in module.modules()))


if __name__ == "__main__":
    unittest.main()
