"""B2 parameter-matched dense tests."""

import unittest

import torch

from src.models.phase_b import ExpertBank
from src.models.phase_b.studies.build_up import (
    DENSE_HIDDEN_DIM,
    DenseEnhancer,
    DenseTokenFFN,
    nearest_dense_hidden_dim,
)


class TestB2(unittest.TestCase):
    def test_dense_ffn_receives_gradient(self):
        module = DenseEnhancer(embed_dim=8, num_levels=4, hidden_dim=19)
        tokens = tuple(torch.randn(2, 6, 8) for _ in range(4))
        alpha = torch.softmax(torch.randn(2, 4), dim=1)
        module(tokens, alpha).fused_tokens.square().mean().backward()
        self.assertGreater(float(module.dense.fc1.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.dense.fc2.weight.grad.abs().sum()), 0.0)

    def test_production_parameter_match_is_exactly_reported(self):
        bank = ExpertBank(embed_dim=768, num_experts=4, hidden_ratio=4)
        expert_parameters = sum(parameter.numel() for parameter in bank.parameters())
        dense_parameters = DenseTokenFFN.parameter_count(
            embed_dim=768,
            hidden_dim=DENSE_HIDDEN_DIM,
        )
        self.assertEqual(expert_parameters, 18_895_872)
        self.assertEqual(DENSE_HIDDEN_DIM, 12_292)
        self.assertEqual(dense_parameters, 18_895_108)
        self.assertEqual(dense_parameters - expert_parameters, -764)
        self.assertAlmostEqual(
            abs(dense_parameters - expert_parameters) / expert_parameters * 100,
            0.004043211130981412,
        )
        self.assertEqual(
            nearest_dense_hidden_dim(
                embed_dim=768,
                target_parameters=expert_parameters,
            ),
            DENSE_HIDDEN_DIM,
        )


if __name__ == "__main__":
    unittest.main()
