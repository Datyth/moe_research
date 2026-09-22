"""B6 hierarchical expert-layer routing tests."""

import unittest

import torch

from src.models.phase_b import HierarchicalMoEEnhancement


class TestB6(unittest.TestCase):
    def test_beta_gamma_shapes_normalization_and_gradients(self):
        module = HierarchicalMoEEnhancement(
            embed_dim=24,
            num_experts=4,
            num_levels=4,
            latent_dim=5,
            expert_hidden_ratio=2,
        )
        tokens = tuple(torch.randn(2, 6, 24) for _ in range(4))
        pools = torch.stack([value.mean(1) for value in tokens], dim=1)
        latent = torch.randn(2, 5)
        indices = torch.tensor([[0, 2], [0, 2]])
        probs = torch.zeros(2, 4).scatter(1, indices, 0.5)
        output = module(tokens, pools, latent, probs, indices)
        self.assertEqual(tuple(output.expert_layer_weights.shape), (2, 2, 4))
        self.assertEqual(tuple(output.layer_weights.shape), (2, 4))
        torch.testing.assert_close(
            output.expert_layer_weights.sum(2),
            torch.ones(2, 2),
        )
        torch.testing.assert_close(output.layer_weights.sum(1), torch.ones(2))

        output.fused_tokens.square().mean().backward()
        scorer_gradient = module.layer_scorer.scorer[0].weight.grad
        self.assertIsNotNone(scorer_gradient)
        self.assertGreater(float(scorer_gradient.abs().sum()), 0.0)
        for expert_id, expert in enumerate(module.experts.experts):
            gradient = expert.fc1.weight.grad
            if expert_id in (0, 2):
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0.0)


if __name__ == "__main__":
    unittest.main()
