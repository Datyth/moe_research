"""B6 hierarchical expert-layer routing tests."""

import copy
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

    def test_external_layer_scorer_matches_default_without_sharing(self):
        torch.manual_seed(9)
        module = HierarchicalMoEEnhancement(
            embed_dim=12,
            num_experts=3,
            num_levels=2,
            latent_dim=4,
            expert_hidden_ratio=1,
        ).eval()
        external = copy.deepcopy(module.layer_scorer)
        self.assertIsNot(external, module.layer_scorer)
        self.assertNotEqual(
            external.expert_embeddings.data_ptr(),
            module.layer_scorer.expert_embeddings.data_ptr(),
        )

        tokens = tuple(torch.randn(2, 5, 12) for _ in range(2))
        pools = torch.stack([value.mean(1) for value in tokens], dim=1)
        latent = torch.randn(2, 4)
        indices = torch.tensor([[0, 2], [1, 2]])
        probs = torch.zeros(2, 3).scatter(1, indices, 0.5)
        default = module(tokens, pools, latent, probs, indices)
        overridden = module(
            tokens,
            pools,
            latent,
            probs,
            indices,
            layer_scorer=external,
        )
        torch.testing.assert_close(
            default.fused_tokens,
            overridden.fused_tokens,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            default.layer_weights,
            overridden.layer_weights,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
