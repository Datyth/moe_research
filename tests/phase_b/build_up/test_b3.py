"""B3 learned and random image-only routing tests."""

import unittest

import torch

from src.models.phase_b import TopKRouter
from src.models.phase_b.studies.build_up import (
    DirectConditioner,
    RandomTopKRouter,
    SparseMoEEnhancer,
)


class TestB3(unittest.TestCase):
    def test_learned_router_is_sparse_and_normalized(self):
        conditioner = DirectConditioner(
            router=TopKRouter(latent_dim=8, num_experts=4, active_experts=2),
            source="image",
        )
        output = conditioner(torch.randn(3, 8))
        self.assertEqual(output.source, "image")
        self.assertTrue(torch.isfinite(output.balance))
        torch.testing.assert_close(output.routing.routing_probs.sum(1), torch.ones(3))
        self.assertTrue((output.routing.routing_probs != 0).sum(1).eq(2).all())

    def test_random_router_is_uniform_and_seed_reproducible(self):
        router = RandomTopKRouter(latent_dim=8, num_experts=4, active_experts=2)
        latent = torch.randn(5, 8)
        torch.manual_seed(11)
        first = router(latent)
        torch.manual_seed(11)
        second = router(latent)
        torch.testing.assert_close(first.expert_indices, second.expert_indices)
        torch.testing.assert_close(first.dense_probs, torch.full((5, 4), 0.25))
        selected = first.routing_probs.gather(1, first.expert_indices)
        torch.testing.assert_close(selected, torch.full((5, 2), 0.5))
        conditioner = DirectConditioner(router=router, source="image")
        torch.manual_seed(11)
        conditioned = conditioner(latent)
        torch.testing.assert_close(conditioned.balance, torch.tensor(1.0))
        self.assertEqual(sum(parameter.numel() for parameter in router.parameters()), 0)

    def test_only_selected_experts_receive_gradient(self):
        module = SparseMoEEnhancer(
            embed_dim=8,
            num_levels=4,
            num_experts=4,
            expert_hidden_ratio=2,
        )
        tokens = tuple(torch.randn(2, 6, 8) for _ in range(4))
        alpha = torch.full((2, 4), 0.25)
        indices = torch.tensor([[0, 2], [0, 2]])
        probs = torch.zeros(2, 4).scatter(1, indices, 0.5)
        output = module(tokens, alpha, probs, indices)
        output.fused_tokens.square().mean().backward()
        for expert_id, expert in enumerate(module.experts.experts):
            gradient = expert.fc1.weight.grad
            if expert_id in (0, 2):
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0.0)


if __name__ == "__main__":
    unittest.main()
