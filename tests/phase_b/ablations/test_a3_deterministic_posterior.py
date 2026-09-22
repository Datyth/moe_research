"""A3 deterministic posterior routing contract."""

import unittest

import torch

from src.models.phase_b import (
    GaussianParameterHead,
    PhaseBRouterHead,
    TopKRouter,
)


class TestA3DeterministicPosterior(unittest.TestCase):
    def test_posterior_mean_is_latent_and_routing_repeats(self):
        torch.manual_seed(4)
        head = PhaseBRouterHead(
            posterior_head=GaussianParameterHead(in_dim=12, latent_dim=5),
            prior_head=GaussianParameterHead(in_dim=7, latent_dim=5),
            router=TopKRouter(
                latent_dim=5,
                num_experts=4,
                active_experts=2,
            ),
        )
        image = torch.randn(3, 7)
        fused = torch.randn(3, 12)

        first = head(image, fused, sample=False)
        second = head(image, fused, sample=False)

        self.assertEqual(first.source, "posterior")
        torch.testing.assert_close(first.latent, first.posterior.mean)
        torch.testing.assert_close(first.latent, second.latent)
        torch.testing.assert_close(
            first.routing.dense_probs,
            second.routing.dense_probs,
        )
        torch.testing.assert_close(
            first.routing.expert_indices,
            second.routing.expert_indices,
        )


if __name__ == "__main__":
    unittest.main()
