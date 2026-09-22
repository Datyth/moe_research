"""B5 posterior-only Gaussian routing tests."""

import unittest

import torch

from src.models.phase_b.studies.build_up import GaussianConditioner


class TestB5(unittest.TestCase):
    def _module(self, stochastic=True):
        return GaussianConditioner(
            in_dim=12,
            latent_dim=5,
            num_experts=4,
            active_experts=2,
            std_floor=1e-4,
            stochastic=stochastic,
        )

    def test_posterior_shapes_positive_scale_and_no_prior_or_kl(self):
        module = self._module()
        module.eval()
        output = module(torch.randn(3, 12))
        self.assertEqual(tuple(output.posterior.mean.shape), (3, 5))
        self.assertEqual(tuple(output.posterior.std.shape), (3, 5))
        self.assertTrue((output.posterior.std > 0).all())
        torch.testing.assert_close(output.latent, output.posterior.mean)
        self.assertIsNone(output.latent_kl)
        self.assertFalse(hasattr(module, "prior_head"))

    def test_stochastic_training_uses_reparameterization(self):
        module = self._module(stochastic=True)
        module.train()
        descriptor = torch.randn(2, 12)
        torch.manual_seed(7)
        output = module(descriptor)
        torch.manual_seed(7)
        posterior = module.posterior_head(descriptor)
        expected = posterior.rsample()
        torch.testing.assert_close(output.latent, expected)

    def test_deterministic_training_uses_mean(self):
        module = self._module(stochastic=False)
        module.train()
        output = module(torch.randn(2, 12))
        torch.testing.assert_close(output.latent, output.posterior.mean)


if __name__ == "__main__":
    unittest.main()
