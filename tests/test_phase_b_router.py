"""Phase B up to the Top-K router: posterior/prior, KL, sparse routing.

These exercise the routing machinery on synthetic descriptors, so they run
without instantiating the ViT-B backbone. A full-model smoke test that builds
the backbone is gated behind RUN_BACKBONE_TESTS=1.
"""

import os
import unittest

import torch

from src.models.phase_b import (
    DiagonalGaussian,
    GaussianParameterHead,
    PhaseBRouterHead,
    TopKRouter,
    gaussian_kl,
    load_balance_loss,
)
from src.models.phase_b.posterior import GaussianParameterHead as _Head  # noqa: F401


class TestGaussianParameterHead(unittest.TestCase):
    def test_outputs_mean_and_positive_std(self):
        head = GaussianParameterHead(in_dim=32, latent_dim=8)
        gauss = head(torch.randn(4, 32))
        self.assertEqual(tuple(gauss.mean.shape), (4, 8))
        self.assertEqual(tuple(gauss.std.shape), (4, 8))
        self.assertTrue(torch.all(gauss.std > 0))

    def test_rejects_wrong_input_dim(self):
        head = GaussianParameterHead(in_dim=16, latent_dim=8)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 15))


class TestReparameterization(unittest.TestCase):
    def test_rsample_matches_mu_plus_sigma_eps(self):
        mean = torch.randn(3, 5)
        std = torch.rand(3, 5) + 0.1
        gauss = DiagonalGaussian(mean=mean, std=std)

        drawn = gauss.rsample(torch.Generator().manual_seed(0))
        eps = torch.randn(3, 5, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(drawn, mean + std * eps)

    def test_rsample_is_differentiable_in_mean_and_std(self):
        head = GaussianParameterHead(in_dim=6, latent_dim=4)
        gauss = head(torch.randn(2, 6))
        gauss.rsample().sum().backward()
        self.assertIsNotNone(head.mean_head.weight.grad)
        self.assertIsNotNone(head.scale_head.weight.grad)


class TestGaussianKL(unittest.TestCase):
    def _gauss(self, mean, std):
        return DiagonalGaussian(mean=mean, std=std)

    def test_zero_when_distributions_match(self):
        g = self._gauss(torch.randn(4, 8), torch.rand(4, 8) + 0.2)
        kl = gaussian_kl(g, self._gauss(g.mean.clone(), g.std.clone()))
        torch.testing.assert_close(kl, torch.zeros(4), atol=1e-6, rtol=0)

    def test_matches_torch_distributions(self):
        q = self._gauss(torch.randn(5, 7), torch.rand(5, 7) + 0.1)
        p = self._gauss(torch.randn(5, 7), torch.rand(5, 7) + 0.1)
        oracle = torch.distributions.kl_divergence(
            torch.distributions.Normal(q.mean, q.std),
            torch.distributions.Normal(p.mean, p.std),
        ).sum(dim=1)
        torch.testing.assert_close(gaussian_kl(q, p), oracle)

    def test_positive_and_asymmetric(self):
        q = self._gauss(torch.zeros(1, 4), torch.ones(1, 4))
        p = self._gauss(torch.ones(1, 4), torch.ones(1, 4) * 2)
        self.assertGreater(gaussian_kl(q, p).item(), 0.0)
        self.assertNotAlmostEqual(
            gaussian_kl(q, p).item(), gaussian_kl(p, q).item(), places=4
        )

    def test_detach_stops_gradient_into_posterior(self):
        qhead = GaussianParameterHead(in_dim=6, latent_dim=4)
        phead = GaussianParameterHead(in_dim=6, latent_dim=4)
        q = qhead(torch.randn(2, 6))
        p = phead(torch.randn(2, 6))
        gaussian_kl(q.detach(), p).mean().backward()
        self.assertIsNone(qhead.mean_head.weight.grad)
        self.assertIsNotNone(phead.mean_head.weight.grad)


class TestTopKRouter(unittest.TestCase):
    def test_routing_is_sparse_and_normalized(self):
        router = TopKRouter(latent_dim=8, num_experts=4, active_experts=2)
        out = router(torch.randn(6, 8))
        self.assertEqual(tuple(out.routing_probs.shape), (6, 4))
        self.assertEqual(tuple(out.expert_indices.shape), (6, 2))
        # Exactly k_e non-zero entries per row.
        self.assertTrue(torch.all((out.routing_probs > 0).sum(dim=1) == 2))
        torch.testing.assert_close(out.routing_probs.sum(dim=1), torch.ones(6))
        torch.testing.assert_close(out.dense_probs.sum(dim=1), torch.ones(6))

    def test_selected_experts_are_the_largest_dense_probs(self):
        router = TopKRouter(latent_dim=8, num_experts=4, active_experts=2)
        out = router(torch.randn(5, 8))
        for b in range(5):
            selected = set(out.expert_indices[b].tolist())
            ranked = out.dense_probs[b].argsort(descending=True)[:2].tolist()
            self.assertEqual(selected, set(ranked))

    def test_gradient_flows_to_router(self):
        router = TopKRouter(latent_dim=8, num_experts=4, active_experts=2)
        router(torch.randn(3, 8)).routing_probs.sum().backward()
        self.assertIsNotNone(router.route.weight.grad)

    def test_rejects_active_greater_than_num_experts(self):
        with self.assertRaises(ValueError):
            TopKRouter(latent_dim=8, num_experts=2, active_experts=3)


class TestLoadBalanceLoss(unittest.TestCase):
    def test_collapse_is_penalized_more_than_balanced(self):
        num_experts = 4
        # Balanced: probabilities spread evenly, selections spread evenly.
        balanced_probs = torch.full((8, num_experts), 0.25)
        balanced_idx = torch.tensor([[0, 1], [2, 3]] * 4)
        balanced = load_balance_loss(
            balanced_probs, balanced_idx, num_experts=num_experts
        )
        # Collapsed: all mass and all selections on expert 0/1.
        collapsed_probs = torch.zeros(8, num_experts)
        collapsed_probs[:, 0] = 0.9
        collapsed_probs[:, 1] = 0.1
        collapsed_idx = torch.tensor([[0, 1]] * 8)
        collapsed = load_balance_loss(
            collapsed_probs, collapsed_idx, num_experts=num_experts
        )
        self.assertGreater(collapsed.item(), balanced.item())

    def test_is_scalar(self):
        loss = load_balance_loss(
            torch.softmax(torch.randn(4, 4), dim=1),
            torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
            num_experts=4,
        )
        self.assertEqual(loss.ndim, 0)


class TestPhaseBRouterHead(unittest.TestCase):
    def _head(self, latent_dim=8, num_experts=4, active_experts=2):
        return PhaseBRouterHead(
            posterior_head=GaussianParameterHead(in_dim=12, latent_dim=latent_dim),
            prior_head=GaussianParameterHead(in_dim=5, latent_dim=latent_dim),
            router=TopKRouter(
                latent_dim=latent_dim,
                num_experts=num_experts,
                active_experts=active_experts,
            ),
        )

    def test_posterior_path_when_mask_present(self):
        head = self._head()
        h_i = torch.randn(3, 5)
        h_q = torch.randn(3, 12)
        stage = head(h_i, h_q, sample=True)
        self.assertEqual(stage.source, "posterior")
        self.assertIsNotNone(stage.posterior)
        self.assertIsNotNone(stage.latent_kl)
        self.assertIsNotNone(stage.balance)
        self.assertEqual(tuple(stage.routing.routing_probs.shape), (3, 4))

    def test_prior_path_when_mask_absent_uses_prior_mean(self):
        head = self._head()
        h_i = torch.randn(3, 5)
        stage = head(h_i, None, sample=True)
        self.assertEqual(stage.source, "prior")
        self.assertIsNone(stage.posterior)
        self.assertIsNone(stage.latent_kl)
        self.assertIsNone(stage.balance)
        torch.testing.assert_close(stage.latent, stage.prior.mean)

    def test_deterministic_posterior_uses_mean(self):
        head = self._head()
        h_i = torch.randn(2, 5)
        h_q = torch.randn(2, 12)
        stage = head(h_i, h_q, sample=False)
        torch.testing.assert_close(stage.latent, stage.posterior.mean)


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone in this test.",
)
class TestPhaseBRouterStageModel(unittest.TestCase):
    def test_forward_adds_router_diagnostic_and_keeps_seg_logits(self):
        from src.models.phase_b import PhaseBRouterStage

        torch.manual_seed(0)
        model = PhaseBRouterStage(image_size=256, freeze_backbone=True)
        model.eval()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()

        with torch.no_grad():
            train_out = model(images, masks=masks)
            infer_out = model(images)

        self.assertIn("phase_b_router", train_out.diagnostics)
        self.assertEqual(train_out.diagnostics["phase_b_router"].source, "posterior")
        self.assertEqual(infer_out.diagnostics["phase_b_router"].source, "prior")


if __name__ == "__main__":
    unittest.main()
