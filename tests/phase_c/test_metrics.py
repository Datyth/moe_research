"""Unit tests for Phase-C distribution and routing-transfer metrics."""

import unittest

import torch

from src.models.phase_b.posterior import DiagonalGaussian, gaussian_kl
from src.models.phase_c.metrics import (
    categorical_routing_kl,
    latent_transfer_metrics,
    routing_js,
    topk_transfer_metrics,
)


class TestPhaseCMetrics(unittest.TestCase):
    def test_gaussian_kl_and_latent_distances_are_finite(self):
        posterior = DiagonalGaussian(
            mean=torch.zeros(3, 4),
            std=torch.ones(3, 4),
        )
        prior = DiagonalGaussian(
            mean=torch.full((3, 4), 0.25, requires_grad=True),
            std=torch.full((3, 4), 1.5, requires_grad=True),
        )
        latent_kl = gaussian_kl(posterior.detach(), prior).mean()
        self.assertTrue(torch.isfinite(latent_kl))
        self.assertGreaterEqual(float(latent_kl.detach()), 0.0)
        distances = latent_transfer_metrics(posterior, prior)
        self.assertEqual(
            set(distances),
            {"mean_distance", "std_distance", "wasserstein2_squared"},
        )
        self.assertTrue(all(torch.isfinite(value) for value in distances.values()))

    def test_categorical_kl_is_finite_and_backpropagates_to_student(self):
        teacher = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
        student_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4]],
            requires_grad=True,
        )
        loss = categorical_routing_kl(teacher, student_logits)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertGreater(float(student_logits.grad.abs().sum()), 0.0)

    def test_topk_set_metrics_ignore_order_and_use_active_count(self):
        metrics = topk_transfer_metrics(
            torch.tensor([[1, 3], [0, 2]], dtype=torch.long),
            torch.tensor([[3, 1], [0, 1]], dtype=torch.long),
            num_experts=4,
        )
        self.assertAlmostEqual(
            float(metrics["exact_topk_set_agreement"]),
            0.5,
        )
        self.assertAlmostEqual(float(metrics["topk_overlap"]), 0.75)

    def test_routing_js_is_finite_symmetric_and_zero_for_identity(self):
        first = torch.tensor([[0.6, 0.2, 0.1, 0.1]])
        second = torch.tensor([[0.1, 0.1, 0.2, 0.6]])
        forward = routing_js(first, second)
        reverse = routing_js(second, first)
        self.assertTrue(torch.isfinite(forward))
        torch.testing.assert_close(forward, reverse)
        torch.testing.assert_close(routing_js(first, first), torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
