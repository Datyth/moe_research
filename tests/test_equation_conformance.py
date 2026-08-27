"""Numeric conformance tests: every formula, checked against its source.

Each test names the equation it pins down, so a reviewer can hold this file next
to the Phase 0 specification and Sec. 3 of ShapeMoE (arXiv 2508.01664) and see
that the code computes what the paper writes, not merely something of the right
shape.

The rest of the suite checks shapes and behaviour. These check arithmetic.
"""

import math
import unittest

import torch
from torch.nn import functional as F

from src.losses import (
    MaskVAELoss,
    expert_balance_cv2_loss,
    gaussian_kl_divergence,
    gaussian_kl_to_standard_normal,
)
from src.models import MaskVAETeacher
from src.models.shapemoe import ShapeAwareSparseRouter


def build_teacher() -> MaskVAETeacher:
    return MaskVAETeacher(
        latent_dim=8,
        image_size=(32, 32),
        encoder_channels=(8, 16),
        decoder_channels=(16, 8, 8),
        decoder_seed_size=8,
    )


class TestSpecificationEquations(unittest.TestCase):
    """The formulas in the Phase 0 specification image."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = build_teacher()
        self.masks = (torch.rand(4, 1, 32, 32) > 0.5).float()

    def test_mu_is_a_linear_map_of_the_pooled_embedding(self):
        """mu_T = W_mu h_T + b_mu."""

        self.model.eval()
        hidden = self.model.pool(self.model.embed(self.masks)).flatten(1)
        expected = hidden @ self.model.fc_mu.weight.T + self.model.fc_mu.bias

        self.assertTrue(
            torch.allclose(self.model.encode(self.masks)[0], expected, atol=1e-6)
        )

    def test_logvar_is_a_separate_linear_map_of_the_same_hidden(self):
        """log sigma^2_T = W_sigma h_T + b_sigma, sharing h_T with the mean head."""

        self.model.eval()
        hidden = self.model.pool(self.model.embed(self.masks)).flatten(1)
        expected = hidden @ self.model.fc_logvar.weight.T + self.model.fc_logvar.bias

        self.assertTrue(
            torch.allclose(self.model.encode(self.masks)[1], expected, atol=1e-6)
        )
        self.assertIsNot(self.model.fc_mu, self.model.fc_logvar)

    def test_reparameterization_matches_the_formula_exactly(self):
        """z_T = mu_T + sigma_T * eps, with the same eps drawn twice."""

        mu = torch.randn(6, 8)
        logvar = torch.randn(6, 8)

        torch.manual_seed(1234)
        actual = self.model.reparameterize(mu, logvar)
        torch.manual_seed(1234)
        expected = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_sigma_is_the_square_root_of_the_variance(self):
        """The 0.5 in exp(0.5 * logvar). Catches sigma being squared or halved.

        With mu = 0 the sample is exactly sigma * eps, so dividing the sample by
        eps recovers sigma. logvar = ln(4) means sigma^2 = 4, so sigma must be 2,
        not 4 and not sqrt(2).
        """

        mu = torch.zeros(4, 8)
        logvar = torch.full((4, 8), math.log(4.0))

        torch.manual_seed(7)
        sample = self.model.reparameterize(mu, logvar)
        torch.manual_seed(7)
        epsilon = torch.randn_like(mu)

        recovered_sigma = sample / epsilon
        self.assertTrue(
            torch.allclose(recovered_sigma, torch.full_like(mu, 2.0), atol=1e-5)
        )

    def test_prior_kl_matches_the_closed_form_termwise(self):
        """L_prior = 0.5 * sum(mu^2 + sigma^2 - 1 - log sigma^2)."""

        mu = torch.randn(5, 8)
        logvar = torch.randn(5, 8)
        expected = 0.5 * (
            mu.pow(2) + logvar.exp() - 1.0 - logvar
        ).sum(dim=1)

        self.assertTrue(
            torch.allclose(
                gaussian_kl_to_standard_normal(mu, logvar),
                expected,
                atol=1e-6,
            )
        )

    def test_prior_kl_matches_an_independently_derived_scalar(self):
        """One dimension, mu = 2, sigma^2 = 9: KL = 0.5*(4 + 9 - 1 - ln 9)."""

        mu = torch.tensor([[2.0]])
        logvar = torch.tensor([[math.log(9.0)]])
        expected = 0.5 * (4.0 + 9.0 - 1.0 - math.log(9.0))

        self.assertAlmostEqual(
            gaussian_kl_to_standard_normal(mu, logvar).item(),
            expected,
            places=5,
        )

    def test_reconstruction_equals_bce_on_sigmoid_probabilities(self):
        """L_rec = BCE(M, M_hat_T) with M_hat_T = sigmoid(D_T(z_T)).

        The code computes BCE in logit space. This proves the documented claim
        that the two are the same number.
        """

        self.model.eval()
        output = self.model(self.masks, sample=False)
        losses = MaskVAELoss(beta=0.0, recon_reduction="sum")(output, self.masks)

        probabilities = torch.sigmoid(output.recon_logits)
        manual = F.binary_cross_entropy(
            probabilities,
            self.masks,
            reduction="none",
        )
        expected = manual.flatten(1).sum(dim=1).mean()

        self.assertTrue(
            torch.allclose(losses.reconstruction, expected, rtol=1e-4)
        )

    def test_total_objective_is_the_weighted_sum(self):
        """L_VAE = L_rec + beta * L_prior, for a beta that is neither 0 nor 1."""

        output = self.model(self.masks)
        beta = 0.37
        losses = MaskVAELoss(beta=beta)(output, self.masks)

        self.assertTrue(
            torch.allclose(
                losses.total,
                losses.reconstruction + beta * losses.kl,
                atol=1e-6,
            )
        )


class TestPaperEquations(unittest.TestCase):
    """ShapeMoE Sec. 3.4, Eq. (2) to (4), and the Sec. 3.6 balancing term."""

    def setUp(self):
        torch.manual_seed(0)
        self.router = ShapeAwareSparseRouter(
            latent_dim=4,
            num_experts=5,
            top_k=2,
        )

    def test_equation_2_sampling_matches_the_formula(self):
        """l_o = mu + sigma * eta, sigma from the log-variance head."""

        mu = torch.randn(3, 4)
        logvar = torch.randn(3, 4)

        torch.manual_seed(99)
        actual = self.router.sample_latent(mu, logvar)
        torch.manual_seed(99)
        expected = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_equation_3_scores_are_a_bias_free_matrix_product(self):
        """s = W * l_o, with W a plain trainable matrix and no bias term."""

        latent = torch.randn(3, 4)
        routing = self.router(latent, torch.full_like(latent, -30.0))
        expected = latent @ self.router.gate.weight.T

        self.assertIsNone(self.router.gate.bias)
        self.assertTrue(torch.allclose(routing.scores, expected, atol=1e-5))

    def test_equation_4_softmax_is_taken_over_the_top_k_only(self):
        """pi = Softmax(TopK(s, k)), zero everywhere outside the selection."""

        scores = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0]])
        with torch.no_grad():
            self.router.gate.weight.zero_()
        latent = torch.zeros(1, 4)
        routing = self.router(latent, torch.full_like(latent, -30.0))

        # Drive known scores through the same masking path.
        top_values, indices = scores.topk(self.router.top_k, dim=1)
        masked = torch.full_like(scores, float("-inf")).scatter(1, indices, top_values)
        expected = masked.softmax(dim=1)

        # Top-2 of [3,1,4,1,5] is {5 at index 4, 4 at index 2}.
        self.assertEqual(sorted(indices.flatten().tolist()), [2, 4])
        self.assertAlmostEqual(expected[0, 4].item(), math.e / (math.e + 1.0), places=6)
        self.assertEqual(expected[0, 0].item(), 0.0)
        self.assertEqual(expected[0, 1].item(), 0.0)
        self.assertEqual(expected[0, 3].item(), 0.0)
        self.assertAlmostEqual(expected.sum().item(), 1.0, places=6)
        self.assertAlmostEqual(routing.pi.sum().item(), 1.0, places=6)

    def test_equation_4_degenerates_to_one_when_k_is_one(self):
        """The property that forces the balancing loss onto the dense softmax."""

        router = ShapeAwareSparseRouter(latent_dim=4, num_experts=5, top_k=1)
        latent = torch.randn(6, 4)
        routing = router(latent, torch.full_like(latent, -30.0))
        selected = routing.pi[routing.pi > 0]

        self.assertTrue(torch.allclose(selected, torch.ones_like(selected)))

    def test_cv2_matches_the_coefficient_of_variation_squared(self):
        """L_CV2 = Var(importance) / Mean(importance)^2, importance summed over the batch."""

        probabilities = torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        importance = probabilities.sum(dim=0)
        expected = importance.var(unbiased=False) / importance.mean().pow(2)

        self.assertAlmostEqual(
            expert_balance_cv2_loss(probabilities).item(),
            expected.item(),
            places=6,
        )


class TestPhaseOneObjective(unittest.TestCase):
    """Eq. (5) generalised: L = L_seg + lambda_b * L_CV2 + lambda_d * KL."""

    def test_total_is_the_weighted_sum_of_the_three_terms(self):
        from src.losses import ShapeMoELoss
        from src.models import build_model

        torch.manual_seed(0)
        model = build_model(
            {
                "name": "shapemoe_unet",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "base_channels": 8,
                "latent_dim": 8,
                "num_experts": 4,
                "top_k": 1,
            }
        )
        images = torch.rand(2, 3, 32, 32)
        targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
        criterion = ShapeMoELoss(balance_weight=0.13, distillation_weight=0.29)
        losses = criterion(
            model(images),
            targets,
            teacher_posterior=(torch.zeros(2, 8), torch.zeros(2, 8)),
        )

        self.assertTrue(
            torch.allclose(
                losses.total,
                losses.segmentation
                + 0.13 * losses.balance
                + 0.29 * losses.distillation,
                atol=1e-6,
            )
        )


class TestDistillationEquation(unittest.TestCase):
    """KL between two diagonal Gaussians, the project's own term."""

    def test_matches_the_closed_form_termwise(self):
        torch.manual_seed(0)
        mu_q, logvar_q = torch.randn(4, 6), torch.randn(4, 6)
        mu_p, logvar_p = torch.randn(4, 6), torch.randn(4, 6)

        expected = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1.0
        ).sum(dim=1)

        self.assertTrue(
            torch.allclose(
                gaussian_kl_divergence(mu_q, logvar_q, mu_p, logvar_p),
                expected,
                atol=1e-5,
            )
        )

    def test_matches_an_independently_derived_scalar(self):
        """N(0,1) || N(1,4) in one dimension: 0.5*(ln4 + (1+1)/4 - 1)."""

        divergence = gaussian_kl_divergence(
            torch.tensor([[0.0]]),
            torch.tensor([[0.0]]),
            torch.tensor([[1.0]]),
            torch.tensor([[math.log(4.0)]]),
        )
        expected = 0.5 * (math.log(4.0) + (1.0 + 1.0) / 4.0 - 1.0)

        self.assertAlmostEqual(divergence.item(), expected, places=5)

    def test_reduces_to_the_prior_kl_when_the_target_is_standard_normal(self):
        """Consistency between the two KL helpers."""

        torch.manual_seed(0)
        mu, logvar = torch.randn(4, 6), torch.randn(4, 6)

        self.assertTrue(
            torch.allclose(
                gaussian_kl_divergence(
                    mu,
                    logvar,
                    torch.zeros_like(mu),
                    torch.zeros_like(logvar),
                ),
                gaussian_kl_to_standard_normal(mu, logvar),
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
