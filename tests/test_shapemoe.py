"""CPU tests for the Phase 1 ShapeMoE segmenter, router, experts and losses."""

import unittest

import torch

from scripts.training.train_shapemoe import run_epoch as run_shapemoe_epoch
from src.losses import (
    ShapeMoELoss,
    expert_balance_cv2_loss,
    gaussian_kl_divergence,
)
from src.models import build_model
from src.models.shapemoe import (
    ExpertMaskHeads,
    ShapeAwareSparseRouter,
    ShapeDistributionEncoder,
)


def build_small_segmenter(**overrides):
    config = {
        "name": "shapemoe_unet",
        "in_channels": 3,
        "num_classes": 1,
        "task": "binary",
        "base_channels": 8,
        "latent_dim": 16,
        "num_experts": 4,
        "top_k": 1,
    }
    config.update(overrides)
    return build_model(config)


class TestShapeDistributionEncoder(unittest.TestCase):
    def test_two_heads_produce_posterior(self):
        encoder = ShapeDistributionEncoder(in_channels=32, latent_dim=16)
        mu, logvar = encoder(torch.rand(4, 32, 8, 8))

        self.assertEqual(mu.shape, (4, 16))
        self.assertEqual(logvar.shape, (4, 16))

    def test_logvar_is_clamped(self):
        encoder = ShapeDistributionEncoder(in_channels=32, latent_dim=16)
        with torch.no_grad():
            encoder.fc_logvar.bias.fill_(500.0)
            encoder.fc_logvar.weight.fill_(0.0)
        _, logvar = encoder(torch.rand(2, 32, 8, 8))

        self.assertTrue(torch.all(logvar <= encoder.logvar_max))

    def test_rejects_non_feature_map(self):
        encoder = ShapeDistributionEncoder(in_channels=32, latent_dim=16)
        with self.assertRaises(ValueError):
            encoder(torch.rand(4, 32))


class TestShapeAwareSparseRouter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.router = ShapeAwareSparseRouter(
            latent_dim=16,
            num_experts=4,
            top_k=1,
        )
        self.mu = torch.randn(8, 16)
        self.logvar = torch.zeros(8, 16)

    def test_pi_is_sparse_and_normalized(self):
        routing = self.router(self.mu, self.logvar)

        self.assertEqual(routing.pi.shape, (8, 4))
        self.assertTrue(torch.allclose(routing.pi.sum(dim=1), torch.ones(8)))
        self.assertTrue(torch.all((routing.pi > 0).sum(dim=1) == 1))

    def test_top_k_two_activates_two_experts(self):
        router = ShapeAwareSparseRouter(latent_dim=16, num_experts=4, top_k=2)
        routing = router(self.mu, self.logvar)

        self.assertTrue(torch.all((routing.pi > 0).sum(dim=1) == 2))
        self.assertTrue(torch.allclose(routing.pi.sum(dim=1), torch.ones(8)))

    def test_selected_expert_is_the_argmax_score(self):
        routing = self.router(self.mu, self.logvar, sample=False)

        self.assertTrue(
            torch.equal(routing.indices.flatten(), routing.scores.argmax(dim=1))
        )

    def test_dense_probabilities_cover_all_experts(self):
        routing = self.router(self.mu, self.logvar)

        self.assertTrue(torch.all(routing.probabilities > 0))
        self.assertTrue(
            torch.allclose(routing.probabilities.sum(dim=1), torch.ones(8))
        )

    def test_sampling_disabled_is_deterministic(self):
        first = self.router(self.mu, self.logvar, sample=False).latent
        second = self.router(self.mu, self.logvar, sample=False).latent

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, self.mu))

    def test_rejects_top_k_above_expert_count(self):
        with self.assertRaises(ValueError):
            ShapeAwareSparseRouter(latent_dim=16, num_experts=2, top_k=3)


class TestExpertMaskHeads(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.heads = ExpertMaskHeads(in_channels=8, num_classes=1, num_experts=3)
        self.features = torch.rand(4, 8, 16, 16)

    def test_routing_selects_the_matching_head(self):
        pi = torch.zeros(4, 3)
        pi[:, 1] = 1.0
        combined = self.heads(self.features, pi)
        direct = self.heads.heads[1](self.features)

        self.assertTrue(torch.allclose(combined, direct, atol=1e-6))

    def test_two_experts_are_blended_by_their_weights(self):
        pi = torch.zeros(2, 3)
        pi[:, 0] = 0.25
        pi[:, 2] = 0.75
        features = self.features[:2]
        combined = self.heads(features, pi)
        expected = 0.25 * self.heads.heads[0](features) + 0.75 * self.heads.heads[
            2
        ](features)

        self.assertTrue(torch.allclose(combined, expected, atol=1e-6))

    def test_unselected_head_receives_no_gradient(self):
        pi = torch.zeros(4, 3)
        pi[:, 0] = 1.0
        self.heads(self.features, pi).sum().backward()

        self.assertIsNotNone(self.heads.heads[0].weight.grad)
        self.assertIsNone(self.heads.heads[2].weight.grad)

    def test_expert_usage_counts_assignments(self):
        pi = torch.zeros(4, 3)
        pi[0:3, 1] = 1.0
        pi[3, 2] = 1.0

        self.assertTrue(
            torch.equal(
                self.heads.expert_usage(pi),
                torch.tensor([0, 3, 1]),
            )
        )


class TestBalanceAndDistillationLosses(unittest.TestCase):
    def test_cv2_is_zero_for_perfect_balance(self):
        probabilities = torch.full((8, 4), 0.25)

        self.assertAlmostEqual(
            expert_balance_cv2_loss(probabilities).item(),
            0.0,
            places=6,
        )

    def test_cv2_grows_when_one_expert_dominates(self):
        balanced = torch.full((8, 4), 0.25)
        skewed = torch.zeros(8, 4)
        skewed[:, 0] = 1.0

        self.assertGreater(
            expert_balance_cv2_loss(skewed).item(),
            expert_balance_cv2_loss(balanced).item(),
        )

    def test_cv2_has_gradient_through_dense_probabilities(self):
        scores = torch.randn(8, 4, requires_grad=True)
        expert_balance_cv2_loss(scores.softmax(dim=1)).backward()

        self.assertGreater(scores.grad.abs().sum().item(), 0.0)

    def test_distillation_kl_is_zero_for_identical_posteriors(self):
        mu = torch.randn(4, 16)
        logvar = torch.randn(4, 16)
        divergence = gaussian_kl_divergence(mu, logvar, mu.clone(), logvar.clone())

        self.assertTrue(torch.allclose(divergence, torch.zeros(4), atol=1e-5))

    def test_distillation_kl_grows_with_posterior_gap(self):
        mu = torch.zeros(2, 16)
        logvar = torch.zeros(2, 16)
        near = gaussian_kl_divergence(mu, logvar, mu + 0.5, logvar).mean()
        far = gaussian_kl_divergence(mu, logvar, mu + 3.0, logvar).mean()

        self.assertGreater(far.item(), near.item())

    def test_distillation_kl_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            gaussian_kl_divergence(
                torch.zeros(4, 16),
                torch.zeros(4, 16),
                torch.zeros(4, 8),
                torch.zeros(4, 8),
            )


class TestShapeMoESegmenter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_small_segmenter()
        self.images = torch.rand(4, 3, 64, 64)
        self.targets = (torch.rand(4, 1, 64, 64) > 0.5).float()

    def test_output_contract_and_diagnostics(self):
        self.model.train()
        output = self.model(self.images)

        self.assertEqual(output.logits.shape, (4, 1, 64, 64))
        for key in ("mu", "logvar", "pi", "router_probabilities", "expert_index"):
            self.assertIn(key, output.diagnostics)

    def test_evaluation_routing_is_deterministic(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(self.images)
            second = self.model(self.images)

        self.assertTrue(torch.equal(first.logits, second.logits))

    def test_training_routing_is_stochastic(self):
        model = build_small_segmenter(latent_dim=16)
        model.train()
        with torch.no_grad():
            first = model(self.images).diagnostics["latent"]
            second = model(self.images).diagnostics["latent"]

        self.assertFalse(torch.allclose(first, second))

    def test_predict_still_honours_the_base_contract(self):
        prediction = self.model.predict(self.images)

        self.assertEqual(prediction.masks.shape, (4, 1, 64, 64))
        self.assertEqual(prediction.probabilities.shape, (4, 1, 64, 64))

    def test_unused_trunk_projection_is_dropped(self):
        names = [name for name, _ in self.model.named_parameters()]

        self.assertFalse(any(name.startswith("trunk.out_conv") for name in names))

    def test_full_objective_reaches_every_trainable_block(self):
        self.model.train()
        criterion = ShapeMoELoss(
            segmentation={"name": "bce_dice"},
            balance_weight=0.1,
            distillation_weight=1.0,
        )
        teacher = (torch.zeros(4, 16), torch.zeros(4, 16))
        losses = criterion(
            self.model(self.images),
            self.targets,
            teacher_posterior=teacher,
        )
        losses.total.backward()

        for name, module in (
            ("router", self.model.router.gate),
            ("shape encoder", self.model.shape_encoder.fc_mu),
            ("variance head", self.model.shape_encoder.fc_logvar),
        ):
            self.assertIsNotNone(module.weight.grad, f"{name} received no gradient")
            self.assertGreater(
                module.weight.grad.abs().sum().item(),
                0.0,
                f"{name} gradient is all zeros",
            )

    def test_distillation_is_optional(self):
        criterion = ShapeMoELoss(distillation_weight=1.0)
        losses = criterion(self.model(self.images), self.targets)

        self.assertEqual(losses.distillation.item(), 0.0)

    def test_loss_rejects_a_plain_segmentation_output(self):
        from src.models.base import SegmentationOutput

        with self.assertRaises(KeyError):
            ShapeMoELoss()(
                SegmentationOutput(logits=torch.zeros(4, 1, 64, 64)),
                self.targets,
            )

    def test_evaluation_epoch_reports_all_segmentation_metrics(self):
        metrics = run_shapemoe_epoch(
            model=self.model,
            loader=[{"image": self.images, "mask": self.targets}],
            criterion=ShapeMoELoss(),
            device=torch.device("cpu"),
            include_surface_metrics=True,
        )

        expected = {
            "loss",
            "segmentation",
            "balance",
            "distillation",
            "dice",
            "iou",
            "hd95",
            "assd",
            "boundary_f1",
        }
        self.assertTrue(expected.issubset(metrics))

    def test_model_config_round_trips(self):
        rebuilt = build_model({"name": "shapemoe_unet", **self.model.model_config()})
        rebuilt.load_state_dict(self.model.state_dict())

        self.assertEqual(rebuilt.num_experts, self.model.num_experts)
        self.assertEqual(rebuilt.latent_dim, self.model.latent_dim)


if __name__ == "__main__":
    unittest.main()
