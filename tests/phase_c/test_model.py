"""Model API tests for Phase-C prior inference and gated B6 integration."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.configs import load_experiment_config
from src.losses import build_loss
from src.models import build_model
from src.models.phase_b.posterior import GaussianParameterHead
from src.models.phase_b.router import TopKRouter
from src.models.phase_c import PhaseCB6PriorDistill
from src.tasks import PhaseCDistillTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CallProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Privileged component was called by prior inference.")


class _ForwardHarness(PhaseCB6PriorDistill):
    """Exercise the real public forward/train methods without constructing SAM."""

    def __init__(self):
        nn.Module.__init__(self)
        self.image_size = 4
        self.prior_head = GaussianParameterHead(
            in_dim=3,
            latent_dim=2,
            std_floor=1e-4,
        )
        self.router_for_test = TopKRouter(
            latent_dim=2,
            num_experts=3,
            active_experts=2,
        )
        for parameter in self.router_for_test.parameters():
            parameter.requires_grad_(False)
        self.frozen_dropout = nn.Dropout(p=0.5)
        self.shape_teacher = _CallProbe()
        self.fusion = _CallProbe()
        self.posterior_branch = _CallProbe()
        self.encoder_calls = 0

    def _encode_images(self, images, *, include_teacher_descriptor=False):
        self.encoder_calls += 1
        return images

    def _prior_from_encoded(self, encoded):
        descriptor = encoded.mean(dim=(2, 3))
        prior = self.prior_head(descriptor)
        routing = self.router_for_test(prior.mean)
        return prior, routing

    def _prior_layer_scorer_module(self):
        return self.frozen_dropout

    def _decode_routing(
        self,
        encoded,
        *,
        latent,
        routing,
        layer_scorer,
        descriptor=None,
        expert_bank=None,
    ):
        scalar = latent[:, :1] + routing.routing_probs[:, :1]
        logits = scalar[:, :, None, None].expand(
            -1,
            1,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        iou = scalar
        gamma = torch.softmax(
            torch.stack([scalar[:, 0], -scalar[:, 0]], dim=1), dim=1
        )
        beta = gamma[:, None, :].expand(-1, 2, -1)
        return SimpleNamespace(
            logits=logits, iou_predictions=iou, gamma=gamma, beta=beta
        )

    def distillation_forward(
        self,
        images,
        masks,
        *,
        decode_prior=False,
        decode_posterior=False,
    ):
        self.posterior_branch(images)
        raise AssertionError("Diagnostic path is outside this unit harness.")


class TestPhaseCModelAPI(unittest.TestCase):
    def test_gaussian_prior_shape_and_positive_standard_deviation(self):
        head = GaussianParameterHead(
            in_dim=256,
            latent_dim=64,
            std_floor=1e-4,
        )
        prior = head(torch.randn(3, 256))
        self.assertEqual(tuple(prior.mean.shape), (3, 64))
        self.assertEqual(tuple(prior.std.shape), (3, 64))
        self.assertTrue(bool((prior.std > 0).all()))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 32896)

    def test_image_only_forward_ignores_mask_and_privileged_modules(self):
        torch.manual_seed(3)
        model = _ForwardHarness().eval()
        images = torch.randn(2, 3, 4, 4)
        mask_a = torch.randn(2, 1, 4, 4)
        mask_b = torch.randn(2, 1, 4, 4)

        without_mask = model(images)
        with_mask_a = model(images, masks=mask_a)
        with_mask_b = model(images, masks=mask_b)

        torch.testing.assert_close(
            without_mask.logits,
            with_mask_a.prior_logits,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            without_mask.logits,
            with_mask_b.prior_logits,
            rtol=0,
            atol=0,
        )
        self.assertEqual(model.encoder_calls, 3)
        self.assertEqual(model.shape_teacher.calls, 0)
        self.assertEqual(model.fusion.calls, 0)
        self.assertEqual(model.posterior_branch.calls, 0)
        self.assertIsNone(with_mask_a.posterior_logits)
        self.assertIsNone(with_mask_b.posterior_logits)

    def test_train_keeps_frozen_children_eval_and_prior_training(self):
        model = _ForwardHarness()
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.prior_head.training)
        self.assertFalse(model.router_for_test.training)
        self.assertFalse(model.frozen_dropout.training)
        self.assertFalse(model.shape_teacher.training)
        self.assertFalse(model.fusion.training)
        self.assertFalse(model.posterior_branch.training)


@unittest.skipUnless(
    os.environ.get("RUN_PHASE_C_INTEGRATION_TESTS") == "1",
    "Set RUN_PHASE_C_INTEGRATION_TESTS=1 for the real SAM/B6 checkpoint test.",
)
class TestPhaseCRealB6Integration(unittest.TestCase):
    def test_teacher_parity_gradients_validation_and_image_only_smoke(self):
        phase_c_config = load_experiment_config(
            PROJECT_ROOT / "configs/phase_c/b6_distill.yaml",
            project_root=PROJECT_ROOT,
        )
        b6_config = load_experiment_config(
            PROJECT_ROOT / "configs/phase_b/build_up/b6_hierarchical.yaml",
            project_root=PROJECT_ROOT,
        )
        shared = {
            "in_channels": 3,
            "num_classes": 1,
            "task": "binary",
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phase_c = build_model({**phase_c_config["model"], **shared}).to(device)
        phase_c.eval()

        total = sum(parameter.numel() for parameter in phase_c.parameters())
        trainable_names = [
            name
            for name, parameter in phase_c.named_parameters()
            if parameter.requires_grad
        ]
        trainable = sum(
            parameter.numel()
            for parameter in phase_c.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(total, 119569347)
        self.assertEqual(trainable, 32896)
        self.assertEqual(
            trainable_names,
            [
                "prior_head.mean_head.weight",
                "prior_head.mean_head.bias",
                "prior_head.scale_head.weight",
                "prior_head.scale_head.bias",
            ],
        )
        metadata = phase_c.phase_c_checkpoint_metadata()
        self.assertEqual(metadata["parameter_counts"]["frozen"], 119536451)
        self.assertEqual(metadata["parameter_counts"]["trainable"], 32896)

        generator = torch.Generator().manual_seed(42)
        images = torch.randn(1, 3, 256, 256, generator=generator).to(device)
        masks = (
            torch.rand(1, 1, 256, 256, generator=generator) > 0.5
        ).float().to(device)

        with torch.inference_mode():
            diagnostic = phase_c(
                images,
                masks=masks,
                include_posterior=True,
            )
            prior_only = phase_c(images)
        phase_c_state = diagnostic.distillation_state
        self.assertIsNotNone(phase_c_state)
        self.assertEqual(tuple(prior_only.logits.shape), (1, 1, 256, 256))
        self.assertTrue(torch.equal(prior_only.logits, diagnostic.prior_logits))

        standalone = build_model({**b6_config["model"], **shared}).to(device)
        teacher_checkpoint = torch.load(
            phase_c_config["model"]["teacher_checkpoint"],
            map_location=device,
            weights_only=False,
        )
        standalone.load_state_dict(
            teacher_checkpoint["model_state_dict"],
            strict=True,
        )
        standalone.eval()
        with torch.inference_mode():
            standalone_logits = standalone(images, masks=masks).logits

        difference = (
            phase_c_state.posterior_logits - standalone_logits
        ).abs()
        message = (
            f"teacher logit mismatch: max={float(difference.max())}, "
            f"mean={float(difference.mean())}"
        )
        self.assertTrue(
            torch.equal(phase_c_state.posterior_logits, standalone_logits),
            message,
        )
        del standalone

        task = PhaseCDistillTask(
            criterion=build_loss(phase_c_config["loss"]),
            lambda_latent=1.0,
            lambda_route=1.0,
            lambda_deploy=0.0,
        )
        phase_c.train()
        step = task.training_step(
            phase_c,
            {"image": images, "mask": masks},
            device,
        )
        step.loss.backward()
        gradient_names = [
            name
            for name, parameter in phase_c.named_parameters()
            if parameter.grad is not None
        ]
        self.assertEqual(gradient_names, trainable_names)
        self.assertFalse(phase_c.backbone.training)
        self.assertFalse(phase_c.conditioner.training)
        self.assertFalse(phase_c.enhancement.training)

        phase_c.eval()
        with torch.inference_mode():
            validation = task.evaluation_step(
                phase_c,
                {"image": images, "mask": masks},
                device,
            )
        self.assertIn("dice", validation.metrics)
        self.assertIn("posterior_dice", validation.metrics)
        self.assertTrue(torch.isfinite(validation.metrics["dice"]))


if __name__ == "__main__":
    unittest.main()
