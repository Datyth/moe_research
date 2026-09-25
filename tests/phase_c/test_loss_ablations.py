"""CPU tests for Phase-C loss ablations C1-C4."""

import copy
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from src.configs import load_experiment_config
from src.configs.experiment import resolve_experiment_config
from src.losses import BCEDiceLoss
from src.models.base import SegmentationOutput
from src.models.phase_b.posterior import GaussianParameterHead, gaussian_kl
from src.models.phase_b.router import TopKRouter
from src.models.phase_c import (
    PhaseCDistillationState,
    categorical_routing_kl,
)
from src.tasks import PhaseCDistillTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs/phase_c"
ABLATION_ROOT = CONFIG_ROOT / "loss_ablations"
TEACHER = (
    PROJECT_ROOT
    / "runs/phase_b_b6_hierarchical/20260921T151109Z_seed-42/best.pt"
)


class TinyAblationModel(nn.Module):
    """Differentiable Phase-C stand-in with observable teacher calls."""

    def __init__(self) -> None:
        super().__init__()
        self.prior_head = GaussianParameterHead(
            in_dim=4,
            latent_dim=2,
            std_floor=1e-4,
        )
        self.posterior_head = GaussianParameterHead(
            in_dim=4,
            latent_dim=2,
            std_floor=1e-4,
        )
        self.router = TopKRouter(
            latent_dim=2,
            num_experts=3,
            active_experts=2,
        )
        for module in (self.posterior_head, self.router):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.teacher_calls = 0
        self.posterior_calls = 0
        self.prior_calls = 0
        self.last_state = None
        self.last_prior_logits = None
        self.last_prior_routing = None

    def _prior_path(self, images):
        descriptor = images.mean(dim=(2, 3))
        prior = self.prior_head(descriptor)
        routing = self.router(prior.mean)
        scalar = prior.mean[:, :1] + routing.routing_probs[:, :1]
        logits = scalar[:, :, None, None].expand(
            -1,
            1,
            images.shape[-2],
            images.shape[-1],
        )
        self.last_prior_logits = logits
        self.last_prior_routing = routing
        return prior, routing, logits

    def forward(self, images, *, masks=None, include_posterior=False):
        if include_posterior:
            raise AssertionError("Tiny prior forward does not provide diagnostics.")
        self.prior_calls += 1
        _, _, logits = self._prior_path(images)
        return SegmentationOutput(logits=logits)

    def distillation_forward(
        self,
        images,
        masks,
        *,
        decode_prior=False,
        decode_posterior=False,
    ):
        self.teacher_calls += 1
        prior, prior_routing, prior_decoded = self._prior_path(images)

        with torch.no_grad():
            self.posterior_calls += 1
            descriptor = masks.mean(dim=(2, 3)).expand(-1, 4)
            posterior = self.posterior_head(descriptor)
            posterior_routing = self.router(posterior.mean)

        prior_logits = prior_decoded if decode_prior else None
        posterior_logits = None
        if decode_posterior:
            scalar = posterior.mean[:, :1]
            posterior_logits = scalar[:, :, None, None].expand(
                -1,
                1,
                masks.shape[-2],
                masks.shape[-1],
            )

        prior_gamma = (
            prior_routing.dense_probs if decode_prior else None
        )
        posterior_gamma = (
            posterior_routing.dense_probs if decode_posterior else None
        )
        state = PhaseCDistillationState(
            posterior=posterior,
            prior=prior,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            prior_gamma=prior_gamma,
            posterior_gamma=posterior_gamma,
        )
        self.last_state = state
        return state


def make_batch():
    generator = torch.Generator().manual_seed(19)
    images = torch.rand(2, 4, 4, 4, generator=generator)
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, :, 1:3, 1:3] = 1
    masks[1, :, :3, 1:3] = 1
    return {"image": images, "mask": masks}


def make_task(objective, latent, route, deploy):
    return PhaseCDistillTask(
        criterion=BCEDiceLoss(),
        latent_objective=objective,
        lambda_latent=latent,
        lambda_route=route,
        lambda_deploy=deploy,
        threshold=0.5,
        boundary_tolerance=1,
    )


class TestPhaseCLossObjectives(unittest.TestCase):
    CASES = {
        "c1": ("gaussian_kl", 0.001, 1.0, 1.0),
        "c2": ("mean_mse", 1.0, 1.0, 1.0),
        "c3": ("none", 0.0, 1.0, 1.0),
        "c4": ("none", 0.0, 0.0, 1.0),
    }

    def test_latent_objective_defaults_and_validation(self):
        default = PhaseCDistillTask(
            criterion=BCEDiceLoss(),
            lambda_latent=1.0,
            lambda_route=1.0,
            lambda_deploy=0.0,
        )
        self.assertEqual(default.latent_objective, "gaussian_kl")

        with self.assertRaisesRegex(ValueError, "latent_objective"):
            make_task("cosine", 1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "lambda_latent"):
            make_task("none", 1.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            make_task("none", 0.0, 0.0, 0.0)

    def test_c1_through_c4_objective_formulas(self):
        batch = make_batch()
        for name, spec in self.CASES.items():
            with self.subTest(name=name):
                torch.manual_seed(7)
                model = TinyAblationModel()
                task = make_task(*spec)
                step = task.training_step(model, batch, torch.device("cpu"))

                if name == "c4":
                    expected_latent = step.loss.new_zeros(())
                    expected_route = step.loss.new_zeros(())
                    expected_deploy = task.criterion(
                        model.last_prior_logits,
                        batch["mask"],
                    )
                    self.assertEqual(model.teacher_calls, 0)
                    self.assertEqual(model.posterior_calls, 0)
                else:
                    state = model.last_state
                    if name == "c1":
                        raw_latent = gaussian_kl(
                            state.posterior.detach(),
                            state.prior,
                        ).mean()
                        expected_latent = 0.001 * raw_latent
                        torch.testing.assert_close(
                            step.metrics["gaussian_kl"],
                            raw_latent,
                        )
                        torch.testing.assert_close(
                            step.metrics["latent_kl"],
                            raw_latent,
                        )
                    elif name == "c2":
                        raw_latent = F.mse_loss(
                            state.prior.mean,
                            state.posterior.mean.detach(),
                            reduction="mean",
                        )
                        expected_latent = raw_latent
                        torch.testing.assert_close(
                            step.metrics["mean_mse"],
                            raw_latent,
                        )
                    else:
                        expected_latent = step.loss.new_zeros(())

                    expected_route = categorical_routing_kl(
                        state.posterior_routing.dense_probs,
                        state.prior_routing.logits,
                    )
                    expected_deploy = task.criterion(
                        state.prior_logits,
                        batch["mask"],
                    )
                    self.assertEqual(model.teacher_calls, 1)
                    self.assertEqual(model.posterior_calls, 1)

                expected = (
                    expected_latent + expected_route + expected_deploy
                )
                torch.testing.assert_close(step.loss, expected)
                torch.testing.assert_close(
                    step.metrics["weighted_latent_loss"],
                    expected_latent,
                )
                torch.testing.assert_close(
                    step.metrics["weighted_route_loss"],
                    expected_route,
                )
                torch.testing.assert_close(
                    step.metrics["weighted_deploy_seg_loss"],
                    expected_deploy,
                )
                torch.testing.assert_close(
                    step.metrics["total_loss"],
                    expected,
                )

    def test_c2_mean_mse_is_elementwise_mean(self):
        model = TinyAblationModel()
        task = make_task("mean_mse", 1.0, 1.0, 1.0)
        step = task.training_step(model, make_batch(), torch.device("cpu"))
        expected = F.mse_loss(
            model.last_state.prior.mean,
            model.last_state.posterior.mean.detach(),
            reduction="mean",
        )
        torch.testing.assert_close(step.metrics["mean_mse"], expected)
        self.assertNotIn("gaussian_kl", step.metrics)
        self.assertNotIn("latent_kl", step.metrics)

    def test_c4_does_not_invoke_teacher_or_disabled_loss_functions(self):
        model = TinyAblationModel()
        task = make_task("none", 0.0, 0.0, 1.0)

        def forbidden(*args, **kwargs):
            raise AssertionError("privileged branch was called")

        model.distillation_forward = forbidden
        model.posterior_head.forward = forbidden
        with patch(
            "src.tasks.phase_c_distill.gaussian_kl",
            side_effect=AssertionError("Gaussian KL was computed"),
        ), patch(
            "src.tasks.phase_c_distill.categorical_routing_kl",
            side_effect=AssertionError("routing KL was computed"),
        ):
            step = task.training_step(
                model,
                make_batch(),
                torch.device("cpu"),
            )

        torch.testing.assert_close(
            step.loss,
            step.metrics["deploy_seg_loss"],
        )
        self.assertEqual(model.prior_calls, 1)
        self.assertEqual(model.posterior_calls, 0)

    def test_mean_and_none_objectives_do_not_compute_gaussian_kl(self):
        for spec in (
            ("mean_mse", 1.0, 1.0, 1.0),
            ("none", 0.0, 1.0, 1.0),
        ):
            with self.subTest(objective=spec[0]), patch(
                "src.tasks.phase_c_distill.gaussian_kl",
                side_effect=AssertionError("Gaussian KL was computed"),
            ):
                model = TinyAblationModel()
                task = make_task(*spec)
                task.training_step(
                    model,
                    make_batch(),
                    torch.device("cpu"),
                )

    def test_c4_evaluation_still_reports_teacher_diagnostics(self):
        model = TinyAblationModel()
        task = make_task("none", 0.0, 0.0, 1.0)
        step = task.evaluation_step(
            model,
            make_batch(),
            torch.device("cpu"),
        )
        self.assertEqual(model.teacher_calls, 1)
        self.assertEqual(model.posterior_calls, 1)
        self.assertIn("posterior_dice", step.metrics)
        self.assertIn("latent_kl", step.metrics)
        self.assertIn("gamma_distance", step.metrics)
        self.assertIn("route_kl", step.metrics)
        self.assertIn("routing_js", step.metrics)

    def test_only_prior_head_receives_gradients_for_every_objective(self):
        for name, spec in self.CASES.items():
            with self.subTest(name=name):
                torch.manual_seed(11)
                model = TinyAblationModel()
                task = make_task(*spec)
                step = task.training_step(
                    model,
                    make_batch(),
                    torch.device("cpu"),
                )
                step.loss.backward()
                trainable_names = [
                    parameter_name
                    for parameter_name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
                gradient_names = [
                    parameter_name
                    for parameter_name, parameter in model.named_parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(trainable_names)
                self.assertTrue(gradient_names)
                self.assertTrue(
                    all(
                        parameter_name.startswith("prior_head.")
                        for parameter_name in trainable_names
                    )
                )
                self.assertTrue(
                    set(gradient_names).issubset(set(trainable_names))
                )
                self.assertTrue(
                    all(
                        parameter_name.startswith("prior_head.")
                        for parameter_name in gradient_names
                    )
                )
                if name == "c1":
                    self.assertEqual(gradient_names, trainable_names)
                else:
                    self.assertEqual(
                        gradient_names,
                        [
                            "prior_head.mean_head.weight",
                            "prior_head.mean_head.bias",
                        ],
                    )

    def test_prior_inference_is_invariant_to_diagnostic_masks(self):
        torch.manual_seed(13)
        model = TinyAblationModel().eval()
        batch = make_batch()
        images = batch["image"]
        mask_a = batch["mask"]
        mask_b = 1.0 - mask_a

        without_mask = model(images).logits
        with_a = model(images, masks=mask_a).logits
        with_b = model(images, masks=mask_b).logits
        torch.testing.assert_close(without_mask, with_a, rtol=0, atol=0)
        torch.testing.assert_close(without_mask, with_b, rtol=0, atol=0)
        self.assertEqual(model.posterior_calls, 0)

    def test_seeded_prior_initialization_is_identical(self):
        checksums = []
        for _ in self.CASES:
            torch.manual_seed(42)
            model = TinyAblationModel()
            digest = hashlib.sha256()
            for name, tensor in sorted(model.prior_head.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(tensor.detach().cpu().numpy().tobytes())
            checksums.append(digest.hexdigest())
        self.assertEqual(len(set(checksums)), 1)


class TestPhaseCLossAblationConfigs(unittest.TestCase):
    def test_existing_configs_default_to_gaussian_kl(self):
        for name in ("b6_distill.yaml", "b6_distill_deploy.yaml"):
            with self.subTest(name=name):
                config = load_experiment_config(
                    CONFIG_ROOT / name,
                    project_root=PROJECT_ROOT,
                )
                self.assertEqual(
                    config["task"]["latent_objective"],
                    "gaussian_kl",
                )

    def test_all_ablation_configs_share_the_100_epoch_recipe(self):
        reference = load_experiment_config(
            CONFIG_ROOT / "b6_distill_deploy_100ep.yaml",
            project_root=PROJECT_ROOT,
        )
        expected_tasks = {
            "c1_balanced_kl": ("gaussian_kl", 0.001, 1.0, 1.0),
            "c2_mean_distill": ("mean_mse", 1.0, 1.0, 1.0),
            "c3_route_seg": ("none", 0.0, 1.0, 1.0),
            "c4_seg_only": ("none", 0.0, 0.0, 1.0),
        }
        for name, expected in expected_tasks.items():
            with self.subTest(name=name):
                config = load_experiment_config(
                    ABLATION_ROOT / f"{name}.yaml",
                    project_root=PROJECT_ROOT,
                )
                self.assertEqual(config["experiment"]["name"], name)
                self.assertEqual(
                    Path(config["experiment"]["output_root"]),
                    PROJECT_ROOT / "runs/phase_c_loss_ablations",
                )
                self.assertEqual(config["seed"], 42)
                self.assertEqual(config["training"]["epochs"], 100)
                self.assertEqual(config["training"]["monitor"], "dice")
                self.assertEqual(config["training"]["monitor_mode"], "max")
                self.assertEqual(
                    Path(config["model"]["teacher_checkpoint"]),
                    TEACHER,
                )
                for section in (
                    "dataset",
                    "model",
                    "loss",
                    "optimizer",
                    "scheduler",
                    "training",
                ):
                    self.assertEqual(config[section], reference[section])
                actual = (
                    config["task"]["latent_objective"],
                    config["task"]["lambda_latent"],
                    config["task"]["lambda_route"],
                    config["task"]["lambda_deploy"],
                )
                self.assertEqual(actual, expected)

    def test_config_validation_rejects_invalid_objectives(self):
        base = load_experiment_config(
            CONFIG_ROOT / "b6_distill.yaml",
            project_root=PROJECT_ROOT,
        )
        cases = (
            ({"latent_objective": "cosine"}, "latent_objective"),
            (
                {"latent_objective": "none", "lambda_latent": 1.0},
                "lambda_latent",
            ),
            (
                {
                    "latent_objective": "none",
                    "lambda_latent": 0.0,
                    "lambda_route": 0.0,
                    "lambda_deploy": 0.0,
                },
                "at least one",
            ),
        )
        for overrides, message in cases:
            invalid = copy.deepcopy(base)
            invalid["task"].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                resolve_experiment_config(invalid, project_root=PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
