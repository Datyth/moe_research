"""CPU tests for the Phase-C objective, gradients, and trainer metrics."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.engine import Trainer, TrainerConfig
from src.losses import BCEDiceLoss
from src.models.phase_b.posterior import GaussianParameterHead, gaussian_kl
from src.models.phase_b.router import TopKRouter
from src.models.phase_c import (
    PhaseCDistillationState,
    categorical_routing_kl,
)
from src.tasks import PhaseCDistillTask


class TinyPhaseCModel(nn.Module):
    """Small differentiable stand-in that preserves the Phase-C contracts."""

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
        self.teacher_block = nn.Linear(2, 2)
        for module in (self.posterior_head, self.router, self.teacher_block):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.last_state = None
        self.last_decode_prior = None
        self.last_decode_posterior = None

    def distillation_forward(
        self,
        images,
        masks,
        *,
        decode_prior=False,
        decode_posterior=False,
    ):
        image_descriptor = images.mean(dim=(2, 3))
        prior = self.prior_head(image_descriptor)
        prior_routing = self.router(prior.mean)

        with torch.no_grad():
            target_descriptor = masks.mean(dim=(2, 3)).expand(-1, 4)
            posterior = self.posterior_head(target_descriptor)
            posterior_routing = self.router(posterior.mean)

        prior_logits = None
        if decode_prior:
            routed_scale = prior_routing.routing_probs[:, :1]
            scalar = prior.mean[:, :1] + routed_scale
            prior_logits = scalar[:, :, None, None].expand(
                -1,
                1,
                masks.shape[-2],
                masks.shape[-1],
            )

        posterior_logits = None
        if decode_posterior:
            scalar = posterior.mean[:, :1]
            posterior_logits = scalar[:, :, None, None].expand(
                -1,
                1,
                masks.shape[-2],
                masks.shape[-1],
            )

        state = PhaseCDistillationState(
            posterior=posterior,
            prior=prior,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
        )
        self.last_state = state
        self.last_decode_prior = decode_prior
        self.last_decode_posterior = decode_posterior
        return state


def make_batch():
    generator = torch.Generator().manual_seed(17)
    images = torch.rand(2, 4, 4, 4, generator=generator)
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, :, 1:3, 1:3] = 1
    masks[1, :, :3, :2] = 1
    return {"image": images, "mask": masks}


def make_task(*, latent=1.0, route=1.0, deploy=0.0):
    return PhaseCDistillTask(
        criterion=BCEDiceLoss(),
        lambda_latent=latent,
        lambda_route=route,
        lambda_deploy=deploy,
        threshold=0.5,
        boundary_tolerance=1,
    )


class TestPhaseCTask(unittest.TestCase):
    def test_default_loss_is_exactly_latent_plus_route_without_decode(self):
        model = TinyPhaseCModel()
        task = make_task()
        step = task.training_step(model, make_batch(), torch.device("cpu"))
        state = model.last_state
        expected_latent = gaussian_kl(state.posterior.detach(), state.prior).mean()
        expected_route = categorical_routing_kl(
            state.posterior_routing.dense_probs,
            state.prior_routing.logits,
        )
        torch.testing.assert_close(step.loss, expected_latent + expected_route)
        torch.testing.assert_close(step.metrics["latent_kl"], expected_latent)
        torch.testing.assert_close(step.metrics["route_kl"], expected_route)
        self.assertNotIn("deploy_seg_loss", step.metrics)
        self.assertFalse(model.last_decode_prior)
        self.assertFalse(model.last_decode_posterior)

    def test_deploy_weight_adds_existing_bce_dice_prior_loss(self):
        model = TinyPhaseCModel()
        task = make_task(latent=2.0, route=3.0, deploy=4.0)
        batch = make_batch()
        step = task.training_step(model, batch, torch.device("cpu"))
        state = model.last_state
        expected_latent = gaussian_kl(state.posterior.detach(), state.prior).mean()
        expected_route = categorical_routing_kl(
            state.posterior_routing.dense_probs,
            state.prior_routing.logits,
        )
        expected_deploy = task.criterion(state.prior_logits, batch["mask"])
        expected = (
            2.0 * expected_latent
            + 3.0 * expected_route
            + 4.0 * expected_deploy
        )
        torch.testing.assert_close(step.loss, expected)
        torch.testing.assert_close(
            step.metrics["deploy_seg_loss"],
            expected_deploy,
        )
        self.assertTrue(model.last_decode_prior)
        self.assertFalse(model.last_decode_posterior)

    def test_only_prior_head_receives_default_phase_c_gradients(self):
        model = TinyPhaseCModel()
        step = make_task().training_step(
            model,
            make_batch(),
            torch.device("cpu"),
        )
        step.loss.backward()

        prior_gradients = [
            parameter.grad
            for parameter in model.prior_head.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in prior_gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in prior_gradients),
            0.0,
        )
        for module in (model.posterior_head, model.router, model.teacher_block):
            self.assertTrue(
                all(parameter.grad is None for parameter in module.parameters())
            )

        state = model.last_state
        torch.testing.assert_close(
            state.posterior_routing.dense_probs.sum(dim=1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            state.prior_routing.dense_probs.sum(dim=1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            state.prior_routing.routing_probs.sum(dim=1),
            torch.ones(2),
        )

    def test_evaluation_reports_prior_oracle_and_transfer_metrics(self):
        model = TinyPhaseCModel()
        task = make_task()
        step = task.evaluation_step(
            model,
            make_batch(),
            torch.device("cpu"),
        )
        required = {
            "dice",
            "iou",
            "hd",
            "hd95",
            "assd",
            "boundary_f1",
            "posterior_dice",
            "posterior_iou",
            "posterior_hd",
            "posterior_hd95",
            "posterior_assd",
            "posterior_boundary_f1",
            "latent_kl",
            "mean_distance",
            "std_distance",
            "wasserstein2_squared",
            "exact_topk_set_agreement",
            "topk_overlap",
            "routing_js",
        }
        self.assertTrue(required.issubset(step.metrics))
        finalized = task.finalize_evaluation_metrics(
            {
                name: float(value.detach())
                for name, value in step.metrics.items()
            }
        )
        self.assertAlmostEqual(
            finalized["transfer_gap_dice"],
            finalized["posterior_dice"] - finalized["dice"],
        )
        self.assertTrue(model.last_decode_prior)
        self.assertTrue(model.last_decode_posterior)

    def test_nonfinite_or_negative_weights_are_rejected(self):
        for name, value in (
            ("lambda_latent", -1.0),
            ("lambda_route", float("nan")),
            ("lambda_deploy", float("inf")),
        ):
            kwargs = {
                "criterion": BCEDiceLoss(),
                "lambda_latent": 1.0,
                "lambda_route": 1.0,
                "lambda_deploy": 0.0,
            }
            kwargs[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                name,
            ):
                PhaseCDistillTask(**kwargs)

    def test_trainer_aggregates_raw_phase_c_training_metrics(self):
        model = TinyPhaseCModel()
        task = make_task()
        batch = make_batch()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = Trainer(
                model=model,
                task=task,
                task_config={
                    "name": "phase_c_distill",
                    "lambda_latent": 1.0,
                    "lambda_route": 1.0,
                    "lambda_deploy": 0.0,
                },
                optimizer=torch.optim.AdamW(
                    model.prior_head.parameters(),
                    lr=1e-3,
                ),
                train_loader=[batch],
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    use_amp=False,
                    log_interval=1,
                    last_checkpoint_path=root / "last.pt",
                    best_checkpoint_path=root / "best.pt",
                    history_path=root / "history.json",
                ),
            )
            history = trainer.train()

        self.assertEqual(len(history), 1)
        self.assertTrue(
            {
                "train_loss",
                "train_latent_kl",
                "train_route_kl",
                "train_total_loss",
            }.issubset(history[0])
        )
        self.assertAlmostEqual(
            history[0]["train_loss"],
            history[0]["train_total_loss"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
