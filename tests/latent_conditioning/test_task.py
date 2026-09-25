"""Loss, schedule, metrics, and collapse-diagnostic tests."""

from __future__ import annotations

import unittest
import warnings

import torch
from torch import nn

from src.losses import BCEDiceLoss
from src.models.latent_conditioning import LatentConditioningState
from src.models.phase_b.posterior import (
    DiagonalGaussian,
    GaussianParameterHead,
    gaussian_kl,
)
from src.models.phase_b.router import TopKRouter, load_balance_loss
from src.tasks import LatentConditioningTask


class _TinyTaskModel(nn.Module):
    def __init__(self, *, use_moe: bool, use_pre_moe_latent: bool):
        super().__init__()
        self.use_moe = use_moe
        self.use_pre_moe_latent = use_pre_moe_latent
        self.posterior_head = GaussianParameterHead(in_dim=5, latent_dim=8)
        self.prior_head = GaussianParameterHead(in_dim=4, latent_dim=8)
        if use_moe:
            self.router = TopKRouter(
                latent_dim=8,
                num_experts=4,
                active_experts=2,
            )
        self.last_decode_posterior = None
        self.last_decode_prior = None
        self.last_state = None

    def joint_forward(
        self,
        images,
        masks,
        *,
        decode_posterior=True,
        decode_prior=False,
    ):
        image_descriptor = images.mean(dim=(2, 3))
        mask_descriptor = masks.mean(dim=(2, 3))
        posterior = self.posterior_head(
            torch.cat([image_descriptor, mask_descriptor], dim=1)
        )
        prior = self.prior_head(image_descriptor)
        posterior_routing = None
        prior_routing = None
        balance = None
        if self.use_moe:
            if self.use_pre_moe_latent:
                posterior_routing = self.router(posterior.mean)
                prior_routing = self.router(prior.mean) if decode_prior else None
            else:
                posterior_routing = self.router(
                    torch.cat(
                        [
                            image_descriptor,
                            torch.zeros(
                                image_descriptor.shape[0],
                                4,
                                device=image_descriptor.device,
                            ),
                        ],
                        dim=1,
                    )
                )
                prior_routing = posterior_routing
            balance = load_balance_loss(
                posterior_routing.dense_probs,
                posterior_routing.expert_indices,
                num_experts=4,
            )

        def decode(latent):
            return latent[:, :1, None, None].expand(
                -1,
                1,
                masks.shape[-2],
                masks.shape[-1],
            )

        state = LatentConditioningState(
            posterior=posterior,
            prior=prior,
            use_moe=self.use_moe,
            use_pre_moe_latent=self.use_pre_moe_latent,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            posterior_balance=balance,
            posterior_logits=decode(posterior.mean) if decode_posterior else None,
            prior_logits=decode(prior.mean) if decode_prior else None,
        )
        self.last_decode_posterior = decode_posterior
        self.last_decode_prior = decode_prior
        self.last_state = state
        return state


def _batch():
    generator = torch.Generator().manual_seed(19)
    images = torch.rand(2, 4, 8, 8, generator=generator)
    masks = torch.zeros(2, 1, 8, 8)
    masks[0, :, 1:6, 2:7] = 1
    masks[1, :, 3:, :5] = 1
    return {"image": images, "mask": masks}


def _task():
    return LatentConditioningTask(
        criterion=BCEDiceLoss(),
        lambda_balance=0.01,
        kl_beta_max=0.1,
        kl_zero_until_epoch=5,
        kl_ramp_end_epoch=20,
        boundary_tolerance=1,
    )


class TestLatentConditioningTask(unittest.TestCase):
    def test_beta_schedule_boundaries(self):
        task = _task()
        for epoch, expected in ((1, 0.0), (5, 0.0), (20, 0.1), (25, 0.1)):
            with self.subTest(epoch=epoch):
                task.set_epoch(epoch)
                self.assertAlmostEqual(task.kl_beta, expected)
        task.set_epoch(10)
        self.assertAlmostEqual(task.kl_beta, 0.1 * 5 / 15)

    def test_c1_loss_decomposition_and_training_decode_policy(self):
        model = _TinyTaskModel(use_moe=True, use_pre_moe_latent=False)
        task = _task()
        task.set_epoch(10)
        batch = _batch()
        step = task.training_step(model, batch, torch.device("cpu"))
        state = model.last_state
        segmentation = task.criterion(state.posterior_logits, batch["mask"])
        kl_sum = gaussian_kl(state.posterior, state.prior).mean()
        expected = (
            segmentation
            + task.kl_beta * kl_sum / 8
            + 0.01 * state.posterior_balance
        )
        torch.testing.assert_close(step.loss, expected)
        torch.testing.assert_close(step.metrics["seg_loss"], segmentation)
        torch.testing.assert_close(step.metrics["latent_kl_sum"], kl_sum)
        torch.testing.assert_close(
            step.metrics["latent_kl_per_dim"],
            kl_sum / 8,
        )
        torch.testing.assert_close(
            step.metrics["balance_loss"],
            state.posterior_balance,
        )
        self.assertTrue(model.last_decode_posterior)
        self.assertFalse(model.last_decode_prior)

    def test_c2_has_no_balance_term_or_metric(self):
        model = _TinyTaskModel(use_moe=False, use_pre_moe_latent=False)
        task = _task()
        task.set_epoch(20)
        batch = _batch()
        step = task.training_step(model, batch, torch.device("cpu"))
        state = model.last_state
        segmentation = task.criterion(state.posterior_logits, batch["mask"])
        kl_sum = gaussian_kl(state.posterior, state.prior).mean()
        torch.testing.assert_close(
            step.loss,
            segmentation + 0.1 * kl_sum / 8,
        )
        self.assertNotIn("balance_loss", step.metrics)
        self.assertIsNone(state.posterior_balance)

    def test_evaluation_uses_prior_metrics_and_posterior_prefixes(self):
        model = _TinyTaskModel(use_moe=False, use_pre_moe_latent=False)
        task = _task()
        batch = _batch()
        step = task.evaluation_step(model, batch, torch.device("cpu"))
        prior = task._segmentation_metrics(
            model.last_state.prior_logits,
            batch["mask"],
        )
        posterior = task._segmentation_metrics(
            model.last_state.posterior_logits,
            batch["mask"],
        )
        torch.testing.assert_close(step.metrics["dice"], prior["dice"])
        torch.testing.assert_close(
            step.metrics["posterior_dice"],
            posterior["dice"],
        )
        for key in (
            "mean_distance",
            "std_distance",
            "wasserstein2_squared",
            "latent_mean_l2",
            "latent_std_l2",
        ):
            self.assertIn(key, step.metrics)
        finalized = task.finalize_evaluation_metrics(
            {
                name: float(value.detach()) if torch.is_tensor(value) else float(value)
                for name, value in step.metrics.items()
            }
        )
        self.assertAlmostEqual(
            finalized["transfer_gap_dice"],
            finalized["posterior_dice"] - finalized["dice"],
        )
        self.assertTrue(model.last_decode_posterior)
        self.assertTrue(model.last_decode_prior)

    def test_route_metrics_are_variant_aware(self):
        expectations = (
            (True, False, False, True),
            (False, False, False, False),
            (True, True, True, False),
        )
        for use_moe, use_pre, has_transfer, has_shared in expectations:
            with self.subTest(use_moe=use_moe, use_pre=use_pre):
                model = _TinyTaskModel(
                    use_moe=use_moe,
                    use_pre_moe_latent=use_pre,
                )
                metrics = _task().evaluation_step(
                    model,
                    _batch(),
                    torch.device("cpu"),
                ).metrics
                self.assertEqual(
                    "exact_topk_set_agreement" in metrics,
                    has_transfer,
                )
                self.assertEqual("routing_js" in metrics, has_transfer)
                self.assertEqual(
                    "routing_entropy_normalized" in metrics,
                    has_shared,
                )
                self.assertEqual(
                    "posterior_routing_entropy_normalized" in metrics,
                    has_transfer,
                )
                if not use_moe:
                    self.assertFalse(
                        any("expert_usage" in name for name in metrics)
                    )

    def test_collapse_warning_requires_three_consecutive_epochs(self):
        task = _task()
        metrics = {
            "dice": 0.4,
            "posterior_dice": 0.5,
            "expert_usage_fraction_0": 0.9,
            "expert_usage_fraction_1": 0.1,
            "expert_usage_fraction_2": 0.0,
            "expert_usage_fraction_3": 0.0,
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for epoch in (1, 2, 3):
                task.set_epoch(epoch)
                task.finalize_evaluation_metrics(metrics)
        self.assertEqual(len(caught), 1)
        self.assertIn("3 consecutive validation epochs", str(caught[0].message))

    def test_gaussian_identity_difference_and_kl_normalization(self):
        head = GaussianParameterHead(in_dim=3, latent_dim=8)
        q = head(torch.randn(2, 3))
        identity = gaussian_kl(q, q).mean()
        torch.testing.assert_close(
            identity,
            torch.zeros_like(identity),
            atol=1e-6,
            rtol=0,
        )
        shifted = DiagonalGaussian(mean=q.mean + 1.0, std=q.std)
        different = gaussian_kl(q, shifted).mean()
        self.assertGreater(float(different.detach()), 0.0)
        self.assertTrue(torch.isfinite(different / 8))

    def test_non_finite_segmentation_loss_fails_with_epoch_context(self):
        task = _task()
        task.set_epoch(7)
        distribution = DiagonalGaussian(
            mean=torch.zeros(1, 8),
            std=torch.ones(1, 8),
        )
        state = LatentConditioningState(
            posterior=distribution,
            prior=distribution,
            use_moe=False,
            use_pre_moe_latent=False,
            posterior_logits=torch.full((1, 1, 2, 2), float("inf")),
        )
        with self.assertRaisesRegex(FloatingPointError, "epoch 7"):
            task._loss_components(state, torch.zeros(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
