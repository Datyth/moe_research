"""Joint objective, schedule, and evaluation-semantic tests."""

import unittest

import torch
from torch import nn

from src.losses import BCEDiceLoss
from src.models.joint_prior_posterior import JointPriorPosteriorState
from src.models.phase_b.posterior import GaussianParameterHead, gaussian_kl
from src.models.phase_b.router import TopKRouter, load_balance_loss
from src.tasks import JointPriorPosteriorTask


class _TinyJointModel(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.posterior_head = GaussianParameterHead(
            in_dim=5,
            latent_dim=latent_dim,
        )
        self.prior_head = GaussianParameterHead(
            in_dim=4,
            latent_dim=latent_dim,
        )
        self.router = TopKRouter(
            latent_dim=latent_dim,
            num_experts=4,
            active_experts=2,
        )
        self.last_state = None
        self.last_decode_posterior = None
        self.last_decode_prior = None

    def joint_forward(
        self,
        images,
        masks,
        *,
        decode_posterior=True,
        decode_prior=False,
    ):
        descriptor = images.mean(dim=(2, 3))
        mask_descriptor = masks.mean(dim=(2, 3))
        posterior = self.posterior_head(
            torch.cat([descriptor, mask_descriptor], dim=1)
        )
        prior = self.prior_head(descriptor)
        posterior_routing = self.router(posterior.mean)
        prior_routing = self.router(prior.mean)
        balance = load_balance_loss(
            posterior_routing.dense_probs,
            posterior_routing.expert_indices,
            num_experts=4,
        )

        def decode(latent, routing):
            scalar = latent[:, :1] + routing.routing_probs[:, :1]
            return scalar[:, :, None, None].expand(
                -1,
                1,
                masks.shape[-2],
                masks.shape[-1],
            )

        state = JointPriorPosteriorState(
            posterior=posterior,
            prior=prior,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            posterior_balance=balance,
            posterior_logits=(
                decode(posterior.mean, posterior_routing)
                if decode_posterior
                else None
            ),
            prior_logits=(
                decode(prior.mean, prior_routing) if decode_prior else None
            ),
        )
        self.last_state = state
        self.last_decode_posterior = decode_posterior
        self.last_decode_prior = decode_prior
        return state


def _batch():
    generator = torch.Generator().manual_seed(19)
    images = torch.rand(2, 4, 4, 4, generator=generator)
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, :, 1:3, 1:3] = 1
    masks[1, :, :3, :2] = 1
    return {"image": images, "mask": masks}


def _task():
    return JointPriorPosteriorTask(
        criterion=BCEDiceLoss(),
        lambda_balance=0.01,
        kl_beta_max=0.1,
        kl_zero_until_epoch=5,
        kl_ramp_end_epoch=20,
        boundary_tolerance=1,
    )


class TestJointPriorPosteriorTask(unittest.TestCase):
    def test_beta_schedule_boundaries(self):
        task = _task()
        for epoch, expected in ((1, 0.0), (5, 0.0), (20, 0.1), (25, 0.1)):
            with self.subTest(epoch=epoch):
                task.set_epoch(epoch)
                self.assertAlmostEqual(task.kl_beta, expected)
        task.set_epoch(10)
        self.assertAlmostEqual(task.kl_beta, 0.1 * 5 / 15)

    def test_total_loss_decomposition_and_training_decode_policy(self):
        model = _TinyJointModel()
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
            + task.lambda_balance * state.posterior_balance
        )
        torch.testing.assert_close(step.loss, expected)
        torch.testing.assert_close(step.metrics["seg_loss"], segmentation)
        torch.testing.assert_close(step.metrics["latent_kl_sum"], kl_sum)
        torch.testing.assert_close(step.metrics["latent_kl_per_dim"], kl_sum / 8)
        self.assertTrue(model.last_decode_posterior)
        self.assertFalse(model.last_decode_prior)

    def test_evaluation_uses_prior_metrics_and_reports_transfer(self):
        model = _TinyJointModel()
        task = _task()
        step = task.evaluation_step(model, _batch(), torch.device("cpu"))
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
            "latent_kl_sum",
            "latent_kl_per_dim",
            "mean_distance",
            "std_distance",
            "wasserstein2_squared",
            "latent_mean_l2",
            "latent_std_l2",
            "exact_topk_set_agreement",
            "topk_set_agreement",
            "topk_overlap",
            "routing_js",
        }
        self.assertTrue(required.issubset(step.metrics))
        torch.testing.assert_close(
            step.metrics["latent_mean_l2"],
            step.metrics["mean_distance"],
        )
        torch.testing.assert_close(
            step.metrics["topk_set_agreement"],
            step.metrics["exact_topk_set_agreement"],
        )
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

    def test_gaussian_kl_identity_difference_and_normalization_are_finite(self):
        for latent_dim in (64, 8):
            with self.subTest(latent_dim=latent_dim):
                head = GaussianParameterHead(in_dim=3, latent_dim=latent_dim)
                q = head(torch.randn(2, 3))
                identity = gaussian_kl(q, q).mean()
                torch.testing.assert_close(
                    identity,
                    torch.zeros_like(identity),
                    atol=1e-6,
                    rtol=0,
                )
                shifted = type(q)(mean=q.mean + 1.0, std=q.std)
                different = gaussian_kl(q, shifted).mean()
                self.assertGreater(float(different.detach()), 0.0)
                self.assertTrue(torch.isfinite(different / latent_dim))


if __name__ == "__main__":
    unittest.main()
