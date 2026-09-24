"""Gradient flow regression for the joint objective."""

import unittest

import torch
from torch import nn

from src.losses import BCEDiceLoss
from src.models.joint_prior_posterior import JointPriorPosteriorState
from src.models.phase_b.posterior import GaussianParameterHead
from src.models.phase_b.router import TopKRouter, load_balance_loss
from src.tasks import JointPriorPosteriorTask


class _TinyGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.trunk = nn.Linear(3, 8)
        self.image_encoder.Adapter = nn.Linear(3, 8)
        self.shape_teacher = nn.Linear(1, 4)
        for module in (self.image_encoder, self.shape_teacher):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.image_descriptor = nn.Linear(8, 6)
        self.posterior_head = GaussianParameterHead(in_dim=10, latent_dim=8)
        self.prior_head = GaussianParameterHead(in_dim=6, latent_dim=8)
        self.router = TopKRouter(latent_dim=8, num_experts=4, active_experts=2)
        with torch.no_grad():
            self.router.route.bias.copy_(torch.tensor([3.0, 2.0, -2.0, -3.0]))
        self.experts = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))
        self.layer_scorer = nn.Linear(8, 1)
        self.enhancement_neck = nn.Linear(8, 8)
        self.lpeg = nn.Linear(8, 8)
        self.mask_decoder = nn.Linear(8, 16)

    def joint_forward(
        self,
        images,
        masks,
        *,
        decode_posterior=True,
        decode_prior=False,
    ):
        pooled_images = images.mean(dim=(2, 3))
        with torch.no_grad():
            frozen_features = (
                self.image_encoder.trunk(pooled_images)
                + self.image_encoder.Adapter(pooled_images)
            )
            shape_features = self.shape_teacher(masks.mean(dim=(2, 3)))
        descriptor = self.image_descriptor(frozen_features)
        posterior = self.posterior_head(
            torch.cat([descriptor, shape_features], dim=1)
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
            active = routing.expert_indices
            probabilities = routing.routing_probs.gather(1, active)
            enhanced = torch.zeros_like(latent)
            for slot in range(active.shape[1]):
                expert_ids = active[:, slot]
                for expert_id in torch.unique(expert_ids).tolist():
                    selected = expert_ids == expert_id
                    expert_output = self.experts[int(expert_id)](latent[selected])
                    enhanced[selected] += (
                        probabilities[selected, slot, None] * expert_output
                    )
            layer_weight = torch.sigmoid(self.layer_scorer(latent))
            enhanced = latent + layer_weight * enhanced
            enhanced = torch.nn.functional.gelu(self.enhancement_neck(enhanced))
            enhanced = enhanced + self.lpeg(enhanced)
            return self.mask_decoder(enhanced).reshape(-1, 1, 4, 4)

        return JointPriorPosteriorState(
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


def _gradient_sum(module):
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class TestJointGradients(unittest.TestCase):
    def test_joint_loss_reaches_both_latents_and_trainable_segmentation_path(self):
        torch.manual_seed(11)
        model = _TinyGradientModel()
        task = JointPriorPosteriorTask(
            criterion=BCEDiceLoss(),
            lambda_balance=0.01,
            kl_beta_max=0.1,
            kl_zero_until_epoch=5,
            kl_ramp_end_epoch=20,
        )
        task.set_epoch(20)
        batch = {
            "image": torch.randn(3, 3, 4, 4),
            "mask": (torch.rand(3, 1, 4, 4) > 0.5).float(),
        }
        step = task.training_step(model, batch, torch.device("cpu"))
        step.loss.backward()

        self.assertGreater(_gradient_sum(model.posterior_head), 0.0)
        self.assertGreater(_gradient_sum(model.prior_head), 0.0)
        self.assertGreater(_gradient_sum(model.router), 0.0)
        self.assertGreater(_gradient_sum(model.image_descriptor), 0.0)
        self.assertGreater(_gradient_sum(model.layer_scorer), 0.0)
        self.assertGreater(_gradient_sum(model.enhancement_neck), 0.0)
        self.assertGreater(_gradient_sum(model.lpeg), 0.0)
        self.assertGreater(_gradient_sum(model.mask_decoder), 0.0)
        self.assertGreater(
            sum(_gradient_sum(expert) for expert in model.experts),
            0.0,
        )
        self.assertEqual(_gradient_sum(model.image_encoder), 0.0)
        self.assertEqual(_gradient_sum(model.shape_teacher), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.image_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.shape_teacher.parameters())
        )

    def test_kl_alone_sends_gradient_to_posterior_and_prior(self):
        torch.manual_seed(23)
        posterior_head = GaussianParameterHead(in_dim=5, latent_dim=8)
        prior_head = GaussianParameterHead(in_dim=3, latent_dim=8)
        posterior = posterior_head(torch.randn(2, 5))
        prior = prior_head(torch.randn(2, 3))
        from src.models.phase_b.posterior import gaussian_kl

        loss = gaussian_kl(posterior, prior).mean() / 8
        loss.backward()
        self.assertGreater(_gradient_sum(posterior_head), 0.0)
        self.assertGreater(_gradient_sum(prior_head), 0.0)


if __name__ == "__main__":
    unittest.main()
