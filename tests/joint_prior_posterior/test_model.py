"""Model API, deterministic routing, and freeze-policy tests."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.models.joint_prior_posterior import (
    JointPriorPosteriorB6,
    JointPriorPosteriorOutput,
    JointPriorPosteriorState,
)
from src.models.phase_b.posterior import GaussianParameterHead
from src.models.phase_b.router import TopKRouter


class _CallProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class _TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(3, 3)
        self.Adapter = nn.Linear(3, 3)


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Module()
        self.network.image_encoder = _TinyImageEncoder()
        self.network.mask_decoder = nn.Linear(3, 3)


class _ForwardHarness(JointPriorPosteriorB6):
    """Exercise public forward/train without constructing ViT-B."""

    def __init__(self, latent_dim=8):
        nn.Module.__init__(self)
        self.image_size = 4
        self.backbone = _TinyBackbone()
        self.shape_teacher = _CallProbe()
        self.image_descriptor = nn.Linear(3, 3)
        self.fusion = nn.Identity()
        self.conditioner = nn.Module()
        self.conditioner.posterior_head = GaussianParameterHead(
            in_dim=4,
            latent_dim=latent_dim,
        )
        self.conditioner.router = TopKRouter(
            latent_dim=latent_dim,
            num_experts=4,
            active_experts=2,
        )
        self.prior_head = GaussianParameterHead(
            in_dim=3,
            latent_dim=latent_dim,
        )
        self.enhancement = nn.Linear(3, 3)
        self.enhancement_neck = nn.Linear(3, 3)
        self._freeze_privileged_sources()

    def _encode_images(self, images):
        return images

    def _prior_from_encoded(self, encoded):
        descriptor = encoded.mean(dim=(2, 3))
        prior = self.prior_head(descriptor)
        return prior, self.conditioner.router(prior.mean)

    def _decode_routing(self, encoded, *, latent, routing):
        scalar = latent[:, :1] + routing.routing_probs[:, :1]
        logits = scalar[:, :, None, None].expand(
            -1,
            1,
            encoded.shape[-2],
            encoded.shape[-1],
        )
        return SimpleNamespace(logits=logits, iou_predictions=scalar)

    def joint_forward(
        self,
        images,
        masks,
        *,
        decode_posterior=True,
        decode_prior=False,
    ):
        self.shape_teacher(masks)
        descriptor = images.mean(dim=(2, 3))
        prior = self.prior_head(descriptor)
        posterior_input = torch.cat([descriptor, masks.mean((2, 3))], dim=1)
        posterior = self.conditioner.posterior_head(posterior_input)
        prior_routing = self.conditioner.router(prior.mean)
        posterior_routing = self.conditioner.router(posterior.mean)
        posterior_logits = (
            self._decode_routing(
                images,
                latent=posterior.mean,
                routing=posterior_routing,
            ).logits
            if decode_posterior
            else None
        )
        prior_logits = (
            self._decode_routing(
                images,
                latent=prior.mean,
                routing=prior_routing,
            ).logits
            if decode_prior
            else None
        )
        return JointPriorPosteriorState(
            posterior=posterior,
            prior=prior,
            posterior_routing=posterior_routing,
            prior_routing=prior_routing,
            posterior_balance=posterior.mean.new_zeros(()),
            posterior_logits=posterior_logits,
            prior_logits=prior_logits,
        )


class TestJointPriorPosteriorModel(unittest.TestCase):
    def test_dynamic_latent_shapes_and_router_contract(self):
        for latent_dim in (64, 8):
            with self.subTest(latent_dim=latent_dim):
                model = _ForwardHarness(latent_dim=latent_dim)
                images = torch.randn(2, 3, 4, 4)
                descriptor = images.mean((2, 3))
                prior = model.prior_head(descriptor)
                routing = model.conditioner.router(prior.mean)
                self.assertEqual(tuple(prior.mean.shape), (2, latent_dim))
                self.assertEqual(tuple(prior.std.shape), (2, latent_dim))
                self.assertEqual(tuple(routing.logits.shape), (2, 4))
                self.assertEqual(tuple(routing.expert_indices.shape), (2, 2))

    def test_prior_forward_ignores_masks_and_diagnostic_requires_them(self):
        torch.manual_seed(4)
        model = _ForwardHarness().eval()
        images = torch.randn(2, 3, 4, 4)
        masks = torch.randn(2, 1, 4, 4)
        without_mask = model(images)
        with_ignored_mask = model(images, masks=masks)
        self.assertIsInstance(without_mask, JointPriorPosteriorOutput)
        torch.testing.assert_close(without_mask.logits, with_ignored_mask.logits)
        self.assertEqual(model.shape_teacher.calls, 0)
        with self.assertRaisesRegex(ValueError, "requires ground-truth masks"):
            model(images, include_posterior=True)

        diagnostic = model(images, masks=masks, include_posterior=True)
        self.assertEqual(model.shape_teacher.calls, 1)
        self.assertEqual(tuple(diagnostic.logits.shape), (2, 1, 4, 4))
        self.assertEqual(tuple(diagnostic.posterior_logits.shape), (2, 1, 4, 4))

    def test_freeze_policy_and_train_modes(self):
        model = _ForwardHarness()
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.backbone.network.image_encoder.parameters()
            )
        )
        adapter_parameters = [
            parameter
            for name, parameter in model.backbone.network.image_encoder.named_parameters()
            if "Adapter" in name
        ]
        self.assertTrue(adapter_parameters)
        self.assertTrue(all(not value.requires_grad for value in adapter_parameters))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.shape_teacher.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.backbone.network.mask_decoder.parameters()
            )
        )

        model.train()
        self.assertFalse(model.backbone.network.image_encoder.training)
        self.assertFalse(model.shape_teacher.training)
        self.assertTrue(model.backbone.network.mask_decoder.training)
        self.assertTrue(model.image_descriptor.training)
        self.assertTrue(model.prior_head.training)
        self.assertTrue(model.conditioner.training)


if __name__ == "__main__":
    unittest.main()
