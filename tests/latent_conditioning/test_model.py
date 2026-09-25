"""Architecture, causal-path, inference, and freeze-policy tests."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from src.models.latent_conditioning import (
    LatentConditioningModel,
    PostLatentConditioning,
    PreMoEContextAdapter,
)
from src.models.phase_b.moe_enhancement import HierarchicalMoEEnhancement
from src.models.phase_b.router import TopKRouter


class _TinyImageEncoder(nn.Module):
    embed_dim = 12

    def __init__(self):
        super().__init__()
        self.embedding = nn.Conv2d(3, 256, kernel_size=1)
        self.Adapter = nn.Linear(3, self.embed_dim)
        self.call_count = 0

    def forward(self, images):
        self.call_count += 1
        pooled = F.adaptive_avg_pool2d(images, (2, 2))
        embeddings = self.embedding(pooled)
        tokens = self.Adapter(pooled.permute(0, 2, 3, 1))
        blocks = [tokens * ((index + 1) / 12) for index in range(12)]
        return embeddings, blocks


class _TinyPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpeg = nn.Conv2d(256, 256, kernel_size=1)

    def forward(
        self,
        *,
        points,
        boxes,
        masks,
        image_embedding,
        batch_size,
    ):
        dense = self.lpeg(image_embedding)
        sparse = dense.new_zeros(batch_size, 0, 256)
        return sparse, dense

    def get_dense_pe(self):
        return self.lpeg.weight.new_zeros(1, 256, 2, 2)


class _TinyMaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        logits = self.head(image_embeddings + dense_prompt_embeddings)
        iou = logits.mean(dim=(2, 3))
        return logits, iou


class _TinyNetwork(nn.Module):
    def __init__(self, image_size=8):
        super().__init__()
        self.image_encoder = _TinyImageEncoder()
        self.prompt_encoder = _TinyPromptEncoder()
        self.mask_decoder = _TinyMaskDecoder()
        self.use_lpeg = True
        self.image_size = image_size

    @staticmethod
    def preprocess(images):
        return images

    def postprocess_masks(self, masks, *, input_size, original_size):
        return F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )


class _TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.network = _TinyNetwork(image_size=kwargs.get("image_size", 8))


class _TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_projection = nn.Linear(4, 256)

    def forward(self, features):
        return self.latent_projection(features)


class _TinyShapeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 4, kernel_size=1)
        self.projector = _TinyProjector()
        self.frozen = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, masks):
        with torch.no_grad():
            features = self.encoder(masks).mean(dim=(2, 3))
            return self.projector(features)


def build_tiny_model(*, use_moe: bool, use_pre_moe_latent: bool):
    with (
        patch(
            "src.models.latent_conditioning.model.EsamModel",
            side_effect=lambda **kwargs: _TinyBackbone(**kwargs),
        ),
        patch(
            "src.models.latent_conditioning.model.load_shape_teacher",
            side_effect=lambda *args, **kwargs: _TinyShapeTeacher(),
        ),
    ):
        return LatentConditioningModel(
            image_size=8,
            checkpoint=None,
            shape_teacher_checkpoint="unused-by-test",
            use_moe=use_moe,
            use_pre_moe_latent=use_pre_moe_latent,
        )


def tiny_batch(batch_size=2):
    generator = torch.Generator().manual_seed(71)
    images = torch.rand(batch_size, 3, 8, 8, generator=generator)
    masks = torch.zeros(batch_size, 1, 8, 8)
    masks[0, :, 1:6, 2:7] = 1
    if batch_size > 1:
        masks[1, :, 3:, :5] = 1
    return images, masks


class TestConditioningModules(unittest.TestCase):
    def test_pre_moe_adapter_has_exact_shape(self):
        module = PreMoEContextAdapter()
        output = module(torch.randn(3, 256), torch.randn(3, 8))
        self.assertEqual(tuple(output.shape), (3, 256))
        self.assertEqual(module.adapter[0].in_features, 264)
        self.assertEqual(module.adapter[0].out_features, 256)

    def test_post_conditioning_has_exact_residual_shape(self):
        module = PostLatentConditioning()
        self.assertEqual(module.latent_projection[0].in_features, 8)
        self.assertEqual(module.latent_projection[0].out_features, 64)
        self.assertEqual(module.fusion.in_channels, 320)
        self.assertEqual(module.fusion.out_channels, 256)
        nn.init.zeros_(module.fusion.weight)
        nn.init.zeros_(module.fusion.bias)
        features = torch.randn(2, 256, 4, 4)
        output = module(features, torch.randn(2, 8))
        torch.testing.assert_close(output, features)


class TestLatentConditioningModel(unittest.TestCase):
    def test_all_variants_have_dz8_and_expected_output_shapes(self):
        images, masks = tiny_batch()
        for use_moe, use_pre in ((True, False), (False, False), (True, True)):
            with self.subTest(use_moe=use_moe, use_pre=use_pre):
                model = build_tiny_model(
                    use_moe=use_moe,
                    use_pre_moe_latent=use_pre,
                )
                state = model.joint_forward(
                    images,
                    masks,
                    decode_posterior=True,
                    decode_prior=True,
                )
                self.assertEqual(tuple(state.posterior.mean.shape), (2, 8))
                self.assertEqual(tuple(state.prior.mean.shape), (2, 8))
                self.assertEqual(tuple(state.posterior_logits.shape), (2, 1, 8, 8))
                self.assertEqual(tuple(state.prior_logits.shape), (2, 1, 8, 8))
                if use_moe:
                    self.assertEqual(tuple(state.posterior_context.shape), (2, 256))
                    self.assertEqual(
                        tuple(state.posterior_routing.logits.shape),
                        (2, 4),
                    )
                    self.assertEqual(
                        tuple(state.posterior_routing.expert_indices.shape),
                        (2, 2),
                    )
                else:
                    self.assertIsNone(state.posterior_context)
                    self.assertIsNone(state.posterior_routing)

    def test_joint_forward_runs_the_encoder_once(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=True)
        images, masks = tiny_batch()
        model.joint_forward(
            images,
            masks,
            decode_posterior=True,
            decode_prior=True,
        )
        self.assertEqual(model.backbone.network.image_encoder.call_count, 1)

    def test_c1_route_is_shared_and_invariant_to_the_mask(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=False)
        images, masks = tiny_batch()
        first = model.joint_forward(
            images,
            masks,
            decode_posterior=False,
            decode_prior=False,
        )
        second = model.joint_forward(
            images,
            1.0 - masks,
            decode_posterior=False,
            decode_prior=False,
        )
        self.assertIs(first.posterior_routing, first.prior_routing)
        torch.testing.assert_close(first.posterior_context, first.prior_context)
        torch.testing.assert_close(
            first.posterior_routing.logits,
            second.posterior_routing.logits,
        )
        self.assertFalse(
            torch.allclose(first.posterior.mean, second.posterior.mean)
        )

    def test_c3_context_and_routing_logits_depend_on_latent(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=True)
        adapter = model.pre_moe_context_adapter.adapter[0]
        router = model.router.route
        with torch.no_grad():
            adapter.weight.zero_()
            adapter.bias.zero_()
            adapter.weight[0, 256] = 1.0
            router.weight.zero_()
            router.bias.zero_()
            router.weight[0, 0] = 1.0
            router.weight[1, 0] = -1.0
        descriptor = torch.zeros(2, 256)
        positive = torch.zeros(2, 8)
        negative = torch.zeros(2, 8)
        positive[:, 0] = 2.0
        negative[:, 0] = -2.0
        first = model._route(model._context(descriptor, positive))
        second = model._route(model._context(descriptor, negative))
        self.assertFalse(torch.allclose(first.logits, second.logits))
        self.assertFalse(
            torch.equal(first.expert_indices, second.expert_indices)
        )

    def test_c2_instantiates_no_moe_modules(self):
        model = build_tiny_model(use_moe=False, use_pre_moe_latent=False)
        self.assertFalse(hasattr(model, "router"))
        self.assertFalse(hasattr(model, "enhancement"))
        self.assertFalse(hasattr(model, "enhancement_neck"))
        self.assertFalse(hasattr(model, "pre_moe_context_adapter"))
        self.assertFalse(
            any(isinstance(module, TopKRouter) for module in model.modules())
        )
        self.assertFalse(
            any(
                isinstance(module, HierarchicalMoEEnhancement)
                for module in model.modules()
            )
        )

    def test_default_forward_is_prior_only_and_never_reads_mask(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=True)
        images, masks = tiny_batch()
        model.shape_teacher.forward = Mock(
            side_effect=AssertionError("target leakage")
        )
        output = model(images, masks=masks)
        self.assertEqual(tuple(output.logits.shape), (2, 1, 8, 8))
        self.assertIsNone(output.posterior_logits)
        model.shape_teacher.forward.assert_not_called()

    def test_posterior_diagnostics_are_explicitly_opt_in(self):
        model = build_tiny_model(use_moe=False, use_pre_moe_latent=False)
        images, masks = tiny_batch()
        with self.assertRaisesRegex(ValueError, "requires masks"):
            model(images, include_posterior=True)
        output = model(images, masks=masks, include_posterior=True)
        self.assertEqual(tuple(output.prior_logits.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(output.posterior_logits.shape), (2, 1, 8, 8))
        self.assertIsNotNone(output.joint_state)

    def test_encoder_adapters_and_teacher_are_frozen_and_stay_eval(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=False)
        encoder = model.backbone.network.image_encoder
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in encoder.Adapter.parameters())
        )
        self.assertTrue(
            all(not p.requires_grad for p in model.shape_teacher.parameters())
        )
        model.train()
        self.assertFalse(encoder.training)
        self.assertFalse(model.shape_teacher.training)
        self.assertTrue(model.image_descriptor.training)
        self.assertTrue(model.posterior_head.training)
        self.assertTrue(model.backbone.network.mask_decoder.training)

    def test_c1_c3_parameter_counts_match_and_c2_is_smaller(self):
        c1 = build_tiny_model(use_moe=True, use_pre_moe_latent=False)
        c2 = build_tiny_model(use_moe=False, use_pre_moe_latent=False)
        c3 = build_tiny_model(use_moe=True, use_pre_moe_latent=True)
        p1 = c1.latent_conditioning_checkpoint_metadata()["parameter_counts"]
        p2 = c2.latent_conditioning_checkpoint_metadata()["parameter_counts"]
        p3 = c3.latent_conditioning_checkpoint_metadata()["parameter_counts"]
        self.assertEqual(p1["total"], p3["total"])
        self.assertEqual(p1["trainable"], p3["trainable"])
        self.assertLess(p2["total"], p1["total"])
        self.assertEqual(p2["groups"]["router"]["total"], 0)
        self.assertEqual(p2["groups"]["experts"]["total"], 0)
        self.assertEqual(p1["groups"]["sam_adapters"]["trainable"], 0)

    def test_non_finite_distribution_fails_fast(self):
        model = build_tiny_model(use_moe=False, use_pre_moe_latent=False)
        images, _ = tiny_batch()
        with torch.no_grad():
            model.prior_head.mean_head.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "prior.mean"):
            model(images)

    def test_invalid_c2_pre_moe_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires model.use_moe=true"):
            build_tiny_model(use_moe=False, use_pre_moe_latent=True)


if __name__ == "__main__":
    unittest.main()
