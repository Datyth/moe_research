"""A2 direct-fuse construction and routing tests."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.models.base import BaseSegmentationModel
from src.models.phase_b import (
    PhaseBA2DirectFuseStage,
    PrivilegedFusion,
)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = SimpleNamespace(
            image_encoder=SimpleNamespace(embed_dim=24),
        )


def _fake_fuse_init(
    instance,
    *,
    in_channels=3,
    num_classes=1,
    task="binary",
    image_size=256,
    descriptor_dim=256,
    levels=(3, 6, 9, 12),
    **kwargs,
):
    BaseSegmentationModel.__init__(
        instance,
        in_channels=in_channels,
        num_classes=num_classes,
        task=task,
    )
    instance.backbone = _Backbone()
    instance.image_size = image_size
    instance.fusion = PrivilegedFusion(
        descriptor_dim=descriptor_dim,
        shape_latent_dim=256,
    )
    instance.shape_teacher = nn.Identity()
    instance.image_descriptor = SimpleNamespace(levels=tuple(levels))


class TestA2DirectFuse(unittest.TestCase):
    def test_hq_512_drives_router_and_layer_scorer_without_gaussians(self):
        with patch(
            "src.models.phase_b.ablations.a2_direct_fuse.PhaseBFuseStage.__init__",
            new=_fake_fuse_init,
        ):
            model = PhaseBA2DirectFuseStage(
                descriptor_dim=256,
                expert_hidden_ratio=1,
            )

        self.assertEqual(model.fused_dim, 512)
        self.assertEqual(model.router.latent_dim, 512)
        self.assertEqual(model.enhancement.layer_scorer.latent_dim, 512)
        self.assertFalse(hasattr(model, "posterior_head"))
        self.assertFalse(hasattr(model, "prior_head"))
        self.assertFalse(hasattr(model, "_head"))

    def test_missing_mask_fails_before_any_encoder_work(self):
        model = PhaseBA2DirectFuseStage.__new__(PhaseBA2DirectFuseStage)
        BaseSegmentationModel.__init__(
            model,
            in_channels=3,
            num_classes=1,
            task="binary",
        )
        with self.assertRaisesRegex(ValueError, "requires masks"):
            model(torch.randn(1, 3, 8, 8))

    def test_rejects_legacy_feb_or_disabled_lpeg_early(self):
        with self.assertRaisesRegex(ValueError, "use_moe=False"):
            PhaseBA2DirectFuseStage(use_moe=True)
        with self.assertRaisesRegex(ValueError, "use_lpeg=True"):
            PhaseBA2DirectFuseStage(use_lpeg=False)


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone in this test.",
)
class TestA2FullBackbone(unittest.TestCase):
    def test_forward_backward_uses_hq_without_gaussian_heads(self):
        torch.manual_seed(0)
        model = PhaseBA2DirectFuseStage(
            image_size=256,
            use_moe=False,
            use_lpeg=True,
            freeze_backbone=True,
        )
        model.train()
        images = torch.randn(1, 3, 256, 256)
        masks = torch.randint(0, 2, (1, 1, 256, 256)).float()
        output = model(images, masks=masks)
        router_stage = output.diagnostics["phase_b_router"]

        self.assertEqual(router_stage.source, "direct_fuse")
        self.assertEqual(tuple(router_stage.latent.shape), (1, 512))
        self.assertIsNone(router_stage.posterior)
        self.assertIsNone(router_stage.prior)
        self.assertIsNone(router_stage.latent_kl)
        self.assertEqual(tuple(output.logits.shape), (1, 1, 256, 256))
        self.assertFalse(model.backbone.network.use_moe)

        output.logits.sum().backward()
        selected = set(
            router_stage.routing.expert_indices.flatten().tolist()
        )
        for expert_id, expert in enumerate(model.enhancement.experts.experts):
            gradient = expert.fc1.weight.grad
            if expert_id in selected:
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertTrue(
                    gradient is None or float(gradient.abs().sum()) == 0.0
                )


if __name__ == "__main__":
    unittest.main()
