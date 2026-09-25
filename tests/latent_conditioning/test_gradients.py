"""Gradient-flow tests across posterior, prior, MoE, and frozen sources."""

from __future__ import annotations

import unittest

import torch

from src.losses import BCEDiceLoss
from src.models.phase_b.posterior import GaussianParameterHead, gaussian_kl
from src.tasks import LatentConditioningTask

from .test_model import build_tiny_model, tiny_batch


def _grad_norm(module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def _task(epoch=20):
    task = LatentConditioningTask(
        criterion=BCEDiceLoss(),
        lambda_balance=0.01,
        kl_beta_max=0.1,
        kl_zero_until_epoch=5,
        kl_ramp_end_epoch=20,
        boundary_tolerance=1,
    )
    task.set_epoch(epoch)
    return task


class TestLatentConditioningGradients(unittest.TestCase):
    def test_kl_without_detach_reaches_both_gaussian_heads(self):
        posterior_head = GaussianParameterHead(in_dim=5, latent_dim=8)
        prior_head = GaussianParameterHead(in_dim=4, latent_dim=8)
        posterior = posterior_head(torch.randn(3, 5))
        prior = prior_head(torch.randn(3, 4))
        (gaussian_kl(posterior, prior).mean() / 8).backward()
        self.assertGreater(_grad_norm(posterior_head), 0.0)
        self.assertGreater(_grad_norm(prior_head), 0.0)

    def test_full_training_path_gradient_policy_for_all_variants(self):
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
                    decode_prior=False,
                )
                loss, _ = _task()._loss_components(state, masks)
                loss.backward()

                self.assertGreater(_grad_norm(model.posterior_head), 0.0)
                self.assertGreater(_grad_norm(model.prior_head), 0.0)
                self.assertGreater(_grad_norm(model.image_descriptor), 0.0)
                self.assertGreater(
                    _grad_norm(model.post_latent_conditioning),
                    0.0,
                )
                self.assertGreater(
                    _grad_norm(model.backbone.network.prompt_encoder.lpeg),
                    0.0,
                )
                self.assertGreater(
                    _grad_norm(model.backbone.network.mask_decoder),
                    0.0,
                )
                self.assertEqual(
                    _grad_norm(model.backbone.network.image_encoder),
                    0.0,
                )
                self.assertEqual(_grad_norm(model.shape_teacher), 0.0)
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in (
                            model.backbone.network.image_encoder.parameters()
                        )
                    )
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in model.shape_teacher.parameters()
                    )
                )

                if use_moe:
                    self.assertGreater(
                        _grad_norm(model.pre_moe_context_adapter),
                        0.0,
                    )
                    self.assertGreater(_grad_norm(model.router), 0.0)
                    self.assertGreater(
                        _grad_norm(model.enhancement.layer_scorer),
                        0.0,
                    )
                    self.assertGreater(_grad_norm(model.enhancement_neck), 0.0)
                    selected = torch.unique(
                        state.posterior_routing.expert_indices
                    ).tolist()
                    self.assertTrue(selected)
                    for expert_id in selected:
                        self.assertGreater(
                            _grad_norm(
                                model.enhancement.experts.experts[expert_id]
                            ),
                            0.0,
                        )
                else:
                    self.assertFalse(hasattr(model, "router"))
                    self.assertFalse(hasattr(model, "enhancement"))

    def test_posterior_has_segmentation_gradient_before_kl_ramp(self):
        model = build_tiny_model(use_moe=False, use_pre_moe_latent=False)
        images, masks = tiny_batch()
        state = model.joint_forward(
            images,
            masks,
            decode_posterior=True,
            decode_prior=False,
        )
        loss, _ = _task(epoch=1)._loss_components(state, masks)
        loss.backward()
        self.assertGreater(_grad_norm(model.posterior_head), 0.0)
        self.assertEqual(_grad_norm(model.prior_head), 0.0)

    def test_c3_posterior_drives_both_pre_and_post_moe_modules(self):
        model = build_tiny_model(use_moe=True, use_pre_moe_latent=True)
        images, masks = tiny_batch()
        state = model.joint_forward(
            images,
            masks,
            decode_posterior=True,
            decode_prior=False,
        )
        loss, _ = _task(epoch=20)._loss_components(state, masks)
        loss.backward()
        self.assertGreater(_grad_norm(model.pre_moe_context_adapter), 0.0)
        self.assertGreater(_grad_norm(model.router), 0.0)
        self.assertGreater(_grad_norm(model.post_latent_conditioning), 0.0)


if __name__ == "__main__":
    unittest.main()
