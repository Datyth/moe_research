"""Contract tests for Phase-C C5 trainable student routing."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.configs import load_experiment_config
from src.experiment import build_optimizer
from src.losses import BCEDiceLoss
from src.models import build_model
from src.models.phase_b.moe_enhancement import HierarchicalMoEEnhancement
from src.models.phase_b.posterior import GaussianParameterHead
from src.models.phase_b.router import TopKRouter
from src.models.phase_c import PhaseCC5TrainableStudentRouting
from src.tasks import PhaseCDistillTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _TinyConditioner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Conv2d(1, 1, kernel_size=1)


class TinyC5Model(PhaseCC5TrainableStudentRouting):
    """Small C5 model that uses the real routers, scorer, and expert bank."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.image_size = 4
        self.embed_dim = 12
        self.teacher_checkpoint_info = {"path": "tiny-b6.pt"}
        self.teacher_checkpoint = "tiny-b6.pt"
        self.prior_head = GaussianParameterHead(
            in_dim=4,
            latent_dim=2,
            std_floor=1e-4,
        )
        self.conditioner = _TinyConditioner()
        self.enhancement = HierarchicalMoEEnhancement(
            embed_dim=12,
            num_experts=3,
            num_levels=2,
            latent_dim=2,
            expert_hidden_ratio=1,
        )
        self.image_descriptor = nn.Linear(12, 4)
        self.backbone = _TinyBackbone()
        self.shape_teacher = nn.Linear(1, 1)
        self.enhancement_neck = nn.Conv2d(1, 1, kernel_size=1)
        self.fusion = nn.Identity()
        self.posterior_calls = 0
        self._initialize_student_routing()

    def _encode_images(self, images, *, include_teacher_descriptor=False):
        with torch.no_grad():
            pooled = images.mean(dim=(2, 3))
            descriptor = self.image_descriptor(pooled)
            tokens = images.permute(0, 2, 3, 1).reshape(
                images.shape[0],
                -1,
                self.embed_dim,
            )
            level_tokens = (tokens, 0.5 * tokens + 0.1)
        descriptor_state = SimpleNamespace(
            descriptor=descriptor,
            level_tokens=level_tokens,
        )
        return SimpleNamespace(
            image_embeddings=images[:, :1],
            descriptor=descriptor_state,
            teacher_descriptor=(
                descriptor_state if include_teacher_descriptor else None
            ),
        )

    def _posterior_from_encoded(self, encoded, masks):
        self.posterior_calls += 1
        with torch.no_grad():
            descriptor = masks.mean(dim=(2, 3)).expand(-1, 4)
            posterior = self.conditioner.posterior_head(descriptor)
            routing = self.teacher_router(posterior.mean)
        return posterior, routing

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
        descriptor = encoded.descriptor if descriptor is None else descriptor
        level_pools = torch.stack(
            [tokens.mean(dim=1) for tokens in descriptor.level_tokens],
            dim=1,
        )
        enhancement = self.enhancement(
            descriptor.level_tokens,
            level_pools,
            latent,
            routing.routing_probs,
            routing.expert_indices,
            layer_scorer=layer_scorer,
            expert_bank=expert_bank,
        )
        side = self.image_size
        raw_logits = enhancement.fused_tokens.mean(dim=2).reshape(
            latent.shape[0],
            1,
            side,
            side,
        )
        logits = self.backbone.decoder(self.enhancement_neck(raw_logits))
        return SimpleNamespace(
            logits=logits,
            iou_predictions=logits.mean(dim=(2, 3)),
            gamma=enhancement.layer_weights,
            beta=enhancement.expert_layer_weights,
        )


def _batch():
    generator = torch.Generator().manual_seed(31)
    images = torch.rand(2, 12, 4, 4, generator=generator)
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, :, 1:3, 1:3] = 1
    masks[1, :, :3, :2] = 1
    return images, masks


def _segmentation_task():
    return PhaseCDistillTask(
        criterion=BCEDiceLoss(),
        latent_objective="none",
        lambda_latent=0.0,
        lambda_route=0.0,
        lambda_deploy=1.0,
        threshold=0.5,
        boundary_tolerance=1,
    )


def _snapshot(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_snapshot_equal(test, expected, module):
    actual = module.state_dict()
    test.assertEqual(set(actual), set(expected))
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)


class TestC5StudentRouting(unittest.TestCase):
    def test_router_and_layer_scorer_are_isolated_identical_copies(self):
        torch.manual_seed(5)
        model = TinyC5Model()

        self.assertIsNot(model.teacher_router, model.student_router)
        self.assertIsNot(
            model.teacher_layer_scorer,
            model.student_layer_scorer,
        )
        for teacher, student in (
            (model.teacher_router, model.student_router),
            (model.teacher_layer_scorer, model.student_layer_scorer),
        ):
            self.assertEqual(
                set(teacher.state_dict()),
                set(student.state_dict()),
            )
            for name, teacher_tensor in teacher.state_dict().items():
                student_tensor = student.state_dict()[name]
                torch.testing.assert_close(
                    teacher_tensor,
                    student_tensor,
                    rtol=0,
                    atol=0,
                )
                self.assertNotEqual(
                    teacher_tensor.data_ptr(),
                    student_tensor.data_ptr(),
                )

    def test_exact_trainable_set_modes_and_optimizer_membership(self):
        model = TinyC5Model()
        expected_prefixes = (
            "prior_head.",
            "student_router.",
            "student_layer_scorer.",
        )
        trainable_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith(expected_prefixes) for name in trainable_names)
        )
        expected_ids = {
            id(parameter)
            for parameter in model.optimizer_parameters()
        }
        self.assertEqual(
            expected_ids,
            {
                id(parameter)
                for parameter in model.parameters()
                if parameter.requires_grad
            },
        )

        optimizer = build_optimizer(
            {
                "optimizer": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "weight_decay": 1e-5,
                }
            },
            model,
        )
        optimizer_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        self.assertEqual(
            {id(parameter) for parameter in optimizer_parameters},
            expected_ids,
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in optimizer_parameters)
        )

        model.train()
        self.assertTrue(model.prior_head.training)
        self.assertTrue(model.student_router.training)
        self.assertTrue(model.student_layer_scorer.training)
        self.assertFalse(model.conditioner.training)
        self.assertFalse(model.enhancement.training)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.image_descriptor.training)

    def test_segmentation_step_updates_every_student_group_only(self):
        torch.manual_seed(7)
        model = TinyC5Model()
        task = _segmentation_task()
        images, masks = _batch()
        optimizer = build_optimizer(
            {
                "optimizer": {
                    "name": "adamw",
                    "lr": 5e-2,
                    "weight_decay": 0.0,
                }
            },
            model,
        )

        trainable_before = {
            name: _snapshot(module)
            for name, module in (
                ("prior_head", model.prior_head),
                ("student_router", model.student_router),
                ("student_layer_scorer", model.student_layer_scorer),
            )
        }
        frozen_modules = {
            "posterior": model.conditioner.posterior_head,
            "teacher_router": model.teacher_router,
            "teacher_layer_scorer": model.teacher_layer_scorer,
            "experts": model.enhancement.experts,
            "image_descriptor": model.image_descriptor,
            "shape_teacher": model.shape_teacher,
            "enhancement_neck": model.enhancement_neck,
            "backbone": model.backbone,
        }
        frozen_before = {
            name: _snapshot(module)
            for name, module in frozen_modules.items()
        }

        model.train()
        step = task.training_step(
            model,
            {"image": images, "mask": masks},
            torch.device("cpu"),
        )
        self.assertEqual(model.posterior_calls, 0)
        step.loss.backward()

        for name, parameters in model.trainable_parameter_groups().items():
            gradient_total = sum(
                float(parameter.grad.abs().sum())
                for parameter in parameters
                if parameter.grad is not None
            )
            self.assertGreater(gradient_total, 0.0, name)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)

        optimizer.step()
        for name, module in (
            ("prior_head", model.prior_head),
            ("student_router", model.student_router),
            ("student_layer_scorer", model.student_layer_scorer),
        ):
            changed = any(
                not torch.equal(module.state_dict()[key], tensor)
                for key, tensor in trainable_before[name].items()
            )
            self.assertTrue(changed, name)
        for name, module in frozen_modules.items():
            _assert_snapshot_equal(self, frozen_before[name], module)

    def test_public_path_is_mask_independent_and_uses_student_routing(self):
        model = TinyC5Model().eval()
        images, masks = _batch()
        mask_b = 1.0 - masks

        with patch.object(
            model.teacher_router,
            "forward",
            side_effect=AssertionError("teacher router used by student"),
        ), patch.object(
            model.teacher_layer_scorer,
            "forward",
            side_effect=AssertionError("teacher scorer used by student"),
        ), patch.object(
            model.student_router,
            "forward",
            wraps=model.student_router.forward,
        ) as student_router, patch.object(
            model.student_layer_scorer,
            "forward",
            wraps=model.student_layer_scorer.forward,
        ) as student_scorer:
            without_mask = model(images).logits
            with_a = model(images, masks=masks).logits
            with_b = model(images, masks=mask_b).logits

        self.assertEqual(student_router.call_count, 3)
        self.assertEqual(student_scorer.call_count, 3)
        torch.testing.assert_close(without_mask, with_a, rtol=0, atol=0)
        torch.testing.assert_close(without_mask, with_b, rtol=0, atol=0)
        self.assertEqual(model.posterior_calls, 0)

    def test_teacher_oracle_and_gamma_diagnostics_are_invariant(self):
        torch.manual_seed(11)
        model = TinyC5Model().eval()
        images, masks = _batch()

        with torch.no_grad():
            before = model.distillation_forward(
                images,
                masks,
                decode_prior=True,
                decode_posterior=True,
            )
            encoded = model._encode_images(images)
            posterior, routing = model._posterior_from_encoded(encoded, masks)
            reference = model._decode_routing(
                encoded,
                latent=posterior.mean,
                routing=routing,
                layer_scorer=model.teacher_layer_scorer,
            )
        torch.testing.assert_close(
            before.posterior_logits,
            reference.logits,
            rtol=0,
            atol=0,
        )
        self.assertEqual(tuple(before.prior_gamma.shape), (2, 2))
        self.assertEqual(tuple(before.posterior_gamma.shape), (2, 2))
        self.assertEqual(tuple(before.prior_beta.shape), (2, 2, 2))
        self.assertEqual(tuple(before.posterior_beta.shape), (2, 2, 2))

        optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=1e-2)
        loss = BCEDiceLoss()(model(images).logits, masks)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            after = model.distillation_forward(
                images,
                masks,
                decode_prior=True,
                decode_posterior=True,
            )
        torch.testing.assert_close(
            before.posterior_logits,
            after.posterior_logits,
            rtol=0,
            atol=0,
        )


@unittest.skipUnless(
    os.environ.get("RUN_PHASE_C_INTEGRATION_TESTS") == "1",
    "Set RUN_PHASE_C_INTEGRATION_TESTS=1 for the real SAM/B6 checkpoint test.",
)
class TestC5RealB6Integration(unittest.TestCase):
    def test_real_counts_initialization_and_teacher_parity(self):
        c5_config = load_experiment_config(
            PROJECT_ROOT / "configs/phase_c/c5_trainable_student_routing.yaml",
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
        model = build_model({**c5_config["model"], **shared}).to(device).eval()
        metadata = model.phase_c_checkpoint_metadata()
        self.assertEqual(metadata["parameter_counts"]["total"], 119675208)
        self.assertEqual(metadata["parameter_counts"]["frozen"], 119536451)
        self.assertEqual(metadata["parameter_counts"]["trainable"], 138757)
        self.assertEqual(
            metadata["parameter_counts"]["trainable_groups"],
            {
                "prior_head": 32896,
                "student_router": 260,
                "student_layer_scorer": 105601,
            },
        )

        for teacher, student in (
            (model.teacher_router, model.student_router),
            (model.teacher_layer_scorer, model.student_layer_scorer),
        ):
            for teacher_parameter, student_parameter in zip(
                teacher.parameters(),
                student.parameters(),
            ):
                torch.testing.assert_close(
                    teacher_parameter,
                    student_parameter,
                    rtol=0,
                    atol=0,
                )
                self.assertNotEqual(
                    teacher_parameter.data_ptr(),
                    student_parameter.data_ptr(),
                )

        generator = torch.Generator().manual_seed(42)
        images = torch.randn(1, 3, 256, 256, generator=generator).to(device)
        masks = (
            torch.rand(1, 1, 256, 256, generator=generator) > 0.5
        ).float().to(device)
        with torch.inference_mode():
            c5_state = model.distillation_forward(
                images,
                masks,
                decode_prior=True,
                decode_posterior=True,
            )
            image_only = model(images)

        standalone = build_model({**b6_config["model"], **shared}).to(device)
        teacher_checkpoint = torch.load(
            c5_config["model"]["teacher_checkpoint"],
            map_location=device,
            weights_only=False,
        )
        standalone.load_state_dict(
            teacher_checkpoint["model_state_dict"],
            strict=True,
        )
        standalone.eval()
        with torch.inference_mode():
            expected = standalone(images, masks=masks).logits
        torch.testing.assert_close(
            c5_state.posterior_logits,
            expected,
            rtol=0,
            atol=0,
        )
        self.assertEqual(tuple(image_only.logits.shape), (1, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
