"""Tests for the configurable Phase-C D3-D13 adaptive student sweep."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import scripts.run_phase_c_comprehensive as sweep_launcher
from scripts.evaluation.evaluate_phase_c_comprehensive import build_summary_row
from scripts.evaluation.evaluate_phase_c_distill import (
    SUPPORTED_PHASE_C_MODELS,
    _checkpoint_configuration,
)
from src.configs import load_experiment_config
from src.experiment import build_optimizer, config_fingerprint
from src.experiments.phase_c_comprehensive import (
    PHASE_C_COMPREHENSIVE_SPECS,
    PhaseCComprehensiveRun,
    load_spec_config,
    matching_runs,
    newest_completed_run,
)
from src.losses import BCEDiceLoss
from src.models import PhaseCAdaptiveStudent, build_model
from src.models.phase_b.image_descriptor import MultiLevelImageDescriptor
from src.models.phase_b.moe_enhancement import HierarchicalMoEEnhancement
from src.models.phase_b.posterior import GaussianParameterHead
from src.models.phase_b.router import TopKRouter
from src.models.phase_c.b6_prior_distill import _EncodedImageState
from src.tasks.phase_c_distill import PhaseCDistillTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASES = {
    "D3": ((True, False, False, False), ("none", 0.0, 0.0, 1.0)),
    "D4": ((False, True, False, False), ("none", 0.0, 0.0, 1.0)),
    "D6": ((True, True, False, False), ("none", 0.0, 1.0, 1.0)),
    "D7": ((True, True, False, False), ("gaussian_kl", 1.0, 1.0, 1.0)),
    "D8": ((True, True, True, False), ("none", 0.0, 0.0, 1.0)),
    "D9": ((True, True, True, False), ("none", 0.0, 1.0, 1.0)),
    "D10": ((True, True, True, False), ("gaussian_kl", 1.0, 1.0, 1.0)),
    "D11": ((True, True, True, True), ("none", 0.0, 0.0, 1.0)),
    "D12": ((True, True, True, True), ("none", 0.0, 1.0, 1.0)),
    "D13": ((True, True, True, True), ("gaussian_kl", 1.0, 1.0, 1.0)),
}

EXPECTED_COUNTS = {
    "D3": (119675208, 33156),
    "D4": (119675208, 138497),
    "D6": (119675208, 138757),
    "D7": (119675208, 138757),
    "D8": (138571080, 19034629),
    "D9": (138571080, 19034629),
    "D10": (138571080, 19034629),
    "D11": (139375049, 19838598),
    "D12": (139375049, 19838598),
    "D13": (139375049, 19838598),
}


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
        self.encoder_probe = nn.Parameter(torch.ones(1))
        self.decoder = nn.Conv2d(1, 1, kernel_size=1)


class TinyAdaptiveStudent(PhaseCAdaptiveStudent):
    """Small real-router/expert/descriptor harness for every D topology."""

    def __init__(self, flags: tuple[bool, bool, bool, bool]) -> None:
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
        self.image_descriptor = MultiLevelImageDescriptor(
            embed_dim=12,
            descriptor_dim=4,
            levels=(1, 2),
            scoring_hidden_dim=3,
        )
        self.backbone = _TinyBackbone()
        self.shape_teacher = nn.Linear(1, 1)
        self.enhancement_neck = nn.Conv2d(1, 1, kernel_size=1)
        self.fusion = nn.Identity()
        self.posterior_calls = 0
        (
            self.train_student_router,
            self.train_student_layer_router,
            self.train_student_experts,
            self.train_student_image_descriptor,
        ) = flags
        self._initialize_adaptive_student()

    def _encode_images(self, images, *, include_teacher_descriptor=False):
        with torch.no_grad():
            first = images.permute(0, 2, 3, 1).contiguous().detach()
            second = (0.5 * first + 0.1).detach()
            block_outputs = (first, second)

        if self.train_student_image_descriptor:
            descriptor = self.student_image_descriptor(list(block_outputs))
        else:
            with torch.no_grad():
                descriptor = self.image_descriptor(list(block_outputs))

        teacher_descriptor = None
        if include_teacher_descriptor:
            if self.train_student_image_descriptor:
                with torch.no_grad():
                    teacher_descriptor = self.image_descriptor(
                        list(block_outputs)
                    )
            else:
                teacher_descriptor = descriptor
        return _EncodedImageState(
            image_embeddings=images[:, :1].detach(),
            descriptor=descriptor,
            teacher_descriptor=teacher_descriptor,
            block_outputs=block_outputs,
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
        logits = enhancement.fused_tokens.mean(dim=2).reshape(
            latent.shape[0], 1, self.image_size, self.image_size
        )
        logits = self.backbone.decoder(self.enhancement_neck(logits))
        return SimpleNamespace(
            logits=logits,
            iou_predictions=logits.mean(dim=(2, 3)),
            gamma=enhancement.layer_weights,
            beta=enhancement.expert_layer_weights,
        )


def _batch():
    generator = torch.Generator().manual_seed(73)
    images = torch.rand(2, 12, 4, 4, generator=generator)
    masks = torch.zeros(2, 1, 4, 4)
    masks[0, :, 1:3, 1:3] = 1
    masks[1, :, :3, 1:3] = 1
    return {"image": images, "mask": masks}


def _task(loss_spec):
    objective, latent, route, deploy = loss_spec
    return PhaseCDistillTask(
        criterion=BCEDiceLoss(),
        latent_objective=objective,
        lambda_latent=latent,
        lambda_route=route,
        lambda_deploy=deploy,
        threshold=0.5,
        boundary_tolerance=1,
    )


def _snapshot(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _changed(before, module):
    return any(
        not torch.equal(before[name], tensor)
        for name, tensor in module.state_dict().items()
    )


class TestAdaptiveStudentTopology(unittest.TestCase):
    def test_exact_trainable_groups_modes_optimizer_and_gradients(self):
        group_for_flag = (
            "student_router",
            "student_layer_scorer",
            "student_experts",
            "student_image_descriptor",
        )
        for ablation_id, (flags, loss_spec) in CASES.items():
            with self.subTest(ablation_id=ablation_id):
                torch.manual_seed(17)
                model = TinyAdaptiveStudent(flags)
                expected_groups = {"prior_head"} | {
                    name
                    for name, enabled in zip(group_for_flag, flags)
                    if enabled
                }
                self.assertEqual(
                    set(model.trainable_parameter_groups()), expected_groups
                )
                expected_ids = {
                    id(parameter)
                    for parameters in model.trainable_parameter_groups().values()
                    for parameter in parameters
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
                            "lr": 1e-2,
                            "weight_decay": 0.0,
                        }
                    },
                    model,
                )
                self.assertEqual(
                    expected_ids,
                    {
                        id(parameter)
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    },
                )

                model.train()
                self.assertTrue(model.prior_head.training)
                modules = (
                    model.student_router,
                    model.student_layer_scorer,
                    getattr(model, "student_experts", None),
                    getattr(model, "student_image_descriptor", None),
                )
                for enabled, module in zip(flags, modules):
                    if module is not None:
                        self.assertEqual(module.training, enabled)
                self.assertFalse(model.backbone.training)
                self.assertFalse(model.conditioner.training)
                self.assertFalse(model.enhancement.training)
                self.assertFalse(model.image_descriptor.training)

                trainable_before = {
                    name: _snapshot(getattr(model, name))
                    for name in expected_groups
                }
                teacher_modules = {
                    "conditioner": model.conditioner,
                    "enhancement": model.enhancement,
                    "descriptor": model.image_descriptor,
                    "backbone": model.backbone,
                    "shape_teacher": model.shape_teacher,
                    "neck": model.enhancement_neck,
                }
                teacher_before = {
                    name: _snapshot(module)
                    for name, module in teacher_modules.items()
                }

                step = _task(loss_spec).training_step(
                    model, _batch(), torch.device("cpu")
                )
                step.loss.backward()
                for name, parameters in model.trainable_parameter_groups().items():
                    gradient = sum(
                        float(parameter.grad.abs().sum())
                        for parameter in parameters
                        if parameter.grad is not None
                    )
                    self.assertGreater(gradient, 0.0, f"{ablation_id}:{name}")
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, f"{ablation_id}:{name}")

                optimizer.step()
                for name in expected_groups:
                    self.assertTrue(
                        _changed(trainable_before[name], getattr(model, name)),
                        f"{ablation_id}:{name}",
                    )
                for name, module in teacher_modules.items():
                    self.assertFalse(
                        _changed(teacher_before[name], module),
                        f"{ablation_id}:{name}",
                    )

    def test_student_copies_are_equal_and_storage_isolated(self):
        for ablation_id, (flags, _) in CASES.items():
            with self.subTest(ablation_id=ablation_id):
                model = TinyAdaptiveStudent(flags)
                pairs = [
                    (model.teacher_router, model.student_router),
                    (model.teacher_layer_scorer, model.student_layer_scorer),
                ]
                if flags[2]:
                    pairs.append((model.teacher_experts, model.student_experts))
                if flags[3]:
                    pairs.append(
                        (
                            model.teacher_image_descriptor,
                            model.student_image_descriptor,
                        )
                    )
                for teacher, student in pairs:
                    self.assertIsNot(teacher, student)
                    for name, tensor in teacher.state_dict().items():
                        copied = student.state_dict()[name]
                        torch.testing.assert_close(tensor, copied, rtol=0, atol=0)
                        self.assertNotEqual(tensor.data_ptr(), copied.data_ptr())

    def test_public_path_is_image_only_and_uses_student_modules(self):
        model = TinyAdaptiveStudent(CASES["D13"][0]).eval()
        batch = _batch()
        images, mask_a = batch["image"], batch["mask"]
        mask_b = 1.0 - mask_a
        with patch.object(
            model.teacher_router,
            "forward",
            side_effect=AssertionError("teacher router used"),
        ), patch.object(
            model.teacher_layer_scorer,
            "forward",
            side_effect=AssertionError("teacher layer scorer used"),
        ), patch.object(
            model.teacher_experts.experts[0],
            "forward",
            side_effect=AssertionError("teacher expert used"),
        ):
            without_mask = model(images).logits
            with_a = model(images, masks=mask_a).logits
            with_b = model(images, masks=mask_b).logits
        torch.testing.assert_close(without_mask, with_a, rtol=0, atol=0)
        torch.testing.assert_close(without_mask, with_b, rtol=0, atol=0)
        self.assertEqual(model.posterior_calls, 0)

    def test_teacher_oracle_is_unchanged_after_d13_step(self):
        model = TinyAdaptiveStudent(CASES["D13"][0]).eval()
        batch = _batch()
        with torch.no_grad():
            before = model.distillation_forward(
                batch["image"],
                batch["mask"],
                decode_prior=True,
                decode_posterior=True,
            ).posterior_logits.detach().clone()
        optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=1e-2)
        model.train()
        step = _task(CASES["D13"][1]).training_step(
            model, batch, torch.device("cpu")
        )
        step.loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            after = model.distillation_forward(
                batch["image"],
                batch["mask"],
                decode_prior=True,
                decode_posterior=True,
            ).posterior_logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)


class TestComprehensiveConfigsAndTooling(unittest.TestCase):
    def test_manifest_order_configs_loss_weights_and_pair_invariants(self):
        self.assertEqual(
            [spec.ablation_id for spec in PHASE_C_COMPREHENSIVE_SPECS],
            ["D3", "D4", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13"],
        )
        configs = {}
        flag_names = (
            "train_student_router",
            "train_student_layer_router",
            "train_student_experts",
            "train_student_image_descriptor",
        )
        for spec in PHASE_C_COMPREHENSIVE_SPECS:
            config = load_spec_config(spec, PROJECT_ROOT)
            configs[spec.ablation_id] = config
            flags, loss_spec = CASES[spec.ablation_id]
            self.assertEqual(config["model"]["name"], "phase_c_adaptive_student")
            self.assertEqual(
                tuple(config["model"][name] for name in flag_names), flags
            )
            self.assertEqual(
                (
                    config["task"]["latent_objective"],
                    config["task"]["lambda_latent"],
                    config["task"]["lambda_route"],
                    config["task"]["lambda_deploy"],
                ),
                loss_spec,
            )
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["training"]["epochs"], 100)
            self.assertEqual(config["training"]["monitor"], "dice")
            self.assertEqual(config["training"]["monitor_mode"], "max")

        for left, right, changed_section in (
            ("D6", "D7", "task"),
            ("D8", "D9", "task"),
            ("D9", "D10", "task"),
            ("D11", "D12", "task"),
            ("D12", "D13", "task"),
        ):
            for section in ("dataset", "model", "loss", "optimizer", "scheduler", "training"):
                self.assertEqual(configs[left][section], configs[right][section])
            self.assertNotEqual(configs[left][changed_section], configs[right][changed_section])

        for left, right, changed_flag in (
            ("D8", "D11", "train_student_image_descriptor"),
        ):
            left_model = dict(configs[left]["model"])
            right_model = dict(configs[right]["model"])
            self.assertNotEqual(left_model.pop(changed_flag), right_model.pop(changed_flag))
            self.assertEqual(left_model, right_model)
            for section in ("dataset", "loss", "optimizer", "scheduler", "training", "task"):
                self.assertEqual(configs[left][section], configs[right][section])

    def test_registry_evaluator_and_expected_count_contract(self):
        self.assertEqual(
            SUPPORTED_PHASE_C_MODELS["phase_c_adaptive_student"],
            "PhaseCAdaptiveStudent",
        )
        self.assertTrue(issubclass(PhaseCAdaptiveStudent, nn.Module))
        self.assertEqual(EXPECTED_COUNTS["D13"], (139375049, 19838598))

        config = load_spec_config(PHASE_C_COMPREHENSIVE_SPECS[-1], PROJECT_ROOT)
        checkpoint = {
            "metadata": {
                "model_config": {
                    **config["model"],
                    "in_channels": 3,
                    "num_classes": 1,
                    "task": "binary",
                },
                "data_config": config["dataset"],
                "loss_config": config["loss"],
            },
            "task_config": {
                **config["task"],
                "threshold": 0.5,
                "boundary_tolerance": 2.0,
                "task": "binary",
            },
        }
        model_config, _, _, task_config = _checkpoint_configuration(
            checkpoint,
            data_root=None,
        )
        self.assertEqual(model_config["name"], "phase_c_adaptive_student")
        self.assertTrue(model_config["train_student_image_descriptor"])
        self.assertEqual(task_config["lambda_latent"], 1.0)

    def test_run_matching_uses_fingerprint_seed_id_and_completed_checkpoint(self):
        spec = PHASE_C_COMPREHENSIVE_SPECS[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_spec_config(spec, PROJECT_ROOT)
            config["experiment"]["output_root"] = str(root / "runs")
            parent = Path(config["experiment"]["output_root"]) / spec.experiment_name
            run_dir = parent / "20260101T000000Z_seed-42"
            run_dir.mkdir(parents=True)
            metadata = {
                "experiment_name": spec.experiment_name,
                "ablation_id": spec.ablation_id,
                "seed": 42,
                "config_fingerprint": config_fingerprint(config),
                "status": "completed",
                "ended_at": "2026-01-01T01:00:00+00:00",
            }
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (run_dir / "best.pt").touch()
            with patch(
                "src.experiments.phase_c_comprehensive.load_spec_config",
                return_value=config,
            ):
                runs = matching_runs(spec, PROJECT_ROOT)
            self.assertEqual(len(runs), 1)
            self.assertEqual(newest_completed_run(runs).path, run_dir)

    def test_summary_row_has_required_schema(self):
        metrics = {
            name: float(index)
            for index, name in enumerate(
                (
                    "dice", "iou", "hd", "hd95", "assd", "boundary_f1",
                    "posterior_dice", "latent_kl", "mean_distance",
                    "std_distance", "wasserstein2_squared", "route_kl",
                    "exact_topk_set_agreement", "topk_overlap", "routing_js",
                    "gamma_distance", "transfer_gap_dice",
                ),
                start=1,
            )
        }
        row = build_summary_row(
            ablation_id="D3",
            metadata={
                "experiment_name": "phase_c_d3_router_seg",
                "run_id": "run",
                "seed": 42,
                "best_val_dice": 0.9,
            },
            evaluation={
                "checkpoint": "/tmp/best.pt",
                "metrics": metrics,
                "phase_c": {
                    "parameter_counts": {"total": 100, "trainable": 10}
                },
            },
        )
        self.assertEqual(row["ID"], "D3")
        self.assertEqual(row["test_dice"], metrics["dice"])
        self.assertEqual(row["gamma_distance"], metrics["gamma_distance"])
        self.assertEqual(row["trainable_parameters"], 10)

    def test_training_launcher_skips_completed_and_preserves_order(self):
        first, second = PHASE_C_COMPREHENSIVE_SPECS[:2]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "completed"
            run_dir.mkdir()
            (run_dir / "best.pt").touch()
            completed = PhaseCComprehensiveRun(
                run_dir,
                {"status": "completed", "ended_at": "2026-01-01"},
            )
            with patch.object(
                sweep_launcher,
                "PHASE_C_COMPREHENSIVE_SPECS",
                (first, second),
            ), patch.object(
                sweep_launcher,
                "matching_runs",
                side_effect=([completed], []),
            ), patch.object(sweep_launcher.subprocess, "run") as run:
                sweep_launcher.main()
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[-1], str(second.config_path(PROJECT_ROOT)))
            self.assertNotIn(str(first.config_path(PROJECT_ROOT)), command)

    def test_training_launcher_stops_on_matching_unfinished_run(self):
        spec = PHASE_C_COMPREHENSIVE_SPECS[0]
        unfinished = PhaseCComprehensiveRun(
            Path("/tmp/phase-c-unfinished"),
            {"status": "failed"},
        )
        with patch.object(
            sweep_launcher,
            "PHASE_C_COMPREHENSIVE_SPECS",
            (spec,),
        ), patch.object(
            sweep_launcher,
            "matching_runs",
            return_value=[unfinished],
        ), patch.object(sweep_launcher.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "cannot be resumed"):
                sweep_launcher.main()
        run.assert_not_called()

    def test_training_launcher_can_preserve_and_restart_uncheckpointed(self):
        spec = PHASE_C_COMPREHENSIVE_SPECS[0]
        unfinished = PhaseCComprehensiveRun(
            Path("/tmp/phase-c-uncheckpointed"),
            {"status": "running"},
        )
        with patch.object(
            sweep_launcher,
            "PHASE_C_COMPREHENSIVE_SPECS",
            (spec,),
        ), patch.object(
            sweep_launcher,
            "matching_runs",
            return_value=[unfinished],
        ), patch.object(sweep_launcher.subprocess, "run") as run:
            sweep_launcher.main(restart_uncheckpointed=True)
        run.assert_called_once()

    def test_training_launcher_requires_resume_for_checkpointed_run(self):
        spec = PHASE_C_COMPREHENSIVE_SPECS[0]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "last.pt").touch()
            unfinished = PhaseCComprehensiveRun(
                run_dir,
                {"status": "running"},
            )
            with patch.object(
                sweep_launcher,
                "PHASE_C_COMPREHENSIVE_SPECS",
                (spec,),
            ), patch.object(
                sweep_launcher,
                "matching_runs",
                return_value=[unfinished],
            ), patch.object(sweep_launcher.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "Resume it"):
                    sweep_launcher.main(restart_uncheckpointed=True)
            run.assert_not_called()


@unittest.skipUnless(
    os.environ.get("RUN_PHASE_C_COMPREHENSIVE_SMOKE") == "1",
    "Set RUN_PHASE_C_COMPREHENSIVE_SMOKE=1 for all real D-config smoke tests.",
)
class TestComprehensiveRealCheckpointSmoke(unittest.TestCase):
    def test_every_config_builds_and_has_exact_parameter_counts(self):
        shared = {"in_channels": 3, "num_classes": 1, "task": "binary"}
        for spec in PHASE_C_COMPREHENSIVE_SPECS:
            with self.subTest(ablation_id=spec.ablation_id):
                config = load_spec_config(spec, PROJECT_ROOT)
                model = build_model({**config["model"], **shared})
                counts = model.phase_c_checkpoint_metadata()["parameter_counts"]
                expected_total, expected_trainable = EXPECTED_COUNTS[
                    spec.ablation_id
                ]
                self.assertEqual(counts["total"], expected_total)
                self.assertEqual(counts["trainable"], expected_trainable)
                self.assertEqual(
                    counts["frozen"], expected_total - expected_trainable
                )

    def test_every_config_forward_backward_validation_and_checkpoint(self):
        shared = {"in_channels": 3, "num_classes": 1, "task": "binary"}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator().manual_seed(42)
        images = torch.randn(1, 3, 256, 256, generator=generator).to(device)
        masks = (
            torch.rand(1, 1, 256, 256, generator=generator) > 0.5
        ).float().to(device)
        batch = {"image": images, "mask": masks}

        for spec in PHASE_C_COMPREHENSIVE_SPECS:
            with self.subTest(ablation_id=spec.ablation_id):
                config = load_spec_config(spec, PROJECT_ROOT)
                model = build_model({**config["model"], **shared}).to(device)
                task = _task(
                    (
                        config["task"]["latent_objective"],
                        config["task"]["lambda_latent"],
                        config["task"]["lambda_route"],
                        config["task"]["lambda_deploy"],
                    )
                )
                optimizer = build_optimizer(config, model)

                sam_parameter = next(model.backbone.parameters())
                teacher_parameter = next(model.conditioner.parameters())
                sam_before = sam_parameter.detach().clone()
                teacher_before = teacher_parameter.detach().clone()
                descriptor_before = None
                if model.train_student_image_descriptor:
                    descriptor_before = next(
                        model.student_image_descriptor.parameters()
                    ).detach().clone()

                model.train()
                step = task.training_step(model, batch, device)
                self.assertTrue(torch.isfinite(step.loss))
                step.loss.backward()
                for name, parameters in model.trainable_parameter_groups().items():
                    gradient = sum(
                        float(parameter.grad.abs().sum())
                        for parameter in parameters
                        if parameter.grad is not None
                    )
                    self.assertGreater(gradient, 0.0, name)
                optimizer.step()
                torch.testing.assert_close(
                    sam_before, sam_parameter, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    teacher_before, teacher_parameter, rtol=0, atol=0
                )
                if descriptor_before is not None:
                    self.assertFalse(
                        torch.equal(
                            descriptor_before,
                            next(model.student_image_descriptor.parameters()),
                        )
                    )

                model.eval()
                with torch.no_grad():
                    validation = task.evaluation_step(model, batch, device)
                self.assertIn("gamma_distance", validation.metrics)
                self.assertIn("posterior_dice", validation.metrics)

                with tempfile.TemporaryDirectory() as temporary:
                    checkpoint_path = Path(temporary) / "smoke.pt"
                    torch.save(
                        {"model_state_dict": model.state_dict()},
                        checkpoint_path,
                    )
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=device,
                        weights_only=False,
                    )
                    model.load_state_dict(
                        checkpoint["model_state_dict"], strict=True
                    )

                del optimizer, task, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def test_d13_teacher_oracle_matches_original_b6(self):
        spec = PHASE_C_COMPREHENSIVE_SPECS[-1]
        config = load_spec_config(spec, PROJECT_ROOT)
        b6_config = load_experiment_config(
            PROJECT_ROOT / "configs/phase_b/build_up/b6_hierarchical.yaml",
            project_root=PROJECT_ROOT,
        )
        shared = {"in_channels": 3, "num_classes": 1, "task": "binary"}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator().manual_seed(91)
        images = torch.randn(1, 3, 256, 256, generator=generator).to(device)
        masks = (
            torch.rand(1, 1, 256, 256, generator=generator) > 0.5
        ).float().to(device)

        adaptive = build_model({**config["model"], **shared}).to(device).eval()
        with torch.inference_mode():
            actual = adaptive.distillation_forward(
                images,
                masks,
                decode_prior=False,
                decode_posterior=True,
            ).posterior_logits

        teacher = build_model({**b6_config["model"], **shared}).to(device)
        checkpoint = torch.load(
            config["model"]["teacher_checkpoint"],
            map_location=device,
            weights_only=False,
        )
        teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
        teacher.eval()
        with torch.inference_mode():
            expected = teacher(images, masks=masks).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
