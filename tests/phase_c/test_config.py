"""Configuration and registry regression tests for Phase C."""

import copy
import unittest
from pathlib import Path

from src.configs import load_experiment_config
from scripts.evaluation.evaluate_phase_c_distill import _checkpoint_configuration
from src.configs.experiment import resolve_experiment_config
from src.experiment import TASK_REGISTRY
from src.models.registry import MODEL_REGISTRY
from src.tasks import PhaseCDistillTask


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs/phase_c"
TEACHER = (
    PROJECT_ROOT
    / "runs/phase_b_b6_hierarchical/20260921T151109Z_seed-42/best.pt"
).resolve()


class TestPhaseCConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b0 = load_experiment_config(
            PROJECT_ROOT / "configs/isic2018_e2.yaml",
            project_root=PROJECT_ROOT,
        )
        cls.b6 = load_experiment_config(
            PROJECT_ROOT / "configs/phase_b/build_up/b6_hierarchical.yaml",
            project_root=PROJECT_ROOT,
        )
        cls.configs = {
            path.stem: load_experiment_config(
                path,
                project_root=PROJECT_ROOT,
            )
            for path in CONFIG_DIR.glob("*.yaml")
        }

    def test_main_config_keeps_b6_data_and_training_recipe(self):
        main = self.configs["b6_distill"]
        for section in (
            "seed",
            "dataset",
            "loss",
            "optimizer",
            "scheduler",
            "training",
        ):
            with self.subTest(section=section):
                self.assertEqual(main[section], self.b6[section])
                self.assertEqual(main[section], self.b0[section])

        architecture_fields = (
            "image_size",
            "checkpoint",
            "use_moe",
            "use_lpeg",
            "freeze_backbone",
            "descriptor_dim",
            "levels",
            "scoring_hidden_dim",
            "router_mode",
            "num_experts",
            "active_experts",
            "expert_hidden_ratio",
            "shape_teacher_checkpoint",
            "freeze_shape_teacher",
            "latent_dim",
            "std_floor",
            "stochastic",
        )
        for field in architecture_fields:
            with self.subTest(field=field):
                self.assertEqual(
                    main["model"][field],
                    self.b6["model"][field],
                )

    def test_teacher_path_task_and_registry_are_explicit(self):
        main = self.configs["b6_distill"]
        self.assertEqual(main["model"]["name"], "phase_c_b6_distill")
        self.assertEqual(
            Path(main["model"]["teacher_checkpoint"]),
            TEACHER,
        )
        self.assertEqual(
            main["task"],
            {
                "name": "phase_c_distill",
                "lambda_latent": 1.0,
                "lambda_route": 1.0,
                "lambda_deploy": 0.0,
            },
        )
        self.assertNotIn("lambda_balance", main["task"])
        self.assertNotIn("evaluation_mode", main["task"])
        self.assertIn("phase_c_b6_distill", MODEL_REGISTRY)
        self.assertIs(TASK_REGISTRY["phase_c_distill"], PhaseCDistillTask)

    def test_config_only_ablations_change_only_loss_weights_and_name(self):
        expected = {
            "b6_latent_only": (1.0, 0.0, 0.0),
            "b6_route_only": (0.0, 1.0, 0.0),
            "b6_distill": (1.0, 1.0, 0.0),
            "b6_distill_deploy": (1.0, 1.0, 1.0),
        }
        main = self.configs["b6_distill"]
        for name, weights in expected.items():
            config = self.configs[name]
            with self.subTest(config=name):
                for section in (
                    "seed",
                    "dataset",
                    "model",
                    "loss",
                    "optimizer",
                    "scheduler",
                    "training",
                ):
                    self.assertEqual(config[section], main[section])
                self.assertEqual(
                    (
                        config["task"]["lambda_latent"],
                        config["task"]["lambda_route"],
                        config["task"]["lambda_deploy"],
                    ),
                    weights,
                )

    def test_evaluator_reconstructs_only_phase_c_checkpoint_contract(self):
        main = self.configs["b6_distill"]
        checkpoint = {
            "metadata": {
                "model_config": copy.deepcopy(main["model"]),
                "data_config": copy.deepcopy(main["dataset"]),
                "loss_config": copy.deepcopy(main["loss"]),
            },
            "task_config": {
                **copy.deepcopy(main["task"]),
                "threshold": 0.5,
                "boundary_tolerance": 2.0,
            },
        }
        model, dataset, loss, task = _checkpoint_configuration(
            checkpoint,
            data_root=PROJECT_ROOT / "dataset/isic2018_task1",
        )
        self.assertEqual(model["name"], "phase_c_b6_distill")
        self.assertEqual(dataset.task, "binary")
        self.assertEqual(loss["name"], "bce_dice")
        self.assertEqual(task["name"], "phase_c_distill")

        wrong_model = copy.deepcopy(checkpoint)
        wrong_model["metadata"]["model_config"]["name"] = (
            "phase_b_b6_hierarchical"
        )
        with self.assertRaisesRegex(ValueError, "phase_c_b6_distill"):
            _checkpoint_configuration(
                wrong_model,
                data_root=None,
            )

    def test_nonfinite_and_negative_phase_c_weights_fail_config_validation(self):
        main = self.configs["b6_distill"]
        for field, value in (
            ("lambda_latent", -0.1),
            ("lambda_route", float("nan")),
            ("lambda_deploy", float("inf")),
        ):
            raw = copy.deepcopy(main)
            raw["task"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                field,
            ):
                resolve_experiment_config(
                    raw,
                    project_root=PROJECT_ROOT,
                )


if __name__ == "__main__":
    unittest.main()
