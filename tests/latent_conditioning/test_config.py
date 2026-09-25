"""Resolved-config fairness and registration tests for C1/C2/C3."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.configs import load_experiment_config, resolve_experiment_config
from src.experiment import TASK_REGISTRY
from src.models.registry import MODEL_REGISTRY
from src.tasks import LatentConditioningTask


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "latent_conditioning"


def _load(name: str) -> dict:
    return load_experiment_config(
        CONFIG_DIR / f"{name}.yaml",
        project_root=ROOT,
    )


def _without(config: dict, *paths: tuple[str, str]) -> dict:
    result = copy.deepcopy(config)
    for section, key in paths:
        result[section].pop(key)
    return result


class TestLatentConditioningConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c1 = _load("c1_moe_post")
        cls.c2 = _load("c2_no_moe_post")
        cls.c3 = _load("c3_moe_pre_post")

    def test_variants_resolve_to_the_controlled_flags(self):
        expected = (
            (self.c1, "latent_c1_moe_post", True, False),
            (self.c2, "latent_c2_no_moe_post", False, False),
            (self.c3, "latent_c3_moe_pre_post", True, True),
        )
        for config, name, use_moe, use_pre in expected:
            with self.subTest(name=name):
                self.assertEqual(config["experiment"]["name"], name)
                self.assertEqual(
                    config["model"]["name"],
                    "latent_conditioning_model",
                )
                self.assertIs(config["model"]["use_moe"], use_moe)
                self.assertIs(
                    config["model"]["use_pre_moe_latent"],
                    use_pre,
                )
                self.assertIs(config["model"]["use_post_moe_latent"], True)
                self.assertEqual(config["model"]["latent_dim"], 8)
                self.assertFalse(config["model"]["stochastic"])

    def test_c1_and_c3_differ_only_by_name_and_pre_moe_flag(self):
        paths = (
            ("experiment", "name"),
            ("model", "use_pre_moe_latent"),
        )
        self.assertEqual(_without(self.c1, *paths), _without(self.c3, *paths))

    def test_c1_and_c2_differ_only_by_name_and_use_moe(self):
        paths = (
            ("experiment", "name"),
            ("model", "use_moe"),
        )
        self.assertEqual(_without(self.c1, *paths), _without(self.c2, *paths))

    def test_baseline_recipe_and_schedule_are_identical(self):
        for config in (self.c1, self.c2, self.c3):
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["dataset"]["image_size"], [256, 256])
            self.assertEqual(config["loss"]["name"], "bce_dice")
            self.assertEqual(config["loss"]["bce_weight"], 0.5)
            self.assertEqual(config["loss"]["dice_weight"], 0.5)
            self.assertEqual(config["optimizer"]["name"], "adamw")
            self.assertEqual(config["optimizer"]["lr"], 1e-4)
            self.assertEqual(config["optimizer"]["weight_decay"], 1e-5)
            self.assertEqual(config["scheduler"]["name"], "warmup_poly")
            self.assertEqual(config["scheduler"]["warmup_steps"], 250)
            self.assertEqual(config["scheduler"]["power"], 0.9)
            self.assertEqual(config["training"]["epochs"], 50)
            self.assertEqual(config["training"]["batch_size"], 8)
            self.assertEqual(config["training"]["num_workers"], 8)
            self.assertTrue(config["training"]["amp"])
            self.assertEqual(config["training"]["amp_dtype"], "bfloat16")
            self.assertEqual(config["training"]["gradient_clip_norm"], 1.0)
            self.assertEqual(config["training"]["monitor"], "dice")
            self.assertEqual(config["training"]["monitor_mode"], "max")
            self.assertEqual(config["task"]["lambda_balance"], 0.01)
            self.assertEqual(config["task"]["kl_beta_max"], 0.1)
            self.assertEqual(config["task"]["kl_zero_until_epoch"], 5)
            self.assertEqual(config["task"]["kl_ramp_end_epoch"], 20)

    def test_checkpoints_and_exact_architecture_are_resolved(self):
        expected_teacher = (
            Path.home()
            / "projects/project_01/nhan/moe_research/runs"
            / "phase_a_s0_small_cnn_10ep"
            / "20260910T185518Z_seed-42/best.pt"
        ).resolve()
        for config in (self.c1, self.c2, self.c3):
            model = config["model"]
            self.assertEqual(
                Path(model["checkpoint"]),
                (ROOT / "checkpoints/sam_vit_b_01ec64.pth").resolve(),
            )
            self.assertEqual(
                Path(model["shape_teacher_checkpoint"]),
                expected_teacher,
            )
            self.assertEqual(model["descriptor_dim"], 256)
            self.assertEqual(model["levels"], [3, 6, 9, 12])
            self.assertEqual(model["moe_context_dim"], 256)
            self.assertEqual(model["latent_projection_dim"], 64)
            self.assertEqual(model["num_experts"], 4)
            self.assertEqual(model["active_experts"], 2)

    def test_registries_include_model_and_task(self):
        self.assertIn("latent_conditioning_model", MODEL_REGISTRY)
        self.assertIs(
            TASK_REGISTRY["latent_conditioning"],
            LatentConditioningTask,
        )

    def test_invalid_kl_schedule_is_rejected(self):
        config = copy.deepcopy(self.c1)
        config["task"]["kl_zero_until_epoch"] = 20
        config["task"]["kl_ramp_end_epoch"] = 20
        with self.assertRaisesRegex(ValueError, "must be less than"):
            resolve_experiment_config(config, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
