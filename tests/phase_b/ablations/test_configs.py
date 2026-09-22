"""Resolved-config regression for the A0-A4 comparison matrix."""

import copy
import unittest
from pathlib import Path

from src.configs import load_experiment_config
from src.models.registry import MODEL_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs" / "phase_b" / "ablations"


class TestPhaseBAblationConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a0 = load_experiment_config(
            PROJECT_ROOT / "configs/phase_b/isic2018_moe_no_feb.yaml",
            project_root=PROJECT_ROOT,
        )
        cls.configs = {
            path.stem: load_experiment_config(path, project_root=PROJECT_ROOT)
            for path in CONFIG_DIR.glob("*.yaml")
        }

    def test_all_configs_share_controlled_training_recipe_and_disable_feb(self):
        for name, config in self.configs.items():
            with self.subTest(config=name):
                for section in (
                    "seed",
                    "dataset",
                    "loss",
                    "optimizer",
                    "scheduler",
                    "training",
                ):
                    self.assertEqual(config[section], self.a0[section])
                self.assertEqual(
                    config["experiment"]["output_root"],
                    self.a0["experiment"]["output_root"],
                )
                self.assertFalse(config["model"]["use_moe"])
                self.assertTrue(config["model"]["use_lpeg"])
                self.assertTrue(config["model"]["freeze_backbone"])
                self.assertEqual(config["model"]["levels"], [3, 6, 9, 12])

    def test_a1_and_a2_only_declare_modules_they_use(self):
        a1 = self.configs["a1_no_expert"]
        for key in (
            "shape_teacher_checkpoint",
            "freeze_shape_teacher",
            "latent_dim",
            "num_experts",
            "active_experts",
            "stochastic",
            "expert_hidden_ratio",
            "moe_num_experts",
            "moe_top_k_ratio",
        ):
            self.assertNotIn(key, a1["model"])

        a2 = self.configs["a2_direct_fuse"]
        for key in (
            "latent_dim",
            "std_floor",
            "stochastic",
            "moe_num_experts",
            "moe_top_k_ratio",
        ):
            self.assertNotIn(key, a2["model"])
        self.assertEqual(a2["model"]["num_experts"], 4)
        self.assertEqual(a2["model"]["active_experts"], 2)
        self.assertEqual(a2["task"]["lambda_latent"], 0.0)
        self.assertEqual(a2["task"]["lambda_balance"], 0.01)

    def test_a3_is_exactly_a0_except_name_and_stochastic_flag(self):
        expected = copy.deepcopy(self.a0)
        expected["experiment"]["name"] = "phase_b_a3_deterministic_posterior"
        expected["model"]["stochastic"] = False
        self.assertEqual(self.configs["a3_deterministic_posterior"], expected)

    def test_a4_is_exactly_a0_except_name_and_model_registry(self):
        expected = copy.deepcopy(self.a0)
        expected["experiment"]["name"] = (
            "phase_b_a4_shape_conditioned_fusion"
        )
        expected["model"]["name"] = "phase_b_a4_shape_conditioned"
        self.assertEqual(
            self.configs["a4_shape_conditioned_fusion"],
            expected,
        )

    def test_model_names_registered_and_checkpoints_exist(self):
        import src.models  # noqa: F401

        for model_name in (
            "phase_b_a1_no_expert",
            "phase_b_no_moe",
            "phase_b_a2_direct_fuse",
            "phase_b_a4_shape_conditioned",
        ):
            self.assertIn(model_name, MODEL_REGISTRY)
        for config in (self.a0, self.configs["a2_direct_fuse"]):
            self.assertTrue(Path(config["model"]["checkpoint"]).is_file())
            self.assertTrue(
                Path(config["model"]["shape_teacher_checkpoint"]).is_file()
            )


if __name__ == "__main__":
    unittest.main()
