"""Configuration invariants for J1 and J2."""

import copy
import unittest
from pathlib import Path

from src.configs import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs/joint_prior_posterior"


class TestJointPriorPosteriorConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.j1 = load_experiment_config(
            CONFIG_DIR / "j1_dz64.yaml",
            project_root=PROJECT_ROOT,
        )
        cls.j2 = load_experiment_config(
            CONFIG_DIR / "j2_dz8.yaml",
            project_root=PROJECT_ROOT,
        )

    def test_latent_dimensions_and_joint_recipe(self):
        self.assertEqual(self.j1["model"]["latent_dim"], 64)
        self.assertEqual(self.j2["model"]["latent_dim"], 8)
        for config in (self.j1, self.j2):
            self.assertEqual(config["model"]["name"], "joint_prior_posterior_b6")
            self.assertFalse(config["model"]["stochastic"])
            self.assertEqual(config["task"]["name"], "joint_prior_posterior")
            self.assertEqual(config["task"]["lambda_balance"], 0.01)
            self.assertEqual(config["task"]["kl_beta_max"], 0.1)
            self.assertEqual(config["task"]["kl_zero_until_epoch"], 5)
            self.assertEqual(config["task"]["kl_ramp_end_epoch"], 20)
            self.assertEqual(config["training"]["monitor"], "dice")
            self.assertEqual(config["training"]["monitor_mode"], "max")

    def test_resolved_configs_differ_only_by_name_and_latent_dimension(self):
        j1 = copy.deepcopy(self.j1)
        j2 = copy.deepcopy(self.j2)
        self.assertNotEqual(j1["experiment"]["name"], j2["experiment"]["name"])
        self.assertNotEqual(j1["model"]["latent_dim"], j2["model"]["latent_dim"])
        del j1["experiment"]["name"]
        del j2["experiment"]["name"]
        del j1["model"]["latent_dim"]
        del j2["model"]["latent_dim"]
        self.assertEqual(j1, j2)

    def test_inherited_scientific_hyperparameters_are_unchanged(self):
        config = self.j1
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["dataset"]["image_size"], [256, 256])
        self.assertEqual(config["loss"]["bce_weight"], 0.5)
        self.assertEqual(config["loss"]["dice_weight"], 0.5)
        self.assertEqual(config["optimizer"]["lr"], 1e-4)
        self.assertEqual(config["optimizer"]["weight_decay"], 1e-5)
        self.assertEqual(config["scheduler"]["name"], "warmup_poly")
        self.assertEqual(config["scheduler"]["warmup_steps"], 250)
        self.assertEqual(config["scheduler"]["power"], 0.9)
        self.assertEqual(config["training"]["epochs"], 50)
        self.assertEqual(config["training"]["batch_size"], 8)
        self.assertEqual(config["training"]["num_workers"], 8)


if __name__ == "__main__":
    unittest.main()
