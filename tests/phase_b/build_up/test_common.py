"""Shared contracts and configuration regression for build-up B1-B6."""

import os
import unittest
from pathlib import Path

import torch

from scripts.evaluation.evaluate_phase_b_build_up import efficiency_metadata
from src.configs import load_experiment_config
from src.models.phase_b.studies.build_up import (
    DenseTokenFFN,
    PhaseBB1MultiLevel,
)
from src.models.phase_b.studies.build_up.common import (
    IMAGE_ONLY,
    inject_enhancement,
    tokens_to_spatial,
)
from src.tasks.phase_b_build_up import PhaseBBuildUpTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs/phase_b/build_up"


class TestBuildUpCommon(unittest.TestCase):
    def test_token_reshape_and_residual_contract(self):
        tokens = torch.randn(2, 16, 8)
        embeddings = torch.randn(2, 4, 4, 4)
        spatial = tokens_to_spatial(tokens, embeddings, embed_dim=8)
        self.assertEqual(tuple(spatial.shape), (2, 8, 4, 4))
        neck = torch.nn.Conv2d(8, 4, 1, bias=False)
        auxiliary, enhanced = inject_enhancement(
            embeddings,
            tokens,
            neck,
            embed_dim=8,
        )
        torch.testing.assert_close(enhanced, embeddings + auxiliary)

    def test_conditional_beta_metric_finalization(self):
        task = PhaseBBuildUpTask(
            criterion=torch.nn.BCEWithLogitsLoss(),
            evaluation_mode=IMAGE_ONLY,
            lambda_balance=0.0,
        )
        finalized = task.finalize_evaluation_metrics(
            {
                "dice": 0.5,
                "__beta_denominator_expert_0": 0.25,
                "__beta_numerator_expert_0_layer_3": 0.1,
                "__beta_denominator_expert_1": 0.0,
                "__beta_numerator_expert_1_layer_3": 0.0,
            }
        )
        self.assertEqual(finalized["mean_beta_expert_0_layer_3"], 0.4)
        self.assertEqual(finalized["mean_beta_expert_1_layer_3"], 0.0)
        self.assertNotIn("__beta_denominator_expert_0", finalized)

    def test_efficiency_metadata_counts_dense_parameters(self):
        model = torch.nn.Sequential(DenseTokenFFN(embed_dim=8, hidden_dim=13))
        metadata = efficiency_metadata(model)
        self.assertEqual(
            metadata["dense_ffn_parameters"],
            DenseTokenFFN.parameter_count(embed_dim=8, hidden_dim=13),
        )
        self.assertEqual(metadata["expert_bank_parameters"], 0)


class TestBuildUpConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b0 = load_experiment_config(
            PROJECT_ROOT / "configs/isic2018_e2.yaml",
            project_root=PROJECT_ROOT,
        )
        cls.configs = {
            path.stem: load_experiment_config(path, project_root=PROJECT_ROOT)
            for path in CONFIG_DIR.glob("*.yaml")
        }

    def test_all_variants_keep_the_b0_recipe_and_disable_legacy_moe(self):
        for name, config in self.configs.items():
            with self.subTest(config=name):
                for section in ("seed", "dataset", "loss", "optimizer", "scheduler", "training"):
                    self.assertEqual(config[section], self.b0[section])
                self.assertFalse(config["model"]["use_moe"])
                self.assertTrue(config["model"]["use_lpeg"])
                self.assertTrue(config["model"]["freeze_backbone"])
                self.assertEqual(config["model"]["levels"], [3, 6, 9, 12])

    def test_evaluation_modes_and_balance_weights(self):
        for name in ("b1_multilevel", "b2_dense", "b3_image_moe", "b3_random_topk"):
            self.assertEqual(self.configs[name]["task"]["evaluation_mode"], "image_only")
        for name in ("b4_shape_direct", "b5_gaussian", "b5_gaussian_deterministic", "b6_hierarchical"):
            self.assertEqual(self.configs[name]["task"]["evaluation_mode"], "posterior_oracle")
        self.assertEqual(self.configs["b1_multilevel"]["task"]["lambda_balance"], 0.0)
        self.assertEqual(self.configs["b2_dense"]["task"]["lambda_balance"], 0.0)
        self.assertEqual(self.configs["b3_image_moe"]["task"]["lambda_balance"], 0.01)


@unittest.skipUnless(
    os.environ.get("RUN_BACKBONE_TESTS") == "1",
    "Set RUN_BACKBONE_TESTS=1 to build the ViT-B backbone.",
)
class TestBuildUpBackbone(unittest.TestCase):
    def test_frozen_backbone_and_decoder_output_shape(self):
        model = PhaseBB1MultiLevel(image_size=256)
        for name, parameter in model.backbone.network.image_encoder.named_parameters():
            if "Adapter" not in name:
                self.assertFalse(parameter.requires_grad)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 256, 256))
        self.assertEqual(tuple(output.logits.shape), (1, 1, 256, 256))
        alpha = output.diagnostics["level_weights"]
        torch.testing.assert_close(alpha.sum(dim=1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
