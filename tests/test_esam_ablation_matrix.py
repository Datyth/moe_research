"""End-to-end wiring check for the E-SAM ablation configs.

Builds every `configs/<dataset>_e[0-3].yaml` model exactly the way
`src.experiment.execute_experiment` does (dataset fields merged into the model
block), but with `checkpoint: None` and a 64px backbone so it stays cheap on
CPU. This proves the configs are not just parseable YAML: each row really
instantiates the architecture its name claims, emits `num_classes` channels for
its dataset, and trains only the parameters Adapter fine-tuning is supposed to
touch.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.configs.experiment import load_experiment_config
from src.models import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

DATASETS = ("acdc", "amos22", "isic2018", "synapse")
ROWS = ("e0", "e1", "e2", "e3")
PROBE_IMAGE_SIZE = 64


def _build_from_config(stem: str):
    config = load_experiment_config(
        CONFIG_DIR / f"{stem}.yaml", project_root=REPO_ROOT
    )
    dataset = config["dataset"]
    model_config = {
        **config["model"],
        "in_channels": dataset["in_channels"],
        "num_classes": dataset["num_classes"],
        "task": dataset["task"],
        # The pretrained SAM weights are not part of the repo, and the probe
        # resolution keeps the CPU test fast; both are irrelevant to wiring.
        "checkpoint": None,
        "image_size": PROBE_IMAGE_SIZE,
    }
    return config, build_model(model_config)


class TestEsamConfigWiring(unittest.TestCase):
    def test_every_ablation_row_builds_and_matches_its_flags(self) -> None:
        for dataset in DATASETS:
            for row in ROWS:
                with self.subTest(config=f"{dataset}_{row}"):
                    stem = f"{dataset}_{row}"
                    config, model = _build_from_config(stem)
                    model_config = config["model"]
                    network = model.network

                    self.assertEqual(model_config["use_moe"], network.use_moe)
                    self.assertEqual(
                        model_config["use_lpeg"], network.prompt_encoder.use_lpeg
                    )
                    # Ablation switches must add/remove the modules themselves,
                    # not merely bypass them in forward().
                    self.assertEqual(
                        model_config["use_moe"], hasattr(network, "ExpertChoiceTokenMoE")
                    )
                    self.assertEqual(
                        model_config["use_lpeg"], network.prompt_encoder.lpeg is not None
                    )

    def test_forward_emits_dataset_channel_count(self) -> None:
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                config, model = _build_from_config(f"{dataset}_e3")
                model.eval()
                batch = torch.zeros(1, config["dataset"]["in_channels"], 64, 64)
                with torch.no_grad():
                    output = model(batch)

                self.assertEqual(
                    tuple(output.logits.shape),
                    (1, config["dataset"]["num_classes"], 64, 64),
                )
                self.assertTrue(torch.isfinite(output.logits).all())

    def test_only_fine_tunable_parameters_train(self) -> None:
        config, model = _build_from_config("acdc_e3")
        self.assertTrue(config["model"]["freeze_backbone"])

        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)

        # Frozen SAM backbone: every image-encoder parameter that trains must be
        # an Adapter; the prompt encoder keeps only LPEG trainable.
        for name in trainable:
            if name.startswith("network.image_encoder."):
                self.assertIn("Adapter", name, name)
            elif name.startswith("network.prompt_encoder."):
                self.assertIn("lpeg", name, name)

        pretrained_frozen = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("network.image_encoder.") and not parameter.requires_grad
        )
        self.assertGreater(pretrained_frozen, 0)

    def test_configs_point_at_the_sam_checkpoint(self) -> None:
        for dataset in DATASETS:
            for row in ROWS:
                config = load_experiment_config(
                    CONFIG_DIR / f"{dataset}_{row}.yaml", project_root=REPO_ROOT
                )
                checkpoint = config["model"].get("checkpoint")
                self.assertIsInstance(checkpoint, str, f"{dataset}_{row}")
                self.assertTrue(checkpoint.endswith(".pth"), f"{dataset}_{row}")


if __name__ == "__main__":
    unittest.main()
