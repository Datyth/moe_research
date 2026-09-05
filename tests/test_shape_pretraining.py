"""Tests for Phase-A configuration and experiment artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from src.configs import resolve_experiment_config
from src.experiments import build_shape_autoencoder, execute_shape_pretraining


class SyntheticShapeDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        mask = torch.zeros(1, 256, 256)
        mask[:, 64:192, 64:192] = 1
        return {
            "image": torch.zeros(3, 256, 256),
            "mask": mask,
            "sample_id": "synthetic",
        }


def raw_shape_config(root: Path) -> dict:
    manifest = root / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return {
        "experiment": {
            "name": "tiny_shape",
            "output_root": str(root / "runs"),
        },
        "seed": 42,
        "dataset": {
            "name": "isic2018",
            "root": str(root / "dataset"),
            "manifest": str(manifest),
            "version": "tiny-shape-v1",
            "task": "binary",
            "num_classes": 1,
            "in_channels": 3,
            "image_size": [256, 256],
        },
        "model": {
            "name": "shape_autoencoder",
            "encoder": {"name": "small_cnn"},
            "projector": {
                "channels": 64,
                "spatial_size": 4,
                "bottleneck_dim": 256,
            },
            "decoder": {
                "start_channels": 128,
                "start_size": 8,
            },
        },
        "loss": {
            "name": "bce_dice",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 0.0003,
            "weight_decay": 0.0001,
        },
        "scheduler": {"name": "none"},
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "amp": False,
            "monitor": "loss",
            "monitor_mode": "min",
            "log_interval": 1,
            "gradient_clip_norm": 1.0,
        },
    }


class TestShapePretraining(unittest.TestCase):
    def test_builder_rejects_inconsistent_fixed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = resolve_experiment_config(
                raw_shape_config(root),
                project_root=root,
            )
            config["model"]["projector"]["bottleneck_dim"] = 128
            with self.assertRaisesRegex(ValueError, "bottleneck_dim"):
                build_shape_autoencoder(config)

    def test_cpu_experiment_creates_phase_a_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = resolve_experiment_config(
                raw_shape_config(root),
                project_root=root,
            )
            loader = DataLoader(SyntheticShapeDataset(), batch_size=1)
            with patch(
                "src.experiments.shape_pretraining.build_loaders",
                return_value=(loader, loader, loader),
            ):
                run_dir = execute_shape_pretraining(config)

            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "config.yaml",
                    "metadata.json",
                    "history.json",
                    "best.pt",
                    "last.pt",
                    "test_metrics.json",
                },
            )
            metrics = json.loads((run_dir / "test_metrics.json").read_text())
            history = json.loads((run_dir / "history.json").read_text())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            checkpoint = torch.load(
                run_dir / "last.pt",
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(set(metrics), {"checkpoint", "split", "loss", "dice"})
            self.assertEqual(
                set(history[0]),
                {"epoch", "train_loss", "val_loss", "val_dice"},
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["monitor_name"], "loss")
            self.assertEqual(metadata["monitor_mode"], "min")
            self.assertNotIn("best_val_dice", metadata)
            self.assertEqual(
                checkpoint["task_config"],
                {"name": "mask_reconstruction", "threshold": 0.5},
            )
            self.assertEqual(checkpoint["monitor_name"], "loss")
            self.assertEqual(checkpoint["monitor_mode"], "min")
            for key in (
                "metrics",
                "best_monitor_value",
                "trainer_config",
                "metadata",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
            ):
                self.assertIn(key, checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
