"""Tests for experiment config, builders, run artifacts and summaries."""

import json
import statistics
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from scripts.summarize_experiments import aggregate_runs, collect_runs
from src.configs import resolve_experiment_config
from src.data import build_dataset
from src.experiment import (
    build_dataset_config,
    build_optimizer,
    build_scheduler,
    execute_experiment,
    file_sha256,
    set_seed,
)
from src.losses import BCEDiceLoss, build_loss, register_loss
from src.models import build_model


class ExperimentFixture:
    def __init__(self, root: Path):
        self.root = root
        self.data_root = root / "dataset"
        self.manifest_path = root / "manifest.json"
        records = {
            "training": [
                self._create_sample("train", "TRAIN_A"),
                self._create_sample("train", "TRAIN_B"),
            ],
            "validation": [self._create_sample("train", "VAL_A")],
            "test": [self._create_sample("test", "TEST_A")],
        }
        self.manifest_path.write_text(json.dumps(records), encoding="utf-8")

    def _create_sample(self, split: str, sample_id: str) -> dict[str, str]:
        image_dir = self.data_root / "images" / split
        mask_dir = self.data_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{sample_id}.jpg"
        mask_path = mask_dir / f"{sample_id}.(1,32,32).npz"
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = 255
        Image.fromarray(image).save(image_path)
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[:, 8:24, 8:24] = 1
        sparse.save_npz(mask_path, sparse.csr_matrix(mask.reshape(1, -1)))
        return {
            "image": str(image_path.relative_to(self.data_root)),
            "label": str(mask_path.relative_to(self.data_root)),
        }

    def raw_config(self) -> dict:
        return {
            "experiment": {
                "name": "tiny_unet",
                "output_root": str(self.root / "runs"),
            },
            "seed": 42,
            "dataset": {
                "name": "isic2018",
                "root": str(self.data_root),
                "manifest": str(self.manifest_path),
                "version": "tiny-v1",
                "task": "binary",
                "num_classes": 1,
                "in_channels": 3,
                "image_size": [32, 32],
            },
            "model": {"name": "unet", "base_channels": 2},
            "loss": {
                "name": "bce_dice",
                "bce_weight": 0.5,
                "dice_weight": 0.5,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
                "weight_decay": 0.0,
            },
            "scheduler": {"name": "none"},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
            },
        }


class TestExperimentFramework(unittest.TestCase):
    def test_config_validation_and_builders(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ExperimentFixture(root)
            config = resolve_experiment_config(
                fixture.raw_config(),
                project_root=root,
            )
            self.assertTrue(Path(config["dataset"]["manifest"]).is_absolute())
            self.assertEqual(config["training"]["prediction_threshold"], 0.5)
            self.assertIsInstance(build_loss(config["loss"]), BCEDiceLoss)

            model = build_model({
                "name": "unet",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "base_channels": 2,
            })
            optimizer = build_optimizer(config, model)
            config["scheduler"] = {"name": "cosine", "eta_min": 0.0}
            self.assertIsInstance(
                build_scheduler(config, optimizer),
                CosineAnnealingLR,
            )
            config["scheduler"] = {
                "name": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 1,
                "min_lr": 0.0,
            }
            self.assertIsInstance(
                build_scheduler(config, optimizer),
                ReduceLROnPlateau,
            )

    def test_invalid_config_and_unknown_loss_are_clear(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ExperimentFixture(root)
            raw = fixture.raw_config()
            del raw["training"]["epochs"]
            with self.assertRaisesRegex(ValueError, "epochs"):
                resolve_experiment_config(raw, project_root=root)
            with self.assertRaisesRegex(ValueError, "Unknown loss"):
                build_loss({"name": "does_not_exist"})
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_loss("bce")(torch.nn.MSELoss)

            invalid_scheduler = fixture.raw_config()
            invalid_scheduler["scheduler"]["name"] = "unknown"
            with self.assertRaisesRegex(ValueError, "scheduler.name"):
                resolve_experiment_config(invalid_scheduler, project_root=root)


    def test_training_seed_does_not_change_frozen_manifest_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ExperimentFixture(root)
            config = resolve_experiment_config(
                fixture.raw_config(),
                project_root=root,
            )
            dataset_config = build_dataset_config(config)
            manifest_hash = file_sha256(fixture.manifest_path)

            set_seed(1)
            first_ids = [
                record["image"]
                for record in build_dataset(dataset_config, "train").records
            ]
            set_seed(999)
            second_ids = [
                record["image"]
                for record in build_dataset(dataset_config, "train").records
            ]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(manifest_hash, file_sha256(fixture.manifest_path))

    def test_tiny_cpu_experiment_creates_standard_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture = ExperimentFixture(root)
            config = resolve_experiment_config(
                fixture.raw_config(),
                project_root=root,
            )
            run_dir = execute_experiment(config)

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
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metrics["checkpoint"], "best.pt")
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["dataset"]["manifest_sha256"], file_sha256(fixture.manifest_path))

    def test_summary_groups_only_matching_config_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            runs_root = Path(temporary_directory)
            values = [0.7, 0.8, 0.9]
            for seed, dice in enumerate(values, start=1):
                run_dir = runs_root / "experiment" / f"run-{seed}"
                run_dir.mkdir(parents=True)
                (run_dir / "config.yaml").write_text("seed: 1\n")
                (run_dir / "metadata.json").write_text(json.dumps({
                    "status": "completed",
                    "experiment_name": "experiment",
                    "config_fingerprint": "same",
                    "run_id": run_dir.name,
                    "seed": seed,
                }))
                (run_dir / "test_metrics.json").write_text(json.dumps({
                    "loss": 1.0 - dice,
                    "dice": dice,
                    "iou": dice - 0.1,
                }))

            rows = collect_runs(runs_root, "experiment")
            aggregates = aggregate_runs(rows)
            self.assertEqual(len(aggregates), 1)
            self.assertEqual(aggregates[0]["n"], 3)
            self.assertAlmostEqual(aggregates[0]["dice_mean"], 0.8)
            self.assertAlmostEqual(
                aggregates[0]["dice_std"],
                statistics.stdev(values),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
