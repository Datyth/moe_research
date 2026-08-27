"""One-epoch artifact smoke tests for both Shape Teacher input modes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from src.shape_teacher.engine import execute_shape_teacher_experiment


class TestShapeTeacherSmokeRuns(unittest.TestCase):
    def _record(self, root: Path, split: str, sample_id: str) -> dict[str, str]:
        directory = root / "dataset" / "labels" / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{sample_id}.(1,32,32).npz"
        mask = np.zeros((1, 32, 32), dtype=np.uint8)
        mask[:, 7:25, 8:24] = 1
        sparse.save_npz(path, sparse.csr_matrix(mask.reshape(1, -1)))
        return {"label": str(path.relative_to(root / "dataset"))}

    def _config(self, root: Path, *, mode: str) -> dict:
        records = {
            "training": [
                self._record(root, "train", f"{mode}_TRAIN_A"),
                self._record(root, "train", f"{mode}_TRAIN_B"),
            ],
            "validation": [self._record(root, "val", f"{mode}_VAL")],
            "test": [self._record(root, "test", f"{mode}_TEST")],
        }
        manifest = root / f"manifest_{mode}.json"
        manifest.write_text(json.dumps(records), encoding="utf-8")
        corruption_settings = {
            "operation_count": [5, 5],
            "probabilities": {
                "erosion": 1.0,
                "dilation": 0.0,
                "random_holes": 0.0,
                "blob_removal": 0.0,
                "boundary_jitter": 0.0,
            },
            "morphology_radius": [1, 1],
        }
        return {
            "experiment": {
                "name": f"teacher_{mode}",
                "output_root": str(root / "runs"),
            },
            "seed": 42,
            "dataset": {
                "name": "mask_only",
                "root": str(root / "dataset"),
                "manifest": str(manifest),
                "version": "tiny-v1",
                "task": "binary",
                "num_classes": 1,
                "in_channels": 3,
                "image_size": [32, 32],
                "image_mean": [0.0, 0.0, 0.0],
                "image_std": [1.0, 1.0, 1.0],
                "mask_threshold": 0.5,
                "foreground_threshold": 0.0,
                "allow_non_binary_source": False,
            },
            "model": {
                "name": "shape_teacher",
                "mask_channels": 1,
                "encoder_channels": [8, 16],
                "feature_dim": 16,
                "latent_dim": 8,
                "decoder_channels": [8, 4],
                "sigma_floor": 1.0e-4,
            },
            "loss": {
                "name": "shape_teacher",
                "lambda_bce": 1.0,
                "lambda_dice": 1.0,
                "dice_epsilon": 1.0e-6,
            },
            "optimizer": {"name": "adamw", "lr": 3.0e-4, "weight_decay": 1.0e-5},
            "scheduler": {"name": "cosine", "eta_min": 0.0},
            "training": {
                "input_mode": mode,
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "device": "cpu",
                "amp": False,
                "warmup_epochs": 0,
                "early_stopping_patience": 0,
                "prediction_threshold": 0.5,
                "boundary_tolerance": 2.0,
                "log_interval": 1,
                "gradient_clip_norm": 5.0,
                "selection_metric": "val_clean_dice",
            },
            "corruption": {
                "enabled": mode == "denoise",
                "training": corruption_settings,
                "evaluation": corruption_settings,
            },
        }

    def test_clean_and_denoise_smoke_runs_write_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("clean", "denoise"):
                run = execute_shape_teacher_experiment(
                    self._config(root, mode=mode), command=["unit-smoke", mode]
                )
                required = {
                    "config.yaml",
                    "metadata.json",
                    "data_audit.json",
                    "history.json",
                    "best.pt",
                    "last.pt",
                    "validation_metrics.json",
                    "test_metrics.json",
                    "qualitative_selection.json",
                    "qualitative_clean.png",
                    "qualitative_corrupted.png",
                    "reconstruction_grid.png",
                }
                self.assertTrue(required.issubset({path.name for path in run.iterdir()}))
                if mode == "denoise":
                    self.assertTrue((run / "corruption_preview.png").is_file())
                history = json.loads((run / "history.json").read_text())
                self.assertIn("train_sigma_mean", history[0])
                self.assertIn("validation_corrupted_soft_dice", history[0])
                metadata = json.loads((run / "metadata.json").read_text())
                self.assertEqual(metadata["status"], "completed")
                selection = json.loads(
                    (run / "qualitative_selection.json").read_text()
                )
                self.assertEqual(
                    [item["category"] for item in selection["representatives"]],
                    ["small", "large", "smooth", "irregular", "difficult"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
