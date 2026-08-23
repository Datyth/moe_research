"""Tests for binary metrics, evaluation and the evaluation CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.engine import compute_binary_dice_iou, evaluate
from src.models import SegmentationOutput, build_model
from scripts.evaluation.evaluate import _build_boundary_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvaluationDataset(Dataset):
    def __init__(self, values: list[float]):
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = self.values[index]
        image = torch.zeros(3, 2, 2)
        image[0].fill_(value)
        return {"image": image, "mask": torch.zeros(1, 2, 2)}


class TensorModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1]


class OutputModel(nn.Module):
    def forward(self, images: torch.Tensor) -> SegmentationOutput:
        return SegmentationOutput(logits=images[:, :1])


class TestBinaryEvaluator(unittest.TestCase):
    def test_boundary_overlay_uses_distinct_gt_prediction_and_shared_colors(self):
        image = np.zeros((7, 7, 3), dtype=np.float32)
        ground_truth = np.zeros((7, 7), dtype=np.float32)
        prediction = np.zeros((7, 7), dtype=np.float32)
        ground_truth[1:4, 1:4] = 1.0
        prediction[3:6, 3:6] = 1.0

        overlay = _build_boundary_overlay(image, ground_truth, prediction)

        colors = {tuple(pixel) for pixel in overlay.reshape(-1, 3)}
        self.assertIn((0.0, 1.0, 0.0), colors)
        self.assertIn((1.0, 0.0, 0.0), colors)
        self.assertIn((1.0, 1.0, 0.0), colors)

    def test_perfect_and_empty_predictions(self):
        targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        perfect_logits = torch.tensor([[[[-20.0, 20.0], [20.0, -20.0]]]])
        dice, iou = compute_binary_dice_iou(perfect_logits, targets)
        self.assertTrue(torch.equal(dice, torch.ones_like(dice)))
        self.assertTrue(torch.equal(iou, torch.ones_like(iou)))

        empty_logits = torch.full((1, 1, 2, 2), -20.0)
        empty_targets = torch.zeros_like(empty_logits)
        empty_dice, empty_iou = compute_binary_dice_iou(
            empty_logits,
            empty_targets,
        )
        self.assertTrue(torch.equal(empty_dice, torch.ones_like(empty_dice)))
        self.assertTrue(torch.equal(empty_iou, torch.ones_like(empty_iou)))
        self.assertTrue(torch.isfinite(empty_dice).all())
        self.assertTrue(torch.isfinite(empty_iou).all())

    def test_wrong_predictions_remain_bounded(self):
        targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        wrong_logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])
        dice, iou = compute_binary_dice_iou(wrong_logits, targets)
        self.assertTrue(((0.0 <= dice) & (dice <= 1.0)).all())
        self.assertTrue(((0.0 <= iou) & (iou <= 1.0)).all())

    def test_evaluator_weights_loss_by_sample_and_restores_mode(self):
        loader = DataLoader(EvaluationDataset([0.0, 1.0, 2.0]), batch_size=2)
        for model in (TensorModel(), OutputModel()):
            with self.subTest(model=type(model).__name__):
                model.train()
                metrics = evaluate(
                    model=model,
                    loader=loader,
                    criterion=nn.MSELoss(),
                    device="cpu",
                )
                self.assertAlmostEqual(metrics["loss"], 5.0 / 3.0, places=6)
                self.assertTrue(0.0 <= metrics["dice"] <= 1.0)
                self.assertTrue(0.0 <= metrics["iou"] <= 1.0)
                self.assertTrue(model.training)

    def test_evaluation_cli_saves_metrics_and_boundary_visualization(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "dataset"
            image_dir = data_root / "images" / "test"
            mask_dir = data_root / "labels" / "test"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)

            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:24, 8:24, 0] = 255
            Image.fromarray(image).save(image_dir / "ISIC_CLI.jpg")
            mask = np.zeros((1, 32, 32), dtype=np.uint8)
            mask[:, 8:24, 8:24] = 1
            sparse.save_npz(
                mask_dir / "ISIC_CLI.(1,32,32).npz",
                sparse.csr_matrix(mask.reshape(1, -1)),
            )
            manifest = {
                "training": [],
                "validation": [],
                "test": [{
                    "image": "images/test/ISIC_CLI.jpg",
                    "label": "labels/test/ISIC_CLI.(1,32,32).npz",
                }],
            }
            (data_root / "dataset.json").write_text(json.dumps(manifest))

            model_config = {
                "name": "unet",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "base_channels": 2,
            }
            model = build_model(model_config)
            checkpoint_path = root / "unet_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "trainer_config": {"prediction_threshold": 0.5},
                "metadata": {
                    "model_config": model_config,
                    "data_config": {
                        "name": "isic2018",
                        "task": "binary",
                        "num_classes": 1,
                        "in_channels": 3,
                        "image_size": [32, 32],
                        "image_mean": [0.485, 0.456, 0.406],
                        "image_std": [0.229, 0.224, 0.225],
                        "mask_threshold": 0.5,
                    },
                    "loss_config": {
                        "name": "bce_dice",
                        "bce_weight": 0.5,
                        "dice_weight": 0.5,
                    },
                },
            }, checkpoint_path)

            output_dir = root / "results"
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "evaluation" / "evaluate.py"),
                    "--data-root", str(data_root),
                    "--checkpoint", str(checkpoint_path),
                    "--output-dir", str(output_dir),
                    "--device", "cpu",
                    "--batch-size", "1",
                    "--num-workers", "0",
                    "--num-visualizations", "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((output_dir / "metrics.json").read_text())
            visualization_path = output_dir / "visualizations" / "ISIC_CLI.png"
            self.assertEqual(metrics["split"], "test")
            self.assertTrue(np.isfinite(metrics["loss"]))
            self.assertTrue(np.isfinite(metrics["dice"]))
            self.assertTrue(np.isfinite(metrics["iou"]))
            self.assertTrue(visualization_path.is_file())
            with Image.open(visualization_path) as visualization:
                self.assertGreater(visualization.width, visualization.height * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
