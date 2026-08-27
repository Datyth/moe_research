"""Tests for binary metrics, evaluation and the evaluation CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.engine import (
    compute_binary_boundary_f1,
    compute_binary_dice_iou,
    compute_binary_hd95_assd,
    compute_binary_surface_metrics,
    evaluate,
)
from src.models import SegmentationOutput, build_model
from scripts.evaluation.evaluate import _build_boundary_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


class MaskPairDataset(Dataset):
    def __init__(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ):
        self.predictions = predictions
        self.targets = targets

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        logits = torch.where(
            self.predictions[index],
            torch.tensor(20.0),
            torch.tensor(-20.0),
        )
        image = torch.zeros(3, *logits.shape)
        image[0] = logits
        return {"image": image, "mask": self.targets[index].unsqueeze(0).float()}


def logits_from_masks(masks: torch.Tensor) -> torch.Tensor:
    return torch.where(masks, torch.tensor(20.0), torch.tensor(-20.0)).float()


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

    def test_identical_masks_have_perfect_region_and_surface_metrics(self):
        targets = torch.zeros((1, 1, 9, 9))
        targets[:, :, 2:7, 2:7] = 1
        logits = logits_from_masks(targets.bool())

        dice, iou = compute_binary_dice_iou(logits, targets)
        hd95, assd, boundary_f1 = compute_binary_surface_metrics(logits, targets)

        self.assertEqual(dice.item(), 1.0)
        self.assertEqual(iou.item(), 1.0)
        self.assertEqual(hd95.item(), 0.0)
        self.assertEqual(assd.item(), 0.0)
        self.assertEqual(boundary_f1.item(), 1.0)

    def test_surface_metrics_handle_both_empty_and_each_one_empty_direction(self):
        targets = torch.zeros((1, 1, 5, 7))
        empty_logits = torch.full_like(targets, -20.0)
        hd95, assd, boundary_f1 = compute_binary_surface_metrics(
            empty_logits,
            targets,
        )
        self.assertEqual((hd95.item(), assd.item(), boundary_f1.item()), (0, 0, 1))

        nonempty = targets.clone()
        nonempty[:, :, 2, 3] = 1
        nonempty_logits = logits_from_masks(nonempty.bool())
        diagonal = np.hypot(4, 6)
        for logits, ground_truth in (
            (empty_logits, nonempty),
            (nonempty_logits, targets),
        ):
            hd95, assd, boundary_f1 = compute_binary_surface_metrics(
                logits,
                ground_truth,
            )
            self.assertAlmostEqual(hd95.item(), diagonal)
            self.assertAlmostEqual(assd.item(), diagonal)
            self.assertEqual(boundary_f1.item(), 0.0)
            self.assertTrue(torch.isfinite(hd95).all())
            self.assertTrue(torch.isfinite(assd).all())

    def test_shifted_square_is_nonperfect_and_tolerance_is_monotonic(self):
        target = torch.zeros((1, 1, 12, 12))
        prediction = torch.zeros_like(target, dtype=torch.bool)
        target[:, :, 2:7, 2:7] = 1
        prediction[:, :, 2:7, 5:10] = True
        logits = logits_from_masks(prediction)

        hd95, assd = compute_binary_hd95_assd(logits, target)
        strict_f1 = compute_binary_boundary_f1(
            logits,
            target,
            boundary_tolerance=0,
        )
        tolerant_f1 = compute_binary_boundary_f1(
            logits,
            target,
            boundary_tolerance=3,
        )

        self.assertGreater(hd95.item(), 0.0)
        self.assertGreater(assd.item(), 0.0)
        self.assertLess(strict_f1.item(), 1.0)
        self.assertGreaterEqual(tolerant_f1.item(), strict_f1.item())

    def test_single_pixel_translation_has_exact_surface_distance(self):
        target = torch.zeros((1, 1, 7, 7))
        prediction = torch.zeros_like(target, dtype=torch.bool)
        target[:, :, 3, 1] = 1
        prediction[:, :, 3, 4] = True
        logits = logits_from_masks(prediction)

        hd95, assd, boundary_f1 = compute_binary_surface_metrics(
            logits,
            target,
            boundary_tolerance=2,
        )
        self.assertEqual(hd95.item(), 3.0)
        self.assertEqual(assd.item(), 3.0)
        self.assertEqual(boundary_f1.item(), 0.0)
        self.assertEqual(
            compute_binary_boundary_f1(
                logits,
                target,
                boundary_tolerance=3,
            ).item(),
            1.0,
        )

    def test_evaluator_averages_surface_metrics_per_sample(self):
        first_target = torch.zeros((8, 8))
        first_target[2:6, 2:6] = 1
        second_target = torch.zeros((8, 8))
        second_target[3, 1] = 1
        first_prediction = first_target.bool()
        second_prediction = torch.zeros((8, 8), dtype=torch.bool)
        second_prediction[3, 4] = True
        predictions = [first_prediction, second_prediction]
        targets = [first_target, second_target]
        loader = DataLoader(
            MaskPairDataset(predictions, targets),
            batch_size=2,
        )

        metrics = evaluate(
            model=OutputModel(),
            loader=loader,
            criterion=nn.MSELoss(),
            device="cpu",
            boundary_tolerance=2,
        )
        stacked_logits = logits_from_masks(torch.stack(predictions).unsqueeze(1))
        stacked_targets = torch.stack(targets).unsqueeze(1)
        hd95, assd, boundary_f1 = compute_binary_surface_metrics(
            stacked_logits,
            stacked_targets,
            boundary_tolerance=2,
        )
        self.assertAlmostEqual(metrics["hd95"], hd95.mean().item())
        self.assertAlmostEqual(metrics["assd"], assd.mean().item())
        self.assertAlmostEqual(
            metrics["boundary_f1"],
            boundary_f1.mean().item(),
        )

    def test_surface_metrics_reject_invalid_inputs(self):
        logits = torch.zeros((1, 1, 5, 5))
        targets = torch.zeros_like(logits)
        with self.assertRaisesRegex(ValueError, "same shape"):
            compute_binary_surface_metrics(logits, torch.zeros((1, 1, 4, 5)))
        with self.assertRaisesRegex(ValueError, r"\[B, 1, H, W\]"):
            compute_binary_surface_metrics(logits[:, 0], targets[:, 0])
        with self.assertRaisesRegex(ValueError, "threshold"):
            compute_binary_surface_metrics(logits, targets, threshold=1.1)
        with self.assertRaisesRegex(ValueError, "boundary_tolerance"):
            compute_binary_surface_metrics(
                logits,
                targets,
                boundary_tolerance=-1,
            )

    def test_wrong_predictions_remain_bounded(self):
        targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        wrong_logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])
        dice, iou = compute_binary_dice_iou(wrong_logits, targets)
        self.assertTrue(((0.0 <= dice) & (dice <= 1.0)).all())
        self.assertTrue(((0.0 <= iou) & (iou <= 1.0)).all())

    def test_evaluator_weights_loss_by_sample_and_restores_mode(self):
        loader = DataLoader(EvaluationDataset([0.0, 1.0, 2.0]), batch_size=2)
        model = OutputModel()
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

    def test_evaluator_rejects_raw_tensor_model_output(self):
        loader = DataLoader(EvaluationDataset([0.0]), batch_size=1)
        with self.assertRaisesRegex(TypeError, "SegmentationOutput"):
            evaluate(
                model=TensorModel(),
                loader=loader,
                criterion=nn.MSELoss(),
                device="cpu",
            )

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
            self.assertTrue(np.isfinite(metrics["hd95"]))
            self.assertTrue(np.isfinite(metrics["assd"]))
            self.assertTrue(np.isfinite(metrics["boundary_f1"]))
            self.assertTrue(visualization_path.is_file())
            with Image.open(visualization_path) as visualization:
                self.assertGreater(visualization.width, visualization.height * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
