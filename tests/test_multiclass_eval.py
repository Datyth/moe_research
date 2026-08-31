"""Tests for multiclass losses, metrics and evaluator dispatch."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine.evaluator import evaluate
from src.losses import CEDiceLoss, build_loss
from src.metrics import (
    compute_multiclass_dice_iou,
    compute_multiclass_surface_metrics,
)
from src.models import SegmentationOutput


def _make_logits_and_labels(
    batch_size: int = 2,
    height: int = 16,
    width: int = 16,
    num_classes: int = 4,
    seed: int = 0,
):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, num_classes, height, width, generator=generator)
    labels = torch.randint(0, num_classes, (batch_size, 1, height, width), generator=generator)
    # Guarantee at least one pixel of every foreground class exists.
    labels[:, 0, 0, 0] = 1
    labels[:, 0, 0, 1] = 2
    labels[:, 0, 0, 2] = 3
    return logits, labels


class TestMulticlassMetrics(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        logits, labels = _make_logits_and_labels()
        # One-hot logits select exactly the label class everywhere.
        perfect = torch.nn.functional.one_hot(
            labels.squeeze(1), num_classes=4
        ).permute(0, 3, 1, 2).float() * 100.0

        dice, iou = compute_multiclass_dice_iou(perfect, labels)
        self.assertTrue(torch.allclose(dice, torch.ones_like(dice)))
        self.assertTrue(torch.allclose(iou, torch.ones_like(iou)))

    def test_wrong_prediction_scores_zero(self):
        logits, labels = _make_logits_and_labels(seed=1)
        # Predict a class that is never the label at label!=0, and shift
        # labels so predictions never match for foreground classes.
        shifted_labels = (labels + 1) % 4
        predictions = torch.nn.functional.one_hot(
            shifted_labels.squeeze(1), num_classes=4
        ).permute(0, 3, 1, 2).float() * 100.0

        dice, _ = compute_multiclass_dice_iou(predictions, labels)
        self.assertTrue(torch.allclose(dice, torch.zeros_like(dice)))

    def test_surface_metrics_shapes_and_finiteness(self):
        logits, labels = _make_logits_and_labels()
        hd, hd95, assd, boundary_f1 = compute_multiclass_surface_metrics(
            logits, labels
        )
        for metric in (hd, hd95, assd, boundary_f1):
            self.assertEqual(metric.shape, (2,))
            self.assertTrue(torch.isfinite(metric).all())

    def test_invalid_class_index_rejected(self):
        logits, labels = _make_logits_and_labels()
        labels[0, 0, 0, 0] = 4
        with self.assertRaises(ValueError):
            compute_multiclass_dice_iou(logits, labels)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_dice_iou_preserves_cuda_device(self):
        logits, labels = _make_logits_and_labels()
        logits = logits.cuda()
        labels = labels.cuda()

        dice, iou = compute_multiclass_dice_iou(logits, labels)

        self.assertEqual(dice.device.type, "cuda")
        self.assertEqual(iou.device.type, "cuda")


class TestCEDiceLoss(unittest.TestCase):
    def test_build_from_config(self):
        loss = build_loss({"name": "ce_dice", "ce_weight": 0.5, "dice_weight": 0.5})
        self.assertIsInstance(loss, CEDiceLoss)

    def test_loss_is_scalar_and_finite(self):
        loss = CEDiceLoss()
        logits, labels = _make_logits_and_labels()
        value = loss(logits, labels)
        self.assertEqual(value.ndim, 0)
        self.assertTrue(torch.isfinite(value))

    def test_loss_decreases_toward_perfect_prediction(self):
        loss = CEDiceLoss()
        logits, labels = _make_logits_and_labels()
        perfect = torch.nn.functional.one_hot(
            labels.squeeze(1), num_classes=4
        ).permute(0, 3, 1, 2).float() * 100.0

        self.assertGreater(loss(logits, labels).item(), loss(perfect, labels).item())

    def test_rejects_binary_logits(self):
        loss = CEDiceLoss()
        logits = torch.randn(2, 1, 8, 8)
        labels = torch.zeros(2, 1, 8, 8)
        with self.assertRaises(ValueError):
            loss(logits, labels)

    def test_backward_produces_gradients(self):
        loss = CEDiceLoss()
        logits, labels = _make_logits_and_labels()
        logits.requires_grad_(True)
        loss(logits, labels).backward()
        self.assertIsNotNone(logits.grad)


class TestConfigValidation(unittest.TestCase):
    def test_acdc_unet_config_resolves_as_multiclass(self):
        from pathlib import Path

        from src.configs import load_experiment_config

        project_root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            project_root / "configs" / "acdc_unet.yaml",
            project_root=project_root,
        )
        self.assertEqual(config["dataset"]["task"], "multiclass")
        self.assertEqual(config["dataset"]["num_classes"], 4)
        self.assertEqual(config["loss"]["name"], "ce_dice")


class _TinyMulticlassModel(nn.Module):
    def forward(self, images):
        logits = torch.randn(
            images.shape[0], 4, images.shape[2], images.shape[3]
        )
        return SegmentationOutput(logits=logits)


class _DictDataset(torch.utils.data.Dataset):
    """Yield evaluation batches in the framework's dict format."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return {"image": self.images[index], "mask": self.labels[index]}


class TestEvaluatorDispatch(unittest.TestCase):
    def test_evaluate_handles_multiclass_batch(self):
        images = torch.randn(3, 3, 16, 16)
        labels = torch.randint(0, 4, (3, 1, 16, 16))
        loader = DataLoader(_DictDataset(images, labels), batch_size=3)

        metrics = evaluate(
            model=_TinyMulticlassModel(),
            loader=loader,
            criterion=CEDiceLoss(),
            device="cpu",
        )

        for key in ("loss", "dice", "iou", "hd", "hd95", "assd", "boundary_f1"):
            self.assertIn(key, metrics)
        for key in ("dice", "iou", "hd", "hd95", "assd"):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))


if __name__ == "__main__":
    unittest.main()
