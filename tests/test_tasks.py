"""Tests for segmentation and mask-reconstruction task semantics."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.engine import evaluate
from src.models import SegmentationOutput
from src.models.shape import ShapeAutoencoderOutput
from src.tasks import MaskReconstructionTask, SegmentationTask


class RecordingCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_target = None

    def forward(self, logits, targets):
        self.last_target = targets
        return torch.mean((logits - targets) ** 2)


class RecordingSegmentationModel(nn.Module):
    def __init__(self, *, return_raw_tensor: bool = False) -> None:
        super().__init__()
        self.return_raw_tensor = return_raw_tensor
        self.last_input = None

    def forward(self, images):
        self.last_input = images
        logits = images[:, :1]
        if self.return_raw_tensor:
            return logits
        return SegmentationOutput(logits=logits)


class RecordingShapeModel(nn.Module):
    def __init__(self, *, return_raw_tensor: bool = False) -> None:
        super().__init__()
        self.return_raw_tensor = return_raw_tensor
        self.last_input = None

    def forward(self, masks):
        self.last_input = masks
        logits = masks * 2 - 1
        if self.return_raw_tensor:
            return logits
        return ShapeAutoencoderOutput(
            reconstruction_logits=logits,
            latent=masks.mean(dim=(2, 3)),
        )


class MaskDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        mask = torch.zeros(1, 4, 4)
        mask[:, index:index + 2, index:index + 2] = 1
        return {"mask": mask}


class TestTasks(unittest.TestCase):
    def test_segmentation_task_uses_image_mask_criterion_and_metrics(self):
        criterion = RecordingCriterion()
        task = SegmentationTask(
            criterion=criterion,
            threshold=0.5,
            boundary_tolerance=2,
        )
        model = RecordingSegmentationModel()
        batch = {
            "image": torch.rand(2, 3, 8, 8),
            "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        }

        training = task.training_step(model, batch, "cpu")
        self.assertIs(model.last_input, batch["image"])
        self.assertTrue(torch.equal(criterion.last_target, batch["mask"]))
        self.assertEqual(training.metrics, {})
        self.assertEqual(training.batch_size, 2)

        evaluation = task.evaluation_step(model, batch, "cpu")
        self.assertEqual(
            set(evaluation.metrics),
            {"dice", "iou", "hd95", "assd", "boundary_f1"},
        )

    def test_segmentation_task_requires_segmentation_output(self):
        task = SegmentationTask(criterion=nn.MSELoss())
        batch = {
            "image": torch.zeros(1, 3, 4, 4),
            "mask": torch.zeros(1, 1, 4, 4),
        }
        with self.assertRaisesRegex(TypeError, "SegmentationOutput"):
            task.training_step(
                RecordingSegmentationModel(return_raw_tensor=True),
                batch,
                "cpu",
            )

    def test_mask_reconstruction_uses_mask_as_input_and_target(self):
        criterion = RecordingCriterion()
        task = MaskReconstructionTask(criterion=criterion)
        model = RecordingShapeModel()
        masks = torch.randint(0, 2, (2, 1, 4, 4)).float()
        step = task.training_step(model, {"mask": masks}, "cpu")

        self.assertIs(model.last_input, masks)
        self.assertTrue(torch.equal(criterion.last_target, masks))
        self.assertEqual(step.metrics, {})
        self.assertEqual(step.batch_size, 2)

    def test_mask_reconstruction_evaluation_reports_only_loss_and_dice(self):
        task = MaskReconstructionTask(criterion=nn.MSELoss())
        metrics = evaluate(
            model=RecordingShapeModel(),
            loader=DataLoader(MaskDataset(), batch_size=2),
            task=task,
            device="cpu",
        )
        self.assertEqual(set(metrics), {"loss", "dice"})

    def test_mask_reconstruction_requires_shape_output(self):
        task = MaskReconstructionTask(criterion=nn.MSELoss())
        with self.assertRaisesRegex(TypeError, "ShapeAutoencoderOutput"):
            task.training_step(
                RecordingShapeModel(return_raw_tensor=True),
                {"mask": torch.zeros(1, 1, 4, 4)},
                "cpu",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
