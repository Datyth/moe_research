"""A1 compatibility and segmentation-only task tests."""

import unittest

import torch

from src.models.phase_b import PhaseBA1NoExpertStage, PhaseBNoMoEStage
from src.models.phase_b.ablations.common import AblationEnhancementOutput
from src.tasks import PhaseBA1NoExpertTask


class _A1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 1, 1)

    def forward(self, images):
        from src.models import SegmentationOutput

        batch_size = images.shape[0]
        alpha = torch.full((batch_size, 4), 0.25, device=images.device)
        stage = AblationEnhancementOutput(
            layer_weights=alpha,
            aux_norm_ratio=torch.ones(batch_size, device=images.device),
            fused_token_norm=torch.full(
                (batch_size,), 2.0, device=images.device
            ),
        )
        return SegmentationOutput(
            logits=self.projection(images),
            diagnostics={
                "level_weights": alpha,
                "level_ids": (3, 6, 9, 12),
                "phase_b_ablation": stage,
            },
        )


class TestA1NoExpert(unittest.TestCase):
    def test_old_and_new_registry_names_share_implementation(self):
        self.assertIs(PhaseBA1NoExpertStage, PhaseBNoMoEStage)

    def test_task_uses_segmentation_loss_and_reports_diagnostics(self):
        task = PhaseBA1NoExpertTask(
            criterion=torch.nn.BCEWithLogitsLoss()
        )
        batch = {
            "image": torch.randn(2, 3, 8, 8),
            "mask": torch.randint(0, 2, (2, 1, 8, 8)).float(),
        }
        model = _A1Model()
        training = task.training_step(model, batch, torch.device("cpu"))
        expected = task.criterion(model(batch["image"]).logits, batch["mask"])
        torch.testing.assert_close(training.loss, expected)

        evaluation = task.evaluation_step(model, batch, torch.device("cpu"))
        for key in (
            "level_weight_entropy",
            "level_weight_max",
            "layer_weight_3",
            "layer_weight_6",
            "layer_weight_9",
            "layer_weight_12",
            "enhancement_aux_ratio",
            "fused_token_norm",
        ):
            self.assertIn(key, evaluation.metrics)


if __name__ == "__main__":
    unittest.main()
