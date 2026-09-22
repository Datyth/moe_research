"""Teacher checkpoint validation tests without constructing SAM."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.phase_c.b6_prior_distill import load_b6_teacher_checkpoint


def _checkpoint(model, *, model_class="PhaseBB6Hierarchical"):
    return {
        "format_version": 2,
        "model_class": model_class,
        "epoch": 7,
        "monitor_name": "dice",
        "monitor_mode": "max",
        "best_monitor_value": 0.9,
        "model_state_dict": model.state_dict(),
        "task_config": {
            "name": "phase_b_build_up",
            "evaluation_mode": "posterior_oracle",
        },
        "metadata": {
            "experiment_name": "phase_b_b6_hierarchical",
            "seed": 42,
            "model_config": {
                "name": "phase_b_b6_hierarchical",
                "levels": [3, 6, 9, 12],
            },
        },
    }


class TestTeacherCheckpoint(unittest.TestCase):
    def test_valid_checkpoint_loads_strictly_and_returns_origin(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(_checkpoint(source), path)
            info = load_b6_teacher_checkpoint(
                target,
                path,
                expected_model_config={"levels": (3, 6, 9, 12)},
            )
        for expected, actual in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(expected, actual)
        self.assertEqual(info["originating_experiment"], "phase_b_b6_hierarchical")
        self.assertEqual(info["epoch"], 7)

    def test_wrong_teacher_class_fails_loudly(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(_checkpoint(model, model_class="WrongModel"), path)
            with self.assertRaisesRegex(ValueError, "model_class"):
                load_b6_teacher_checkpoint(model, path)

    def test_incompatible_state_dict_fails_strictly(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(_checkpoint(source), path)
            with self.assertRaises(RuntimeError):
                load_b6_teacher_checkpoint(target, path)


if __name__ == "__main__":
    unittest.main()
