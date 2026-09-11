"""Tests for segmentation training, validation, resume and checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.engine import Trainer, TrainerConfig
from src.losses import BCEDiceLoss
from src.models import build_model


class TinySegmentationDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.rand(3, 32, 32, generator=generator),
            "mask": torch.randint(0, 2, (1, 32, 32), generator=generator).float(),
        }


def build_tiny_trainer(
    root: Path,
    *,
    with_validation: bool,
    epochs: int = 1,
    scheduler_name: str = "none",
    early_stopping_patience: int | None = None,
) -> Trainer:
    model_config = {
        "name": "unet",
        "in_channels": 3,
        "num_classes": 1,
        "task": "binary",
        "base_channels": 2,
    }
    model = build_model(model_config)
    loader = DataLoader(TinySegmentationDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=2)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=0
        )
    return Trainer(
        model=model,
        criterion=BCEDiceLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader if with_validation else None,
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            last_checkpoint_path=root / "unet_last.pt",
            best_checkpoint_path=root / "unet_best.pt",
            history_path=root / "history.json",
            use_amp=False,
            log_interval=1,
            early_stopping_patience=early_stopping_patience,
        ),
        checkpoint_metadata={"model_config": model_config},
    )


class TestTrainer(unittest.TestCase):
    def test_train_validate_and_save_best_last_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = build_tiny_trainer(root, with_validation=True).train()
            best_checkpoint = torch.load(
                root / "unet_best.pt", map_location="cpu", weights_only=False
            )
            last_checkpoint = torch.load(
                root / "unet_last.pt", map_location="cpu", weights_only=False
            )
            saved_history = json.loads((root / "history.json").read_text())

            self.assertEqual(saved_history, history)
            self.assertEqual(len(history), 1)
            self.assertEqual(
                set(history[0]),
                {"epoch", "train_loss", "val_loss", "val_dice", "val_iou"},
            )
            self.assertEqual(last_checkpoint["format_version"], 2)
            self.assertEqual(best_checkpoint["model_class"], "UNetModel")
            self.assertIn("optimizer_state_dict", last_checkpoint)
            self.assertIn("scaler_state_dict", last_checkpoint)
            self.assertIn("scheduler_state_dict", last_checkpoint)
            self.assertEqual(
                best_checkpoint["best_val_dice"],
                max(entry["val_dice"] for entry in history),
            )

    def test_training_without_validation_saves_only_last(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            history = build_tiny_trainer(root, with_validation=False).train()

            self.assertTrue((root / "unet_last.pt").is_file())
            self.assertFalse((root / "unet_best.pt").exists())
            self.assertIsNone(history[0]["val_loss"])
            self.assertIsNone(history[0]["val_dice"])
            self.assertIsNone(history[0]["val_iou"])

    def test_resume_restores_state_and_appends_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_trainer = build_tiny_trainer(
                root, with_validation=True, epochs=1, scheduler_name="cosine"
            )
            first_history = first_trainer.train()
            first_checkpoint = torch.load(
                root / "unet_last.pt", map_location="cpu", weights_only=False
            )

            resumed_trainer = build_tiny_trainer(
                root, with_validation=True, epochs=2, scheduler_name="cosine"
            )
            completed_epoch = resumed_trainer.resume(root / "unet_last.pt")
            resumed_history = resumed_trainer.train()
            resumed_checkpoint = torch.load(
                root / "unet_last.pt", map_location="cpu", weights_only=False
            )

            self.assertEqual(completed_epoch, 1)
            self.assertEqual([entry["epoch"] for entry in resumed_history], [1, 2])
            self.assertEqual(resumed_history[0], first_history[0])
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertIsNotNone(resumed_checkpoint["scheduler_state_dict"])
            self.assertGreaterEqual(
                resumed_checkpoint["best_val_dice"],
                first_checkpoint["best_val_dice"],
            )


    def test_reduce_on_plateau_state_resumes_and_updates_lr(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_trainer = build_tiny_trainer(
                root,
                with_validation=True,
                scheduler_name="plateau",
            )
            first_trainer.train()
            saved_checkpoint = torch.load(
                root / "unet_last.pt", map_location="cpu", weights_only=False
            )

            resumed_trainer = build_tiny_trainer(
                root,
                with_validation=True,
                epochs=2,
                scheduler_name="plateau",
            )
            resumed_trainer.resume(root / "unet_last.pt")
            self.assertEqual(
                resumed_trainer.scheduler.state_dict(),
                saved_checkpoint["scheduler_state_dict"],
            )
            initial_lr = resumed_trainer.optimizer.param_groups[0]["lr"]
            resumed_trainer._step_scheduler({"loss": float("inf")})
            self.assertEqual(
                resumed_trainer.optimizer.param_groups[0]["lr"],
                initial_lr * 0.5,
            )

    def test_early_stopping_halts_after_patience_epochs_without_improvement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = build_tiny_trainer(
                root,
                with_validation=True,
                epochs=10,
                early_stopping_patience=2,
            )

            # First epoch improves (establishes best_val_dice); the rest plateau,
            # so training should stop after 2 more (patience) epochs, at epoch 3.
            val_dice_sequence = iter([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

            def fake_evaluate(**kwargs):
                return {"loss": 0.1, "dice": next(val_dice_sequence), "iou": 0.1}

            with patch("src.engine.trainer.evaluate", side_effect=fake_evaluate):
                history = trainer.train()

            self.assertEqual(len(history), 3)
            self.assertEqual(trainer.epochs_without_improvement, 2)
            self.assertEqual(trainer.best_val_dice, 0.5)

    def test_early_stopping_disabled_by_default_runs_all_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = build_tiny_trainer(root, with_validation=True, epochs=4)

            with patch(
                "src.engine.trainer.evaluate",
                return_value={"loss": 0.1, "dice": 0.5, "iou": 0.1},
            ):
                history = trainer.train()

            self.assertEqual(len(history), 4)

    def test_early_stopping_streak_survives_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_trainer = build_tiny_trainer(
                root, with_validation=True, epochs=3, early_stopping_patience=5
            )
            val_dice_sequence = iter([0.5, 0.5, 0.5])
            with patch(
                "src.engine.trainer.evaluate",
                side_effect=lambda **kwargs: {
                    "loss": 0.1, "dice": next(val_dice_sequence), "iou": 0.1
                },
            ):
                first_trainer.train()

            self.assertEqual(first_trainer.epochs_without_improvement, 2)

            resumed_trainer = build_tiny_trainer(
                root, with_validation=True, epochs=4, early_stopping_patience=5
            )
            resumed_trainer.resume(root / "unet_last.pt")

            self.assertEqual(resumed_trainer.epochs_without_improvement, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
