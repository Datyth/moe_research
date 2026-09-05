"""Tests for segmentation training, validation, resume and checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.engine import Trainer, TrainerConfig
from src.losses import BCEDiceLoss
from src.models import build_model
from src.models.shape import ShapeAutoencoderOutput
from src.tasks import MaskReconstructionTask, SegmentationTask


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
    monitor: str = "dice",
    monitor_mode: str = "max",
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
    criterion = BCEDiceLoss()
    task = SegmentationTask(criterion=criterion)
    return Trainer(
        model=model,
        task=task,
        task_config={
            "name": "segmentation",
            "threshold": 0.5,
            "boundary_tolerance": 2.0,
        },
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
            monitor=monitor,
            monitor_mode=monitor_mode,
        ),
        checkpoint_metadata={"model_config": model_config},
    )


class TinyMaskDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        mask = torch.zeros(1, 4, 4)
        mask[:, index:index + 2, index:index + 2] = 1
        return {"mask": mask}


class TinyShapeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, masks):
        return ShapeAutoencoderOutput(
            reconstruction_logits=masks * self.scale,
            latent=masks.mean(dim=(2, 3)),
        )


def build_tiny_mask_trainer(root: Path, *, epochs: int = 1) -> Trainer:
    model = TinyShapeModel()
    loader = DataLoader(TinyMaskDataset(), batch_size=2)
    task = MaskReconstructionTask(criterion=torch.nn.MSELoss())
    return Trainer(
        model=model,
        task=task,
        task_config={"name": "mask_reconstruction", "threshold": 0.5},
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        train_loader=loader,
        val_loader=loader,
        config=TrainerConfig(
            epochs=epochs,
            device="cpu",
            last_checkpoint_path=root / "shape_last.pt",
            best_checkpoint_path=root / "shape_best.pt",
            history_path=root / "shape_history.json",
            use_amp=False,
            log_interval=1,
            monitor="loss",
            monitor_mode="min",
        ),
    )


class TestTrainer(unittest.TestCase):

    def test_same_trainer_optimizes_mask_reconstruction_with_loss_min(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = build_tiny_mask_trainer(root)
            history = trainer.train()
            checkpoint = torch.load(
                root / "shape_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                set(history[0]),
                {"epoch", "train_loss", "val_loss", "val_dice"},
            )
            self.assertEqual(checkpoint["monitor_name"], "loss")
            self.assertEqual(checkpoint["monitor_mode"], "min")
            self.assertNotIn("best_val_dice", checkpoint)

    def test_generic_loss_min_checkpoint_resumes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = build_tiny_mask_trainer(root, epochs=1)
            first.train()
            resumed = build_tiny_mask_trainer(root, epochs=2)
            self.assertEqual(resumed.resume(root / "shape_last.pt"), 1)
            history = resumed.train()
            self.assertEqual([entry["epoch"] for entry in history], [1, 2])
            self.assertIsNotNone(resumed.best_monitor_value)

    def test_legacy_v2_resume_requires_dice_max(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_tiny_trainer(root, with_validation=True).train()
            checkpoint_path = root / "unet_last.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            for key in ("monitor_name", "monitor_mode", "best_monitor_value"):
                checkpoint.pop(key)
            torch.save(checkpoint, checkpoint_path)

            compatible = build_tiny_trainer(root, with_validation=True, epochs=2)
            self.assertEqual(compatible.resume(checkpoint_path), 1)
            incompatible = build_tiny_trainer(
                root,
                with_validation=True,
                epochs=2,
                monitor="loss",
                monitor_mode="min",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                incompatible.resume(checkpoint_path)

    def test_missing_monitor_metric_raises_after_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = build_tiny_trainer(
                Path(temporary_directory),
                with_validation=True,
                monitor="does_not_exist",
            )
            with self.assertRaisesRegex(ValueError, "was not returned"):
                trainer.train()

    def test_trainer_config_rejects_invalid_monitoring(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TrainerConfig(monitor="")
        with self.assertRaisesRegex(ValueError, "monitor_mode"):
            TrainerConfig(monitor_mode="sideways")

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
                {
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_dice",
                    "val_iou",
                    "val_hd95",
                    "val_assd",
                    "val_boundary_f1",
                },
            )
            self.assertEqual(last_checkpoint["format_version"], 2)
            self.assertEqual(best_checkpoint["model_class"], "UNetModel")
            self.assertIn("optimizer_state_dict", last_checkpoint)
            self.assertIn("scaler_state_dict", last_checkpoint)
            self.assertIn("scheduler_state_dict", last_checkpoint)
            self.assertEqual(last_checkpoint["monitor_name"], "dice")
            self.assertEqual(last_checkpoint["monitor_mode"], "max")
            self.assertEqual(
                last_checkpoint["task_config"]["name"],
                "segmentation",
            )
            for key in ("metrics", "best_monitor_value", "trainer_config", "metadata"):
                self.assertIn(key, last_checkpoint)
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
            self.assertEqual(set(history[0]), {"epoch", "train_loss"})

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
                resumed_checkpoint["best_monitor_value"],
                first_checkpoint["best_monitor_value"],
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

if __name__ == "__main__":
    unittest.main(verbosity=2)
