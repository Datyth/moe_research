"""Smoke tests for segmentation training, validation and checkpoints."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import torch
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
            "mask": torch.randint(
                0,
                2,
                (1, 32, 32),
                generator=generator,
            ).float(),
        }


def build_tiny_trainer(
    root: Path,
    *,
    with_validation: bool,
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
    return Trainer(
        model=model,
        criterion=BCEDiceLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        train_loader=loader,
        val_loader=loader if with_validation else None,
        config=TrainerConfig(
            epochs=1,
            device="cpu",
            last_checkpoint_path=root / "unet_last.pt",
            best_checkpoint_path=root / "unet_best.pt",
            history_path=root / "history.json",
            use_amp=False,
            log_interval=1,
        ),
        checkpoint_metadata={"model_config": model_config},
    )


class TestTrainer(unittest.TestCase):
    def test_train_validate_and_save_best_last_history(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = build_tiny_trainer(root, with_validation=True)

            history = trainer.train()
            best_checkpoint = torch.load(
                root / "unet_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            last_checkpoint = torch.load(
                root / "unet_last.pt",
                map_location="cpu",
                weights_only=False,
            )
            saved_history = json.loads((root / "history.json").read_text())

            self.assertEqual(len(history), 1)
            self.assertEqual(saved_history, history)
            self.assertEqual(
                set(history[0]),
                {"epoch", "train_loss", "val_loss", "val_dice", "val_iou"},
            )
            for metric_name in ("train_loss", "val_loss", "val_dice", "val_iou"):
                self.assertTrue(torch.isfinite(torch.tensor(history[0][metric_name])))
            self.assertEqual(best_checkpoint["epoch"], 1)
            self.assertEqual(last_checkpoint["epoch"], 1)
            self.assertEqual(best_checkpoint["model_class"], "UNetModel")
            self.assertIn("optimizer_state_dict", last_checkpoint)
            self.assertIn("scaler_state_dict", last_checkpoint)
            self.assertEqual(
                best_checkpoint["best_val_dice"],
                max(entry["val_dice"] for entry in history),
            )
            self.assertEqual(
                best_checkpoint["metadata"]["model_config"]["name"],
                "unet",
            )

    def test_training_without_validation_saves_only_last(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = build_tiny_trainer(root, with_validation=False)

            history = trainer.train()

            self.assertTrue((root / "unet_last.pt").is_file())
            self.assertFalse((root / "unet_best.pt").exists())
            self.assertIsNone(history[0]["val_loss"])
            self.assertIsNone(history[0]["val_dice"])
            self.assertIsNone(history[0]["val_iou"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
