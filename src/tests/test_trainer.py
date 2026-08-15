"""Smoke test for the training-only segmentation Trainer."""

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


class TestTrainer(unittest.TestCase):
    def test_train_one_epoch_and_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = (
                Path(temporary_directory) / "unet_test.pt"
            )
            model = build_model({
                "name": "unet",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "base_channels": 2,
            })
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
            )
            trainer = Trainer(
                model=model,
                criterion=BCEDiceLoss(),
                optimizer=optimizer,
                train_loader=DataLoader(
                    TinySegmentationDataset(),
                    batch_size=2,
                ),
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    checkpoint_path=checkpoint_path,
                    use_amp=False,
                    log_interval=1,
                ),
            )

            history = trainer.train()
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            self.assertEqual(len(history), 1)
            self.assertTrue(torch.isfinite(torch.tensor(history[0])))
            self.assertTrue(checkpoint_path.is_file())
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(
                checkpoint["model_class"],
                "UNetModel",
            )
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
