"""Tests for WarmupPolyLR (paper-parity scheduler) and its Trainer wiring."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.engine import Trainer, TrainerConfig, WarmupPolyLR
from src.losses import BCEDiceLoss
from src.models import build_model


class TinySegmentationDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        return {
            "image": torch.rand(3, 32, 32, generator=generator),
            "mask": torch.randint(0, 2, (1, 32, 32), generator=generator).float(),
        }


class TestWarmupPolyLR(unittest.TestCase):
    def _scheduler(self, *, total_steps=100, warmup_steps=10, power=0.9, lr=1.0):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=lr)
        return WarmupPolyLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            power=power,
        )

    def _lr_at(self, step, **kwargs):
        scheduler = self._scheduler(**kwargs)
        for _ in range(step):
            scheduler.step()
        return scheduler.get_last_lr()[0]

    def test_linear_warmup_ramp(self):
        self.assertAlmostEqual(self._lr_at(5, warmup_steps=10), 0.5, places=6)
        self.assertAlmostEqual(self._lr_at(10, warmup_steps=10), 1.0, places=6)

    def test_poly_decay_after_warmup(self):
        # Step 55: progress = (55-10)/90 = 0.5 -> lr = 0.5**0.9.
        expected = 1.0 * (1.0 - 0.5) ** 0.9
        self.assertAlmostEqual(
            self._lr_at(55, total_steps=100, warmup_steps=10, power=0.9),
            expected,
            places=6,
        )
        # At total_steps the LR reaches zero.
        self.assertAlmostEqual(
            self._lr_at(100, total_steps=100, warmup_steps=10, power=0.9),
            0.0,
            places=6,
        )

    def test_decay_is_monotone_after_warmup(self):
        scheduler = self._scheduler()
        previous = None
        for _ in range(100):
            scheduler.step()
            if scheduler.last_epoch <= scheduler.warmup_steps:
                continue  # warmup ramps UP; decay starts afterwards
            current = scheduler.get_last_lr()[0]
            if previous is not None:
                self.assertLessEqual(current, previous + 1e-12)
            previous = current

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            self._scheduler(total_steps=10, warmup_steps=10)
        with self.assertRaises(ValueError):
            self._scheduler(total_steps=0)
        with self.assertRaises(ValueError):
            self._scheduler(power=1.5)

    def test_state_dict_round_trip(self):
        scheduler = self._scheduler()
        for _ in range(7):
            scheduler.step()
        state = scheduler.state_dict()
        rebuilt = self._scheduler()
        rebuilt.load_state_dict(state)
        self.assertEqual(rebuilt.last_epoch, 7)
        self.assertAlmostEqual(
            rebuilt.get_last_lr()[0], scheduler.get_last_lr()[0], places=9
        )


class TestTrainerPerIterationStepping(unittest.TestCase):
    def test_warmup_poly_steps_once_per_optimizer_step(self):
        model = build_model(
            {
                "name": "unet",
                "in_channels": 3,
                "num_classes": 1,
                "task": "binary",
                "base_channels": 2,
            }
        )
        loader = DataLoader(TinySegmentationDataset(), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupPolyLR(
            optimizer, total_steps=4, warmup_steps=2, power=0.9
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trainer = Trainer(
                model=model,
                criterion=BCEDiceLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=loader,
                val_loader=None,
                config=TrainerConfig(
                    epochs=1,
                    device="cpu",
                    last_checkpoint_path=root / "last.pt",
                    best_checkpoint_path=root / "best.pt",
                    history_path=root / "history.json",
                    use_amp=False,
                    log_interval=1,
                ),
            )
            trainer.train()
            # 4 samples / batch size 2 -> 2 optimizer steps in one epoch
            # (plus the construction step, so last_epoch == 2).
            self.assertEqual(scheduler.last_epoch, 2)
            checkpoint = torch.load(root / "last.pt", weights_only=False)
            self.assertEqual(
                checkpoint["scheduler_state_dict"]["last_epoch"], 2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
