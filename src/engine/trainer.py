"""Training loop with validation and checkpointing for segmentation models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .evaluator import evaluate


HistoryEntry = dict[str, int | float | None]


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for a segmentation training experiment."""

    epochs: int = 1
    device: str = "cuda"
    last_checkpoint_path: Path = Path("checkpoints/unet_last.pt")
    best_checkpoint_path: Path = Path("checkpoints/unet_best.pt")
    history_path: Path = Path("results/unet_training_history.json")
    prediction_threshold: float = 0.5
    use_amp: bool = True
    log_interval: int = 20
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must be in [0, 1].")
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive.")
        if Path(self.last_checkpoint_path) == Path(self.best_checkpoint_path):
            raise ValueError("last and best checkpoint paths must be different.")


class Trainer:
    """Train a segmentation model, validate it and save best/last state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig,
        val_loader: DataLoader | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.device = torch.device(config.device)
        self.best_val_dice: float | None = None

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        if len(train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch.")
        if val_loader is not None and len(val_loader) == 0:
            raise ValueError("val_loader must contain at least one batch.")

        self.amp_enabled = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.amp_enabled,
        )

        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self) -> list[HistoryEntry]:
        """Run all epochs and return train/validation metric history."""

        history: list[HistoryEntry] = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            validation_metrics = None
            if self.val_loader is not None:
                validation_metrics = evaluate(
                    model=self.model,
                    loader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                    threshold=self.config.prediction_threshold,
                )

            entry: HistoryEntry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": (
                    validation_metrics["loss"]
                    if validation_metrics is not None
                    else None
                ),
                "val_dice": (
                    validation_metrics["dice"]
                    if validation_metrics is not None
                    else None
                ),
                "val_iou": (
                    validation_metrics["iou"]
                    if validation_metrics is not None
                    else None
                ),
            }
            history.append(entry)

            val_dice = entry["val_dice"]
            if isinstance(val_dice, float) and (
                self.best_val_dice is None or val_dice > self.best_val_dice
            ):
                self.best_val_dice = val_dice
                self._save_checkpoint(
                    path=self.config.best_checkpoint_path,
                    epoch=epoch,
                    metrics=entry,
                )

            self._save_checkpoint(
                path=self.config.last_checkpoint_path,
                epoch=epoch,
                metrics=entry,
            )
            self._save_history(history)
            self._print_epoch_summary(entry)

        return history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        number_of_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            images, targets = self._prepare_batch(batch)
            batch_size = images.shape[0]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                output = self.model(images)
                logits = self._extract_logits(output)
                loss = self.criterion(logits, targets)

            if loss.ndim != 0:
                raise ValueError(
                    "criterion must return a scalar loss, got "
                    f"shape {tuple(loss.shape)}."
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected at epoch {epoch}, step {step}."
                )

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size

            if (
                step == 1
                or step % self.config.log_interval == 0
                or step == number_of_batches
            ):
                running_loss = total_loss / total_samples
                print(
                    f"Epoch {epoch}/{self.config.epochs} "
                    f"- batch {step}/{number_of_batches} "
                    f"- loss: {running_loss:.6f}"
                )

        return total_loss / total_samples

    def _prepare_batch(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        if "image" not in batch or "mask" not in batch:
            raise KeyError(
                "Each training batch must contain 'image' and 'mask'."
            )

        images = batch["image"].to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        targets = batch["mask"].to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        return images, targets

    @staticmethod
    def _extract_logits(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "logits"):
            return output.logits
        raise TypeError(
            "Model output must be a Tensor or expose a 'logits' attribute."
        )

    def _save_checkpoint(
        self,
        *,
        path: Path,
        epoch: int,
        metrics: HistoryEntry,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".tmp"
        )

        checkpoint = {
            "epoch": epoch,
            "model_class": type(self.model).__name__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": metrics["train_loss"],
            "val_loss": metrics["val_loss"],
            "val_dice": metrics["val_dice"],
            "val_iou": metrics["val_iou"],
            "best_val_dice": self.best_val_dice,
            "trainer_config": asdict(self.config),
            "metadata": self.checkpoint_metadata,
        }
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

    def _save_history(self, history: list[HistoryEntry]) -> None:
        history_path = Path(self.config.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = history_path.with_suffix(history_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
            file.write("\n")
        temporary_path.replace(history_path)

    def _print_epoch_summary(self, metrics: HistoryEntry) -> None:
        print(f"Epoch {metrics['epoch']}/{self.config.epochs} completed")
        print(f"Train Loss : {metrics['train_loss']:.6f}")
        if metrics["val_loss"] is not None:
            print(f"Val Loss   : {metrics['val_loss']:.6f}")
            print(f"Val Dice   : {metrics['val_dice']:.6f}")
            print(f"Val IoU    : {metrics['val_iou']:.6f}")
        print(f"Last checkpoint: {self.config.last_checkpoint_path}")
        if self.best_val_dice is not None:
            print(f"Best checkpoint: {self.config.best_checkpoint_path}")
