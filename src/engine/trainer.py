"""Training loop with validation and checkpointing for segmentation models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .evaluator import evaluate
from src.models import SegmentationOutput



HistoryEntry = dict[str, int | float | None]


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for a segmentation training experiment."""

    epochs: int = 1
    device: str = "cuda"
    last_checkpoint_path: Path = Path("runs/default/last.pt")
    best_checkpoint_path: Path = Path("runs/default/best.pt")
    history_path: Path = Path("runs/default/history.json")
    prediction_threshold: float = 0.5
    boundary_tolerance: float = 2
    use_amp: bool = True
    amp_dtype: str = "float16"
    log_interval: int = 20
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must be in [0, 1].")
        if (
            isinstance(self.boundary_tolerance, bool)
            or not isinstance(self.boundary_tolerance, (int, float))
            or not math.isfinite(float(self.boundary_tolerance))
            or self.boundary_tolerance < 0
        ):
            raise ValueError("boundary_tolerance must be a non-negative number.")
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
        scheduler: Any | None = None,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig,
        val_loader: DataLoader | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.device = torch.device(config.device)
        self.start_epoch = 1
        self.history: list[HistoryEntry] = []
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
        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )
        # bfloat16 doesn't need loss scaling; enabled=False makes GradScaler
        # a documented no-op passthrough, so no separate code path is needed.
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.amp_enabled and self.amp_dtype is torch.float16,
        )

        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self) -> list[HistoryEntry]:
        """Run all epochs and return train/validation metric history."""

        history = self.history

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            validation_metrics = None
            if self.val_loader is not None:
                validation_metrics = evaluate(
                    model=self.model,
                    loader=self.val_loader,
                    criterion=self.criterion,
                    device=self.device,
                    threshold=self.config.prediction_threshold,
                    boundary_tolerance=self.config.boundary_tolerance,
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
            self._step_scheduler(validation_metrics)


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
    def resume(self, checkpoint_path: Path) -> int:
        """Restore a versioned last checkpoint and return its completed epoch."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
            raise ValueError(
                "Resume requires a version-2 experiment checkpoint. "
                "Legacy checkpoints remain evaluation-only."
            )

        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is None and scheduler_state is not None:
            raise ValueError("Checkpoint contains a scheduler but config uses none.")
        if self.scheduler is not None and scheduler_state is None:
            raise ValueError("Checkpoint has no scheduler state required by config.")
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        completed_epoch = int(checkpoint["epoch"])
        self.start_epoch = completed_epoch + 1
        best_val_dice = checkpoint.get("best_val_dice")
        self.best_val_dice = (
            None if best_val_dice is None else float(best_val_dice)
        )

        history_path = Path(self.config.history_path)
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Resume history not found next to checkpoint: {history_path}"
            )
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if not isinstance(history, list):
            raise ValueError("Resume history must contain a JSON list.")
        if history and int(history[-1]["epoch"]) != completed_epoch:
            raise ValueError(
                "Resume history and last checkpoint disagree on completed epoch."
            )
        self.history = history
        return completed_epoch

    def _step_scheduler(self, validation_metrics: dict[str, float] | None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if validation_metrics is None:
                raise ValueError("reduce_on_plateau requires validation metrics.")
            self.scheduler.step(validation_metrics["loss"])
        else:
            self.scheduler.step()


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
                dtype=self.amp_dtype,
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
        if not isinstance(output, SegmentationOutput):
            raise TypeError(
                "Model forward must return SegmentationOutput, got "
                f"{type(output).__name__}."
            )
        return output.logits

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
            "format_version": 2,
            "epoch": epoch,
            "model_class": type(self.model).__name__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
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
