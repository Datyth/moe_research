"""Task-agnostic training, validation, and checkpointing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.tasks import Task

from .evaluator import evaluate, validate_step_output


HistoryEntry = dict[str, int | float]


@dataclass(frozen=True)
class TrainerConfig:
    """Engine-level configuration for a training experiment."""

    epochs: int = 1
    device: str = "cuda"
    last_checkpoint_path: Path = Path("runs/default/last.pt")
    best_checkpoint_path: Path = Path("runs/default/best.pt")
    history_path: Path = Path("runs/default/history.json")
    use_amp: bool = True
    amp_dtype: str = "float16"
    log_interval: int = 20
    gradient_clip_norm: float | None = None
    monitor: str = "dice"
    monitor_mode: str = "max"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if Path(self.last_checkpoint_path) == Path(self.best_checkpoint_path):
            raise ValueError("last and best checkpoint paths must be different.")
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("monitor must be a non-empty string.")
        if (
            not isinstance(self.monitor_mode, str)
            or self.monitor_mode not in {"min", "max"}
        ):
            raise ValueError("monitor_mode must be 'min' or 'max'.")


class Trainer:
    """Optimize any model through a task-defined batch contract."""

    def __init__(
        self,
        *,
        model: nn.Module,
        task: Task,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig,
        scheduler: Any | None = None,
        val_loader: DataLoader | None = None,
        task_config: dict[str, Any] | None = None,
        checkpoint_metadata: dict[str, Any] | None = None,
    ) -> None:
        if task_config is not None and not isinstance(task_config, dict):
            raise TypeError("task_config must be a dictionary or None.")
        self.model = model
        self.task = task
        self.task_config = dict(task_config or {})
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.device = torch.device(config.device)
        self.start_epoch = 1
        self.history: list[HistoryEntry] = []
        self.best_monitor_value: float | None = None

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
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.amp_enabled and self.amp_dtype is torch.float16,
        )
        self.model.to(self.device)
        self.task.criterion.to(self.device)

    def train(self) -> list[HistoryEntry]:
        """Run configured epochs and return dynamic metric history."""

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            validation_metrics: dict[str, float] | None = None
            if self.val_loader is not None:
                validation_metrics = evaluate(
                    model=self.model,
                    loader=self.val_loader,
                    task=self.task,
                    device=self.device,
                )
                if self.config.monitor not in validation_metrics:
                    raise ValueError(
                        f"Monitored metric {self.config.monitor!r} was not returned "
                        "by validation."
                    )

            entry: HistoryEntry = {"epoch": epoch, "train_loss": train_loss}
            if validation_metrics is not None:
                entry.update(
                    {
                        f"val_{name}": value
                        for name, value in validation_metrics.items()
                    }
                )
            self.history.append(entry)
            self._step_scheduler(validation_metrics)

            if validation_metrics is not None:
                monitored_value = validation_metrics[self.config.monitor]
                if self._is_improved(monitored_value):
                    self.best_monitor_value = monitored_value
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
            self._save_history()
            self._print_epoch_summary(entry)

        return self.history

    def resume(self, checkpoint_path: Path) -> int:
        """Restore a compatible version-2 checkpoint and history."""

        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
            raise ValueError("Resume requires a version-2 experiment checkpoint.")

        monitor_fields = (
            "monitor_name",
            "monitor_mode",
            "best_monitor_value",
        )
        present_monitor_fields = [name in checkpoint for name in monitor_fields]
        if any(present_monitor_fields) and not all(present_monitor_fields):
            raise ValueError("Checkpoint has incomplete generic monitor metadata.")
        is_generic_checkpoint = all(present_monitor_fields)
        if is_generic_checkpoint:
            saved_monitor = checkpoint["monitor_name"]
            saved_mode = checkpoint["monitor_mode"]
            saved_best = checkpoint["best_monitor_value"]
        else:
            saved_monitor = "dice"
            saved_mode = "max"
            saved_best = checkpoint.get("best_val_dice")
        if (
            saved_monitor != self.config.monitor
            or saved_mode != self.config.monitor_mode
        ):
            raise ValueError(
                "Checkpoint monitor configuration "
                f"{saved_monitor!r}/{saved_mode!r} does not match current "
                f"{self.config.monitor!r}/{self.config.monitor_mode!r}."
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
        self.best_monitor_value = (
            None if saved_best is None else float(saved_best)
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
        if not history or int(history[-1]["epoch"]) != completed_epoch:
            raise ValueError(
                "Resume history and last checkpoint disagree on completed epoch."
            )
        if is_generic_checkpoint:
            saved_metrics = checkpoint.get("metrics")
            if not isinstance(saved_metrics, dict):
                raise ValueError("Generic checkpoint metrics must be a dictionary.")
            if history[-1] != saved_metrics:
                raise ValueError(
                    "Resume history and checkpoint metrics disagree on the saved epoch."
                )
        self.history = history
        return completed_epoch

    def _is_improved(self, value: float) -> bool:
        if self.best_monitor_value is None:
            return True
        if self.config.monitor_mode == "max":
            return value > self.best_monitor_value
        return value < self.best_monitor_value

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

        for batch_number, batch in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                step = self.task.training_step(self.model, batch, self.device)
            validate_step_output(step)
            loss = step.loss

            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.detach().item() * step.batch_size
            total_samples += step.batch_size
            if (
                batch_number == 1
                or batch_number % self.config.log_interval == 0
                or batch_number == number_of_batches
            ):
                print(
                    f"Epoch {epoch}/{self.config.epochs} "
                    f"- batch {batch_number}/{number_of_batches} "
                    f"- loss: {total_loss / total_samples:.6f}"
                )

        if total_samples == 0:
            raise ValueError("train_loader produced zero samples.")
        return total_loss / total_samples

    def _save_checkpoint(
        self,
        *,
        path: Path,
        epoch: int,
        metrics: HistoryEntry,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        checkpoint = {
            "format_version": 2,
            "epoch": epoch,
            "model_class": type(self.model).__name__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": dict(metrics),
            "monitor_name": self.config.monitor,
            "monitor_mode": self.config.monitor_mode,
            "best_monitor_value": self.best_monitor_value,
            "trainer_config": asdict(self.config),
            "task_config": dict(self.task_config),
            "metadata": self.checkpoint_metadata,
        }
        if self.config.monitor == "dice" and self.config.monitor_mode == "max":
            checkpoint["best_val_dice"] = self.best_monitor_value
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)

    def _save_history(self) -> None:
        history_path = Path(self.config.history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = history_path.with_suffix(history_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
            file.write("\n")
        temporary_path.replace(history_path)

    def _print_epoch_summary(self, metrics: HistoryEntry) -> None:
        print(f"Epoch {metrics['epoch']}/{self.config.epochs} completed")
        print(f"Train Loss : {metrics['train_loss']:.6f}")
        for name, value in metrics.items():
            if name.startswith("val_"):
                label = name[4:].replace("_", " ").title()
                print(f"Val {label} : {value:.6f}")
        print(f"Last checkpoint: {self.config.last_checkpoint_path}")
        if self.best_monitor_value is not None:
            print(f"Best checkpoint: {self.config.best_checkpoint_path}")
