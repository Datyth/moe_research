"""Basic training loop for segmentation models."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for a training-only experiment."""

    epochs: int = 1
    device: str = "cuda"
    checkpoint_path: Path = Path("checkpoints/unet_initial.pt")
    use_amp: bool = True
    log_interval: int = 20
    gradient_clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive.")


class Trainer:
    """Train a segmentation model and save the latest checkpoint."""

    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        config: TrainerConfig,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.device = torch.device(config.device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        if len(train_loader) == 0:
            raise ValueError("train_loader must contain at least one batch.")

        self.amp_enabled = config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.amp_enabled,
        )

        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self) -> list[float]:
        """Run all configured epochs and return mean training loss history."""

        history: list[float] = []

        for epoch in range(1, self.config.epochs + 1):
            mean_loss = self._train_epoch(epoch)
            history.append(mean_loss)
            self._save_checkpoint(epoch=epoch, mean_loss=mean_loss)

            print(
                f"Epoch {epoch}/{self.config.epochs} completed "
                f"- mean loss: {mean_loss:.6f}"
            )
            print(f"Checkpoint: {self.config.checkpoint_path}")

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
                    f"criterion must return a scalar loss, got "
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
        epoch: int,
        mean_loss: float,
    ) -> None:
        checkpoint_path = Path(self.config.checkpoint_path)
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
            "train_loss": mean_loss,
            "trainer_config": asdict(self.config),
        }

        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
