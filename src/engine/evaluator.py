"""Task-agnostic model evaluation."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.tasks import Task, TaskStepOutput


def _scalar_value(value: Tensor | float, *, name: str) -> float:
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}.")
        result = float(value.detach().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        raise TypeError(f"{name} must be a scalar tensor or number.")
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} must be finite.")
    return result


def validate_step_output(step: TaskStepOutput) -> None:
    """Validate the shared per-batch task contract."""

    if not isinstance(step, TaskStepOutput):
        raise TypeError(
            "Task steps must return TaskStepOutput, got "
            f"{type(step).__name__}."
        )
    if not isinstance(step.loss, Tensor):
        raise TypeError("TaskStepOutput.loss must be a tensor.")
    _scalar_value(step.loss, name="TaskStepOutput.loss")
    if not isinstance(step.metrics, dict):
        raise TypeError("TaskStepOutput.metrics must be a dictionary.")
    if "loss" in step.metrics:
        raise ValueError("TaskStepOutput.metrics must not contain 'loss'.")
    for name, value in step.metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("TaskStepOutput metric names must be non-empty strings.")
        _scalar_value(value, name=f"TaskStepOutput.metrics[{name!r}]")
    if (
        isinstance(step.batch_size, bool)
        or not isinstance(step.batch_size, int)
        or step.batch_size <= 0
    ):
        raise ValueError("TaskStepOutput.batch_size must be a positive integer.")


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    task: Task,
    device: str | torch.device,
) -> dict[str, float]:
    """Return sample-weighted mean loss and task-defined metrics."""

    if len(loader) == 0:
        raise ValueError("loader must contain at least one batch.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    was_training = model.training
    model.to(resolved_device)
    task.criterion.to(resolved_device)
    total_loss = 0.0
    total_metrics: dict[str, float] = {}
    expected_metric_keys: tuple[str, ...] | None = None
    total_samples = 0

    try:
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                step = task.evaluation_step(model, batch, resolved_device)
                validate_step_output(step)
                metric_keys = tuple(step.metrics)
                if expected_metric_keys is None:
                    expected_metric_keys = metric_keys
                    total_metrics = {name: 0.0 for name in metric_keys}
                elif set(metric_keys) != set(expected_metric_keys):
                    raise ValueError(
                        "Task evaluation metric keys changed across batches: "
                        f"expected {sorted(expected_metric_keys)}, got "
                        f"{sorted(metric_keys)}."
                    )

                total_loss += (
                    _scalar_value(step.loss, name="TaskStepOutput.loss")
                    * step.batch_size
                )
                for name, value in step.metrics.items():
                    total_metrics[name] += (
                        _scalar_value(
                            value,
                            name=f"TaskStepOutput.metrics[{name!r}]",
                        )
                        * step.batch_size
                    )
                total_samples += step.batch_size
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("loader produced zero samples.")
    return {
        "loss": total_loss / total_samples,
        **{
            name: total_metrics[name] / total_samples
            for name in (expected_metric_keys or ())
        },
    }
