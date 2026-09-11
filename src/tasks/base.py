"""Minimal contract between learning tasks and the generic engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor, nn


@dataclass
class TaskStepOutput:
    """One batch summarized as scalar batch means."""

    loss: Tensor
    metrics: dict[str, Tensor | float]
    batch_size: int


class Task(Protocol):
    """Structural protocol implemented by trainable learning tasks."""

    criterion: nn.Module

    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        ...

    def evaluation_step(
        self,
        model: nn.Module,
        batch: Any,
        device: Any,
    ) -> TaskStepOutput:
        ...
