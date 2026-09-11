"""LR schedulers beyond torch's built-ins.

`WarmupPolyLR` reproduces the MoE-SAM (E-SAM) released training recipe: a
linear warmup over a fixed number of *iterations* (250 in the official
`trainer.py`), followed by polynomial decay with exponent 0.9:

    lr(step) = base_lr * step / warmup_steps                     (step <= warmup)
    lr(step) = base_lr * (1 - p) ** power                        (step  > warmup)

where `p = (step - warmup_steps) / (total_steps - warmup_steps)`.

This is a per-iteration scheduler: the `Trainer` steps it once per optimizer
step (not once per epoch like `CosineAnnealingLR`). `Trainer` detects it via
`PER_ITERATION_SCHEDULERS`.
"""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer

try:  # torch >= 2.0 exposes LRScheduler; fall back for older versions.
    from torch.optim.lr_scheduler import LRScheduler as _Base
except ImportError:  # pragma: no cover - legacy torch only
    from torch.optim.lr_scheduler import _LRScheduler as _Base


class WarmupPolyLR(_Base):
    """Linear warmup (iterations) followed by polynomial decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 250,
        power: float = 0.9,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if warmup_steps >= total_steps:
            raise ValueError(
                "warmup_steps must be smaller than total_steps, got "
                f"warmup_steps={warmup_steps}, total_steps={total_steps}."
            )
        if not math.isfinite(power) or not 0.0 < power <= 1.0:
            raise ValueError("power must be a finite float in (0, 1].")
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.power = float(power)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        step = max(self.last_epoch, 0)
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
        else:
            progress = (step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            scale = max(0.0, 1.0 - progress) ** self.power
        return [base_lr * scale for base_lr in self.base_lrs]


def is_per_iteration_scheduler(scheduler: Any) -> bool:
    """Whether the scheduler must be stepped once per optimizer step."""
    return isinstance(scheduler, WarmupPolyLR)


PER_ITERATION_SCHEDULERS = (WarmupPolyLR,)

__all__ = ["WarmupPolyLR", "is_per_iteration_scheduler", "PER_ITERATION_SCHEDULERS"]
