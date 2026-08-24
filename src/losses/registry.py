"""Registry and factory for segmentation losses."""

from collections.abc import Callable
from typing import Any

from torch import nn


LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str):
    """Register a loss class under a stable configuration name."""

    def decorator(loss_class):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = loss_class
        return loss_class

    return decorator


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Build a registered loss from a flat configuration mapping."""

    config = config.copy()
    try:
        loss_name = config.pop("name")
    except KeyError as error:
        raise ValueError("Loss configuration requires a 'name'.") from error

    try:
        loss_class = LOSS_REGISTRY[loss_name]
    except KeyError as error:
        available = ", ".join(sorted(LOSS_REGISTRY))
        raise ValueError(
            f"Unknown loss '{loss_name}'. Available: {available}"
        ) from error

    return loss_class(**config)
