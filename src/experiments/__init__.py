"""Research experiment entry points."""

from .shape_pretraining import build_shape_autoencoder, execute_shape_pretraining

__all__ = [
    "build_shape_autoencoder",
    "execute_shape_pretraining",
]
