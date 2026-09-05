"""Shape representation models."""

from .autoencoder import ShapeAutoencoder, ShapeAutoencoderOutput
from .decoder import ReconstructionDecoder
from .projector import SpatialProjector
from .small_cnn import SmallCNN

__all__ = [
    "SmallCNN",
    "SpatialProjector",
    "ReconstructionDecoder",
    "ShapeAutoencoder",
    "ShapeAutoencoderOutput",
]
