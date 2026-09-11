"""Spatial bottleneck projector for shape features."""

from torch import nn


class SpatialProjector(nn.Module):
    """Project 256x16x16 features to a 256-dimensional latent."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_projection = nn.Conv2d(
            256,
            64,
            kernel_size=1,
            bias=True,
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.latent_projection = nn.Linear(
            64 * 4 * 4,
            256,
            bias=True,
        )

    def forward(self, features):
        features = self.channel_projection(features)
        features = self.spatial_pool(features)
        features = self.flatten(features)
        return self.latent_projection(features)
