"""Latent-only decoder for binary mask reconstruction."""

from torch import nn


class _UpsamplingBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class ReconstructionDecoder(nn.Module):
    """Decode a 256-dimensional latent into raw 256x256 mask logits."""

    def __init__(self) -> None:
        super().__init__()
        self.initial_projection = nn.Linear(
            256,
            128 * 8 * 8,
            bias=True,
        )
        self.initial_activation = nn.GELU()
        self.upsampling_blocks = nn.Sequential(
            _UpsamplingBlock(128, 128),
            _UpsamplingBlock(128, 64),
            _UpsamplingBlock(64, 32),
            _UpsamplingBlock(32, 16),
            _UpsamplingBlock(16, 16),
        )
        self.output_projection = nn.Conv2d(
            16,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, latent):
        features = self.initial_activation(self.initial_projection(latent))
        features = features.reshape(latent.shape[0], 128, 8, 8)
        features = self.upsampling_blocks(features)
        return self.output_projection(features)
