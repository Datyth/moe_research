"""Small convolutional encoder for binary lesion masks."""

from torch import nn


class SmallCNN(nn.Module):
    """Encode a 256x256 mask into a 256x16x16 feature map."""

    def __init__(self) -> None:
        super().__init__()
        channels = (1, 32, 64, 128, 256)
        stages = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                )
            )
        self.stages = nn.Sequential(*stages)

    def forward(self, masks):
        return self.stages(masks)
