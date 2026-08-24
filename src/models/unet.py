import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseSegmentationModel, SegmentationOutput
from .registry import register_model

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        
        # ENCODER
        self.inc = DoubleConv(in_channels, base_channels)             # 3 => 32
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2) # 32 => 64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4) # 64 => 128
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 4, base_channels * 8) # 128 => 256
        )
        
        # BOTTLENECK 
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(base_channels * 8, base_channels * 16) # 256 => 512
        )

        # DECODER 
        # Use ConvTranspose2d to upsample
        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_channels * 16, base_channels * 8) # 512 + 256 (skip) => 256
        
        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_channels * 8, base_channels * 4)  # 256 + 128 (skip) => 128
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_channels * 4, base_channels * 2)  # 128 + 64 (skip) => 64
        
        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_channels * 2, base_channels)      # 64 + 32 (skip) => 32
        
        # Output heads
        # Conv 1x1 để chuyển số kênh về đúng num_classes
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Encoder 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 2. Bottleneck
        x5 = self.down4(x4)
        
        # 3. Decoder + Skip connections
        # Block 1
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1) # channels dim
        x = self.conv1(x)
        
        # Block 2
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        # Block 3
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        # Block 4
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        # 4. Return logits
        logits = self.out_conv(x)
        return logits

@register_model("unet")
class UNetModel(BaseSegmentationModel):
    def __init__(self, *, in_channels: int = 3, num_classes: int = 1, task: str = "binary", base_channels: int = 32):
        super().__init__(in_channels = in_channels, num_classes = num_classes, task = task)
        self.network = UNet(in_channels = in_channels, out_channels = num_classes, base_channels = base_channels)

    def forward(self, images: Tensor, **kwargs) -> SegmentationOutput:
        logits = self.network(images)
        return SegmentationOutput(logits = logits)