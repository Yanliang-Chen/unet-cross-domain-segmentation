"""Standard U-Net for binary semantic segmentation."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Four-level encoder/decoder U-Net."""

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        depth = 4
        widths = [base_channels * (2 ** level) for level in range(depth + 1)]
        self.encoders = nn.ModuleList(
            DoubleConv(in_channels if i == 0 else widths[i - 1], widths[i])
            for i in range(depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(widths[-2], widths[-1])
        self.decoders = nn.ModuleList(
            DoubleConv(widths[i + 1] + widths[i], widths[i])
            for i in reversed(range(depth))
        )
        self.out = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))
        return self.out(x)


def build_model(config):
    architecture = config.get("architecture", "unet")
    if architecture != "unet":
        raise ValueError(f"Unknown architecture: {architecture}")
    return UNet(
        in_channels=int(config["input_channels"]),
        base_channels=int(config["base_channels"]),
    )
