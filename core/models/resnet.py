import torch
import torch.nn as nn


# TODO: make more generic
class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        n_blocks: int,
        n_linear: int,
        base_channels: int = 64,
        init_conv_kernel_size: int = 7,
        init_conv_stride: int = 2,
        maxpool_kernel_size: int = 3,
        maxpool_stride: int = 2,
        maxpool_padding: int = 1,
        resblock_kernel_size: int = 3,
    ):
        super(ResNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=init_conv_kernel_size,
                stride=init_conv_stride,
                padding=init_conv_kernel_size // 2,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=maxpool_kernel_size,
                stride=maxpool_stride,
                padding=maxpool_padding,
            ),
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock(base_channels, kernel_size=resblock_kernel_size)
                for _ in range(n_blocks)
            ]
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(base_channels, n_linear),
            nn.ReLU(inplace=True),
            nn.Linear(n_linear, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x
