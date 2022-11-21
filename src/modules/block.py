import torch.nn as nn
from ..utils import exists

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_gruops=groups, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.projection(x)
        x = self.group_norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x
