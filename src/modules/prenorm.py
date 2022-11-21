import torch.nn as nn

class PreNorm(nn.Module):
    def __init__(self, num_channels, fn):
        super().__init__()
        self.fn = fn
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)

    def forward(self, x):
        x = self.group_norm(x)
        return self.fn(x)