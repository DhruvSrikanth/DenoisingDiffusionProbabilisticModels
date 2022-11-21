import torch.nn as nn
from ..utils import exists
from einops import rearrange

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, channel_scale_factor=2, normalize=True):
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.GELU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=in_channels)
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.ds_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, groups=in_channels))

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * channel_scale_factor, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=1, num_channels=out_channels * channel_scale_factor), 
            nn.Conv2d(in_channels=out_channels * channel_scale_factor, out_channels=out_channels, kernel_size=3, padding=1),
        )

        self.residual_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            condition = self.time_projection(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        

        h = self.net(h)
        return h + self.residual_connection(x)