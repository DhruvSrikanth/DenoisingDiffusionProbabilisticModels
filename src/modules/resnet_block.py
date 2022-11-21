import torch.nn as nn
from ..utils import exists
from .block import Block
from einops import rearrange

class ResNetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, groups=8):
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.SiLU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=out_channels) 
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.block1 = Block(in_channels=in_channels, out_channels=out_channels, groups=groups)
        self.block2 = Block(in_channels=out_channels, out_channels=out_channels, groups=groups)
        self.residual_connection = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.residual_connection(x)