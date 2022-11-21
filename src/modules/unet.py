import torch
import torch.nn as nn
from ..utils import default, exists
from .upsample import Upsample
from .downsample import Downsample
from .positional_embedding import SinusoidalPositionEmbeddings
from .residual import Residual
from .convnext_block import ConvNextBlock
from .resnet_block import ResNetBlock
from .attention import Attention
from .linear_attention import LinearAttention

from .prenorm import PreNorm

from functools import partial


class DDPM(nn.Module):
    def __init__(self, n_features, init_channels=None, out_channels=None, channel_scale_factors=(1, 2, 4, 8), in_channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_scale_factor=2):
        super().__init__()

        # determine dimensions
        self.in_channels = in_channels

        init_channels = default(init_channels, n_features // 3 * 2)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=7, padding=3)

        dims = [init_channels, *map(lambda m: n_features * m, channel_scale_factors)]
        resolution_translations = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, channel_scale_factor=convnext_scale_factor)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = n_features * 4
            self.time_projection = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=n_features),
                nn.Linear(in_features=n_features, out_features=time_dim),
                nn.GELU(),
                nn.Linear(in_features=time_dim, out_features=time_dim),
            )
        else:
            time_dim = None
            self.time_projection = None

        # layers
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        num_resolutions = len(resolution_translations)

        for idx, (in_chan, out_chan) in enumerate(resolution_translations):
            is_last = idx >= (num_resolutions - 1)
            self.encoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=in_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=out_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=out_chan, fn=LinearAttention(num_channels=out_chan))),
                        Downsample(num_channels=out_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        bottleneck_capacity = dims[-1]
        self.bottleneck = [
            nn.ModuleList(
                [
                    block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim), 
                    Residual(fn=PreNorm(bottleneck_capacity, Attention(bottleneck_capacity))), 
                    block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim), 
                ]
            )
        ]

        

        for idx, (in_chan, out_chan) in enumerate(reversed(resolution_translations[1:])):
            is_last = idx >= (num_resolutions - 1)

            self.decoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=out_chan * 2, out_channels=in_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=in_chan, out_channels=in_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=in_chan, fn=LinearAttention(num_channels=in_chan))),
                        Upsample(num_channels=in_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_chan = default(out_channels, in_channels)
        self.final_conv = nn.Sequential(
            block_klass(in_channels=n_features, out_channels=n_features), 
            nn.Conv2d(in_channels=n_features, out_channels=out_chan, kernel_size=1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_projection(time) if exists(self.time_projection) else None

        noisy_latent_representation_stack = []

        # downsample
        for block1, block2, attn, downsample in self.encoder:
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            noisy_latent_representation_stack.append(x)
            x = downsample(x)
        
        # bottleneck
        for block1, attn, block2 in self.bottleneck:
            x = block1(x, time_emb=t)
            x = attn(x)
            x = block2(x, time_emb=t)


        # upsample
        for block1, block2, attn, upsample in self.decoder:
            x = torch.cat((x, noisy_latent_representation_stack.pop()), dim=1)
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)