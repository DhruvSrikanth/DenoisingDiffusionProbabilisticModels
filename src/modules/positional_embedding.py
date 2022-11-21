import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.partial_embeddings = math.log(10000) / (self.half_dim - 1)
        
    
    def forward(self, time):
        device = time.device 
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -self.partial_embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
