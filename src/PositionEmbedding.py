import torch
import math
import torch.nn as nn


# got from colab
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # Calculate frequencies for sine and cosine functions
        embeddings = math.log(10000) / (half_dim - 1)
        # Calculate exponential of the negative frequencies
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Multiply the timestep by the negative frequencies
        embeddings = time[:, None] * embeddings[None, :]
        # Calculate the sine and cosine for each timestep in the batch
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
