import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import diff_tools as dm
import pdb
import random


# Conv block for  in depth feature extraction with a skip connection
class DoubConvBlock(nn.Module):
    # ADAPTED FROM https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, inp_channels, outp_channels, residual=None):
        super().__init__()

        self.residual = residual
        self.outp_channels = outp_channels
        self.endNorm = nn.GroupNorm(1, outp_channels)

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=inp_channels, out_channels=outp_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, outp_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=outp_channels, out_channels=outp_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, outp_channels),
        )

    def forward(self, x):
        if self.residual:
            # add element wise skip connection to help gradient flow
            feature_map = F.gelu(x + self.convblock(x))
            return feature_map
        else:
            feature_map = self.convblock(x)
            return feature_map


class ImageEncoder(nn.Module):
    def __init__(self, c_in=3, embedding_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=2, padding=1),  # [32,32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [16,16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [8,8]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [4,4]
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, embedding_dim)  # Adjust based on your image size
        )

    def forward(self, x):
        x1 = self.encoder(x)
        return x1
