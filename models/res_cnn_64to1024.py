# res_cnn_64to1024.py

"""
Residual CNN for Image Reconstruction, to reproduce 1024*1024 resolution from a low resolution

This module defines a Variant Residual Convolutional Neural Network using PyTorch Lightning.
The model is designed to generate 1024*1024 image from 64*64 resolution where the encoder
compresses the input into a latent representation, and the decoder reconstructs the image
from this representation while leveraging skip connections to preserve spatial details.

Author: Yu Yao
Date: 2025-01-30
"""

import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from einops.layers.torch import Rearrange

import torch.nn.functional as F
from basic_blocks import conv4x4, conv3x3, conv2x2, convT2x2, conv127x127_reflect


class VRES_CNN_64to1024(L.LightningModule):
    """
    Variant Residual Convolutional Neural Network (RES_CNN) for image reconstruction.

    Args:
        resolution (int): The input of low resolution to be trained while the output is fixed at 1024*1024 resolution
        Assume the default low-resolution is 64*64 resolution, a.k.a the resolution is 64
    """

    def __init__(self, input_resolution=64):
        super().__init__()
        self.resolution = input_resolution

        # Encoder
        self.encoder = nn.Sequential(
            conv4x4(1,16),
            conv3x3(16,32),
            conv4x4(32,512),
            conv3x3(512,1024),
            conv2x2(1024,2048),
            conv2x2(2048,4096),
        )
        
        # Latent representation
        self.channel_mixing = nn.Sequential(            
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(4096, 4096*16),
            Rearrange("b (c h w) -> b c h w", h=64, w=64), # c = 16
        )

        # Decoder
        self.decoder_1 = convT2x2(16,8)

        self.decoder_2 = convT2x2(8+1,4)

        self.decoder_3 = convT2x2(4+1,2)

        self.decoder_4 = convT2x2(2+1,1)

        self.decoder_5 = conv127x127_reflect(1+1,1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
                - "decoded": The reconstructed image with resolution of 1024*1024.
        """
        # Encoder
        latent_representation = self.encoder(x)

        latent_representation = self.channel_mixing(latent_representation)

        # Decoder with skip connection
        decoded = self.decoder_1(latent_representation)
        decoded = torch.cat([decoded, F.interpolate(x, scale_factor=2, mode='nearest')], dim=1)

        decoded = self.decoder_2(decoded)
        decoded = torch.cat([decoded, F.interpolate(x, scale_factor=4, mode='nearest')], dim=1)

        decoded = self.decoder_3(decoded)
        decoded = torch.cat([decoded, F.interpolate(x, scale_factor=8, mode='nearest')], dim=1)

        decoded = self.decoder_4(decoded)
        decoded = torch.cat([decoded, F.interpolate(x, scale_factor=16, mode='nearest')], dim=1)

        decoded = self.decoder_5(decoded)

        return decoded
  