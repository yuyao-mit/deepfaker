# res_cnn.py

"""
Residual Convolutional Neural Network (RES_CNN) for Image Reconstruction

This module defines a Residual Convolutional Neural Network (RES_CNN) using PyTorch Lightning.
The model is designed for tasks like image reconstruction or autoencoding, where the encoder
compresses the input into a latent representation, and the decoder reconstructs the image
from this representation while leveraging skip connections to preserve spatial details.

Key Features:
- Encoder: A series of convolutional layers that downsample the input image into a latent representation.
- Decoder: A series of transposed convolutional layers that upsample the latent representation back to the
  original image size, with skip connections to preserve spatial details.
- Skip Connections: Downsampled versions of the input image are concatenated with the decoder’s feature maps
  at different stages to improve reconstruction quality.
- Dynamic Resolution: The model architecture dynamically adjusts based on the `resolution` parameter.

Classes:
1. RES_CNN: The main model class for image reconstruction.

Usage:
1. Initialize the model:
   model = RES_CNN(resolution=256)
2. Train the model using PyTorch Lightning:
   trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
   trainer.fit(model, datamodule=data)

Author: Yu Yao
Date: 2025-01-22
"""

# Standard library imports
import os

# Third-party imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Tools
from einops.layers.torch import Rearrange
import lightning as L


def conv16x16(in_channel: int, out_channel: int, kernel_size: int = 16, stride: int = 16, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    """
    Create a 16x16 convolutional layer.
    """
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


def conv8x8(in_channel: int, out_channel: int, kernel_size: int = 8, stride: int = 8, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


def conv4x4(in_channel: int, out_channel: int, kernel_size: int = 4, stride: int = 4, groups: int = 1, padding: int = 0) -> nn.Conv2d:
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


def conv3x3(in_channel: int, out_channel: int, kernel_size: int = 3, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


def conv2x2(in_channel: int, out_channel: int, kernel_size: int = 2, stride: int = 2, groups: int = 1, padding: int = 0) -> nn.Conv2d:
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1,
    )


class RES_CNN(L.LightningModule):
    """
    Residual Convolutional Neural Network (RES_CNN) for image reconstruction.

    Args:
        resolution (int): The target resolution for the input and output images.
                          This parameter dynamically adjusts the model's architecture.
    """

    def __init__(self, resolution):
        super(RES_CNN, self).__init__()
        self.resolution = resolution

        # Encoder
        self.encoder = nn.Sequential(
            conv16x16(1, self.resolution),
            nn.BatchNorm2d(self.resolution),
            nn.SiLU(inplace=True),
            conv3x3(self.resolution, self.resolution),
            nn.BatchNorm2d(self.resolution),
            nn.SiLU(inplace=True),
            conv2x2(self.resolution, self.resolution * 2),
            nn.BatchNorm2d(self.resolution * 2),
            nn.SiLU(inplace=True),
            conv2x2(self.resolution * 2, self.resolution * 4),
            nn.BatchNorm2d(self.resolution * 4),
            nn.SiLU(inplace=True),
            conv4x4(self.resolution * 4, self.resolution * 16),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(self.resolution * 16, self.resolution * 16),  # 4096*4096
        )

        # Decoder
        self.decoder_1 = nn.Sequential(
            Rearrange("b (c h w) -> b c h w", h=16, w=16),  # c = 16
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(16 + 1, 8, kernel_size=2, stride=2, padding=0),  # h=32
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(8 + 1, 4, kernel_size=2, stride=2, padding=0),  # h=64
            nn.BatchNorm2d(4),
            nn.SiLU(inplace=True),
        )

        self.decoder_4 = nn.Sequential(
            nn.ConvTranspose2d(4 + 1, 2, kernel_size=2, stride=2, padding=0),  # h=128
            nn.BatchNorm2d(2),
            nn.SiLU(inplace=True),
        )

        self.decoder_5 = nn.Sequential(
            nn.ConvTranspose2d(2 + 1, 1, kernel_size=2, stride=2, padding=0),  # h=256
            nn.Tanh(),
        )

        # Skip connections
        self.skip = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128
            nn.MaxPool2d(kernel_size=4, stride=4),  # 64
            nn.MaxPool2d(kernel_size=8, stride=8),  # 32
            nn.MaxPool2d(kernel_size=16, stride=16),  # 16
        ])

    def forward(self, x):
        """
        Forward pass for the RES_CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).

        Returns:
            dict: A dictionary containing:
                - "encoded": The latent representation of the input.
                - "decoded": The reconstructed image.
        """
        # Encoder
        latent_representation = self.encoder(x)

        # Skip connections
        skip0 = self.skip[0](x)
        skip1 = self.skip[1](x)
        skip2 = self.skip[2](x)
        skip3 = self.skip[3](x)

        # Decoder with skip connections
        decoded = self.decoder_1(latent_representation)
        decoded = torch.cat([decoded, skip3], dim=1)

        decoded = self.decoder_2(decoded)
        decoded = torch.cat([decoded, skip2], dim=1)

        decoded = self.decoder_3(decoded)
        decoded = torch.cat([decoded, skip1], dim=1)

        decoded = self.decoder_4(decoded)
        decoded = torch.cat([decoded, skip0], dim=1)

        decoded = self.decoder_5(decoded)

        return {"encoded": latent_representation, "decoded": decoded}
  