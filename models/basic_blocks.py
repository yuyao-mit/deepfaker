# basic_blocks.py

import torch
from torch import nn
from torch.nn.utils import spectral_norm

################################################################################################
# DOWNSAMPING
################################################################################################

def conv8x8(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 8,
    stride: int = 8,
    groups: int = 1,
    padding: int = 1
) -> nn.Sequential:
    """
    Returns an 8x8 convolutional layer with spectral norm, BatchNorm2d, and SiLU activation.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H//8, W//8]
    """
    conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(conv.weight, mean=0.0, std=1.0)
    spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channel),
        nn.SiLU(inplace=True)
    )

def conv4x4(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 4,
    stride: int = 4,
    groups: int = 1,
    padding: int = 0
) -> nn.Sequential:
    """
    Returns a 4x4 convolutional layer with spectral norm, BatchNorm2d, and SiLU activation.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H//4, W//4]
    """
    conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(conv.weight, mean=0.0, std=1.0)
    spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channel),
        nn.SiLU(inplace=True)
    )

def conv3x3(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    padding: int = 1
) -> nn.Sequential:
    """
    Returns a 3x3 convolutional layer with spectral norm, BatchNorm2d, and SiLU activation.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H, W]
    """
    conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(conv.weight, mean=0.0, std=1.0)
    spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channel),
        nn.SiLU(inplace=True)
    )

def conv2x2(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 2,
    stride: int = 2,
    groups: int = 1,
    padding: int = 0
) -> nn.Sequential:
    """
    Returns a 2x2 convolutional layer with spectral norm, BatchNorm2d, and SiLU activation.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H//2, W//2]
    """
    conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(conv.weight, mean=0.0, std=1.0)
    spectral_norm(conv)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channel),
        nn.SiLU(inplace=True)
    )


################################################################################################
# UPSAMPING
################################################################################################

def convT8x8(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 8,
    stride: int = 8,
    groups: int = 1,
    padding: int = 1,
    output_padding: int = 0
) -> nn.Sequential:
    """
    TransposedConv 8x8 with pre-BatchNorm2d, spectral norm, and SiLU.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H*8, W*8]
    """
    convtrans = nn.ConvTranspose2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(convtrans.weight, mean=0.0, std=1.0)
    spectral_norm(convtrans)
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        convtrans,
        nn.SiLU(inplace=True)
    )

def convT4x4(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 4,
    stride: int = 4,
    groups: int = 1,
    padding: int = 0,
    output_padding: int = 0
) -> nn.Sequential:
    """
    TransposedConv 4x4 with pre-BatchNorm2d, spectral norm, and SiLU.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H*4, W*4]
    """
    convtrans = nn.ConvTranspose2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(convtrans.weight, mean=0.0, std=1.0)
    spectral_norm(convtrans)
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        convtrans,
        nn.SiLU(inplace=True)
    )

def convT3x3(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    padding: int = 1,
    output_padding: int = 0
) -> nn.Sequential:
    """
    TransposedConv 3x3 with pre-BatchNorm2d, spectral norm, and SiLU.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H, W]
    """
    convtrans = nn.ConvTranspose2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(convtrans.weight, mean=0.0, std=1.0)
    spectral_norm(convtrans)
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        convtrans,
        nn.SiLU(inplace=True)
    )

def convT2x2(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 2,
    stride: int = 2,
    groups: int = 1,
    padding: int = 0,
    output_padding: int = 0
) -> nn.Sequential:
    """
    TransposedConv 2x2 with pre-BatchNorm2d, spectral norm, and SiLU.
    # Input -> Output: [B, in_channel, H, W] -> [B, out_channel, H*2, W*2]
    """
    convtrans = nn.ConvTranspose2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=1
    )
    nn.init.normal_(convtrans.weight, mean=0.0, std=1.0)
    spectral_norm(convtrans)
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        convtrans,
        nn.SiLU(inplace=True)
    )


################################################################################################
# FINAL ADJUSTMENTS
################################################################################################

def conv127x127_reflect(
    in_channel: int,
    out_channel: int,
    kernel_size: int = 127,
    stride: int = 1,
    groups: int = 1
) -> nn.Sequential:
    """
    128x128 convolution with reflection padding, spectral norm, pre-BatchNorm2d, and Tanh.
    """
    conv = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,  
        groups=groups,
        bias=False,
        dilation=1
    )
    
    nn.init.normal_(conv.weight, mean=0.0, std=1.0)
    spectral_norm(conv)
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReflectionPad2d(63),
        conv,
        nn.Tanh()
    )
