#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
# Standard library
import os
import struct
import time
import random

# Third-party library
import numpy as np
from math import e
import matplotlib.pyplot as plt
from tqdm import tqdm 
from tqdm.notebook import tqdm as notebook_tqdm

# Pytorch
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange, Reduce
from torch.cuda.amp import autocast, GradScaler

import torch.nn.functional as F
from einops import rearrange 
from typing import List
import math
from timm.utils import ModelEmaV3 
from collections import OrderedDict


# In[2]:


# Global Variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir='/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deep_faker/demo/checkpoint_demo/DiT/64'
dataset_dir = '/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deep_faker/dataset/raw'
batch_size = 32
mean_value = 0.23286105
std_value = 0.19771799

device


# In[3]:


def set_seed(seed: int = 42):
    """
    Sets a fixed seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# # 1. Rethinking Data Format

# In[4]:


class MultiscaleImageDataset(Dataset):
    """
    Custom dataset for loading multiscale images stored in .tiff format.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory containing the raw images (./Multiscale_Image_Dataset/raw/).
            transform (callable, optional): Optional transform to be applied on each image.
        """
        self.root_dir = root_dir
        self.transform = transform

        # List all .tiff files in the raw folder
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".tiff")
        ]

        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image

class CropBottom(object):
    """
    Custom transformation to crop the bottom 100 pixels of an image.
    """
    def __call__(self, image):
        # Convert the image to a PIL image
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Get the dimensions of the image
        width, height = image.size

        # Crop the bottom 100 pixels
        cropped_image = image.crop((0, 0, width, height - 100))

        return cropped_image






# ### 3.1.2 Train with mixed precision

# In[4]:


def train(
    dataloader,
    checkpoint_dir,
    batch_size,
    checkpoint_interval: int = 50,
    num_time_steps: int = 1000,
    num_epochs: int = 10000,
    seed: int = -1,
    ema_decay: float = 0.9999,
    lr: float = 2e-5,
    checkpoint_path: str = None,
    accumulation_steps: int = 1,
):
    """
    Train function with mixed precision (FP16) and gradient accumulation.

    Args:
        dataloader (DataLoader): PyTorch DataLoader for training data.
        checkpoint_dir (str): Directory to save checkpoints.
        batch_size (int): Batch size per step. (Make sure dataloader uses this batch size)
        checkpoint_interval (int): Save a checkpoint every N epochs.
        num_time_steps (int): Number of diffusion time steps.
        num_epochs (int): Number of training epochs.
        seed (int): Random seed. If -1, a random seed is chosen.
        ema_decay (float): EMA decay constant.
        lr (float): Learning rate for the optimizer.
        checkpoint_path (str): Optional path to a checkpoint to resume from.
        accumulation_steps (int): Number of steps to accumulate gradients before an optimizer step.

    Returns:
        float: Total training time in seconds.
    """
    # --- Setup and seeding ---
    if seed == -1:
        used_seed = random.randint(0, 2**32 - 1)
        set_seed(used_seed)
    else:
        used_seed = seed
        set_seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    train_loader = dataloader

    # --- Create scheduler, model, optimizer, EMA ---
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)
    unet_model = UNET().to(device)

    # Optional data parallel
    model = torch.nn.DataParallel(unet_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    criterion = nn.MSELoss(reduction='mean')

    # Mixed precision: create GradScaler
    scaler = GradScaler()

    # For logging losses per epoch
    epoch_losses = []

    # --- Load from checkpoint if available ---
    start_epoch = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        # If we saved "losses" before, load them
        if "losses" in checkpoint:
            epoch_losses = checkpoint["losses"]
        print(f"Loaded checkpoint from {checkpoint_path}, resuming at epoch {start_epoch+1}")

    start_time = time.time()

    # --- Training loop ---
    for epoch_idx in range(start_epoch, start_epoch + num_epochs):
        model.train()
        total_loss = 0.0

        # Zero grad at the start of each epoch
        optimizer.zero_grad(set_to_none=True)

        # Mini-batch loop
        for bidx, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{start_epoch+num_epochs}")):
            x = x.to(device)
            # Pad if needed
            x = F.pad(x, (2, 2, 2, 2))

            B = x.shape[0]
            t = torch.randint(0, num_time_steps, (B,), device=device)
            e = torch.randn_like(x, device=device)

            # Compute alpha(t) from scheduler
            _, a_cum = scheduler(t)
            a_cum = a_cum.view(B, 1, 1, 1)

            # Forward diffusion: x_t
            x_t = torch.sqrt(a_cum) * x + torch.sqrt(1 - a_cum) * e

            # --- Mixed precision forward ---
            with autocast():
                output = model(x_t, t)
                loss = criterion(output, e)

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            # Backprop
            scaler.scale(loss).backward()

            # Keep track of total loss (unscaled)
            total_loss += loss.item() * accumulation_steps

            # Update weights after "accumulation_steps" mini-batches
            if (bidx + 1) % accumulation_steps == 0 or (bidx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # EMA update
                ema.update(model)

        # Calculate epoch-wise average loss
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch_idx+1} | Loss {avg_loss:.5f}")

        # --- Save checkpoint periodically ---
        if (epoch_idx + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_idx+1}.pth")
            torch.save({
                "epoch": epoch_idx + 1,
                "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "losses": epoch_losses,
                "seed": used_seed
            }, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path}")

    # --- End of training, record total time ---
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # --- Optionally save final checkpoint ---
    final_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_idx+1}.pth")
    torch.save({
        "epoch": epoch_idx + 1,
        "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "losses": epoch_losses,
        "seed": used_seed
    }, final_ckpt_path)
    print(f"Final checkpoint saved at {final_ckpt_path}")

    return total_training_time


# ## 3.2 Train Process

# In[6]:


class SinusoidalEmbeddings(nn.Module):
    """
    Sinusoidal embeddings to encode positional information for time steps.
    """
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        self.time_steps = time_steps
        self.embed_dim = embed_dim
        self.register_buffer("embedding", self.build_embedding())

    def build_embedding(self):
        # Create sinusoidal embeddings
        positions = torch.arange(0, self.time_steps, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -torch.arange(0, self.embed_dim, 2).float()
            * (torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )
        encoding = torch.zeros((self.time_steps, self.embed_dim))
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        return encoding

    def forward(self, x, t):
        """
        x: (B, ..., H, W) any shape as long as x.shape[0] == B
        t: (B,) time steps
        """
        batch_size = x.shape[0]
        # self.embedding[t] => (B, embed_dim)
        embeddings = self.embedding[t]              # (B, embed_dim)
        embeddings = rearrange(embeddings, 'b e -> b e 1 1')  
        # (B, e) -> (B, e, 1, 1), now each time-step has a spatial embedding
        # Expand spatially to match x’s spatial dims
        embeddings = embeddings.expand(batch_size, self.embed_dim, x.size(-2), x.size(-1))
        return embeddings



class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        # embeddings: (B, embed_dim, 1, 1) => at least embed_dim >= C
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x


class Attention(nn.Module):
    """
    Multi-head attention block with explicit Q, K, V projections.
    """
    def __init__(self, C: int, num_heads:int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 1) Flatten spatial dims => (B, H*W, C)
        x_reshaped = rearrange(x, 'b c h w -> b (h w) c')  # L = H*W

        # 2) Project Q, K, V
        qkv = self.proj1(x_reshaped)                       # (B, L, 3*C)
        q, k, v = torch.split(qkv, C, dim=2)               # each (B, L, C)

        # 3) Split heads => (B, L, num_heads, C//num_heads)
        #    then rearrange to (B, num_heads, L, head_dim)
        q = rearrange(q, 'b l (heads d) -> b heads l d', heads=self.num_heads)
        k = rearrange(k, 'b l (heads d) -> b heads l d', heads=self.num_heads)
        v = rearrange(v, 'b l (heads d) -> b heads l d', heads=self.num_heads)

        # 4) Scaled dot-product attention (PyTorch 2.0+)
        attn = F.scaled_dot_product_attention(q, k, v, 
                                              is_causal=False, 
                                              dropout_p=self.dropout_prob)
        # attn shape => (B, heads, L, d)

        # 5) Merge heads back => (B, L, heads*d) = (B, L, C)
        attn = rearrange(attn, 'b heads l d -> b l (heads d)')

        # 6) Final linear projection + reshape to (B, C, H, W)
        out = self.proj2(attn)                            # (B, L, C)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out




class DDPM_Scheduler(nn.Module):
    """
    Simple scheduler for a DDPM-like approach.
    We store beta, alpha as buffers so they will be moved to 'device' along with the model.
    """
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_time_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        # Register as buffers
        self.register_buffer('beta', beta, persistent=False)
        self.register_buffer('alpha', alpha, persistent=False)
        self.register_buffer('alpha_cum', alpha_cum, persistent=False)
        self.num_time_steps = num_time_steps

    def forward(self, t: torch.Tensor):
        """
        t: (B,) time steps
        returns (beta[t], alpha_cum[t]) for each sample
        """
        t = t.long()
        return self.beta[t], self.alpha_cum[t]


class UnetLayer(nn.Module):
    """
    A single U-Net layer with optional attention and upsampling/downsampling.
    """
    def __init__(self, upscale: bool, attention: bool, num_groups: int, 
                 dropout_prob: float, C: int, num_heads: int):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(num_groups, C)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)

        # If upscale is True, this halves the channels: 384 -> 192
        self.upscale = nn.ConvTranspose2d(C, C // 2, kernel_size=2, stride=2) if upscale else None
        
        # If attention is True, create an MHA with embed_dim == C (e.g. 384)
        self.attention = nn.MultiheadAttention(C, num_heads) if attention else None

        # Projection layers for embeddings and inputs
        self.embedding_proj = None
        self.input_proj = None

    def forward(self, x, embeddings):
        # Possibly match input channels to conv1
        if self.input_proj is None and x.size(1) != self.conv1.in_channels:
            self.input_proj = nn.Conv2d(x.size(1), self.conv1.in_channels, kernel_size=1).to(x.device)
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Possibly match embedding channels to x
        if self.embedding_proj is None:
            self.embedding_proj = nn.Conv2d(embeddings.size(1), x.size(1), kernel_size=1).to(x.device)
        embeddings = self.embedding_proj(embeddings)

        # Add embeddings
        x = x + embeddings

        # Convolutional path
        x = self.relu(self.gn(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.gn(self.conv2(x)))
        residual = x  # store for skip connections

        # >>> ATTENTION BEFORE UPSAMPLING <<<
        if self.attention is not None:
            B, C, H, W = x.shape
            # Reshape to (sequence_len=L, batch_size=B, embed_dim=C)
            x_flat = rearrange(x, 'b c h w -> (h w) b c')
            x_flat, _ = self.attention(x_flat, x_flat, x_flat)
            # Reshape back to (B, C, H, W)
            x = rearrange(x_flat, '(h w) b c -> b c h w', h=H, w=W)

        # Upsample AFTER attention
        if self.upscale is not None:
            x = self.upscale(x)

        return x, residual



class UNET(nn.Module):
    """
    A U-Net style architecture that can be used for diffusion or other tasks.
    """
    def __init__(self,
                 Channels: List[int] = [64, 128, 256, 512, 512, 384],
                 Attentions: List[bool] = [False, True, False, False, False, True],
                 Upscales: List[bool] = [False, False, False, True, True, True],
                 num_groups: int = 32,
                 dropout_prob: float = 0.1,
                 num_heads: int = 8,
                 input_channels: int = 1,
                 output_channels: int = 1,
                 time_steps: int = 1000):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1] // 2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Sinusoidal embeddings
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))

        # Construct the layers
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        """
        x: (B, 1, H, W) if grayscale
        t: (B,) time-steps
        """
        x = self.shallow_conv(x)
        residuals = []
        
        # Down-sampling path
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)

        # Up-sampling path
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            out_i, _ = layer(x, embeddings)

            # Ensure matching dimensions
            r_idx = self.num_layers - i - 1
            diffY = residuals[r_idx].size(2) - out_i.size(2)
            diffX = residuals[r_idx].size(3) - out_i.size(3)
            out_i = F.pad(out_i, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            # Skip connection
            x = torch.cat((out_i, residuals[r_idx]), dim=1)

        x = self.relu(self.late_conv(x))
        return self.output_conv(x)


# In[ ]:



normalized_transform = transforms.Compose([
    transforms.Resize((1700, 1600)),
    CropBottom(),
    transforms.ToTensor(),         
    transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 3)),
    transforms.Normalize((mean_value,), (std_value,)), 
    transforms.Resize((64, 64)),  
])

normalized_dataset = MultiscaleImageDataset(root_dir=dataset_dir, transform=normalized_transform)
normalized_data_loader = DataLoader(normalized_dataset, batch_size=8, shuffle=True)


# In[8]:

checkpoint_path ="/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deep_faker/demo/checkpoint_demo/DiT/64/checkpoint_epoch_26000.pth"

model = UNET(
    Channels=[64, 128, 256, 512, 512, 384],
    Attentions=[False, True, False, False, False, True],
    Upscales=[False, False, False, True, True, True],
    num_groups=16,
    dropout_prob=0.1,
    num_heads=8,
    input_channels=1,
    output_channels=1,
    time_steps=1000
).to(device)

torch.cuda.empty_cache()

trained_model = train(
    dataloader=normalized_data_loader,
    checkpoint_dir=checkpoint_dir,
    num_epochs=10000,
    batch_size=8,
    checkpoint_path = checkpoint_path,
    lr=1e-4
)

