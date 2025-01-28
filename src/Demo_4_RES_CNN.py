#!/usr/bin/env python
# coding: utf-8

# Standard library
import os
import struct
import time
import random
import argparse

# Third-party library
import numpy as np
from math import e
from typing import List
import math

# Pytorch
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFilter

# Tools
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner


# Files
from models.res_cnn import RES_CNN
from utils.dataloader_1 import Data


# # UTILS
class Multiscale_Generation(L.LightningModule):
    def __init__(self,model,train_loader):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.train_loader = train_loader

    def training_step(self, batch, batch_idx):
        # batch is images
        images = batch
        outputs = self.model(images)["decoded"]
        loss = self.loss_fn(outputs, images)
        self.log("train_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.005,weight_decay=1e-9)


# In[5]:


def main_worker(
    num_nodes: int,
    ckpt_interval: int = 1000,
    ckpt_dir: str = "/work2/10214/yu_yao/frontera/deep_faker/ckpt/res_cnn/256",
    epochs: int = 1_000_000,
):
    """
    Main training entry point.

    Args:
        num_nodes (int): Number of nodes for distributed training.
        ckpt_interval (int): Save a checkpoint every N epochs.
        ckpt_dir (str): Directory to save checkpoints.
        epochs (int): Total number of epochs to train.
        batch_size (int): Batch size for training.
    """
    # 1. Initialize the data module (assuming it supports a batch_size argument)
    data = Data()
    data.setup()
    train_loader = data.train_dataloader()

    # 2. Initialize the model to be trained
    model_to_be_trained = RES_CNN()
    model = Multiscale_Generation(model=model_to_be_trained,train_loader=train_loader)

    # 3. Set up the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:02d}-step-{step:04d}",
        save_top_k=-1,
        every_n_epochs=ckpt_interval,
    )

    # 4. Create the Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_dir,
        accelerator="auto",
        devices="auto",
        strategy="ddp",  
        num_nodes=num_nodes,
    )

    # 6. Train
    trainer.fit(model, data)

##############################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RES_CNN model with Lightning.")
    parser.add_argument("--num_nodes", type=int, required=True, help="Number of nodes for DDP")
    parser.add_argument("--ckpt_interval", type=int, default=1000, help="Save a checkpoint every N epochs")
    parser.add_argument("--ckpt_dir", type=str, default="/work2/10214/yu_yao/frontera/deep_faker/ckpt/res_cnn/256/brightness_1", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=1_000_000, help="Number of epochs to train")

    args = parser.parse_args()

    main_worker(
        num_nodes=args.num_nodes,
        ckpt_interval=args.ckpt_interval,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
    )
