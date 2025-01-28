# evaluation.py
import os
import struct
import time
import random
import argparse
import sys
import torch
import matplotlib.pyplot as plt
from dataloader import Data
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eval_reconstruction(
    model, 
    brightness,
    device,
    dataset_dir="/home/gridsan/yyao/Research_Projects/Microstructure_Enough/deep_faker/dataset/raw",
    batch_size=32, 
    resolution=256, 
    seed=42
):
    
    # Prepare the data
    data = Data(
        root_dir=dataset_dir,
        brightness=brightness,
        batch_size=batch_size,
        resolution=resolution,
        seed=seed
    )
    data.setup()
    test_loader = data.val_dataloader()
    test_images = next(iter(test_loader)).to(device)
    
    # Run the model inference
    with torch.no_grad():
        output = model(test_images)
        if isinstance(output, dict):
            output = output["decoded"]

    test_images = test_images.detach().to(device)
    test_images = test_images.squeeze(1)

    output = output.detach().to(device)
    output = output.squeeze(1)
    
    # Plot original and reconstructed images
    rows, cols = 8, 4
    fig, axes = plt.subplots(rows, cols * 2, figsize=(24, 24))

    for i in range(rows * cols):
        ax_original = axes[i // cols, (i % cols) * 2]
        ax_original.imshow(test_images[i].cpu().numpy(), cmap="gray")
        ax_original.axis("off")
        ax_original.set_title(f"Original {i + 1}", fontsize=10)

        ax_reconstructed = axes[i // cols, (i % cols) * 2 + 1]
        ax_reconstructed.imshow(output[i].cpu().numpy(), cmap="gray")
        ax_reconstructed.axis("off")
        ax_reconstructed.set_title(f"Reconstructed {i + 1}", fontsize=10)

    plt.tight_layout()
    plt.show()


def loss_visualization(start_epoch, epoch_interval, metrics_path):
    # Load the metrics CSV file
    df = pd.read_csv(metrics_path)

    # Convert the relevant columns to numeric, handling any invalid values
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df['train_loss'] = pd.to_numeric(df['train_loss'], errors='coerce')

    # Filter rows where the epoch matches the specified interval and start epoch
    df = df[df['epoch'] >= start_epoch]
    df = df[(df['epoch'] - start_epoch) % epoch_interval == 0]

    # Plot the train loss over epochs using seaborn for a cleaner plot
    sns.lineplot(data=df, x='epoch', y='train_loss')
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.show()
