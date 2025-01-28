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

# Pytorch
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFilter


# Global Variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir='./checkpoint_demo/256/latent_dim_4096'
dataset_dir = "../dataset/raw/"
batch_size = 32


# # 1. Load Data


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


# In[39]:

bright_transform = transforms.Compose([
    transforms.Resize((1700, 1600)),  # Ensure initial size (1700, 1600)
    CropBottom(),                 # Crop bottom 100 pixels
    transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 3)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((128, 128)),
])

dataset = MultiscaleImageDataset(root_dir=dataset_dir, transform=bright_transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)



# # 2. Model Demo

# # 2.1 Autoencoder

# In[47]:


class Autoencoder(nn.Module):
    def __init__(self, input_dim=128):
        """
        Simple autoencoder with fully connected layers.
        
        Args:
            input_dim (int): Flattened input size (default is for 64x64 grayscale images).
            latent_dim (int): Dimensionality of the latent space.
        """
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim*input_dim, int(input_dim*input_dim*2)),
            nn.ReLU(),
            nn.Linear(int(input_dim*input_dim*2), int(input_dim*input_dim/8)),
            nn.ReLU(),
            nn.Linear(int(input_dim*input_dim/8), int(input_dim*input_dim/16))                        
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(int(input_dim*input_dim/16), int(input_dim*input_dim/8)),
            nn.ReLU(),
            nn.Linear(int(input_dim*input_dim/8), int(input_dim*input_dim/4)),
            nn.ReLU(),
            nn.Linear(int(input_dim*input_dim/4), int(input_dim*input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim*input_dim/2), int(input_dim*input_dim)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, width, height].

        Returns:
            dict: Encoded latent representation and reconstructed image.
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        encoded = self.encoder(x)
        encoded = encoded + torch.randn_like(encoded) * 0.1
        decoded = self.decoder(encoded)        
        decoded = decoded.view(batch_size, 1, self.input_dim, self.input_dim)  
        return {"encoded": encoded, "decoded": decoded}


# # 3. Train

# In[48]:

def train_deepfaker(model, train_loader, num_epochs=1500, criterion=None, batch_size=32, device=None, checkpoint_interval=50, checkpoint_dir='./checkpoint_demo'):
    """
    Trains the DeepFaker model and saves checkpoints at specified intervals with multi-GPU support.

    Args:
        model (torch.nn.Module): The DeepFaker model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        num_epochs (int): Number of epochs to train. Default is 100.
        criterion (callable): Loss function. Default is MSELoss().
        batch_size (int): Batch size for training. Default is 32.
        device (torch.device): Device for training (e.g., 'cpu' or 'cuda'). Default is 'cuda' if available.
        checkpoint_interval (int): Save a checkpoint every `checkpoint_interval` epochs. Default is 10.

    Returns:
        float: Total training time in seconds.
    """
    # DEFAULT SETTINGS AND REGULAR CHECK
    if criterion is None:
        criterion = torch.nn.MSELoss()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Multi-GPU
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    epoch_losses = []

    # Start training
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        # Train loop without tqdm
        for images in train_loader:
            images = images.to(device)

            # Forward
            outputs = model(images)["decoded"]
            loss = criterion(outputs, images)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": epoch_losses,
                "seed":42,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    return total_training_time


#################################################################################################################################################

'''   
def main():
    # Initialize the model
    deep_faker = Autoencoder()

    # Train the model
    total_training_time = train_deepfaker(
        model=deep_faker, 
        train_loader=data_loader, 
        num_epochs=1800,          
        criterion=torch.nn.MSELoss(),
        batch_size=32,       
        device=device,
        checkpoint_interval=50
    )

    # Print training completion time
    print(f"Training completed in {total_training_time:.2f} seconds")

if __name__ == "__main__":
    main()
'''

def main(input_dim):
    # Initialize the model with input_dim
    torch.cuda.empty_cache()

    deep_faker = Autoencoder(input_dim=input_dim)

    # Train the model
    total_training_time = train_deepfaker(
        model=deep_faker, 
        train_loader=data_loader, 
        num_epochs=10000,          
        criterion=torch.nn.MSELoss(),
        batch_size=32,       
        device=device,
        checkpoint_interval=100
    )

    # Print training completion time
    print(f"Training completed in {total_training_time:.2f} seconds")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the DeepFaker model.")
    parser.add_argument("--input_dim", type=int, required=True, help="The input dimension for the Autoencoder.")
    args = parser.parse_args()

    # Call main with the input_dim argument
    main(input_dim=args.input_dim)




