# dataloader.py

"""
Data Loading Module for Image Datasets

This module provides a PyTorch Lightning DataModule for loading and preprocessing image datasets.
It is designed to handle .tiff images, apply transformations, and split the dataset into training
and validation sets. The module is modular and can be easily integrated into PyTorch Lightning
training pipelines.

Key Features:
- Custom transform pipeline including resizing, cropping, normalization, and final resizing.
- Support for reproducible dataset splitting using a random seed.
- Integration with PyTorch Lightning for seamless training and validation.
- Error handling for missing directories or empty datasets.

Classes:
1. CropBottom: A custom transform to crop the bottom 100 pixels from an image.
2. ImageDataset: A PyTorch Dataset class for loading images from a list of file paths.
3. Data: A PyTorch Lightning DataModule for managing the dataset and data loaders.

Usage:
1. Initialize the Data module:
   data = Data(root_dir="/path/to/images", batch_size=32, resolution=256, seed=42)
2. Set up the dataset:
   data.setup()
3. Access data loaders:
   train_loader = data.train_dataloader()
   val_loader = data.val_dataloader()
4. Use the data loaders in a PyTorch Lightning Trainer:
   trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
   trainer.fit(model, train_loader, val_loader)

Author: Yu Yao
Date: 2025-01-22
"""

# Standard library imports
import os
import random

# Third-party imports
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import lightning as L
import torch

# Constants for normalization
MEAN = 0.23286105
STD = 0.19771799

class CropBottom:
    """
    A custom transform to crop the bottom 100 pixels from an image.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The cropped image.
    """
    def __call__(self, image):
        width, height = image.size
        return image.crop((0, 0, width, height - 100))


class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading images from a list of file paths.

    Args:
        image_paths (list): List of paths to image files.
        transform (callable, optional): A transform to be applied to the images.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image


class Data(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing image datasets.

    Args:
        root_dir (str): Path to the directory containing the images.
        batch_size (int): Batch size for data loaders (default: 32).
        resolution (int): Target resolution for resizing images (default: 256).
        seed (int): Random seed for reproducibility (default: 42).
        transform (callable, optional): Custom transform pipeline (default: None).
    """
    def __init__(self, root_dir, brightness, batch_size=32, resolution=256, seed=42, transform=None):
        super(Data, self).__init__()
        self.root_dir = root_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.seed = seed
        self.brightness = brightness

        # Collect image paths
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".tiff")
        ]

        # Error handling for missing directory or empty dataset
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")
        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {root_dir}")

        # Default transform pipeline
        self.transform = transform or transforms.Compose([
            transforms.Resize((1700, 1600)),  # Initial resizing
            CropBottom(),  # Crop bottom 100 pixels
            transforms.ToTensor(),  # Convert to tensor
            transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, self.brightness)),
            transforms.Normalize((MEAN,), (STD,)),  # Normalize
            transforms.Resize((self.resolution, self.resolution)),  # Final resizing
        ])

    def setup(self, stage=None):
        """
        Set up the dataset by splitting it into training and validation sets.

        Args:
            stage (str, optional): Current stage (e.g., 'fit', 'test'). Defaults to None.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        dataset = ImageDataset(self.image_paths, transform=self.transform)
        train_size = int(0.9 * len(dataset))  # 90% training, 10% validation
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        """
        Create a DataLoader for the training set.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation set.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)