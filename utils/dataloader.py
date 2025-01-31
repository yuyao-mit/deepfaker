# dataloader.py

"""
Data Loading Module for Image Datasets

This module provides a PyTorch Lightning DataModule for loading and preprocessing image datasets.
It is designed to handle .tiff images, apply transformations, and split the dataset into training,
validation, and test sets. The module is modular and can be easily integrated into PyTorch Lightning
training pipelines.

Key Features:
- Custom transform pipeline including resizing, cropping, normalization, and final resizing.
- Support for reproducible dataset splitting using a random seed.
- Integration with PyTorch Lightning for seamless training, validation, and testing.
- Error handling for missing directories or empty datasets.

Classes:
1. CropBottom: A custom transform to crop the bottom 100 pixels from an image.
2. ImageDataset: A PyTorch Dataset class for loading images from a list of file paths.
3. Data: A PyTorch Lightning DataModule for managing the dataset and data loaders.

Usage:
1. Initialize the Data module:
   data = Data(root_dir="/path/to/images", brightness=1.0, batch_size=32, resolution=256, seed=42)
2. Set up the dataset:
   data.setup()
3. Access data loaders:
   train_loader = data.train_dataloader()
   val_loader = data.val_dataloader()
   test_loader = data.test_dataloader()
4. Use the data loaders in a PyTorch Lightning Trainer:
   trainer = L.Trainer(max_epochs=10, accelerator="gpu", devices=1)
   trainer.fit(model, train_loader, val_loader)

Author: Yu Yao
Date: 2025-01-30
"""

# Standard library imports
import os
import random
from typing import List, Optional, Callable

# Third-party imports
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

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
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        return image.crop((0, 0, width, height - 100))

class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading images from a list of file paths.

    Args:
        image_paths (list): List of paths to image files.
        transform (callable, optional): A transform to be applied to the images.
    """
    def __init__(self, image_paths: List[str], transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
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
        brightness (float): Brightness factor for image adjustment (1.0 = no change).
        batch_size (int): Batch size for data loaders.
        resolution (int): Target resolution for resizing images (default: 256).
        seed (int): Random seed for reproducibility (default: 42).
        transform (callable, optional): Custom transform pipeline (default: None).
    """
    def __init__(
        self,
        root_dir: str,
        brightness: float,
        batch_size: int,
        resolution: int = 256,
        seed: int = 42,
        num_workers: int = 8,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.brightness = brightness
        self.batch_size = batch_size
        self.resolution = resolution
        self.seed = seed
        self.num_workers = num_workers

        # Ensure the directory exists before collecting image paths
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")

        # Collect image paths
        self.image_paths = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.lower().endswith(".tiff")
        ]

        # Error handling for empty dataset
        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {self.root_dir}")

        # Default transform pipeline if none is provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((1700, 1600)),  # Initial resizing
            CropBottom(),                     # Crop bottom 100 pixels
            transforms.ToTensor(),            # Convert to tensor
            transforms.Lambda(
                lambda img: transforms.functional.adjust_brightness(img, self.brightness)
            ),
            transforms.Normalize((MEAN,), (STD,)),          # Normalize
            transforms.Resize((self.resolution, self.resolution)),  # Final resizing
        ])

        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset by splitting it into training, validation, and test sets.

        Args:
            stage (str, optional): Stage to set up ('fit', 'validate', 'test', etc.).
        """
        # For reproducible splitting
        generator = torch.Generator().manual_seed(self.seed)
        random.seed(self.seed)

        dataset = ImageDataset(self.image_paths, transform=self.transform)

        # Determine split sizes
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training set.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation set.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test set.

        Returns:
            DataLoader: Test DataLoader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
