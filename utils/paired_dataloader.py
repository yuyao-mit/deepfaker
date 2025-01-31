# paired_dataloader.py

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



class PairedResolutionDataset(Dataset):
    """
    Loads each image twice with two different transform pipelines:
      - Low-resolution transform
      - High-resolution transform
    Returns a tuple (low_res_tensor, high_res_tensor).
    """
    def __init__(self, image_paths, transform_lr, transform_hr):
        self.image_paths = image_paths
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L") 

        lr_img = self.transform_lr(img)   # 64x64 after transform
        hr_img = self.transform_hr(img)   # 1024x1024 after transform

        return (lr_img, hr_img)



class Data(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule that returns pairs of images:
    (LR 64×64, HR 1024×1024)
    """
    def __init__(
        self,
        root_dir: str,
        brightness: float,
        batch_size: int,
        resolution_lr: int = 64,
        resolution_hr: int = 1024,
        seed: int = 42,
        num_workers: int = 8
    ):
        super().__init__()
        self.root_dir = root_dir
        self.brightness = brightness
        self.batch_size = batch_size
        self.resolution_lr = resolution_lr
        self.resolution_hr = resolution_hr
        self.seed = seed
        self.num_workers = num_workers

        # Check directory existence
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} does not exist.")

        # Gather image paths
        self.image_paths = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.lower().endswith(".tiff")
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No .tiff images found in {self.root_dir}")

        # Define transforms for low-res and high-res
        self.transform_lr = transforms.Compose([
            transforms.Resize((1700, 1600)),
            CropBottom(),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda img: transforms.functional.adjust_brightness(img, self.brightness)
            ),
            transforms.Normalize((MEAN,), (STD,)),
            transforms.Resize((self.resolution_lr, self.resolution_lr))  # => 64x64
        ])

        self.transform_hr = transforms.Compose([
            transforms.Resize((1700, 1600)),
            CropBottom(),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda img: transforms.functional.adjust_brightness(img, self.brightness)
            ),
            transforms.Normalize((MEAN,), (STD,)),
            transforms.Resize((self.resolution_hr, self.resolution_hr))  # => 1024x1024
        ])

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        generator = torch.Generator().manual_seed(self.seed)
        random.seed(self.seed)

        dataset = PairedResolutionDataset(
            self.image_paths,
            transform_lr=self.transform_lr,
            transform_hr=self.transform_hr
        )

        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

