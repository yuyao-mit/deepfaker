# Demo_6.py
# Standard library
import os
import sys
import argparse
import struct
import time

# Lightning
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# Project files
sys.path.append(os.path.abspath("../models"))
from res_cnn_64to1024 import VRES_CNN_64to1024
from lightning_caller import Resolution_Recovery

sys.path.append(os.path.abspath("../utils"))
from paired_dataloader import Data

DATASET_DIR = "../dataset/raw"

def main(
    model_to_be_trained: L.LightningModule,
    max_epochs: int,
    ckpt_interval: int,
    ckpt_dir: str,
    num_nodes: int,
    strategy: str = "ddp"
):
    L.seed_everything(42)

    # Data initialization
    data = Data(
        root_dir=DATASET_DIR,
        brightness=1.0,
        batch_size=32,
        resolution_lr=64,
        resolution_hr=1024
    )
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch-{epoch:07d}",
        save_top_k=-1,
        every_n_epochs=ckpt_interval
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_dir,
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        num_nodes=num_nodes
    )

    # Train
    trainer.fit(
        model=model_to_be_trained,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Faker Training Initiated.")
    parser.add_argument("--max_epochs", type=int, required=True,
                        help="Number of epochs to train")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory to store checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=1000,
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to checkpoint for resuming")
    parser.add_argument("--strategy", type=str, default="ddp",
                        help="PyTorch Lightning training strategy")
    parser.add_argument("--num_nodes", type=int, required=True,
                        help="Number of nodes")
    args = parser.parse_args()

    # Restore model or create a new one
    if args.ckpt_path is not None:
        model_to_be_trained = Resolution_Recovery.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            model=VRES_CNN_64to1024(input_resolution=64)
        )
    else:
        model_to_be_trained = Resolution_Recovery(
            model=VRES_CNN_64to1024(input_resolution=64)
        )

    # Run training
    main(
        model_to_be_trained=model_to_be_trained,
        max_epochs=args.max_epochs,
        ckpt_interval=args.ckpt_interval,
        ckpt_dir=args.ckpt_dir,
        strategy=args.strategy,
        num_nodes=args.num_nodes
    )
