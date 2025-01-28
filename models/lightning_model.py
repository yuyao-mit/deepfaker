# lightning_model.py

import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F

class Multiscale_Generation(L.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self,x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # batch is images
        images = batch
        outputs = self.model(images)["decoded"]
        loss = self.loss_fn(outputs, images)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # batch is images
        images = batch
        outputs = self.model(images)["decoded"]
        loss = self.loss_fn(outputs, images)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch is images
        images = batch
        outputs = self.model(images)["decoded"]
        loss = self.loss_fn(outputs, images)
        self.log("val_loss", loss)
        return loss  

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.005,weight_decay=1e-9)