import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


def compute_class_weights(csv_path, num_classes=7, ignore_index=0):
    df = pd.read_csv(csv_path)
    counts = df.iloc[:, 1:].sum().values  # assuming columns 1..7 are class frequencies
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum()  # normalize
    class_weights = torch.tensor(weights, dtype=torch.float32)
    class_weights[ignore_index] = 0.0  # background ignored
    return class_weights


class PatSegLightningModule(pl.LightningModule):
    def __init__(self, model, lr=1e-4, num_classes=7):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
