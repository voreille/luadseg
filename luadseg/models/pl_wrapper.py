import math

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
    def __init__(self, model, lr=1e-4, num_classes=7, tile_size=256):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        scores = self.dice_score(logits, y)
        for i, score in enumerate(scores):
            self.log(f"val_dice_class_{i}", score, prog_bar=(i != 0))

        return loss

    def dice_score(preds, targets, eps=1e-6):
        num_classes = preds.shape[1]
        preds = preds.argmax(dim=1)

        scores = []
        for c in range(num_classes):
            pred_c = (preds == c).float()
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * intersection + eps) / (union + eps)
            scores.append(dice.item())
        return scores

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = []
        for i in range(len(x)):
            tiles, image_infos = self.tile_image(x[i])
            tile_preds = self(tiles)
            tile_preds = F.softmax(tile_preds, dim=1)
            pred = self.aggregate_tile_preds(tile_preds, image_infos)
            preds.append(pred)

        preds = torch.stack(preds)
        dice = self.compute_dice(preds, y)
        return dice

    @torch.no_grad()
    def aggregate_tile_preds(self, preds, image_infos):
        """
        Reconstruct full-size prediction from overlapping tile predictions.

        Args:
            preds (Tensor): (n_tiles, num_classes, tile_size, tile_size)
            spatial_shape (tuple): (H, W)

        Returns:
            Tensor: (num_classes, H, W)
        """

        h = image_infos["height"]
        w = image_infos["width"]
        stride_x = image_infos["stride_x"]
        stride_y = image_infos["stride_y"]
        n_tiles_x = image_infos["n_tiles_x"]
        n_tiles_y = image_infos["n_tiles_y"]

        full_probs = torch.zeros(self.num_classes, h, w, device=preds.device)
        norm_map = torch.zeros(self.num_classes, h, w, device=preds.device)

        idx = 0
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                y0 = i * stride_y
                x0 = j * stride_x
                y1 = min(y0 + self.tile_size, h)
                x1 = min(x0 + self.tile_size, w)

                pred_tile = preds[idx, :, : y1 - y0, : x1 - x0]
                full_probs[:, y0:y1, x0:x1] += pred_tile
                norm_map[:, y0:y1, x0:x1] += 1
                idx += 1

        full_probs /= norm_map.clamp(min=1e-5)
        return full_probs

    @torch.no_grad()
    def calculate_stride(self, image_dim, minimum_overlap=0.5):
        """Calculate stride so that tiles of size tile_size cover the image with at least minimum overlap."""
        if image_dim <= self.tile_size:
            raise ValueError("Image dimension must be larger than tile size")

        # Compute maximum allowed stride
        max_stride = int(self.tile_size * (1 - minimum_overlap))
        if max_stride < 1:
            raise ValueError("Minimum overlap too high; stride becomes zero or negative.")

        # Compute number of steps needed to cover the image
        n_tiles = math.ceil((image_dim - self.tile_size) / max_stride) + 1

        # Distribute tiles symmetrically across the image
        if n_tiles == 1:
            return 0

        # Adjust stride so that tiles span the full image
        stride = (image_dim - self.tile_size) // (n_tiles - 1)

        return int(stride)

    

    @torch.no_grad()
    def tile_image(self, x, minimum_overlap=0.5):
        """
        Tile the input image tensor into patches of shape (tile_size, tile_size).

        Args:
            x (torch.Tensor): Input image of shape (C, H, W).
            tile_size (int): Size of the square tile.
            minimum_overlap (float): Minimum overlap ratio between tiles.

        Returns:
            torch.Tensor: A tensor of shape (n_tiles, C, tile_size, tile_size).
        """
        C, H, W = x.shape
        # Compute strides for height and width
        stride_y = self.calculate_stride(H, minimum_overlap=minimum_overlap)
        stride_x = self.calculate_stride(W, minimum_overlap=minimum_overlap)

        # Use tensor.unfold to extract sliding windows
        # For the height dimension (dim=1) and width dimension (dim=2)
        tiles = x.unfold(1, self.tile_size, stride_y).unfold(2, self.tile_size, stride_x)
        # tiles shape: (C, n_tiles_y, n_tiles_x, tile_size, tile_size)

        # Rearrange dimensions so that each tile is a separate sample
        tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
        # New shape: (n_tiles_y, n_tiles_x, C, tile_size, tile_size)
        n_tiles_y, n_tiles_x, C, _, _ = tiles.shape

        # Flatten the first two dimensions to get a list of tiles
        tiles = tiles.view(-1, C, self.tile_size, self.tile_size)

        return tiles, {
            "height": H,
            "width": W,
            "stride_x": stride_x,
            "stride_y": stride_y,
            "n_tiles_x": n_tiles_x,
            "n_tiles_y": n_tiles_y,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
