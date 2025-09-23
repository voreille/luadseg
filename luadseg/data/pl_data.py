import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from luadseg.data.data import SegDataset


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=4, num_workers=4, tile_size=256, fold=0):
        super().__init__()
        self.root_dir = root_dir
        self.fold = fold

        self.images_dir = root_dir / "images"
        self.masks_dir = root_dir / "mask"
        self.validation_root_dir = root_dir / f"validation_tiles_{tile_size}"
        self.validation_image_dir = self.validation_root_dir / "image"
        self.validation_mask_dir = self.validation_root_dir / "mask"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]
        if not self.validation_root_dir.exists():
            raise FileNotFoundError(
                f"Validation directory {self.validation_root_dir} does not exist. Please run the preprocessing script first."
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tile_size = tile_size

    def _get_split_ids(self):
        return (
            self.split_df[self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]["image_id"].unique().tolist(),
        )

    def setup(self, stage=None):
        train_ids, val_ids, test_ids = self._get_split_ids()
        if stage == "fit" or stage is None:
            val_tile_ids = [
                f.stem
                for f in (self.validation_image_dir).glob("*.png")
                if f.stem.split("_tile")[0] in val_ids
            ]

            self.train_dataset = SegDataset(
                train_ids,
                self.images_dir,
                self.masks_dir,
                transform=self.get_transforms(type="train"),
            )
            self.val_dataset = SegDataset(
                val_tile_ids,
                self.validation_image_dir,
                self.validation_mask_dir,
                transform=self.get_transforms(type="val"),
            )
        if stage == "test" or stage is None:
            self.test_dataset = SegDataset(
                test_ids,
                self.images_dir,
                self.masks_dir,
                transform=self.get_transforms(type="test"),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True, 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        # Need bs=1 cause ROIs have different sizes
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)

    def predict_dataloader(self):
        return self.test_dataloader()

    def get_transforms(self, type="train"):
        """Get data augmentation transforms for training or validation."""

        if type == "train":
            size_precrop = np.sqrt(2) * self.tile_size
            return A.Compose(
                [
                    A.RandomCrop(
                        height=size_precrop, width=size_precrop, p=1.0, pad_if_needed=True
                    ),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                    A.CenterCrop(height=self.tile_size, width=self.tile_size, p=1.0),
                    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                strict=True,
                seed=137,
            )
        elif type == "val":
            return A.Compose(
                [
                    A.Resize(self.tile_size, self.tile_size, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                strict=True,
                seed=137,
            )

        elif type == "test":
            return A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                strict=True,
                seed=137,
            )
