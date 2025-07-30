import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from luadseg.data.data import SegDataset


def get_transforms(type="train", tile_size=256):
    """Get data augmentation transforms for training or validation."""

    if type == "train":
        size_precrop = np.sqrt(2) * tile_size
        return A.Compose(
            [
                A.RandomCrop(height=size_precrop, width=size_precrop, p=1.0, pad_if_needed=True),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.CenterCrop(height=tile_size, width=tile_size, p=1.0),
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
                A.Resize(tile_size, tile_size, p=1.0),
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


def stratified_train_val_split(train_ids, class_ratio_csv, val_split=0.2):
    df = pd.read_csv(class_ratio_csv)
    df = df[df["image_id"].isin(train_ids)]

    labels = df.set_index("image_id").loc[train_ids]["dominant_class"].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=137)
    train_idx, val_idx = next(sss.split(train_ids, labels))

    train_split = [train_ids[i] for i in train_idx]
    val_split = [train_ids[i] for i in val_idx]
    return train_split, val_split


def get_stratified_folds(image_ids, class_ratio_csv, n_splits=5, fold_idx=0):
    df = pd.read_csv(class_ratio_csv)
    df = df[df["image_id"].isin(image_ids)]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=137)
    for i, (train_idx, test_idx) in enumerate(skf.split(df["image_id"], df["dominant_class"])):
        if i == fold_idx:
            return df["image_id"].iloc[train_idx].tolist(), df["image_id"].iloc[test_idx].tolist()


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=4, tile_size=256, fold=0):
        super().__init__()
        self.root_dir = root_dir
        self.fold = fold

        self.images_dir = root_dir / "images"
        self.masks_dir = root_dir / "mask"
        self.validation_dir = root_dir / f"validation_tiles_{tile_size}"

        split_df = pd.read_csv(root_dir / "split_df.csv")
        self.split_df = split_df[split_df["fold"] == fold]
        if not self.validation_dir.exists():
            raise FileNotFoundError(
                f"Validation directory {self.validation_dir} does not exist. Please run the preprocessing script first."
            )

        self.batch_size = batch_size
        self.tile_size = tile_size

    def _get_split_ids(self):
        return (
            self.split_df[self.split_df["is_train"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_val"]]["image_id"].unique().tolist(),
            self.split_df[self.split_df["is_test"]]["image_id"].unique().tolist(),
        )

    def setup(self, stage=None):
        train_ids, val_split_ids, test_ids = self._get_split_ids()

        self.train_dataset = SegDataset(
            train_ids,
            self.images_dir,
            self.masks_dir,
            transform=get_transforms(type="train", tile_size=self.tile_size),
        )
        self.val_dataset = SegDataset(
            val_split_ids,
            self.images_dir,
            self.masks_dir,
            transform=get_transforms(type="val", tile_size=self.tile_size),
        )
        self.test_dataset = SegDataset(
            test_ids, self.images_dir, self.masks_dir, transform=get_transforms(type="test")
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        # Need bs=1 cause ROIs have different sizes
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)
