from pathlib import Path

import cv2
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self, image_ids, images_directory, masks_directory, transform=None):
        self.image_ids = image_ids
        self.images_directory = Path(images_directory)
        self.masks_directory = Path(masks_directory)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = cv2.imread(self.images_directory / f"{image_id}.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            self.masks_directory / f"{image_id}.png",
            cv2.IMREAD_UNCHANGED,
        )
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
