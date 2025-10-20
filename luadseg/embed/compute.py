import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class IndexDataset(Dataset):
    """Dataset that loads tiles from a Parquet index."""

    def __init__(self,
                 root_dir: Path,
                 transform=None,
                 target_stain_img: Optional[Path] = None):
        """Initialize dataset from index file."""
        self.root_dir = root_dir
        self.transform = transform

        if target_stain_img is not None:
            self.target_stain_img = target_stain_img.resolve()
            self.stain_fn = self._get_stain_fn()
        else:
            self.target_stain_img = None
            self.stain_fn = None

        # Load index
        self.index_df = pd.read_parquet(root_dir / "index.parquet")

    def _get_stain_fn(self):
        import cv2
        import torchstain
        from torchvision import transforms

        target = cv2.cvtColor(
            cv2.imread(self.target_stain_img),  # type: ignore
            cv2.COLOR_BGR2RGB)

        T = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x * 255)])

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        normalizer.fit(T(target))

        def normalize_fn(img):
            t_to_transform = T(img)
            norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
            return Image.fromarray(norm.numpy().astype(np.uint8))

        return normalize_fn

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        """Load tile image."""

        tile_path = self.root_dir / self.index_df.iloc[idx]['tile_path']
        tile_idx = self.index_df.iloc[idx]['tile_idx']
        image = Image.open(tile_path).convert("RGB")
        if self.stain_fn:
            image = self.stain_fn(image)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, tile_idx


def embed_tiles(
    encoder: nn.Module,
    preprocess: Any,
    root_dir: Path,
    out_path: Path,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
    autocast_dtype: Optional[torch.dtype] = None,
    dataset_name: Optional[str] = None,
    logger: Optional[Any] = None,
    encoder_metadata: Optional[dict] = None,
    embedding_dim: int = 1536,
    target_stain_img: Optional[Path] = None,
) -> None:
    """
    Embed tiles using pretrained SSL encoder and save to HDF5.
    
    Args:
        encoder: Pretrained model to use as encoder.
        preprocess: Preprocessing function for input images.
        root_dir: Directory with 'index.parquet' and tile images.
        out_path: Output PT file path for embeddings.
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for DataLoader.
        device: Device to run the model on ('cuda' or 'cpu').
        autocast_dtype: Dtype for automatic mixed precision (e.g., torch.float16).
        dataset_name: Optional name of the dataset to store in metadata.
        logger: Optional logger for logging progress.
        encoder_metadata: Optional metadata dictionary about the encoder.
        embedding_dim: Dimensionality of the output embeddings.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if encoder_metadata is None:
        encoder_metadata = {}

    logger.info(f"Loading encoder: {encoder_metadata.get('id', 'unknown')}")

    encoder.eval()

    # Create output directories
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset index
    logger.info(f"Loading dataset index from: {root_dir / 'index.parquet'}")
    dataset = IndexDataset(
        root_dir,
        transform=preprocess,
        target_stain_img=target_stain_img,
    )
    logger.info(f"Found {len(dataset)} tiles in index")

    if len(dataset) == 0:
        logger.warning("No tiles found in index")
        return

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for indexing
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
    )

    # Process all tiles
    all_embeddings = []
    all_indices = []

    logger.info("Starting embedding computation...")

    use_amp = autocast_dtype is not None

    for batch_idx, (images, indices) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        # Forward pass with optional AMP
        with torch.inference_mode():
            if use_amp and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    embeddings = encoder(images)
            else:
                embeddings = encoder(images)

        # Move to CPU and store
        embeddings = embeddings.cpu()
        all_embeddings.append(embeddings)
        all_indices.extend(indices.tolist())

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indices = torch.tensor(all_indices, dtype=torch.int32)
        order = torch.argsort(all_indices)
        all_embeddings = all_embeddings[order]
        all_indices = all_indices[order]

        torch.save(
            {
                "embeddings": all_embeddings,
                "tile_idx": all_indices,
                "encoder": encoder_metadata,
                "dataset_name": dataset_name,
                "root_dir": str(root_dir),
            },
            out_path,
        )

        logger.info(f"Saved {len(all_embeddings)} embeddings to {out_path}")
    else:
        logger.warning("No embeddings were generated.")

    logger.info("Embedding completed successfully!")
