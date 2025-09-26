import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from luadseg.embed.store import create_embeddings_file, write_embeddings_batch
from luadseg.models.foundation_models import load_model


class IndexDataset(Dataset):
    """Dataset that loads tiles from a Parquet index."""

    def __init__(self, root_dir: Path, transform=None):
        """Initialize dataset from index file."""
        self.root_dir = root_dir
        self.transform = transform

        # Load index
        self.index_df = pd.read_parquet(root_dir / "index.parquet")
        self.index_df.set_index('tile_idx', inplace=True)

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        """Load tile image."""

        tile_path = self.root_dir / self.index_df.iloc[idx]['tile_path']
        image = Image.open(tile_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, idx  # Return index for tracking


def embed_tiles(
    encoder_name: str,
    root_dir: Path,
    out_path: Path,
    weights_path: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    device: str = "cuda",
    use_amp: bool = False,
    dataset_name: Optional[str] = None,
    logger: Optional[Any] = None,
    apply_torch_scripting: bool = True,
) -> None:
    """
    Embed tiles using pretrained SSL encoder and save to HDF5.
    
    Args:
        encoder_name: Name of the encoder (e.g. "UNI2", "H-optimus-0")
        root_dir: Root directory containing the dataset
        out_path: Output HDF5 file path
        weights_path: Optional path to custom weights
        batch_size: Batch size for inference
        num_workers: Number of DataLoader workers  
        device: Device to use ("cuda" or "cpu")
        use_amp: Whether to use automatic mixed precision
        dataset_name: Dataset name for metadata
        logger: Optional logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading encoder: {encoder_name}")

    # Create encoder and preprocessing
    encoder, preprocess, embedding_dim, autocast_dtype = load_model(
        encoder_name,
        device=device,
        apply_torch_scripting=apply_torch_scripting,
    )
    # Load custom weights if provided
    if weights_path:
        logger.info(f"Loading custom weights from: {weights_path}")
        encoder.load_state_dict(torch.load(weights_path, map_location=device))

    encoder.eval()
    logger.info(f"Encoder loaded. Embedding dimension: {embedding_dim}")

    # Create output directories
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset index
    logger.info(f"Loading dataset index from: {root_dir / 'index.parquet'}")
    dataset = IndexDataset(root_dir, transform=preprocess)
    logger.info(f"Found {len(dataset)} tiles in index")

    if len(dataset) == 0:
        logger.warning("No tiles found in index")
        return

    # Initialize HDF5 file
    create_embeddings_file(
        out_path,
        encoder_metadata={
            "encoder": encoder_name,
            "encoder_weights": weights_path
        },
        dataset_name=dataset_name,
        emb_dim=embedding_dim,
    )

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
        embeddings = embeddings.cpu().numpy()
        all_embeddings.append(embeddings)
        all_indices.extend(indices.tolist())

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Generated {len(all_embeddings)} embeddings")

        # For ANORAK (no splits), store everything under 'all' split
        split_name = "all"

        # Write to HDF5
        write_embeddings_batch(
            out_path,
            split=split_name,
            embeddings=all_embeddings,
        )

        logger.info(f"Saved embeddings to: {out_path}")

        # Save embedding metadata with index mapping
        meta_path = out_path.with_suffix('.meta.json')
        metadata = {
            "encoder": encoder_name,
            "encoder_weights": weights_path,
            "dataset_name": dataset_name,
            "num_embeddings": len(all_embeddings),
            "embedding_dim": embedding_dim,
            "root_dir": str(root_dir),
            "split": split_name,
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to: {meta_path}")

    logger.info("Embedding completed successfully!")
